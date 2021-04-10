# Copyright 2020 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Model function definition, including both architecture and loss."""
import functools
import re
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
import coco_metric
import efficientdet_arch
import hparams_config
import nms_np
import utils
from keras import anchors
from keras import efficientdet_keras
from keras import postprocess

_DEFAULT_BATCH_SIZE = 64


def update_learning_rate_schedule_parameters(params):
  """Updates params that are related to the learning rate schedule."""
  # params['batch_size'] is per-shard within model_fn if strategy=tpu.
  batch_size = (
      params['batch_size'] * params['num_shards']
      if params['strategy'] in ['tpu', 'gpus'] else params['batch_size'])
  # Learning rate is proportional to the batch size
  params['adjusted_learning_rate'] = (
      params['learning_rate'] * batch_size / _DEFAULT_BATCH_SIZE)

  if 'lr_warmup_init' in params:
    params['adjusted_lr_warmup_init'] = (
        params['lr_warmup_init'] * batch_size / _DEFAULT_BATCH_SIZE)

  steps_per_epoch = params['num_examples_per_epoch'] / batch_size
  params['lr_warmup_step'] = int(params['lr_warmup_epoch'] * steps_per_epoch)
  params['first_lr_drop_step'] = int(params['first_lr_drop_epoch'] *
                                     steps_per_epoch)
  params['second_lr_drop_step'] = int(params['second_lr_drop_epoch'] *
                                      steps_per_epoch)
  params['total_steps'] = int(params['num_epochs'] * steps_per_epoch)
  params['steps_per_epoch'] = steps_per_epoch


def stepwise_lr_schedule(adjusted_learning_rate, adjusted_lr_warmup_init,
                         lr_warmup_step, first_lr_drop_step,
                         second_lr_drop_step, global_step):
  """Handles linear scaling rule, gradual warmup, and LR decay."""
  # adjusted_lr_warmup_init is the starting learning rate; LR is linearly
  # scaled up to the full learning rate after `lr_warmup_step` before decaying.
  logging.info('LR schedule method: stepwise')
  linear_warmup = (
      adjusted_lr_warmup_init +
      (tf.cast(global_step, dtype=tf.float32) / lr_warmup_step *
       (adjusted_learning_rate - adjusted_lr_warmup_init)))
  learning_rate = tf.where(global_step < lr_warmup_step, linear_warmup,
                           adjusted_learning_rate)
  lr_schedule = [[1.0, lr_warmup_step], [0.1, first_lr_drop_step],
                 [0.01, second_lr_drop_step]]
  for mult, start_global_step in lr_schedule:
    learning_rate = tf.where(global_step < start_global_step, learning_rate,
                             adjusted_learning_rate * mult)
  return learning_rate


def cosine_lr_schedule(adjusted_lr, adjusted_lr_warmup_init, lr_warmup_step,
                       total_steps, step):
  """Cosine learning rate scahedule."""
  logging.info('LR schedule method: cosine')
  linear_warmup = (
      adjusted_lr_warmup_init +
      (tf.cast(step, dtype=tf.float32) / lr_warmup_step *
       (adjusted_lr - adjusted_lr_warmup_init)))
  decay_steps = tf.cast(total_steps - lr_warmup_step, tf.float32)
  cosine_lr = 0.5 * adjusted_lr * (
      1 + tf.cos(np.pi * tf.cast(step, tf.float32) / decay_steps))
  return tf.where(step < lr_warmup_step, linear_warmup, cosine_lr)


def polynomial_lr_schedule(adjusted_lr, adjusted_lr_warmup_init, lr_warmup_step,
                           power, total_steps, step):
  logging.info('LR schedule method: polynomial')
  linear_warmup = (
      adjusted_lr_warmup_init +
      (tf.cast(step, dtype=tf.float32) / lr_warmup_step *
       (adjusted_lr - adjusted_lr_warmup_init)))
  polynomial_lr = adjusted_lr * tf.pow(
      1 - (tf.cast(step, tf.float32) / total_steps), power)
  return tf.where(step < lr_warmup_step, linear_warmup, polynomial_lr)


def learning_rate_schedule(params, global_step):
  """Learning rate schedule based on global step."""
  lr_decay_method = params['lr_decay_method']
  if lr_decay_method == 'stepwise':
    return stepwise_lr_schedule(params['adjusted_learning_rate'],
                                params['adjusted_lr_warmup_init'],
                                params['lr_warmup_step'],
                                params['first_lr_drop_step'],
                                params['second_lr_drop_step'], global_step)

  if lr_decay_method == 'cosine':
    return cosine_lr_schedule(params['adjusted_learning_rate'],
                              params['adjusted_lr_warmup_init'],
                              params['lr_warmup_step'], params['total_steps'],
                              global_step)

  if lr_decay_method == 'polynomial':
    return polynomial_lr_schedule(params['adjusted_learning_rate'],
                                  params['adjusted_lr_warmup_init'],
                                  params['lr_warmup_step'],
                                  params['poly_lr_power'],
                                  params['total_steps'], global_step)

  if lr_decay_method == 'constant':
    return params['adjusted_learning_rate']

  raise ValueError('unknown lr_decay_method: {}'.format(lr_decay_method))


def focal_loss(y_pred, y_true, alpha, gamma, normalizer, label_smoothing=0.0):
  """Compute the focal loss between `logits` and the golden `target` values.

  Focal loss = -(1-pt)^gamma * log(pt)
  where pt is the probability of being classified to the true class.

  Args:
    y_pred: A float tensor of size [batch, height_in, width_in,
      num_predictions].
    y_true: A float tensor of size [batch, height_in, width_in,
      num_predictions].
    alpha: A float scalar multiplying alpha to the loss from positive examples
      and (1-alpha) to the loss from negative examples.
    gamma: A float scalar modulating loss from hard and easy examples.
    normalizer: Divide loss by this value.
    label_smoothing: Float in [0, 1]. If > `0` then smooth the labels.

  Returns:
    loss: A float32 scalar representing normalized total loss.
  """
  with tf.name_scope('focal_loss'):
    normalizer = tf.cast(normalizer, dtype=y_pred.dtype)

    # compute focal loss multipliers before label smoothing, such that it will
    # not blow up the loss.
    pred_prob = tf.sigmoid(y_pred)
    p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
    alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
    modulating_factor = (1.0 - p_t)**gamma

    # apply label smoothing for cross_entropy for each entry.
    if label_smoothing:
      y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing
    ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)

    # compute the final loss and return
    return (1 / normalizer) * alpha_factor * modulating_factor * ce


def _box_loss(box_outputs, box_targets, num_positives, delta=0.1):
  """Computes box regression loss."""
  # delta is typically around the mean value of regression target.
  # for instances, the regression targets of 512x512 input with 6 anchors on
  # P3-P7 pyramid is about [0.1, 0.1, 0.2, 0.2].
  normalizer = num_positives * 4.0
  mask = tf.not_equal(box_targets, 0.0)
  box_loss = tf.losses.huber_loss(
      box_targets,
      box_outputs,
      weights=mask,
      delta=delta,
      reduction=tf.losses.Reduction.SUM)
  box_loss /= normalizer
  return box_loss


def detection_loss(cls_outputs, box_outputs, labels, params):
  """Computes total detection loss.

  Computes total detection loss including box and class loss from all levels.
  Args:
    cls_outputs: an OrderDict with keys representing levels and values
      representing logits in [batch_size, height, width, num_anchors].
    box_outputs: an OrderDict with keys representing levels and values
      representing box regression targets in [batch_size, height, width,
      num_anchors * 4].
    labels: the dictionary that returned from dataloader that includes
      groundtruth targets.
    params: the dictionary including training parameters specified in
      default_haprams function in this file.

  Returns:
    total_loss: an integer tensor representing total loss reducing from
      class and box losses from all levels.
    cls_loss: an integer tensor representing total class loss.
    box_loss: an integer tensor representing total box regression loss.
  """
  # Sum all positives in a batch for normalization and avoid zero
  # num_positives_sum, which would lead to inf loss during training
  num_positives_sum = tf.reduce_sum(labels['mean_num_positives']) + 1.0
  positives_momentum = params.get('positives_momentum', None) or 0
  if positives_momentum > 0:
    # normalize the num_positive_examples for training stability.
    moving_normalizer_var = tf.Variable(
        0.0,
        name='moving_normalizer',
        dtype=tf.float32,
        synchronization=tf.VariableSynchronization.ON_READ,
        trainable=False,
        aggregation=tf.VariableAggregation.MEAN)
    num_positives_sum = tf.keras.backend.moving_average_update(
        moving_normalizer_var,
        num_positives_sum,
        momentum=params['positives_momentum'])
  elif positives_momentum < 0:
    num_positives_sum = utils.cross_replica_mean(num_positives_sum)

  levels = cls_outputs.keys()
  cls_losses = []
  box_losses = []
  for level in levels:
    # Onehot encoding for classification labels.
    cls_targets_at_level = tf.one_hot(
        labels['cls_targets_%d' % level],
        params['num_classes'],
        dtype=cls_outputs[level].dtype)

    if params['data_format'] == 'channels_first':
      bs, _, width, height, _ = cls_targets_at_level.get_shape().as_list()
      cls_targets_at_level = tf.reshape(cls_targets_at_level,
                                        [bs, -1, width, height])
    else:
      bs, width, height, _, _ = cls_targets_at_level.get_shape().as_list()
      cls_targets_at_level = tf.reshape(cls_targets_at_level,
                                        [bs, width, height, -1])
    box_targets_at_level = labels['box_targets_%d' % level]

    cls_loss = focal_loss(
        cls_outputs[level],
        cls_targets_at_level,
        params['alpha'],
        params['gamma'],
        normalizer=num_positives_sum,
        label_smoothing=params['label_smoothing'])

    if params['data_format'] == 'channels_first':
      cls_loss = tf.reshape(cls_loss,
                            [bs, -1, width, height, params['num_classes']])
    else:
      cls_loss = tf.reshape(cls_loss,
                            [bs, width, height, -1, params['num_classes']])

    cls_loss *= tf.cast(
        tf.expand_dims(tf.not_equal(labels['cls_targets_%d' % level], -2), -1),
        cls_loss.dtype)
    cls_loss_sum = tf.reduce_sum(cls_loss)
    cls_losses.append(tf.cast(cls_loss_sum, tf.float32))

    if params['box_loss_weight']:
      box_losses.append(
          _box_loss(
              box_outputs[level],
              box_targets_at_level,
              num_positives_sum,
              delta=params['delta']))

  # Sum per level losses to total loss.
  cls_loss = tf.add_n(cls_losses)
  box_loss = tf.add_n(box_losses) if box_losses else tf.constant(0.)

  total_loss = (
      cls_loss + params['box_loss_weight'] * box_loss)

  return total_loss, cls_loss, box_loss

def reg_l2_loss(weight_decay, trainable_vars, regex=r'.*(kernel|weight):0$'):
  """Return regularization l2 loss loss."""
  var_match = re.compile(regex)
  return weight_decay * tf.add_n([
      tf.nn.l2_loss(v)
      for v in trainable_vars
      if var_match.match(v.name)
  ])
  
@tf.autograph.experimental.do_not_convert
def _model_fn(features, labels, mode, params, model, variable_filter_fn=None):
  """Model definition entry.

  Args:
    features: the input image tensor with shape [batch_size, height, width, 3].
      The height and width are fixed and equal.
    labels: the input labels in a dictionary. The labels include class targets
      and box targets which are dense label maps. The labels are generated from
      get_input_fn function in data/dataloader.py
    mode: the mode of TPUEstimator including TRAIN and EVAL.
    params: the dictionary defines hyperparameters of model. The default
      settings are in default_hparams function in this file.
    model: the model outputs class logits and box regression outputs.
    variable_filter_fn: the filter function that takes trainable_variables and
      returns the variable list after applying the filter rule.

  Returns:
    tpu_spec: the TPUEstimatorSpec to run training, evaluation, or prediction.

  Raises:
    RuntimeError: if both ckpt and backbone_ckpt are set.
  """
  
  is_tpu = params['strategy'] == 'tpu'
  with tf.GradientTape() as tape:
    cls_out_list, box_out_list = model(features, params['is_training_bn'])
    cls_outputs, box_outputs = {}, {}
    for i in range(params['min_level'], params['max_level'] + 1):
      cls_outputs[i] = cls_out_list[i - params['min_level']]
      box_outputs[i] = box_out_list[i - params['min_level']]

    levels = cls_outputs.keys()
    for level in levels:
      cls_outputs[level] = tf.cast(cls_outputs[level], tf.float32)
      box_outputs[level] = tf.cast(box_outputs[level], tf.float32)

    # Set up training loss and learning rate.
    update_learning_rate_schedule_parameters(params)
    global_step = model.global_step
    learning_rate = learning_rate_schedule(params, global_step)

    # cls_loss and box_loss are for logging. only total_loss is optimized.
    det_loss, cls_loss, box_loss = detection_loss(
        cls_outputs, box_outputs, labels, params)
    reg_l2loss = reg_l2_loss(params['weight_decay'], model.trainable_variables)
    total_loss = det_loss + reg_l2loss

  trainable_vars = model.trainable_variables
  gradients = tape.gradient(total_loss, trainable_vars)
  model.optimizer.apply_gradients(zip(gradients, trainable_vars))

  model.loss_tracker.update_state(total_loss)
  return {'loss': total_loss}

def efficientdet_model_fn(features, labels, mode, params, model):
  """EfficientDet model."""
  variable_filter_fn = functools.partial(
      efficientdet_arch.freeze_vars, pattern=params['var_freeze_expr'])
  return _model_fn(
      features,
      labels,
      mode,
      params,
      model=model,
      variable_filter_fn=variable_filter_fn)


def get_model_arch(model_name='efficientdet-d0'):
  """Get model architecture for a given model name."""
  if 'efficientdet' in model_name:
    return efficientdet_arch.efficientdet

  raise ValueError('Invalide model name {}'.format(model_name))


def get_model_fn(model_name='efficientdet-d0'):
  """Get model fn for a given model name."""
  if 'efficientdet' in model_name:
    return efficientdet_model_fn

  raise ValueError('Invalide model name {}'.format(model_name))
