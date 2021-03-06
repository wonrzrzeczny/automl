import nvidia.dali as dali

import nvidia.dali.plugin.tf as dali_tf
import tensorflow as tf

from absl import logging
from glob import glob
from keras import anchors

import DALI.ops as ops


class EfficientDetPipeline():
    def __init__(self, file_pattern,
                 batch_size, image_size, seed,
                 num_threads=1, device_id=None):

        self._batch_size = batch_size
        self._image_size = image_size
        self._tfrecord_files = glob(file_pattern)
        self._tfrecord_idxs = [filename + "_idx" for filename in self._tfrecord_files]
       
        self._num_shards = num_threads
        self._shard_id = 0 if device_id is None else device_id
        self._device = "cpu" if device_id is None else "gpu"

        self._anchors = anchors.Anchors(3, 7, 3, [1.0, 2.0, 0.5], 4.0, image_size)
        self._boxes = self._get_boxes()

        self._pipe = dali.pipeline.Pipeline(
            batch_size = batch_size,
            num_threads = num_threads,
            device_id = device_id,
            seed = seed
        )
        self._define_pipeline()


    def _get_boxes(self):
        boxes_l = self._anchors.boxes[: ,0] / self._image_size[0]
        boxes_t = self._anchors.boxes[: ,1] / self._image_size[1]
        boxes_r = self._anchors.boxes[: ,2] / self._image_size[0]
        boxes_b = self._anchors.boxes[:, 3] / self._image_size[1]
        boxes = tf.transpose(tf.stack([boxes_l, boxes_t, boxes_r, boxes_b]))
        return tf.reshape(boxes, boxes.shape[0] * 4).numpy().tolist()


    def _define_pipeline(self):
        with self._pipe:
            images, bboxes, classes = ops.input(
                self._tfrecord_files,
                self._tfrecord_idxs,
                device=self._device,
                shard_id=self._shard_id,
                num_shards=self._num_shards
            )

            images, bboxes = ops.normalize_flip(self._device, images, bboxes)
            images, bboxes, classes = ops.random_crop_resize_2(self._device, images, bboxes, classes, self._image_size)

            enc_bboxes, enc_classes = dali.fn.box_encoder(bboxes, classes, anchors = self._boxes, offset = True)
            # split into layers by size
            enc_bboxes_layers, enc_classes_layers = self._unpack_labels(enc_bboxes, enc_classes)

            # interleave enc_bboxes_layers and enc_classes_layers
            enc_layers = [item for pair in zip(enc_classes_layers, enc_bboxes_layers) for item in pair]

            self._pipe.set_outputs(images, *enc_layers)


    def _unpack_labels(self, enc_bboxes, enc_classes):
        # from keras/anchors.py

        enc_bboxes_layers = []
        enc_classes_layers = []

        count = 0
        for level in range(self._anchors.min_level, self._anchors.max_level + 1):
            feat_size = self._anchors.feat_sizes[level]
            steps = feat_size['height'] * feat_size['width'] * self._anchors.get_anchors_per_location()

            enc_bboxes_layers.append(
                dali.fn.reshape(
                    dali.fn.slice(enc_bboxes, (count, 0), (steps, 4), axes = [0, 1]),
                    [feat_size['height'], feat_size['width'], -1]
                )
            )
            enc_classes_layers.append(
                dali.fn.reshape(
                    dali.fn.slice(enc_classes, count, steps, axes = [0]),
                    [feat_size['height'], feat_size['width'], -1]
                )
            )

            count += steps

        return enc_bboxes_layers, enc_classes_layers


    def _format_data(self, batch_size, images, *cls_box_args):
        labels = {}

        for level in range(self._anchors.min_level, self._anchors.max_level + 1):
            i = 2 * (level - self._anchors.min_level)
            labels['cls_targets_%d' % level] = cls_box_args[i]
            labels['box_targets_%d' % level] = cls_box_args[i + 1]

        labels['mean_num_positives'] = 0.0

        return images, labels


    def __call__(self, params):
        output_shapes = [(self._batch_size, self._image_size[0], self._image_size[1], 3)]
        output_dtypes = [tf.float32]

        for level in range(self._anchors.min_level, self._anchors.max_level + 1):
            feat_size = self._anchors.feat_sizes[level]
            output_shapes.append((self._batch_size, feat_size['height'],
                feat_size['width'], self._anchors.get_anchors_per_location()))
            output_shapes.append((self._batch_size, feat_size['height'],
                feat_size['width'], self._anchors.get_anchors_per_location() * 4))
            output_dtypes.append(tf.int32)
            output_dtypes.append(tf.float32)

        dataset = dali_tf.DALIDataset(
            pipeline = self._pipe,
            batch_size = self._batch_size,
            output_shapes=tuple(output_shapes),
            output_dtypes=tuple(output_dtypes)
        )
        dataset = dataset.map(
            lambda *args: self._format_data(self._batch_size, *args))

        return dataset



    def build(self):
        self._pipe.build()


    def run(self):
        return self._pipe.run()
