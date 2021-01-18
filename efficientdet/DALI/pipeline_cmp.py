import nvidia.dali as dali

import nvidia.dali.plugin.tf as dali_tf
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np


import dataloader
from keras import anchors

import ops




def run_all(file_root, annotations_file, batch_size, steps):

    run_normalize(file_root, annotations_file, batch_size, steps)
    run_anchors(file_root, annotations_file, batch_size, steps)



def run_normalize(file_root, annotations_file, batch_size, steps):

    input_pipeline = build_input_pipeline(file_root, annotations_file, batch_size, steps)

    dali_pipeline = dali.pipeline.Pipeline(
        batch_size = batch_size,
        num_threads = 1,
        device_id = 0
    )
    with dali_pipeline:
        images, bboxes, classes = ops.input(file_root, annotations_file, 0, 1, False)
        images, bboxes = ops.normalize_flip(images, bboxes, 0)
        dali_pipeline.set_outputs(images)
    dali_pipeline.build()

    for _ in range(steps):
        images_tf, _, _ = input_pipeline.run()
        (images_dali,) = dali_pipeline.run()

        for i in range(batch_size):
            image_tf = images_tf.at(i)
            tf_processor = dataloader.InputProcessor(image_tf, None)
            tf_processor.normalize_image()
            image_tf = tf_processor._image

            image_dali = images_dali.at(i)
            maxdiff = np.max(np.abs(image_dali - image_tf))
            if maxdiff > 1e-6:
                print("Normalize check fail, maxdiff =", maxdiff)
                quit()


def run_resize(file_root, annotations_file, batch_size, steps):

    input_pipeline = build_input_pipeline(file_root, annotations_file, batch_size, steps)

    dali_pipeline = dali.pipeline.Pipeline(
        batch_size = batch_size,
        num_threads = 1,
        device_id = 0
    )
    with dali_pipeline:
        images, bboxes, classes = ops.input(file_root, annotations_file, 0, 1, False)
        images = dali.fn.resize(images, resize_x = 512, resize_y = 512)
        dali_pipeline.set_outputs(images)
    dali_pipeline.build()

    for _ in range(steps):
        images_tf, _, _ = input_pipeline.run()
        (images_dali,) = dali_pipeline.run()

        for i in range(batch_size):
            image_tf = images_tf.at(i)
            tf_processor = dataloader.InputProcessor(image_tf, (512, 512))
            tf_processor.set_scale_factors_to_output_size()
            image_tf = tf.cast(tf_processor.resize_and_crop_image(), tf.int32)

            image_dali = images_dali.at(i)
            maxdiff = np.max(np.abs(image_dali - image_tf))
            print(image_dali)
            print(image_tf)
            if maxdiff > 1e-6:
                print("Resize check fail, maxdiff =", maxdiff)
                plt.imshow(image_dali)
                plt.savefig('dali.png')
                plt.clf()
                plt.imshow(image_tf)
                plt.savefig('tf.png')
                plt.clf()
                quit()


def run_anchors(file_root, annotations_file, batch_size, steps):

    input_pipeline = build_input_pipeline(file_root, annotations_file, batch_size, steps)
    input_anchors = anchors.Anchors(3, 7, 3, [1.0, 2.0, 0.5], 4.0, (512, 512))

    def _get_boxes():
        boxes_l = input_anchors.boxes[: ,0] / 512
        boxes_t = input_anchors.boxes[: ,1] / 512
        boxes_r = input_anchors.boxes[: ,2] / 512
        boxes_b = input_anchors.boxes[:, 3] / 512
        boxes = tf.transpose(tf.stack([boxes_l, boxes_t, boxes_r, boxes_b]))
        return tf.reshape(boxes, boxes.shape[0] * 4).numpy().tolist()

    anchor_labeler = anchors.AnchorLabeler(input_anchors, 90)

    dali_pipeline = dali.pipeline.Pipeline(
        batch_size = batch_size,
        num_threads = 1,
        device_id = 0
    )
    with dali_pipeline:
        images, bboxes, classes = ops.input(file_root, annotations_file, 0, 1, False)
        enc_bboxes, enc_classes = dali.fn.box_encoder(bboxes, classes, anchors = _get_boxes())
        dali_pipeline.set_outputs(images, enc_bboxes, enc_classes)
    dali_pipeline.build()

    for _ in range(steps):
        images_tf, bboxes_tf, classes_tf = input_pipeline.run()
        images_dali, bboxes_dali, classes_dali = dali_pipeline.run()

        for i in range(batch_size):
            bbox_tf = tf.convert_to_tensor(bboxes_tf.at(i)) * 512
            class_tf = tf.reshape(tf.convert_to_tensor(classes_tf.at(i), dtype=tf.float32), [-1, 1])
            cls_tf, box_tf, _ = anchor_labeler.label_anchors(bbox_tf, class_tf)
            cls_tf = tf.concat([tf.reshape(cls_tf[i], [-1]) for i in range(3, 8)], axis=0)
            box_tf = tf.concat([tf.reshape(box_tf[i], [-1, 4]) for i in range(3, 8)], axis=0)

            cls_dali = classes_dali.at(i)
            box_dali = bboxes_dali.at(i)

            np.savetxt('__dali.txt', cls_dali)
            np.savetxt('__tf.txt', cls_tf)

            print(_get_boxes()[4 * 41632 : 4 * 41632 + 4])
            print(box_dali[41632])
            print(bbox_tf)
            quit()




def build_input_pipeline(file_root, annotations_file, batch_size, steps):
    pipeline = dali.pipeline.Pipeline(
        batch_size = batch_size,
        num_threads = 1,
        device_id = 0
    )
    with pipeline:
        images, bboxes, classes = ops.input(file_root, annotations_file, 0, 1, False)
        pipeline.set_outputs(images, bboxes, classes)
    pipeline.build()
    return pipeline
