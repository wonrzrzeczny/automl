from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types

import nvidia.dali.plugin.tf as dali_tf
import tensorflow as tf

from keras import anchors


class EfficientDetPipeline(Pipeline):
    def __init__(self, file_root, annotations_file,
                 batch_size, image_size,
                 num_threads, device_id, seed):
        super(EfficientDetPipeline, self).__init__(
            batch_size,
            num_threads,
            device_id,
            seed
        )

        self.image_size = image_size
        self.boxes = anchors.Anchors(3, 7, 3, [1.0, 2.0, 0.5], 1.0, image_size).boxes
        boxes_l = self.boxes[: ,0] / image_size[0]
        boxes_t = self.boxes[: ,1] / image_size[1]
        boxes_r = self.boxes[: ,2] / image_size[0]
        boxes_b = self.boxes[:, 3] / image_size[1]
        self.boxes = tf.transpose(tf.stack([boxes_l, boxes_t, boxes_r, boxes_b]))
        self.boxes = tf.reshape(self.boxes, self.boxes.shape[0] * 4).numpy().tolist()
        self.box_encode = ops.BoxEncoder(anchors=self.boxes)

        self.input = ops.COCOReader(
            file_root = file_root,
            annotations_file = annotations_file,
            ltrb = True,
            shard_id = device_id,
            num_shards = num_threads,
            ratio = True,
            random_shuffle = True
        )
        self.decode = ops.ImageDecoder(device = 'cpu', output_type = types.RGB)
        self.normalize = ops.CropMirrorNormalize(
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            output_layout=types.NHWC,
            mirror=1,
        )
        self.bbflip = ops.BbFlip(ltrb=True)
        self.crop = ops.RandomBBoxCrop(
            bbox_layout="xyXY",
            scaling=[0.1, 2.0],
            allow_no_crop=False
        )
        self.slice = ops.Slice(out_of_bounds_policy='pad')
        self.resize = ops.Resize(resize_x=self.image_size[0], resize_y=self.image_size[1])
        self.coin_flip = ops.CoinFlip()

    def define_graph(self):
        # skip_crowd_during_training
        inputs, bboxes, labels = self.input()
        images = self.decode(inputs)

        # grid_mask
        # autoaugment

        images = self.normalize(images)
        bboxes = self.bbflip(bboxes)

        anchors, shapes, bboxes = self.crop(bboxes)
        images = self.slice(images, anchors, shapes)
        images = self.resize(images)

        enc_bboxes, enc_labels = self.box_encode(bboxes, labels)

        # Prepare output

        return images, enc_bboxes, enc_labels

    def __call__(self, params):
        dataset = dali_tf.DALIDataset(
          pipeline = self,
          batch_size = self.batch_size,
          output_shapes=((self.batch_size, self.image_size[0], self.image_size[1], 3), (self.batch_size, 100)),
          output_dtypes = (tf.float32, tf.float32)
        )
        return dataset
