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

        self._image_size = image_size

        self._anchors = anchors.Anchors(3, 7, 3, [1.0, 2.0, 0.5], 4.0, image_size)
        self._boxes = self._get_boxes()

        self.box_encode = ops.BoxEncoder(anchors=self._boxes)

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
        self.slice2D = ops.Slice(axes=[0,1])
        self.slice1D = ops.Slice(axes=[0])
        self.reshape = ops.Reshape()
        self.resize = ops.Resize(resize_x=self._image_size[0], resize_y=self._image_size[1])
        self.coin_flip = ops.CoinFlip()

    def _get_boxes(self):
        boxes_l = self._anchors.boxes[: ,0] / self._image_size[0]
        boxes_t = self._anchors.boxes[: ,1] / self._image_size[1]
        boxes_r = self._anchors.boxes[: ,2] / self._image_size[0]
        boxes_b = self._anchors.boxes[:, 3] / self._image_size[1]
        boxes = tf.transpose(tf.stack([boxes_l, boxes_t, boxes_r, boxes_b]))
        return tf.reshape(boxes, boxes.shape[0] * 4).numpy().tolist()

    def _unpack_labels(self, enc_bboxes, enc_classes):
        # from keras/anchors.py

        enc_bboxes_layers = []
        enc_classes_layers = []

        count = 0
        for level in range(self._anchors.min_level, self._anchors.max_level + 1):
            feat_size = self._anchors.feat_sizes[level]
            steps = feat_size['height'] * feat_size['width'] * self._anchors.get_anchors_per_location()

            enc_bboxes_layers.append(self.reshape(
                    self.slice2D(enc_bboxes, (count, 0), (steps, 4)), [feat_size['height'], feat_size['width'], -1])
            )
            enc_classes_layers.append(self.reshape(
                    self.slice1D(enc_classes, count, steps), [feat_size['height'], feat_size['width'], -1])
            )

            count += steps

        return enc_bboxes_layers, enc_classes_layers

    def define_graph(self):

        inputs, bboxes, classes = self.input() # skip_crowd_during_training
        images = self.decode(inputs)

        # grid_mask
        # autoaugment

        images = self.normalize(images)
        bboxes = self.bbflip(bboxes)

        anchors, shapes, bboxes = self.crop(bboxes)
        images = self.slice(images, anchors, shapes)
        images = self.resize(images)

        # label anchors
        enc_bboxes, enc_classes = self.box_encode(bboxes, classes)
        # enc_bboxes are in [x, y, w, h] format ???

        # spliting labels into feature size classes
        enc_bboxes_layers, enc_classes_layers = \
            self._unpack_labels(enc_bboxes, enc_classes)

        # no way to return values as dictionary (like original efficientdet impl does)
        enc_layers = [item for pair in zip(enc_classes_layers, enc_bboxes_layers) for item in pair]
        return (images, *enc_layers) #mean_num_positives, source_ids, groundtruth_data, image_scales, image_masks

    def _format_data(self, batch_size, images, *cls_box_args):
        labels = {}

        for level in range(self._anchors.min_level, self._anchors.max_level + 1):
            i = 2 * (level - self._anchors.min_level)
            labels['cls_targets_%d' % level] = cls_box_args[i]
            labels['box_targets_%d' % level] = cls_box_args[i + 1]

        labels['mean_num_positives'] = 0.0

        return images, labels

    def __call__(self, params):

        output_shapes = [(self.batch_size, self._image_size[0], self._image_size[1], 3)]
        output_dtypes = [tf.float32]

        for level in range(self._anchors.min_level, self._anchors.max_level + 1):
            feat_size = self._anchors.feat_sizes[level]
            output_shapes.append((self.batch_size, feat_size['height'],
                feat_size['width'], self._anchors.get_anchors_per_location()))
            output_shapes.append((self.batch_size, feat_size['height'],
                feat_size['width'], self._anchors.get_anchors_per_location() * 4))
            output_dtypes.append(tf.int32)
            output_dtypes.append(tf.float32)

        dataset = dali_tf.DALIDataset(
            pipeline = self,
            batch_size = self.batch_size,
            output_shapes=tuple(output_shapes),
            output_dtypes=tuple(output_dtypes)
        )
        dataset = dataset.map(
            lambda *args: self._format_data(self.batch_size, *args))

        return dataset
