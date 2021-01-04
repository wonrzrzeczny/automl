import nvidia.dali as dali

def normalize_flip(images, bboxes, p = 0.5):
    flip = dali.fn.coin_flip(probability = p)
    images = dali.fn.crop_mirror_normalize(
        images,
        mirror = flip,
        mean = [0.485 * 255, 0.456 * 255, 0.406 * 255],
        std = [0.229 * 255, 0.224 * 255, 0.225 * 255],
        output_layout = dali.types.NHWC
    )
    bboxes =  dali.fn.bb_flip(bboxes, horizontal = flip, ltrb = True)
    return images, bboxes

def random_crop(images, bboxes, classes, scaling=[0.1, 2.0]):
    anchors, shapes, bboxes, classes = dali.fn.random_bbox_crop(
        bboxes, classes,
        bbox_layout = "xyXY",
        scaling = scaling,
        allow_no_crop = False
    )
    images = dali.fn.slice(images, anchors, shapes,
        out_of_bounds_policy = 'pad')
    return images, bboxes, classes
