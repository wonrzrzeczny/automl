import matplotlib.pyplot as plt
import matplotlib.patches as patches

import os, sys

def run_dali():

    from DALI.pipeline import EfficientDetPipeline

    dali_extra = os.environ['DALI_EXTRA_PATH']
    file_root = os.path.join(dali_extra, 'db', 'coco', 'images')
    annotations_file = os.path.join(dali_extra, 'db', 'coco', 'instances.json')

    batch_size = 8
    image_size = (256, 256)
    num_threads = 1
    device_id = 0
    seed = int.from_bytes(os.urandom(4), 'little')

    pipeline = EfficientDetPipeline(
        file_root, annotations_file,
        batch_size, image_size,
        num_threads, device_id, seed
    )

    pipeline.build()

    (images, cls_3, box_3, cls_4, box_4, cls_5, box_5,
             cls_6, box_6, cls_7, box_7) = pipeline.run()

    for i, image in enumerate(images):
        plt.imshow(image)
        plt.savefig('dali/image' + str(i) + '.png')
        plt.clf()


def run_dali_fn():

    from DALI.fn_pipeline import EfficientDetPipeline

    dali_extra = os.environ['DALI_EXTRA_PATH']
    file_root = os.path.join(dali_extra, 'db', 'coco', 'images')
    annotations_file = os.path.join(dali_extra, 'db', 'coco', 'instances.json')

    batch_size = 8
    image_size = (256, 256)
    num_threads = 1
    device_id = 0
    seed = int.from_bytes(os.urandom(4), 'little')

    pipeline = EfficientDetPipeline(
        file_root, annotations_file,
        batch_size, image_size,
        num_threads, device_id, seed
    )

    pipeline.build()

    (images, cls_3, box_3, cls_4, box_4, cls_5, box_5,
             cls_6, box_6, cls_7, box_7) = pipeline.run()

    for i, image in enumerate(images):
        plt.imshow(image)
        plt.savefig('dali/image' + str(i) + '.png')
        plt.clf()


def run_tf(tfrecord_pattern):

    from dataloader import InputReader
    from hparams_config import default_detection_configs

    train_input_fn = InputReader(
        tfrecord_pattern,
        is_training=True,
        use_fake_data=False,
        max_instances_per_image=100
    )

    params = default_detection_configs()
    params.image_size = 256
    params.grid_mask = False
    params.tf_random_seed = 1234

    dataset = train_input_fn(params, batch_size=8)

    for elem in dataset.take(1):
        images, labels = elem
        for i, image in enumerate(images):
            plt.imshow(image)
            plt.savefig('tf/out' + str(i) + '.png')
            plt.clf()


def run_cmp():

    dali_extra = os.environ['DALI_EXTRA_PATH']
    file_root = os.path.join(dali_extra, 'db', 'coco', 'images')
    annotations_file = os.path.join(dali_extra, 'db', 'coco', 'instances.json')

    import pipeline_cmp
    pipeline_cmp.run_all(file_root, annotations_file, 32, 2)

if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == 'tf':
        run_tf(sys.argv[2])
    elif len(sys.argv) == 2 and sys.argv[1] == 'cmp':
        run_cmp()
    else:
        run_dali_fn()
