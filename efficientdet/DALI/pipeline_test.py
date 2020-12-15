import matplotlib.pyplot as plt

import os, sys

def run_dali():

    from DALI.pipeline import EfficientDetPipeline

    dali_extra = os.environ['DALI_EXTRA_PATH']
    file_root = os.path.join(dali_extra, 'db', 'coco', 'images')
    annotations_file = os.path.join(dali_extra, 'db', 'coco', 'instances.json')

    batch_size = 8
    image_size = (640, 640)
    num_threads = 1
    device_id = 0
    seed = int.from_bytes(os.urandom(4), 'little')

    pipeline = EfficientDetPipeline(
        file_root, annotations_file,
        batch_size, image_size,
        num_threads, device_id, seed
    )

    pipeline.build()

    images, enc_bboxes, enc_labels = pipeline.run()

    for i, image in enumerate(images):
        for j in range(len(enc_labels.at(i))):
            if enc_labels.at(i)[j] != 0:
                print(j, ':', enc_labels.at(i)[j], enc_bboxes.at(i)[j])
        plt.imshow(image)
        plt.savefig('dali/image' + str(i) + '.png')
        plt.clf()


def run_recon(tfrecord_pattern):

    from dataloader import InputReader
    from hparams_config import default_detection_configs

    train_input_fn = InputReader(
        tfrecord_pattern,
        is_training=True,
        use_fake_data=False,
        max_instances_per_image=100
    )

    params = default_detection_configs()
    params.image_size = 640
    params.grid_mask = False

    dataset = train_input_fn(params, batch_size=32)

    for i, elem in enumerate(dataset.take(8)):
        images, labels = elem
        plt.imshow(images[0])
        plt.savefig('recon/out' + str(i) + '.png')
        plt.clf()
#        print(labels)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == 'recon':
        run_recon(sys.argv[2])
    else:
        run_dali()
