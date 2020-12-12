from pipeline import EfficientDetPipeline

import matplotlib.pyplot as plt

import os

dali_extra = os.environ['DALI_EXTRA_PATH']
file_root = os.path.join(dali_extra, 'db', 'coco', 'images')
annotations_file = os.path.join(dali_extra, 'db', 'coco', 'instances.json')

batch_size = 32
image_size = (640, 480)
num_threads = 1
device_id = 0
seed = int.from_bytes(os.urandom(4), 'little')

pipeline = EfficientDetPipeline(
    file_root, annotations_file,
    batch_size, image_size,
    num_threads, device_id, seed)

pipeline.build()

images, labels = pipeline.run()

for i, image in enumerate(images):
    plt.imshow(image)
    plt.savefig('out/image' + str(i) + '.png')
    plt.clf()
