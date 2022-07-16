import os

import numpy as np
import src.data.constants as c
import os.path as osp
import skimage.exposure as ske
import src.data.utils.utils as utils
from time import time
import random

if __name__ == '__main__':
    modes = ('train', 'val', 'test')
    sample_ratios = (0.1,)*3
    db_version = 'hist_eq'

    tic = time()
    random.seed(42)
    utils.set_cwd(__file__)
    root_dir = c.DATA_DIR if osp.exists(c.DATA_DIR) else c.PROJECT_DATA_DIR

    for mode, sample_ratio in zip(modes, sample_ratios):

        src_dir = osp.join(root_dir, mode, c.IMG_DIR)
        dst_dir = osp.join(root_dir, 'db_versions',
                           db_version, mode, c.IMG_DIR)
        utils.make_dir(src_dir)
        utils.make_dir(dst_dir)

        kernel_size = 10
        clip_limit = 0.92
        images = sorted([image for image in os.listdir(src_dir)
                        if '.npy' in image])
        already_sampled = sorted([image
                                  for image in os.listdir(dst_dir)
                                  if '.png' in image])
        # remove images which have already been sampled
        images = [image for image in images
                  if image not in already_sampled]

        k = int(len(images) * sample_ratio) - len(already_sampled)
        if k <= 0:
            print((f'With the chosen ratio,'
                   f'no more images can be sampled from {mode}'
                   f'dataset as there are'
                   f'already more than enough sampled images.'))
            continue

        sampled_imgs = sorted(random.sample(images, k=k))

        for sample in sampled_imgs:
            image_arr = np.load(osp.join(src_dir, sample))
            bright = ske.equalize_adapthist(
                image_arr, kernel_size, clip_limit)
            name = sample[:-4] + '.png'
            # np.save(osp.join(dst_dir, image), bright)
            utils.imsave(osp.join(dst_dir, name), bright, resize=False)

    print(f'imgs_to_histeq.py completed in {utils.time_report(tic, time())}.')
