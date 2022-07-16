import os
from time import time

from matplotlib import pyplot as plt
import src.data.constants as c
import os.path as osp
import random
import src.data.utils.utils as utils

if __name__ == '__main__':
    modes = ('train', 'val', 'test')
    db_version = 'hist_eq'
    sample_ratios = (0.1,) * 3  # how much of the datasets to sample

    random.seed(42)
    utils.set_cwd(__file__)
    tic = time()
    root_dir = c.DATA_DIR if osp.exists(c.DATA_DIR) else c.PROJECT_DATA_DIR

    for mode, sample_ratio in zip(modes, sample_ratios):
        load_path = osp.join(root_dir, db_version,
                             mode, c.IMG_DIR)
        sample_dir = osp.join(root_dir, db_version, mode,
                              c.MASK_DIR)
        utils.make_dir(sample_dir)

        images = sorted((image for image in os.listdir(load_path)
                        if '.png' in image))
        already_sampled = sorted([image
                                  for image in os.listdir(sample_dir)
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

        sampled_imgs = random.sample(images, k=k)

        for sample in sampled_imgs:
            sampled_img = plt.imread(osp.join(load_path, sample))
            utils.imsave(osp.join(sample_dir, sample), sampled_img)

    print(f'choose_annotate.py completed in {utils.time_report(tic, time())}.')
