import os
from time import time

from matplotlib import pyplot as plt
import src.data.constants as c
import os.path as osp
import random
import src.data.utils.utils as utils
import numpy as np

if __name__ == '__main__':
    mode = 'train'
    db_version = 'hist_eq'
    ratio_sample = 0.1  # how much of the dataset to sample

    random.seed(42)
    utils.set_cwd(__file__)
    tic = time()
    root_dir = c.DATA_DIR if osp.exists(c.DATA_DIR) else c.PROJECT_DATA_DIR

    load_path = osp.join(root_dir, mode,
                         c.IMG_DIR, db_version)
    sample_dir = osp.join(load_path, 'sampled_for_labeling')
    utils.make_dir(sample_dir)

    images = sorted((image for image in os.listdir(load_path)
                     if '.png' in image))
    already_sampled = sorted([image for image in os.listdir(sample_dir)])
    # remove images which have already been sampled
    images = [image for image in images
              if image not in already_sampled]

    k = int(len(images) * ratio_sample) - len(already_sampled)
    if k <= 0:
        print(('With the chosen ratio,'
               'no more images can be sampled as there are'
               'already more than enough sampled images.'))
        exit()

    sampled_imgs = random.sample(images, k=k)

    for sample in sampled_imgs:
        sampled_img = plt.imread(osp.join(load_path, sample))
        utils.imsave(osp.join(sample_dir, sample), sampled_img)

    print(f'choose_annotate.py completed in {utils.time_report(tic, time())}.')
