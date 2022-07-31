import os

import pandas as pd
import src.data.constants as c
import os.path as osp
import skimage.exposure as ske
import src.data.utils.utils as utils
from time import time
import random
from aicsimageio import AICSImage


if __name__ == '__main__':
    # TODO: requested sample sizes using `sample_ratios`
    # should result in those exact requested sample sizes,
    # regardless of how many images there are in the sample
    # directory already. I.e. if there are 5 files in the sample
    # dir, and there are 100 requested, then only 95 will be sampled
    # -- excluding the ones already there.
    # modes = ('train', 'val', 'test')
    modes = ('val',)
    sample_ratios = (0.1,)*3
    db_version = 'hist_eq'

    tic = time()
    random.seed(42)
    utils.set_cwd(__file__)
    root_dir = utils.get_data_dir()

    for mode, sample_ratio in zip(modes, sample_ratios):

        src_dir = osp.join(root_dir, c.DB_VERS_DIR,
                           c.VANILLA_VERSION, mode, c.IMG_DIR)
        dst_dir = osp.join(root_dir, c.DB_VERS_DIR,
                           db_version, mode, c.IMG_DIR)
        utils.make_dir(src_dir)
        utils.make_dir(dst_dir)

        # parameters for adaptive histogram equalization
        kernel_size = 10
        clip_limit = 0.92
        # images present in main dataset
        images = sorted([image for image in os.listdir(src_dir)
                        if '.npy' in image])
        # images which have already been sampled
        already_sampled = sorted([image
                                  for image in os.listdir(dst_dir)
                                  if '.png' in image])
        # remove images which have already been sampled
        images = [image for image in images
                  if image not in already_sampled]

        k = int(len(images) * sample_ratio) - len(already_sampled)
        if k <= 0:
            print((f'With the chosen ratio,'
                   f' no more images can be sampled from {mode}'
                   f' dataset as there are'
                   f' already more than enough sampled images.'))
            continue

        record_path = osp.join(root_dir, mode,
                               c.IMG_DIR, 'slice_record.csv')
        slice_record = pd.read_csv(
            record_path, sep='\t', header=0, dtype={'name': str})

        sampled_imgs = sorted(random.sample(images, k=k))

        raw_file_path = ''
        for sample in sampled_imgs:
            index = int(osp.splitext(sample)[0])
            name, file, t, z = slice_record.iloc[index].values

            raw_temp = osp.join(c.RAW_DATA_DIR, file)

            if raw_file_path != raw_temp:
                raw_data = AICSImage(raw_temp)
                raw_file_path = raw_temp

            image_arr = utils.get_raw_array(raw_data, t, z, name)
            bright = ske.equalize_adapthist(
                image_arr, kernel_size, clip_limit)
            name = sample[:-4]

            utils.imsave(
                osp.join(dst_dir, f'{name}.png'), bright, resize=False)
            utils.imsave(
                osp.join(dst_dir, f'{name}_orig.png'), image_arr, resize=False)

    print(f'imgs_to_histeq.py completed in {utils.time_report(tic, time())}.')
