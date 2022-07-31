import os
import os.path as osp
from time import time
from aicsimageio import AICSImage
import numpy as np

import pandas as pd
import imageio

import src.data.constants as c
import skimage.exposure as ske
import cv2
import src.data.utils.utils as utils


def slice_to_mip(db_version, mode, resize,
                 which, ext='jpg'):
    root_dir = c.DATA_DIR if osp.exists(c.DATA_DIR) else c.PROJECT_DATA_DIR

    images = sorted([img for img in os.listdir(
        osp.join(root_dir, mode, c.IMG_DIR)) if '.npy' in img])

    record_path = osp.join(root_dir, 'db_versions',
                           db_version, mode,
                           c.IMG_DIR, 'slice_record.csv')
    slice_record = pd.read_csv(
        record_path, sep='\t', header=0, dtype={'name': str})

    folder = f'{which}s_{mode}'
    output_path = osp.join(root_dir, c.EXTRACT_DIR, folder)
    utils.make_dir(output_path)

    clip_limit = 0.8
    raw_file_path = ''
    for image in images:
        index = int(osp.splitext(image)[0])
        name, file, t, _ = slice_record.iloc[index].values

        raw_temp = osp.join(c.RAW_DATA_DIR, file)

        if raw_file_path != raw_temp:
            raw_data = AICSImage(raw_temp)
            raw_file_path = raw_temp

        t = int(t) if which == 'timepoint' else None

        if not osp.exists(osp.join(output_path, f'{name}_mip.jpg')):
            data = utils.get_raw_array(raw_data, t).compute()
            data_mip = np.max(data, axis=0)
            data_mip_eq = ske.equalize_adapthist(
                data_mip, clip_limit=clip_limit)

            name_str = f'{name}_mip.{ext}'
            mip_name = osp.join(output_path, name_str)
            eq_name = osp.join(output_path, f'{name}_mip_eq.{ext}')
            utils.imsave(mip_name, data_mip, resize=resize)
            utils.imsave(eq_name, data_mip_eq, resize=resize)
            del data_mip, data_mip_eq

        # make movie from MIP to aid the annotation process
        T = raw_data.dims['T'][0]
        mip_to_movie(raw_data, range(t-10, min(t+10, T)),
                     osp.join(output_path, 'movies',
                     f'{name}_movie.mp4'))

        print(f'{name}_mip.{ext}')
        print(f'{name}_mip_eq.{ext}')
        print('saved to', osp.join(output_path))

    return output_path


def mip_to_movie(raw_data, time_range, output_path):
    data_movie = utils.get_raw_array(
        raw_data, time_range).compute()
    data_movie_mip = np.max(data_movie, axis=1)

    movie_frames = []
    for frame in data_movie_mip:
        bright = ske.equalize_adapthist(frame, clip_limit=0.8)
        bright = utils.normalize(bright, 0, 255, out=cv2.CV_8UC1)
        movie_frames.append(bright)

    utils.make_dir(osp.dirname(output_path))
    imageio.mimsave(output_path, movie_frames, fps=1)


if __name__ == '__main__':
    db_version = 'hist_eq'
    mode = 'val'
    resize = False

    # which:
    # Should the extract be one timepoint across slices,
    # or one slice across timepoints?
    which = 'timepoint'  # or which = 'slice'

    tic = time()
    utils.set_cwd(__file__)

    output_path = slice_to_mip(db_version, mode, resize, which)

    print('Performed MIP on all images from',
          mode, 'and saved in\n\t', output_path)
    print(f'slice_to_mip.py completed in {utils.time_report(tic, time())}.')
