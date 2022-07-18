from os import listdir
import os
import os.path as osp
from aicsimageio import AICSImage
import numpy as np

import pandas as pd
import src.data.constants as c
import src.data.utils.utils as utils
import tifffile


def slice_to_mip(db_version, mode, resize,
                 which, ext='jpg'):
    data_dir = c.DATA_DIR if osp.exists(c.DATA_DIR) else c.PROJECT_DATA_DIR

    images = sorted([img for img in listdir(
        osp.join(data_dir, mode, c.IMG_DIR)) if '.npy' in img])

    record_path = osp.join(data_dir, 'db_versions',
                           db_version, mode,
                           c.IMG_DIR, 'slice_record.csv')
    slice_record = pd.read_csv(
        record_path, sep='\t', header=0, dtype={'name': str})

    folder = f'{which}s_{mode}'
    output_path = osp.join(data_dir, c.EXTRACT_DIR, folder)
    utils.make_dir(output_path)

    raw_file_path = ''
    for image in images:
        index = int(osp.splitext(image)[0])
        name, file, t, _ = slice_record.iloc[index].values
        # skip if already created
        if osp.exists(osp.join(output_path, f'{name}_mip.jpg')):
            # print('path exists', osp.join(output_path, f'{name}_mip.jpg'))
            continue
        # else:
        #     print('path doesn\'t exist', osp.join(output_path, f'{name}.jpg'))

        raw_temp = osp.join(c.RAW_DATA_DIR, file)
        if raw_file_path != raw_temp:
            raw_file_path = raw_temp

        raw_file_path = osp.join(c.RAW_DATA_DIR, file)
        data = AICSImage(raw_file_path)

        t = int(t) if which == 'timepoint' else None

        data = utils.get_raw_array(data, t).compute()
        data_mip = np.max(data, axis=0)
        del data

        name_str = f'{name}_mip.{ext}'
        utils.imsave(osp.join(output_path, name_str), data_mip, resize=resize)
        print('saved to', osp.join(output_path, name_str))

    return output_path


if __name__ == '__main__':
    db_version = 'hist_eq'
    mode = 'train'
    resize = False
    # which:
    # Should the extract be one timepoint across slices,
    # or one slice across timepoints?
    which = 'timepoint'  # or which = 'slice'
    utils.set_cwd(__file__)

    output_path = slice_to_mip(db_version, mode, resize, which)

    print('Performed MIP on all images from',
          mode, 'and saved in\n\t', output_path)
