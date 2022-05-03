from os import listdir
from os.path import join, splitext
import numpy as np

import pandas as pd
import src.data.constants as c
import src.data.utils.utils as utils


def slice_to_mip(mode, resize, which, ext='jpg'):
    images = sorted([img for img in listdir(
        join(c.DATA_DIR, mode, c.IMG_DIR)) if '.npy' in img])

    record_path = join(c.DATA_DIR, mode, c.IMG_DIR, 'slice_record.csv')
    slice_record = pd.read_csv(
        record_path, sep='\t', header=0, dtype={'name': str})

    for image in images:
        index = int(splitext(image)[0])
        name, file, t, z = slice_record.iloc[index].values
        raw_file_path = join(c.RAW_DATA_DIR, file)

        idx = int(t) if which == 'timepoint' else int(z)
        data = utils.get_raw_array(raw_file_path, which, idx)
        data_mip = np.max(data, axis=0)
        del data
        folder = f'{which}s_{mode}'
        output_path = join(c.DATA_DIR, c.EXTRACT_DIR, folder)
        utils.make_dir(output_path)

        name_str = f'index_{t:05d}_mip.{ext}'

        utils.imsave(join(output_path, name_str), data_mip, resize=resize)

    return output_path


if __name__ == '__main__':
    mode = 'train'
    resize = False
    # which:
    # Should the extract be one timepoint across slices,
    # or one slice across timepoints?
    which = 'timepoint'  # or which = 'slice'
    utils.setcwd(__file__)

    output_path = slice_to_mip(mode, resize, which)

    print('Performed MIP on all images from',
          mode, 'and saved in\n\t', output_path)
