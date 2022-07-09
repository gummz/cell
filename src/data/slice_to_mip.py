from os import listdir
import os
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

    folder = f'{which}s_{mode}'
    output_path = join(c.DATA_DIR, c.EXTRACT_DIR, folder)
    utils.make_dir(output_path)

    for image in images:
        index = int(splitext(image)[0])
        name, file, t, _ = slice_record.iloc[index].values
        # skip if already created
        if os.path.exists(join(output_path, f'{name}_mip.jpg')):
            # print('path exists', join(output_path, f'{name}_mip.jpg'))
            continue
        # else:
        #     print('path doesn\'t exist', join(output_path, f'{name}.jpg'))

        raw_file_path = join(c.RAW_DATA_DIR, file)

        t = int(t) if which == 'timepoint' else None

        data = utils.get_raw_array(raw_file_path, t).compute()
        data_mip = np.max(data, axis=0)
        del data

        name_str = f'{name}_mip.{ext}'
        utils.imsave(join(output_path, name_str), data_mip, resize=resize)
        print('saved to', join(output_path, name_str))

    return output_path


if __name__ == '__main__':
    mode = 'train'
    resize = False
    # which:
    # Should the extract be one timepoint across slices,
    # or one slice across timepoints?
    which = 'timepoint'  # or which = 'slice'
    utils.set_cwd(__file__)

    output_path = slice_to_mip(mode, resize, which)

    print('Performed MIP on all images from',
          mode, 'and saved in\n\t', output_path)
