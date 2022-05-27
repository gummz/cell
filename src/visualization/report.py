from aicsimageio import AICSImage
import src.data.utils.utils as utils
import src.data.constants as c
import os.path as osp
import numpy as np


def building_dataset(file_path, array):
    fig_dir = osp.join(c.FIG_DIR, 'building_dataset')
    utils.make_dir(fig_dir)

    array_mip = np.max(array)
    slices = np.random.randint(low=0,
                               high=len(array_mip), size=3)
    array_slices = array[slices]
    utils.imsave(osp.join(fig_dir, 'MIP'), array_mip)
    for i, z_slice in enumerate(array_slices):
        utils.imsave(osp.join(fig_dir, i), z_slice)


def function2():
    pass


if __name__ == '__main__':
    file_path = osp.join(c.RAW_DATA_DIR, c.PRED_FILE)
    array = utils.get_raw_array(file_path, t=100)
    args = (file_path, array)

    to_output = [building_dataset, function2]
    for output in to_output:
        output(*args)
