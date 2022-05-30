from aicsimageio import AICSImage
from matplotlib import pyplot as plt
import src.data.utils.utils as utils
import src.data.constants as c
import os.path as osp
import numpy as np
from time import time


def building_dataset():
    '''
    To show that, despite the maximum intensity projection
    yielding an image containing many cells, the individual
    slices can still be mostly barren.
    '''
    file_path = osp.join(c.RAW_DATA_DIR, c.PRED_FILE)
    array = utils.get_raw_array(file_path, t=50).compute()
    fig_dir = osp.join(c.FIG_DIR, 'report', 'building_dataset')
    utils.make_dir(fig_dir)

    array_mip = np.max(array, axis=0)
    slices = np.random.randint(low=0, high=array.shape[0],
                               size=3)
    array_slices = array[slices]
    utils.imsave(osp.join(fig_dir, 'MIP'), array_mip)
    for i, z_slice in enumerate(array_slices):
        utils.imsave(osp.join(fig_dir, str(i)), z_slice)


def statistics():
    '''
    To be used in Statistics section of thesis.
    Histograms of mean pixel intensity of raw files.
    '''
    save_path = osp.join(c.FIG_DIR, 'report', 'statistics')
    utils.make_dir(save_path)
    files = tuple(c.RAW_FILES_TRAIN.keys())
    files = utils.add_ext(files)
    # hists = []
    for file in files:
        path = osp.join(c.RAW_DATA_DIR, file)
        datafile = AICSImage(path)
        T = datafile.dims['T'][0]
        Z = datafile.dims['Z'][0]
        bar = np.zeros((T, Z))
        # hist = np.zeros(T)
        del datafile
        for t in range(T):
            array = utils.get_raw_array(path, t).compute()
            bar[t] = np.mean(array, axis=(1, 2))
            # hist[t] = np.histogram(array)
        # hists.append(hist)
        print(bar)
        plt.bar(bar)
        plt.title(f'File: {file}')
        plt.xlabel('Slice index')
        plt.ylabel('Mean pixel value')
        plt.savefig(osp.join(save_path, f'{file}_bar'))
        plt.close()

        plt.hist(array.flatten())
        plt.title(f'Histogram for: {file}')
        plt.savefig(osp.join(save_path, f'{file}_hist'))
        plt.close()


if __name__ == '__main__':
    tic = time()
    utils.setcwd(__file__)
    file_path = osp.join(c.RAW_DATA_DIR, c.PRED_FILE)
    array = utils.get_raw_array(file_path, t=100)
    args = (file_path, array)

    to_output = [building_dataset, statistics]
    for output in to_output:
        output()

    elapsed = utils.time_report(tic, time())
    print('Report completed after', elapsed)
