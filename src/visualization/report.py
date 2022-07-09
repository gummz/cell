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
    mode = 'train'
    files = tuple(c.RAW_FILES[mode].keys())
    files = utils.add_ext(files)

    bar = []  # np.zeros((T, Z))
    hist = []  # np.zeros(T)
    for file in files:
        print('Now on file', file)
        save_path = osp.join(c.FIG_DIR, 'report', 'statistics', file)
        utils.make_dir(save_path)

        path = osp.join(c.RAW_DATA_DIR, file)
        raw_file = AICSImage(path)
        for t in range(400):
            try:
                array = utils.get_raw_array(raw_file, t).compute()
            except Exception as e:
                print(e)
                break
            finally:
                bar.append(np.mean(array, axis=0))
                hist.append(np.histogram(array))

        plt.bar(range(t - 1), bar)
        plt.title(f'File: {file}')
        plt.xlabel('Time index')
        plt.ylabel('Mean pixel value')
        plt.savefig(osp.join(save_path, f'{file}_bar.jpg'))
        plt.close()

        plt.hist(array.flatten())
        plt.title(f'Histogram for: {file}')
        plt.savefig(osp.join(save_path, f'{file}_hist.jpg'))
        plt.close()


def calculate_histogram(raw_file):
    '''
    Returns histogram of one raw data file.
    '''
    T = raw_file.dims['T'][0]
    hist = np.empty(T)
    for timepoint in raw_file:
        hist[timepoint] = np.histogram(timepoint)


if __name__ == '__main__':
    print('report.py start')
    tic = time()
    utils.set_cwd(__file__)
    name = c.PRED_FILE
    array = utils.get_raw_array(file_path, t=100)
    args = (file_path, array)

    to_output = [statistics, building_dataset]
    for output in to_output:
        output()

    elapsed = utils.time_report(tic, time())
    print('Report completed after', elapsed)
