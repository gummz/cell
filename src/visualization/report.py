from aicsimageio import AICSImage
from matplotlib import pyplot as plt
import pandas as pd
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
    name = c.PRED_FILE
    name = utils.add_ext(name)
    file_path = osp.join(c.RAW_DATA_DIR, name)
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


def barplots():
    '''
    To be used in Statistics section of thesis.
    Bar plots of mean pixel intensity of raw files.
    '''
    mode = 'train'
    files = tuple(c.RAW_FILES[mode].keys())
    files = utils.add_ext(files)

    bar = []  # np.zeros((T, Z))
    hist = []  # np.zeros(T)
    fontsize = 25
    for file in files:
        print('Now on file', file)
        save_path = osp.join(c.FIG_DIR, 'report', 'barplots', file)
        utils.make_dir(save_path)

        path = osp.join(c.RAW_DATA_DIR, file)
        raw_file = AICSImage(path)
        for t in range(400):
            if t < 100:
                continue

            try:
                array = utils.get_raw_array(raw_file, t).compute()
            except IndexError:
                break
            finally:
                bar.append(np.mean(array, axis=(1, 2)))
                hist.append(np.histogram(array))
            break

        x = len(bar[0])
        plt.bar(range(x), bar[0])
        plt.title(f'File: {file}\nTimepoint: {t}',
                  fontsize=fontsize + 5)
        plt.xlabel('Slice index', fontsize=fontsize + 5)
        plt.ylabel('Mean pixel value', fontsize=fontsize + 5)

        plt.savefig(osp.join(save_path, f'{file}_bar.jpg'))
        plt.close()

        # save selected slices based on barplot
        present_slices = (0, 15, 25)
        for z_slice in present_slices:
            figure = plt.figure()
            plt.title(f'Timepoint {t}, slice {z_slice}',
                      fontsize=fontsize)
            plt.imshow(array[z_slice])
            plt.axis('off')
            utils.imsave(osp.join(save_path, f'{file}_{t}_{z_slice}'), figure)
            plt.close()

        break


def distributions():
    datasets = c.RAW_FILES.items()
    save_path = osp.join(c.FIG_DIR, 'report', 'distributions')
    n_timepoints = 1
    n_slices = 5
    for name, dataset in datasets:
        # add extensions to filenames in `dataset`
        dataset = utils.add_ext(dataset)
        hists = []
        for filename in dataset:
            bins, hist = calc_hist(n_timepoints, n_slices, filename)
            hists.append(hist)
        mean_hist = np.mean(hists)

        figure = plt.figure(figsize=(10, 10))
        plt.title(f'Histogram of pixel intensities for\n' +
                  f'{name} set (sample of {n_timepoints} timepoints)')
        plt.bar(bins, mean_hist)
        utils.imsave(osp.join(save_path, f'hist_{name}'), figure)

        hists = []


def calc_hist(n_timepoints, n_slices, name):
    '''
    Calculates histogram for one data file.
    Sample from number of timepoints: n_timepoints.
    '''
    file = AICSImage(osp.join(c.RAW_DATA_DIR, name))
    dims = file.dims
    T = dims['T'][0]
    Z = dims['Z'][0]
    tps = tuple(np.random.randint(0, T, size=n_timepoints))
    slices = np.random.randint(0, Z, size=n_slices)
    data = utils.get_raw_array(
        file, tps).squeeze().compute()

    bins, hist = np.histogram(data[slices].flatten(), bins=10)
    return bins, hist


def manual_select_pr_curve():
    '''
    To be used in Statistics section of thesis.
    Manual selection of PR curve.
    '''
    fig_dir = osp.join(c.FIG_DIR, 'report', 'manual_select_pr_curve')
    utils.make_dir(fig_dir)

    manual_select_dir = osp.join(c.EXPERIMENT_DIR, 'manual_select')
    manual = pd.read_csv(osp.join(
        manual_select_dir, 'manual_select_study_manual.csv')).to_dict('records')
    auto = pd.read_csv('manual_select_study_auto.csv').to_dict('records')
    manual.plot(x='slice', y='manual_select', color='red', label='Manual')
    plt.savefig(osp.join(fig_dir, 'manual_select.jpg'))


def eval_gridsearch_pr_curve():
    '''
    To be used in Statistics section of thesis.
    Evaluation of gridsearch PR curve.
    '''
    fig_dir = osp.join(c.FIG_DIR, 'report', 'eval_gridsearch_pr_curve')
    utils.make_dir(fig_dir)

    eval_gridsearch_dir = osp.join(c.EXPERIMENT_DIR, 'eval_gridsearch')
    broad = pd.read_csv(osp.join(
        eval_gridsearch_dir, 'eval_gridsearch_study_0to1.csv'))
    print(broad.columns)
    broad = broad[broad['match_threshold'] > 0]
    # narrow = pd.read_csv(osp.join(
    #     eval_gridsearch_dir, 'eval_gridsearch_study_0.9to1.csv')).to_dict('records')
    # narrow = narrow[narrow.match_threshold > 0]
    broad.plot(x='precision', y='recall', color='red', label='Broad')
    plt.savefig(osp.join(fig_dir, 'manual_select.jpg'))


if __name__ == '__main__':
    print('report.py start')
    tic = time()
    utils.set_cwd(__file__)
    name = c.PRED_FILE
    name = utils.add_ext(name)
    file_path = osp.join(c.RAW_DATA_DIR, name)

    args = {
        barplots: None,
        distributions: None,
    }
    to_output = [distributions]
    for output in to_output:
        output()

    elapsed = utils.time_report(tic, time())
    print('Report completed after', elapsed)
