from __future__ import annotations
import os
import logging
import pickle
from os import listdir
from os.path import join
from time import time
from aicsimageio import AICSImage
import cv2
import numpy as np
import pandas as pd
import skimage
import src.data.constants as c
import src.data.utils.utils as utils
import src.visualization.plot as plot
import src.visualization.utils as viz
from matplotlib import pyplot as plt


def eval_track(tracked_centroids, time_range, filename, location):
    # use `tracked_centroids`'s frame, x, y, p
    # to overlay the points onto MIP of raw image

    # choose colormap
    cmap = 'tab20'

    # choose marker size
    marker_size = 10

    # folder is: pred/eval/track/
    # create relevant directories
    tracking_folder = 'with_tracking'
    utils.make_dir(join(location, tracking_folder))

    # in order to make a color map that uniquely identifies
    # each cell, the total number of cells is needed
    particles = pd.unique(tracked_centroids['particle'])
    # get unique colors for each detected cell
    colors = viz.get_colors(len(particles), cmap)
    colors_dict = dict(zip(particles, colors))

    path = join(c.RAW_DATA_DIR, filename)
    raw_file = AICSImage(path)
    channels = [c.CELL_CHANNEL, c.TUBE_CHANNEL]
    columns = ['x', 'y', 'particle', 'intensity']

    # overlay points in each frame on corresponding
    # raw MIP image
    frames = tracked_centroids.groupby('frame')
    # filter according to time range (frames could contain more
    # than necessary)
    frames = (frame for frame in frames if frame[0] in time_range)
    for t, (_, frame) in zip(time_range, frames):
        exists = True  # assume image exists until proven otherwise
        name = f'{t:05d}'

        # don't output if it's already there
        if True:  # not os.path.exists(join(location, f'{name}.png')):
            exists = False

            images = output_raw_images(location, raw_file, channels, t, name)
            combined, cells_scale, tubes_mip = images

        X, Y, P, I = (frame[c] for c in columns)

        plt.figure(figsize=(20, 12))

        # plot markers on beta cell channel and combined
        # image
        for i in range(2):
            plt.subplot(1, 3, i + 1)
            plot_markers(marker_size, colors_dict, X, Y, P, I)

        if exists:
            combined = plt.imread(join(location, f'{name}.png'))
            cells_scale = plt.imread(join(location, f'{name}_cells.png'))
            tubes_mip = plt.imread(join(location, f'{name}_tubes.png'))

        save = join(location, tracking_folder, f'{t:05d}.png')

        output_figure(filename, t, cells_scale, tubes_mip, combined, save)


def output_raw_images(location, raw_file, channels, t, name):
    data = utils.get_raw_array(
        raw_file, t, ch=channels).compute()

    cells_mip = np.max(data[:, :, :, channels[0]], axis=0)
    tubes_mip = np.max(data[:, :, :, channels[1]], axis=0)

    cells_mip = utils.normalize(cells_mip, 0, 1, out=cv2.CV_32FC1)
    # cells_scale = skimage.exposure.rescale_intensity(
    #     cells_mip, in_range=(0, 255), out_range=(220, 255))
    cells_scale = skimage.exposure.equalize_adapthist(
        cells_mip, clip_limit=0.8)

    # print('after rescale >=0', np.sum(cells_scale >= 0))
    # print('after rescale', np.sum(cells_scale >= 220))
    # cells_nl = utils.normalize(cells_scale, 220, 255, cv2.CV_8UC1)
    # cells_nl = cv2.fastNlMeansDenoising(cells_nl, None, 11, 7, 21)
    # cells_mip = utils.normalize(cells_mip, 0, 1, cv2.CV_32FC1)
    # tubes_mip = utils.normalize(tubes_mip, 0, 255, cv2.CV_8UC1)
    tubes_mip = utils.normalize(tubes_mip, 0, 1, cv2.CV_32FC1)
    # cells_scale = utils.normalize(cells_scale, 0, 1, cv2.CV_32FC1)
    # print('after normalize', np.sum(cells_scale >= 220/255))
    # exit()
    # cells_nl = utils.normalize(cells_nl, 0, 1, cv2.CV_32FC1)
    # cells_nl = np.where(cells_nl > 1, 1, cells_nl)

    cells_scale = np.where(cells_scale > 1, 1, cells_scale)
    cells_scale = np.where(cells_scale < 0, 0, cells_scale)

    tubes_mip = np.where(tubes_mip > 1, 1, tubes_mip)
    tubes_mip = np.where(tubes_mip < 0, 0, tubes_mip)

    # cells_nl = np.where(cells_nl < 0, 0, cells_nl)
    combined = np.zeros((1024, 1024, 4), dtype=np.float32)
    combined[:, :, 0] = cells_scale
    combined[:, :, 1] = tubes_mip
    combined[:, :, 3] = 1.

    tubes_rgb = np.zeros((1024, 1024, 4), dtype=np.float32)
    tubes_rgb[:, :, 1] = tubes_mip
    tubes_rgb[:, :, 3] = 1.
    # normalize ...
    # cells_scale = utils.normalize(cells_scale, 200, 255, cv2.CV_8UC1)
    # cells_rgb = np.zeros((*cells_scale.shape, 3))
    # cells_rgb[:, :, 1] = cells_scale
    # cv2.cvtColor(cells_scale.astype(np.float32),
    #                         cv2.COLOR_GRAY2RGB)

    # tubes_rgb = np.zeros((*tubes_mip.shape, 3))
    # tubes_rgb[:, :, 2] = tubes_mip
    # cv2.cvtColor(tubes_mip.astype(np.float32),
    #                         cv2.COLOR_GRAY2RGB)

    # linear blend between cells and tubes
    # alpha = 0.90
    # blended = cv2.addWeighted(cells_nl.astype(np.float32), alpha,
    #                           tubes_mip.astype(np.float32), 1 - alpha,
    #                           0, None, dtype=-1)
    # print(np.unique(combined))
    # print(np.unique(cells_scale))
    # print(np.unique(cells_mip))
    # print(np.unique(tubes_mip))

    utils.imsave(join(location, f'{name}.png'),
                 combined, False)

    # output cells and tubes
    utils.imsave(join(location, f'{name}_cells.png'),
                 cells_scale, False, 'Reds')
    utils.imsave(join(location, f'{name}_cells_raw.png'),
                 cells_mip, False, 'viridis')
    utils.imsave(join(location, f'{name}_tubes.png'),
                 tubes_rgb, False)

    return combined, cells_scale, tubes_mip


def output_figure(filename, t, cells_scale, tubes_mip, combined, save):
    plt.subplot(131)
    plt.title('Beta cell channel (0) with amplified signal')
    plt.imshow(cells_scale, cmap='Reds')

    plt.subplot(132)
    plt.imshow(combined, extent=(0, 1024, 1024, 0))
    plt.xlabel('X dimension')
    plt.xlim(left=0, right=1024)
    plt.ylabel('Y dimension')
    plt.ylim(bottom=1024, top=0)
    plt.title('Tracked cells with tubes overlay')

    plt.subplot(133)
    plt.title('Tubes channel (1)')
    plt.imshow(tubes_mip)

    plt.suptitle(
        f'Tracked beta cells for\nfile: {filename}, timepoint: {t}',
        fontsize=30)

    plt.tight_layout()
    plt.savefig(save, dpi=300)
    plt.close()


def plot_markers(marker_size, colors, X, Y, P, I):
    marker = 'x'
    for x, y, p, i in zip(X, Y, P, I):
        # if i < c.ACTIVE_THRESHOLD:
        #     marker = 'x'
        # else:
        #     marker = 'o'
        plt.plot(x, y, c=colors[p], marker=marker,
                 markersize=marker_size, markeredgewidth=3)


if __name__ == '__main__':
    n_frames = 30
    tic = time()
    mode = 'test'
    utils.setcwd(__file__)
    files = c.RAW_FILES[mode]
    range_ok in files.items()  # add extensions (.lsm etc.)
    files = [files]
    print(files)
    # name = files[3]  # c.PRED_FILE
    for i, (name, range_ok) in enumerate(files.items()):
        print('Outputting tracks for file', name, '...', end='')

        load_location = join(c.PROJECT_DATA_DIR, c.PRED_DIR,
                             name)
        tracked_centroids = pickle.load(
            open(join(load_location, 'tracked_centroids.pkl'),
                 'rb'))

        save_location = join(c.PROJECT_DATA_DIR, c.PRED_DIR,
                             'eval', 'track_2D', name)
        utils.make_dir(save_location)

        # unique_frames = tuple(pd.unique(tracked_centroids['frame']))
        # time_range = range(min(unique_frames), max(unique_frames) + 1)
        T = len(os.listdir(join(load_location, 'timepoints')))
        time_start = np.random.randint(0, T - n_frames)
        time_range = range(time_start, time_start + n_frames)
        eval_track(tracked_centroids, time_range, name, save_location)

        print('Done', end='')

    # frames = tuple(tracked_centroids.groupby('frame'))
    # time_range = range(len(frames))
    # plot.create_movie(join(save_location, 'with_tracking'),
    #                   time_range)

    elapsed = utils.time_report(tic, time())
    print(f'eval_track finished in {elapsed}.')
