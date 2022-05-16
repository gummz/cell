from __future__ import annotations
import os

import pickle
from os import listdir
from os.path import join
from time import time
import cv2

import numpy as np
import skimage
import src.data.constants as c
import src.data.utils.utils as utils
import src.visualization.plot as plot
import src.visualization.utils as viz
from matplotlib import pyplot as plt


def eval_track(tracked_centroids, time_range,
               filename, location):
    # use `tracked_centroids`'s frame, x, y, p
    # to overlay the points onto MIP of raw image

    # choose colormap
    cmap = 'tab20'

    # folder is: pred/eval/track/
    # create relevant directories
    utils.make_dir(location)
    tracking_folder = 'with_tracking'
    utils.make_dir(join(location, tracking_folder))

    # in order to make a color map that uniquely identifies
    # each cell, the total number of cells is needed
    max_item = len(tracked_centroids.groupby('particle'))
    # get unique colors for each detected cell
    colors = viz.get_colors(max_item, cmap)
    path = join(c.RAW_DATA_DIR, filename)
    channels = [c.CELL_CHANNEL, c.TUBE_CHANNEL]
    columns = ['x', 'y', 'particle', 'intensity']

    exists = True  # assume image exists until proven otherwise
    # overlay points in each frame on corresponding
    # raw MIP image
    frames = tuple(tracked_centroids.groupby('frame'))
    for t, (_, frame) in zip(time_range, frames):

        name = f'{t:05d}'

        # don't output if it's already there
        # if f'{name}.png' in images:
        #     print(f'{name}.png', 'in images')
        if not os.path.exists(join(location, f'{name}.png')):
            exists = False

            data = utils.get_raw_array(
                path, t, ch=channels).compute()

            cells_mip = np.max(data[:, :, :, channels[0]], axis=0)
            cells_scale = skimage.exposure.rescale_intensity(
                cells_mip, in_range='image', out_range=(200, 255))
            # normalize ...
            # cells_scale = utils.normalize(cells_scale, 200, 255, cv2.CV_8UC1)
            cells_rgb = np.zeros((*cells_scale.shape, 3))
            cells_rgb[:, :, 1] = cells_scale
            # cv2.cvtColor(cells_scale.astype(np.float32),
            #                         cv2.COLOR_GRAY2RGB)

            tubes_mip = np.max(data[:, :, :, channels[1]], axis=0)
            # tubes_mip = utils.normalize(tubes_mip, 0, 255, cv2.CV_8UC1)
            tubes_rgb = np.zeros((*tubes_mip.shape, 3))
            tubes_rgb[:, :, 2] = tubes_mip
            # cv2.cvtColor(tubes_mip.astype(np.float32),
            #                         cv2.COLOR_GRAY2RGB)

            # linear blend between cells and tubes
            alpha = 0.95
            blended = cv2.addWeighted(cells_rgb, alpha, tubes_rgb, 1 - alpha,
                                      0, None, dtype=-1)
            
            utils.imsave(join(location, f'{name}.png'), blended[:, :, 1])

            # output cells and tubes for debugging
            utils.imsave(
                join(location, f'{name}_cells_scale.jpg'), cells_rgb)
            utils.imsave(
                join(location, f'{name}_cells_mip.jpg'), cells_mip)
            utils.imsave(join(location, f'{name}_tubes.jpg'), tubes_rgb)

        X, Y, P, I = (frame[c] for c in columns)

        for x, y, p, i in zip(X, Y, P, I):
            if i < c.ACTIVE_THRESHOLD:
                marker = 'x'
            else:
                marker = 'o'
            plt.plot(x, y, c=colors[p], marker=marker)

        if exists:
            blended = plt.imread(join(location, f'{name}.png'))
        save = join(location, tracking_folder, f'{t:05d}.png')

        plt.imshow(blended, extent=(0, 1024, 1024, 0))
        # plt.xlim(left=0, right=1024)
        plt.xlabel('X dimension')
        plt.xlim(left=0, right=1024)
        # plt.ylim(bottom=0, top=1024)
        plt.ylabel('Y dimension')
        plt.ylim(bottom=1024, top=0)
        plt.title(f'Tracked beta cells for\nfile: {filename}, timepoint: {t}')
        plt.savefig(save, dpi=300)
        plt.close()


if __name__ == '__main__':
    tic = time()
    utils.setcwd(__file__)
    # output 2D movie
    name = c.PRED_FILE

    tracked_centroids = pickle.load(
        open(join(c.DATA_DIR, c.PRED_DIR, c.PRED_FILE, 'tracked_centroids.pkl'), 'rb'))
    location = join(c.DATA_DIR, c.PRED_DIR,
                    'eval', 'track_2D', c.PRED_FILE)
    utils.make_dir(location)

    time_range = range(10)
    eval_track(tracked_centroids, time_range,
               c.PRED_FILE, location)

    plot.create_movie(join(location, 'with_tracking'),
                      time_range)

    elapsed = utils.time_report(tic, time())
    print(f'eval_track finished in {elapsed}.')
