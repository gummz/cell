from __future__ import annotations

import pickle
from os import listdir
from os.path import join
from time import time

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
    max_array = [frame.particle.values
                 for _, frame in tracked_centroids]
    max_array = np.concatenate(max_array)
    # get unique colors for each detected cell
    colors = viz.get_colors(max_array, cmap)
    path = join(c.RAW_DATA_DIR, filename)
    channels = [c.CELL_CHANNEL, c.TUBE_CHANNEL]
    columns = ['x', 'y', 'particle', 'intensity']
    # eval tracking images already created
    images = [image for image in listdir(location) if '.png' in image]

    # overlay points in each frame on corresponding
    # raw MIP image
    iterable = zip(time_range, tracked_centroids, images)
    for t, (_, frame), image in iterable:
        data = utils.get_raw_array(
            path, t, ch=channels).compute()
        name = f'{t:05d}'

        # don't output if it's already there
        if f'{name}.png' in images:
            print(f'{name}.png', 'in images')
            continue
        else:
            print(name, 'not in images')

        cells_mip = np.max(data[:, :, :, channels[0]], axis=0)
        cells_mip = skimage.exposure.rescale_intensity(
            cells_mip, in_range='image', out_range=(25, 255))

        tubes_mip = np.max(data[:, :, :, channels[1]], axis=0)

        joined_img = np.where(
            cells_mip > 50, cells_mip, tubes_mip)

        utils.imsave(join(location, f'{name}.png'), joined_img)

        # output cells and tubes for debugging
        utils.imsave(join(location, f'{name}_cells.jpg'), cells_mip)
        utils.imsave(join(location, f'{name}_tubes.jpg'), tubes_mip)

        save = join(location, tracking_folder, f'{t:05d}.png')

        X, Y, P, I = (frame[c] for c in columns)

        for x, y, p, i in zip(X, Y, P, I):
            if i < c.ACTIVE_THRESHOLD:
                marker = 'x'
            else:
                marker = 'o'
            plt.plot(x, y, col=colors[p], marker=marker)

        plt.imshow(joined_img)

        # plt.xlim(left=0, right=1024)
        plt.xlabel('X dimension')
        # plt.ylim(bottom=0, top=1024)
        plt.ylabel('Y dimension')
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

    time_range = range(10)
    eval_track(tracked_centroids, time_range,
               c.PRED_FILE, location)

    plot.create_movie(join(location, 'with_tracking'),
                      time_range)

    elapsed = utils.time_report(tic, time())
    print(f'eval_track finished in {elapsed}.')
