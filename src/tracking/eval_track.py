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
    # get unique particles from first frame
    p_tracks = get_p_tracks(frames)
    p_track = p_tracks[batch_idx - 1]
    for t, (_, frame) in zip(time_range, frames):
        exists = True  # assume image exists until proven otherwise
        name = f'{t:05d}'

        # don't output if it's already there
        if True:  # not os.path.exists(join(location, f'{name}.png')):
            exists = False

            images = output_raw_images(location, raw_file, channels, t, name)
            combined, cells_scale, tubes_mip = images

        X, Y, P, I = (frame[c] for c in columns)

        plt.figure(figsize=(20, 14))

        for i in range(2):
            plot_markers(marker_size, colors_dict, X, Y,
                         P, I, t, time_range, p_track)

        if exists:
            combined, cells, cells_xz, cells_yz, tubes = load_existing(
                location, name)

        save = join(location, tracking_folder, f'{t:05d}.png')

def load_existing(location, name):
    combined = plt.imread(join(location, f'{name}.png'))
    cells = plt.imread(join(location, f'{name}_cells.png'))
    cells_xz = plt.imread(join(location, f'{name}_cells_xz.png'))
    cells_yz = plt.imread(join(location, f'{name}_cells_yz.png'))
    tubes = plt.imread(join(location, f'{name}_tubes.png'))
    return combined, cells, cells_xz, cells_yz, tubes


def output_raw_images(location, raw_file, channels, t, name):
    data = utils.get_raw_array(
        raw_file, t, ch=channels).compute()

    cells_mip = np.max(data[:, :, :, channels[0]], axis=0)
    cells_mip_xz = np.max(data[:, :, :, channels[0]], axis=2)
    cells_mip_yz = np.max(data[:, :, :, channels[0]], axis=1)
    tubes_mip = np.max(data[:, :, :, channels[1]], axis=0)

    cells_mip = utils.normalize(cells_mip, 0, 1, out=cv2.CV_32FC1)
    cells_mip_xz = utils.normalize(cells_mip_xz, 0, 1, out=cv2.CV_32FC1)
    cells_mip_yz = utils.normalize(cells_mip_yz, 0, 1, out=cv2.CV_32FC1)

    cells_scale = skimage.exposure.equalize_adapthist(
        cells_mip, clip_limit=0.8)
    cells_mip_xz = skimage.exposure.equalize_adapthist(
        cells_mip_xz, clip_limit=0.8)
    cells_mip_yz = skimage.exposure.equalize_adapthist(
        cells_mip_yz, clip_limit=0.8)

    tubes_mip = utils.normalize(tubes_mip, 0, 1, cv2.CV_32FC1)

    cells_scale = np.where(cells_scale > 1, 1, cells_scale)
    cells_scale = np.where(cells_scale < 0, 0, cells_scale)
    cells_mip_xz = np.where(cells_mip_xz > 1, 1, cells_mip_xz)
    cells_mip_xz = np.where(cells_mip_xz < 0, 0, cells_mip_xz)
    cells_mip_yz = np.where(cells_mip_yz > 1, 1, cells_mip_yz)
    cells_mip_yz = np.where(cells_mip_yz < 0, 0, cells_mip_yz)

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

    utils.imsave(join(location, f'{name}.png'),
                 combined, False)

    # output cells and tubes
    utils.imsave(join(location, f'{name}_cells.png'),
                 cells_scale, False, 'Reds')
    utils.imsave(join(location, f'{name}_cells_xz.png'),
                 cells_mip_xz, False, 'Reds')
    utils.imsave(join(location, f'{name}_cells_yz.png'),
                 cells_mip_yz, False, 'Reds')
    utils.imsave(join(location, f'{name}_cells_raw.png'),
                 cells_mip, False, 'viridis')
    utils.imsave(join(location, f'{name}_tubes.png'),
                 tubes_rgb, False)

    return (combined, cells_scale, cells_mip_xz,
            cells_mip_yz, tubes_mip)


def output_tracks(filename, t, cells_scale, cells_mip_xz, cells_mip_yz, tubes_mip, combined, save, time_range):

    fontsize = 25
    plt.subplot(231)
    plt.title('Beta cell channel (0)\n with amplified signal', fontsize=fontsize)
    plt.xlabel('X dimension', fontsize=fontsize)
    plt.ylabel('Y dimension', fontsize=fontsize)
    plt.xticks(fontsize=fontsize, rotation=20)
    plt.yticks(fontsize=fontsize)
    plt.imshow(cells_scale, cmap='Reds')

    plt.subplot(232)
    plt.imshow(combined, extent=(0, 1024, 1024, 0))
    plt.xlabel('X dimension', fontsize=fontsize)
    plt.xlim(left=0, right=1024)
    plt.xticks(fontsize=fontsize, rotation=20)
    plt.yticks(fontsize=fontsize)
    plt.ylabel('Y dimension', fontsize=fontsize)
    plt.ylim(top=0, bottom=1024)
    plt.title('Tracked cells with \ntubes overlay', fontsize=fontsize)

    plt.subplot(233)
    plt.title('Tubes channel (1)', fontsize=fontsize)
    plt.xlabel('X dimension', fontsize=fontsize)
    plt.ylabel('Y dimension', fontsize=fontsize)
    plt.xticks(fontsize=fontsize, rotation=20)
    plt.yticks(fontsize=fontsize)
    plt.imshow(tubes_mip, cmap='viridis')
    plt.ylabel('Y dimension', fontsize=fontsize)
    plt.ylim(bottom=1024, top=0)
    plt.title('Tubes channel (1)', fontsize=fontsize)

    plt.subplot(234)
    plt.title('MIP, XZ', fontsize=fontsize)
    plt.xlabel('X dimension', fontsize=fontsize)
    plt.ylabel('Z dimension', fontsize=fontsize)
    plt.xticks(fontsize=fontsize, rotation=20)
    plt.yticks(fontsize=fontsize)
    plt.imshow(cells_mip_xz, cmap='Reds', extent=(0, 1024, 40, 0), aspect=10)
    plt.ylim(bottom=40, top=0)

    plt.subplot(236)
    plt.title('MIP, YZ', fontsize=fontsize)
    plt.xlabel('Y dimension', fontsize=fontsize)
    plt.ylabel('Z dimension', fontsize=fontsize)
    plt.xticks(fontsize=fontsize, rotation=20)
    plt.yticks(fontsize=fontsize)
    plt.imshow(cells_mip_yz, cmap='Reds', extent=(0, 1024, 40, 0), aspect=10)
    plt.ylim(bottom=40, top=0)

    start, end = min(time_range), max(time_range)
    plt.suptitle(
        f'Tracked beta cells for\nfile: {filename}, timepoint: {t}, \ntotal timepoints: ({start}-{end})',
        fontsize=30)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(save, dpi=200)
    plt.close()


def plot_markers(marker_size, colors, X, Y, P, I,
                 t, time_range, p_track):
    marker = 'x'
    marker_size_plt = marker_size
    for x, y, p, i in zip(X, Y, P, I):
        if p == p_track:
            marker = 'o'
            if t == min(time_range):
                marker_size_plt = marker_size * 3
            else:
                marker_size_plt = marker_size

        plt.plot(x, y, marker=marker,
                 markersize=marker_size_plt,
                 markeredgewidth=2,
                 markerfacecolor=(0, 0, 0, 0),
                 markeredgecolor=colors[p])

        # reset markers to default size for next timepoint
        if p == p_track:
            marker = 'x'
            marker_size_plt = marker_size


if __name__ == '__main__':
    n_frames = 30
    tic = time()
    mode = 'test'
    utils.set_cwd(__file__)

    # name = files[3]  # c.PRED_FILE
    for i, (name, range_ok) in enumerate(files):
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
