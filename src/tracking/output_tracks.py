from __future__ import annotations
import os
import os.path as osp
from time import time
from aicsimageio import AICSImage
import cv2
import numpy as np
import pandas as pd
import skimage
import src.data.constants as c
import src.data.utils.utils as utils
import src.visualization.utils as viz
from matplotlib import pyplot as plt
import imageio
from PIL import Image


def make_tracks(tracked_centroids, time_range,
                filename, location,
                batch_idx, loops=False):
    '''
    Optional to overlay Kasra's loops over beta cells.
    '''
    # choose colormap
    cmap = 'tab20'

    # choose marker size
    marker_size = 13

    # folder is: pred/eval/track/
    # create relevant directories
    tracking_folder = osp.join('with_tracking', f'batch_{batch_idx}')
    utils.make_dir(osp.join(location, tracking_folder))

    if tracked_centroids:
        # in order to make a color map that uniquely identifies
        # each cell, the total number of cells is needed
        particles = pd.unique(tracked_centroids['particle'])
        # get unique colors for each detected cell
        colors = viz.get_colors(len(particles), cmap)
        colors_dict = dict(zip(particles, colors))

        # overlay points in each frame on corresponding
        # raw MIP image
        frames = tracked_centroids.groupby('frame')
        # filter according to time range (there could be more
        # than necessary)
        frames = [frame for frame in frames if frame[0] in time_range]
        # get unique particles from first frame
        p_tracks = get_p_tracks(frames)
        if batch_idx > len(p_tracks):
            # all available particles at the first frame are used;
            # nothing to do
            return

        p_track = p_tracks[batch_idx - 1]
    else:
        # without centroids, there are no frames; so
        # to make for-loop work, just define frames as time range
        frames = zip(time_range, time_range)

    cells_path = osp.join(c.RAW_DATA_DIR, filename)
    cells_file = AICSImage(cells_path)
    if loops:
        # example loops_path:
        # /dtu-compute/tubes/results/test/LI_2019-02-05_emb5_pos4...
        #       /cyctpy15-pred-0.7-semi-40_2019-02-05_emb5_pos4.tif
        name = filename[:-4]  # remove extension
        loops_path = osp.join(c.TUBES_DIR,
                              c.LOOP_FILES_LOC[name],
                              name,
                              name.replace(c.RAW_DATA_PREFIX,
                                           c.CYC_PREFIX) + '.tif')
        loops_file = AICSImage(loops_path)
    channels = [c.CELL_CHANNEL, c.TUBE_CHANNEL]
    columns = ['x', 'y', 'particle', 'intensity']

    for t, (_, frame) in zip(time_range, frames):
        exists = True  # assume image exists until proven otherwise
        name = f'{t:05d}'

        # don't output if images are already there
        if not os.path.exists(osp.join(location, f'{name}_xz.png')):
            exists = False

            images = output_raw_images(location, cells_file, channels, t, name)
            combined, cells, cells_xz, cells_yz, tubes = images

        plt.figure(figsize=(20, 14))

        if tracked_centroids:
            X, Y, P, I = (frame[c] for c in columns)
            for i in range(2):
                plt.subplot(2, 3, i + 1)
                # TODO: make it possible to plot an arbitrary
                # number of times (e.g. for ground truth
                # and predicted)
                # for example a list of things to plot
                # e.g. [ground_truth, predicted, loops]
                # instead of this if-else tracked_centroids,
                # loops, ...
                plot_markers(marker_size, colors_dict, X, Y,
                             P, I, t, time_range, p_track)

        if exists:
            combined, cells, cells_xz, cells_yz, tubes = load_existing(
                location, name)

        save = osp.join(location, tracking_folder, f'{t:05d}.png')
        if loops:
            loops_t = get_loops(loops_file, t)

        output_tracks(filename, t,
                      cells, cells_xz, cells_yz,
                      tubes, combined,
                      save, time_range,
                      None if not loops else loops_t)

    create_track_movie(filename, save, time_range)


def get_loops(loops_file, t):
    loops_t = utils.get_raw_array(loops_file, 0)
    loops_t = loops_t[t].compute()
    loops_t = utils.normalize(loops_t, 0, 1, out=cv2.CV_8UC1)
    return loops_t


def create_track_movie(filename, save, time_range):
    images = []
    save_dir = osp.dirname(save)
    for image in sorted(os.listdir(save_dir)):
        if image.endswith('.png') and int(image.split('.')[0]) in time_range:
            images.append(plt.imread(osp.join(save_dir, image)))

    imageio.mimsave(osp.join(save_dir,
                             f'movie_{filename}.mp4'),
                    images, fps=1)


def load_existing(location, name):
    '''
    Loads existing images. It is assumed that the images exist
    in storage.
    '''
    combined = plt.imread(osp.join(location, f'{name}.png'))
    cells = plt.imread(osp.join(location, f'{name}_cells.png'))
    cells_xz = plt.imread(osp.join(location, f'{name}_cells_xz.png'))
    cells_yz = plt.imread(osp.join(location, f'{name}_cells_yz.png'))
    tubes = plt.imread(osp.join(location, f'{name}_tubes.png'))
    return combined, cells, cells_xz, cells_yz, tubes


def get_p_tracks(frames):
    first_frame_ptc = pd.unique(frames[0][1]['particle'])
    n_tracks = min(len(first_frame_ptc), 5)
    p_tracks = np.random.choice(first_frame_ptc, n_tracks, replace=False)
    return sorted(p_tracks)


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

    combined = np.zeros((1024, 1024, 4), dtype=np.float32)
    combined[:, :, 0] = cells_scale
    combined[:, :, 1] = tubes_mip
    combined[:, :, 3] = 1.

    tubes_rgb = np.zeros((1024, 1024, 4), dtype=np.float32)
    tubes_rgb[:, :, 1] = tubes_mip
    tubes_rgb[:, :, 3] = 1.

    utils.imsave(osp.join(location, f'{name}.png'),
                 combined, False)

    # output cells and tubes
    utils.imsave(osp.join(location, f'{name}_cells.png'),
                 cells_scale, False, 'Reds')
    utils.imsave(osp.join(location, f'{name}_cells_xz.png'),
                 cells_mip_xz, False, 'Reds')
    utils.imsave(osp.join(location, f'{name}_cells_yz.png'),
                 cells_mip_yz, False, 'Reds')
    utils.imsave(osp.join(location, f'{name}_cells_raw.png'),
                 cells_mip, False, 'viridis')
    utils.imsave(osp.join(location, f'{name}_tubes.png'),
                 tubes_rgb, False)

    return (combined, cells_scale, cells_mip_xz,
            cells_mip_yz, tubes_mip)


def output_tracks(filename, t,
                  cells_scale, cells_mip_xz, cells_mip_yz,
                  tubes_mip, combined,
                  save, time_range, loops=None):

    fontsize = 25
    x_dim_str = 'X dimension'
    y_dim_str = 'Y dimension'
    z_dim_str = 'Z dimension'

    plt.subplot(231)
    plt.title('Beta cell channel (0)\n with amplified signal',
              fontsize=fontsize)
    plt.xlabel(x_dim_str, fontsize=fontsize)
    plt.ylabel(y_dim_str, fontsize=fontsize)
    plt.xticks(fontsize=fontsize, rotation=20)
    plt.yticks(fontsize=fontsize)
    plt.imshow(cells_scale, cmap='Reds')
    if loops:
        plt.imshow(loops, cmap='Greens', alpha=0.5)

    plt.subplot(232)
    plt.imshow(combined, extent=(0, 1024, 1024, 0))
    plt.xlabel(x_dim_str, fontsize=fontsize)
    plt.xlim(left=0, right=1024)
    plt.xticks(fontsize=fontsize, rotation=20)
    plt.yticks(fontsize=fontsize)
    plt.ylabel(y_dim_str, fontsize=fontsize)
    plt.ylim(top=0, bottom=1024)
    plt.title('Tracked cells with \ntubes overlay', fontsize=fontsize)

    plt.subplot(233)
    plt.title('Tubes channel (1)', fontsize=fontsize)
    plt.xlabel(x_dim_str, fontsize=fontsize)
    plt.ylabel(y_dim_str, fontsize=fontsize)
    plt.xticks(fontsize=fontsize, rotation=20)
    plt.yticks(fontsize=fontsize)
    plt.imshow(tubes_mip, cmap='viridis')
    plt.ylabel(y_dim_str, fontsize=fontsize)
    plt.ylim(bottom=1024, top=0)
    plt.title('Tubes channel (1)', fontsize=fontsize)

    plt.subplot(234)
    plt.title('MIP, XZ', fontsize=fontsize)
    plt.xlabel(x_dim_str, fontsize=fontsize)
    plt.ylabel(z_dim_str, fontsize=fontsize)
    plt.xticks(fontsize=fontsize, rotation=20)
    plt.yticks(fontsize=fontsize)
    plt.imshow(cells_mip_xz, cmap='Reds', extent=(0, 1024, 40, 0), aspect=10)
    plt.ylim(bottom=40, top=0)

    plt.subplot(236)
    plt.title('MIP, YZ', fontsize=fontsize)
    plt.xlabel(y_dim_str, fontsize=fontsize)
    plt.ylabel(z_dim_str, fontsize=fontsize)
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
                 t, time_range, p_track,
                 loops=None):
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


def get_time_range(n_frames, range_ok, load_location):
    T = len(os.listdir(osp.join(load_location, 'timepoints')))
    T = range_ok[1] if range_ok else T
    time_start = np.random.randint(0, T - n_frames)
    time_range = range(time_start, time_start + n_frames)
    return time_range


def get_present_timerange(batch_idx, save_location):
    '''Fetches the time range already present in the
    specified directory (save_location) and batch.
    '''
    iterable = os.listdir(osp.join(save_location,
                                   'with_tracking',
                                   f'batch_{batch_idx}'))
    present = sorted([int(osp.splitext(file)[0])
                      for file in iterable
                      if 'png' in file])

    return present


def tracks_to_movie(time_range, location):
    '''Saves a movie from a list of images.
    Saves the movie in the specified location,
    where it is assumed the images are (in `location`).
    '''
    utils.make_dir(osp.dirname(location))
    # append each image to the movie frames list
    # and convert the image to grayscale
    # (since the dataset is grayscale)
    movie_frames = []
    for image in sorted(os.listdir(location)):
        name, ext = osp.splitext(image)
        image_path = osp.join(location, image)
        if ext == '.png' and int(name) in time_range:
            movie_frames.append(Image.open(image_path))  # .convert('L'))

    raw_data_name, _ = osp.splitext(
        osp.dirname(osp.dirname(osp.dirname(location))))
    start, stop = time_range[0], time_range[-1]
    movie_name = f'{raw_data_name}_movie_{start}-{stop}.mp4'

    imageio.mimsave(movie_name, movie_frames, fps=1)


if __name__ == '__main__':
    n_frames = 10  # set to 'max' for all frames, or choose range
    mode = 'val'
    debug = False

    np.random.seed(42)  # so that cell sampling is reproducible
    tic = time()
    utils.set_cwd(__file__)

    files = c.RAW_FILES[mode]
    # add extensions (.lsm etc.)
    # to make the file-finding process extension-agnostic
    files = utils.add_ext(files.keys())
    files = dict(zip(files, c.RAW_FILES[mode].values()))

    loops = None  # c.LOOP_FILES_LOC

    # each batch index marks a particle in the first frame
    # batch_idx = 2: particle at index 2-1=1 is selected
    # batch_idx = 3: particle at index 3-1=2 is selected
    for batch_idx in (1,):
        for i, (name, range_ok) in enumerate(files.items()):

            load_location = osp.join(c.PROJECT_DATA_DIR,
                                     c.PRED_DIR,
                                     'reject_not', name)
            save_location = osp.join(c.PROJECT_DATA_DIR,
                                     c.PRED_DIR,
                                     'eval', 'track_2D', name)
            utils.make_dir(osp.join(save_location,
                                    'with_tracking',
                                    f'batch_{batch_idx}'))

            tracked_centroids = None  # pickle.load(
            # open(osp.join(load_location, 'tracked_centroids.pkl'),
            #      'rb'))

            if n_frames == 'max':  # max frames requested
                unique_frames = tuple(pd.unique(tracked_centroids['frame']))
                time_range = range(min(unique_frames), max(unique_frames) + 1)
            else:  # specific time range requested
                time_range = get_time_range(n_frames, range_ok, load_location)

            # if '2019-02-05_emb5_pos4' not in name:
            #     continue

            # Keep this commented for now.
            # Probably better to just take whatever cells are
            # available at the first timepoint,
            # or else we introduce bias.
            # if batch_idx > 3:
            #     # some files did not have enough particles to track
            #     # at the beginning of the file,
            #     # so we need to adjust the time range
            #     if '2018-11-20_emb7_pos3' in name:
            #         time_range = range(152, 152 + n_frames)
            #     if '2018-11-20_emb6_pos2' in name:
            #         time_range = range(160, 160 + n_frames)
            print(
                f'Outputting track range \n{tuple(time_range)} for file\n', name, f'batch {batch_idx}...\n')

            # debug: double check np.random.seed(42) works
            # i.e. that the same random numbers are generated each time
            present = get_present_timerange(batch_idx, save_location)
            print('Time range found in folder:\n', present)

            make_tracks(tracked_centroids, time_range,
                        name, save_location,
                        batch_idx, loops)

            tracks_to_movie(time_range, osp.join(
                save_location, 'with_tracking', f'batch_{batch_idx}'))

            if debug:
                print('Debugging mode, exiting.')
                break

            print('Done\n')

    elapsed = utils.time_report(tic, time())
    print(f'eval_track finished in {elapsed}.')
