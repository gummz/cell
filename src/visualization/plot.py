from os import listdir
from os.path import join, splitext

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import subprocess as sp

import pandas as pd
import src.data.constants as c
import cv2
import src.data.utils.utils as utils
from matplotlib.colors import ListedColormap
import skvideo.io
import src.visualization.utils as viz


def create_movie(location, time_range=None):

    if time_range:
        images = sorted([image for image in listdir(location)
                         if '.png' in image and
                         int(splitext(image)[0]) in time_range])
        start, end = min(time_range), max(time_range)
    else:
        images = sorted([image for image in listdir(location)
                         if '.png' in image])
        start, end = 0, len(images)

    name = f'movie_prediction_{start}_{end}.mp4'

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(join(location, name), fourcc, 1, (1200, 1548))

    # rate = '1'
    # inputdict = {'-r': rate}
    # outputdict = {'-vcodec': 'libx264', '-crf': '25', '-pix_fmt': 'yuv420p', '-vf': 'fps=1;scale=1200:1548'}
    # vid_out = skvideo.io.FFmpegWriter(join(location, name),
    #                                   inputdict, outputdict)
    # vcodecs:
    #    libx264
    # mpeg4
    # mp4als
    for i, image in enumerate(images):
        array = plt.imread(join(location, image))
        array = array[:, :, :-1]
        # print(array.shape)
        # print(np.histogram(array))
        array = utils.normalize(array, 0, 255, cv2.CV_8UC1)
        video.write(array)
        # print(array.shape)
        # print(np.histogram(array))
        print('array written is shape', array.shape)
        # vid_out.writeFrame(array)

    # vid_out.close()
    cv2.destroyAllWindows()
    video.release()


def prepare_3d(timepoint):
    points = np.array(timepoint)
    _, X, Y, Z, I, _, P = [array[0]
                           for array in np.split(points.T, 7)]
    return (X, Y, Z, I,
            np.array(P, dtype=int), get_cmap())


def get_cmap():
    # Choose colormap
    cmap = plt.cm.RdBu

    # Get the colormap colors
    my_cmap = cmap(np.arange(cmap.N))

    # Set alpha
    my_cmap[:, -1] = np.linspace(0, 1, cmap.N)

    # Create new colormap
    my_cmap = ListedColormap(my_cmap)

    return my_cmap


def save_figures(centroids, save):
    cmap = 'tab20'
    utils.make_dir(save)
    file = save.split('/')[-2]

    particles = pd.unique(centroids['particle'])
    # get unique colors for each detected cell
    colors = viz.get_colors(len(particles), cmap)
    colors_dict = dict(zip(particles, colors))

    unique_frames = tuple(pd.unique(centroids['frame']))
    time_range = range(min(unique_frames), max(unique_frames))
    frames = centroids.groupby('frame')
    for j, (_, frame) in zip(time_range, frames):
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        ax.invert_yaxis()

        ax.set_xlim3d(0, 1025)
        ax.set_ylim3d(0, 50)
        ax.set_zlim3d(1025, 0)
        ax.set_xlabel('X dimension')
        ax.set_ylabel('Z dimension')
        ax.set_zlabel('Y dimension')
        ax.set_title(f'File: {file}\nTimepoint: {j}')

        X, Y, Z, I, P, _ = prepare_3d(frame)
        marker = 'x'
        for x, y, z, i, p in zip(X, Y, Z, I, P):
            # if i < c.ACTIVE_THRESHOLD:
            #     marker = 'x'
            # else:
            #     marker = 'o'

            # switch Z and Y because we want Z to be depth
            ax.scatter(x, z, y, color=colors_dict[p], marker=marker)
            # s = f'{int(x)},{int(y)},{int(z)}'
            ax.text(x, z, y, s=p, color=colors_dict[p])

        plt.savefig(join(save, f'{j:05d}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    operation = '3d_plot'
    save = join(c.FIG_DIR, operation)
    utils.make_dir(save)


# # files = RAW_FILES[START_IDX:]
# files = listdir(c.RAW_DATA_DIR)
# # Filter out already used files
# files = [file for file in files if file in c.RAW_FILES]
# file_paths = [join(c.RAW_DATA_DIR, file) for file in files]
# file_paths_tiff = [file for file in file_paths if '.lsm' in file]

# fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

# file_idx = 0
# file_path = file_paths_tiff[file_idx]
# with tiff.TiffFile(file_path) as f:
#     pages = np.array(f.pages[::2])

#     pages = pages.reshape(
#         (c.TIMEPOINTS[file_idx], c.RAW_FILE_DIMENSIONS[files[file_idx]]))
#     timepoint = pages[0, :]
#     timepoint = [slice.asarray()[0, :, :] for slice in timepoint]
#     timepoint = np.array(timepoint)
#     timepoint = np.moveaxis(timepoint, 0, 2)
#     size = 512
#     r = range(size)
#     r38 = np.linspace(0, 1, 38)
#     X, Y, Z = np.meshgrid(r, r, r38)
#     print(X.shape, Y.shape, Z.shape)

#     # Choose colormap
#     cmap = plt.cm.RdBu

#     # Get the colormap colors
#     my_cmap = cmap(np.arange(cmap.N))

#     # Set alpha
#     my_cmap[:, -1] = np.linspace(0, 1, cmap.N)

#     # Create new colormap
#     my_cmap = ListedColormap(my_cmap)

#     # X = timepoint[0]
#     # Y = timepoint[1]
#     # Z = timepoint[2]
#     # # ax.plot_surface(mesh)
#     timepoint = np.where(timepoint > 100, timepoint, 0)
#     ax.scatter(X, Y, Z, c=timepoint[0:size, 0:size, :], cmap=my_cmap)

#     plt.savefig(join(save, 'plot.jpg'))
