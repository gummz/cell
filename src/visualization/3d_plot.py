from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from os.path import join
from os import listdir, makedirs
import numpy as np
import pickle
import src.data.constants as c
import src.data.utils.utils as utils
import tifffile as tiff


utils.setcwd(__file__)

operation = '3d_plot'
save = join(c.FIG_DIR, operation)
utils.make_dir(save)


def prepare_3d(timepoint):
    points = np.array(timepoint)
    return points, get_cmap()


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


if __name__ == '__main__':
    name = 'LI_2019-02-05_emb5_pos3.lsm'
    file = 'LI_2019-02-05_emb5_pos3.lsm_0_3.csv'
    chains_tot = np.load(open(join(c.DATA_DIR, c.PRED_DIR, name, file), 'rb'))
    figures = []
    for timepoint in chains_tot:
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        points, cmap = prepare_3d(timepoint)
        X, Y, Z = points[:, :3]
        ax.scatter(X, Y, Z, c=points[:, 3], cmap=cmap)
        figures.append(fig)

    print('3d_plot.py complete')
