import matplotlib.pyplot as plt
import numpy as np
from os.path import join
from os import makedirs, listdir
from aicsimageio import AICSImage
import os
import src.data.constants as c


def draw_cells(files, file_paths, consecutive, operation):
    for i, (file, file_path) in enumerate(zip(files, file_paths)):
        czi = AICSImage(file_path)
        T = czi.dims['T']

        # Create sample indexes
        if consecutive:
            randint = np.random.randint(low=0, high=T - 9)
            idx = np.arange(randint, randint + 9)
        else:
            idx = sorted([int(i) for i in np.random.randint(0, T, 9)])

        timepoints = czi.get_image_dask_data('TZYX', T=idx, C=0)
        timepoints_max = get_MIP(timepoints.compute())
        sample_images = timepoints_max

        for j, (t, image) in enumerate(zip(idx, sample_images)):
            # Create plot of sample images
            plt.subplot(3, 3, j + 1)
            plt.imshow(image)
            plt.axis('off')
            plt.title(f'Timepoint {t}', fontsize=6)

        title = f'Filename: {file_path.split("/")[-1]}\nTimepoints: {T}'
        plt.suptitle(title, fontsize=10)
        save = join(c.FIG_DIR, operation, f'{file}.{c.IMG_EXT}')
        plt.savefig(save)


def get_MIP(timepoints: np.array):
    '''Returns Maximum Intensity Projection of
    `timepoints` array'''
    timepoints_max = np.max(timepoints, axis=1)
    return timepoints_max


if __name__ == '__main__':
    # Set working directory to file location
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    # files = RAW_FILES[START_IDX:]
    files = listdir(c.RAW_DATA_DIR)
    # Filter out already used files
    files = [file for file in files if (file not in c.RAW_FILES) & (
        file not in c.RAW_FILES_GENERALIZE)]
    # files_lsm = [file for file in files if 'lsm' in file]
    # files_czi = [file for file in files if 'czi' in file]

    file_paths = [join(c.RAW_DATA_DIR, file) for file in files]
    # file_paths_lsm = [file for file in file_paths if 'lsm' in file]
    # file_paths_czi = [file for file in file_paths if 'czi' in file]
    # file = 'LI_2019-01-17_emb7_pos3.lsm'
    # file_path = join(RAW_DATA_DIR, file)

    # whether to draw the pages in a consecutive order or random
    consecutive = False
    operation = 'draw_cells'
    # Create `FIG_DIR` directory
    try:
        makedirs(join(c.FIG_DIR, operation))
    except FileExistsError:
        pass

    draw_cells(files, file_paths, consecutive, operation)

# .lsm files
# for i, (file, file_path) in enumerate(zip(files_lsm, file_paths_lsm)):
#     lsm = AICSImage(file_path)
#     T = lsm.dims['T']

#     # Create sample indexes
#     if consecutive:
#         randint = np.random.randint(low=0, high=T - 9)
#         idx = np.arange(randint, randint + 9)
#     else:
#         idx = sorted([int(i) for i in np.random.randint(0, T, 9)])

#     timepoints = lsm.get_image_dask_data('TZYX', T=idx, C=0)
#     timepoints_max = np.max(timepoints.compute(), axis=1)
#     sample_images = timepoints_max

#     for j, (t, image) in enumerate(zip(idx, sample_images)):
#         # Create plot of sample images
#         plt.subplot(3, 3, j + 1)
#         plt.imshow(image)
#         plt.axis('off')
#         plt.title(f'Timepoint {t}', fontsize=18)

#     title = f'Filename: {file_path.split("/")[-1]}\nTimepoints: {T}'
#     plt.suptitle(title, fontsize=24)
#     save = join(c.FIG_DIR, operation, f'{file}.{c.IMG_EXT}')
#     plt.savefig(save)

print('draw_cells.py complete')

# The images do change slightly. Successive time points were
# plotted, so it makes sense that the images would only
# change slightly.
