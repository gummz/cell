import matplotlib.pyplot as plt
import numpy as np
from os.path import join, splitext
from os import listdir
from aicsimageio import AICSImage
import os
import src.data.constants as c
import src.data.utils.utils as utils


def draw_cells(files, file_paths, consecutive, operation):
    for i, (file, file_path) in enumerate(zip(files, file_paths)):
        data = AICSImage(file_path)
        if '.czi' not in file_path:
            T = data.dims['T'][0]
        else:
            dims = utils.get_czi_dims(data.metadata)
            T = dims['T']

        # Create sample indexes
        if consecutive:  # sample timepoints in a row
            randint = np.random.randint(low=0, high=T - 9)
            idx = np.arange(randint, randint + 9)
        else:
            idx = sorted([int(i) for i in np.random.randint(0, T, 9)])

        timepoints = data.get_image_dask_data('TZYX', T=idx, C=0)
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
    print('draw_cells.py start')
    utils.setcwd(__file__)
    print(os.getcwd())

    files = listdir(c.RAW_DATA_DIR)

    operation = 'draw_cells'
    utils.make_dir(join(c.FIG_DIR, operation))

    files_drawn = listdir(f'{c.FIG_DIR}/{operation}')
    files_drawn = [splitext(file)[0] for file in files_drawn]
    
    # Filter out files already used in train/test
    # files = [file for file in files if (file not in c.RAW_FILES) & (
    #     file not in c.RAW_FILES_GENERALIZE)]

    # Filter out files already drawn
    for file in files:
        if file in files_drawn:
            files.remove(file)
    
    file_paths = [join(c.RAW_DATA_DIR, file) for file in files]

    # whether to draw the pages in a consecutive order or random
    consecutive = False
    # Create `FIG_DIR` directory

    draw_cells(files, file_paths, consecutive, operation)

print('draw_cells.py complete')
