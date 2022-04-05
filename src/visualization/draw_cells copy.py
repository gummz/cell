import matplotlib.pyplot as plt
import numpy as np
from os.path import join
from os import makedirs, listdir
from aicsimageio import AICSImage
import aicsimageio
import tifffile as tiff
import os

import src.data.constants as c

# Set working directory to file location
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# files = RAW_FILES[START_IDX:]
files = listdir(c.RAW_DATA_DIR)
# Filter out already used files
files = [file for file in files if file not in c.RAW_FILES]
file_paths = [join(c.RAW_DATA_DIR, file) for file in files]
file_paths_lsm = [file for file in file_paths if 'lsm' in file]
file_paths_czi = [file for file in file_paths if 'czi' in file]
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

# .czi files
for t, (file, file_path) in enumerate(zip(files, file_paths_czi)):
    czi = AICSImage(file_path)
    T = czi.dims['T']

    # Create sample indexes
    if consecutive:
        randint = np.random.randint(low=0, high=T - 9)
        idx = np.arange(randint, randint + 9)
    else:
        idx = sorted([int(i) for i in np.random.randint(0, T, 9)])

    timepoint = czi.get_image_dask_data('TZYX', T=idx, C=0)
    timepoint_max = np.max(timepoint.compute(), axis=1)
    sample_images = timepoint_max

    for i, (t, image) in enumerate(zip(idx, sample_images)):
        # Create plot of sample images
        plt.subplot(3, 3, i + 1)
        plt.imshow(image)
        plt.axis('off')
        plt.title(f'Timepoint {t}', fontsize=18)

    title = f'Filename: {file_path.split("/")[-1]}\nTimepoints: {T}'
    plt.suptitle(title, fontsize=24)
    save = join(c.FIG_DIR, operation, f'{file}.jpg')
    plt.savefig(save)

# .tif files
for t, (file, file_path) in enumerate(zip(files, file_paths_lsm)):
    with tiff.TiffFile(file_path) as f:
        pages = f.pages  # f.pages[1::2]
        n_img = len(pages)

        # Create sample indexes
        if consecutive:
            randint = np.random.randint(low=0, high=n_img - 9)
            idx = np.arange(randint, randint + 9)
        else:
            idx = sorted([int(i) for i in np.random.randint(0, n_img, 9)])

        sample_images = [pages[i].asarray()[0, :, :] for i in idx]
        plt.figure(figsize=(20, 20))
        for i, (t, image) in enumerate(zip(idx, sample_images)):
            # Create plot of sample images
            plt.subplot(3, 3, i + 1)
            plt.imshow(image)
            plt.axis('off')
            plt.title(f'Page {t}', fontsize=18)

        title = f'Filename: {file_path.split("/")[-1]}\nPages: {len(f.pages)}'
        plt.suptitle(title, fontsize=24)
        save = join(FIG_DIR, operation, f'{file}.jpg')
        plt.savefig(save)

print('draw_cells.py complete')

# The images do change slightly. Successive time points were
# plotted, so it makes sense that the images would only
# change slightly.
