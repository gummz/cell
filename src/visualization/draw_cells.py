import matplotlib.pyplot as plt
import numpy as np
from os.path import join
from os import makedirs, listdir
from aicsimageio import AICSImage
import tifffile as tiff
import os

from src.data.constants import RAW_FILES, RAW_DATA_DIR, START_IDX, FIG_DIR, RAW_FILES_GENERALIZE

# Set working directory to file location
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# files = RAW_FILES[START_IDX:]
files = listdir(RAW_DATA_DIR)
# Filter out already used files
files = [file for file in files if file not in RAW_FILES]
file_paths = [join(RAW_DATA_DIR, file) for file in files]
file_paths_tiff = [file for file in file_paths if 'tiff' in file]
file_paths_czi = [file for file in file_paths if 'czi' in file]
# file = 'LI_2019-01-17_emb7_pos3.lsm'
# file_path = join(RAW_DATA_DIR, file)

# Create `FIG_DIR` directory
consecutive = False
operation = 'draw_cells'
try:
    makedirs(join(FIG_DIR, operation))
except FileExistsError:
    pass


for j, (file, file_path) in enumerate(zip(files, file_paths_tiff)):
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
        for i, (j, image) in enumerate(zip(idx, sample_images)):
            # Create plot of sample images
            plt.subplot(3, 3, i + 1)
            plt.imshow(image)
            plt.axis('off')
            plt.title(f'Page {j}', fontsize=18)

        title = f'Filename: {file_path.split("/")[-1]}\nPages: {len(f.pages)}'
        plt.suptitle(title, fontsize=24)
        save = join(FIG_DIR, operation, f'{file}.jpg')
        plt.savefig(save)

print('draw_cells.py complete')

# The images do change slightly. Successive time points were
# plotted, so it makes sense that the images would only
# change slightly.
