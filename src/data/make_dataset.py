import os
from os import listdir
from os.path import join
from time import time
import cv2
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

from constants import (MEDIAN_FILTER_KERNEL, RAW_DATA_DIR, RAW_FILE_DIMENSIONS,
                       RAW_FILES, RAW_CUTOFFS, IMG_DIR, DBG_EVERY, START_IDX)

raw_data_dir = RAW_DATA_DIR
files = RAW_FILES[START_IDX:]
cutoffs = RAW_CUTOFFS
file_paths = [join(raw_data_dir, file) for file in files]
Ds = RAW_FILE_DIMENSIONS

# Create `IMG_DIR` folder, in case it doesn't exist
folder_name = IMG_DIR
folder = f'../data/interim/{folder_name}'
try:
    os.mkdir(folder)
except FileExistsError:
    pass

# Get files in imgs folder - need to know what's in there so
# we don't start the index at 0
images = [image for image in listdir(folder) if '.npy' in image]

# This is the index we will start on, in case there are already
# data files in there
# So, we are only adding to the existing list of files in /imgs/
# -1 for zero-indexing, +1 because we want to start at the next free index
img_idx = len(images) - 1 + 1
idx = img_idx if img_idx > 0 else 0  # numbering for images
# How often to print out with matplotlib
debug_every = DBG_EVERY

tic = time()
for j, (file, file_path) in enumerate(zip(files, file_paths)):
    file = files[j]
    cutoff = cutoffs[file]
    with tiff.TiffFile(file_path) as f:
        n = len(f.pages)
        print(f'Number of pages: {n}')

        # Pages are the images of each file
        pages = f.pages[::2]  # only every other page has beta cells
        D = Ds[file]  # dimension of current file
        # pages / D
        # We want this to pass; otherwise, the z-dimension doesn't add up.
        # assert type(int(len(pages)) / int(D)) == int

        for i, page in enumerate(pages):
            if i % D == 0:
                # Save the dimension in case we can't use all
                # frames, in which case we need to stop early
                # in image_list below
                old_D = D
                # If the next MIP goes beyond cutoff, take
                # only the z-values up to cutoff
                if cutoff is not None and i + D > cutoff:
                    D = cutoff
                name = f'{idx:05d}'
                idx = idx + 1
                save = os.path.join(folder, name)
                image_list = [page.asarray()[0, :, :]
                              for page in pages[i: i + D]]

                image_max = np.max(image_list, axis=0)
                image_max = cv2.normalize(image_max, image_max, alpha=0, beta=255,
                                          dtype=cv2.CV_8UC1, norm_type=cv2.NORM_MINMAX)
                image_max = cv2.fastNlMeansDenoising(
                    image_max, None, 11, 7, 21)
                np.save(save, image_max)

                # Save intermittently to .jpg for debugging
                if i % (old_D * debug_every) == 0:
                    dirs = os.path.dirname(save)
                    file = os.path.basename(save)
                    plt.imsave(f'{dirs}/_{file}.jpg', image_max)

            # -1 because index was updated to +1 above
            # print(f'Image {idx-1} saved.')

toc = time()
print(f'make_dataset.py complete after {(tic-toc)/60: .1f}')
