import os
from os import listdir
from os.path import join, splitext
from time import time
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import src.data.constants as c
from aicsimageio import AICSImage
import cv2

raw_data_dir = c.RAW_DATA_DIR
files = c.RAW_FILES_GENERALIZE.keys()
raw_files = listdir(raw_data_dir)
temp_files = []
for file in files:
    if f'{file}.lsm' in raw_files:
        tmp = f'{file}.lsm'
    elif f'{file}.czi' in raw_files:
        tmp = f'{file}.czi'
    elif f'{file}.ims' in raw_files:
        tmp = f'{file}.ims'
    temp_files.append(tmp)
files = temp_files


cutoffs = c.RAW_CUTOFFS_TEST
file_paths = [join(raw_data_dir, file) for file in files]
cell_ch = c.CELL_CHANNEL

# Create `IMG_DIR` folder, in case it doesn't exist
folder_name = c.IMG_DIR_TEST
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
debug_every = c.DBG_EVERY

tic = time()
n_timepoints = 10
n_slices = 5
idx = 0
for j, (file, file_path) in enumerate(zip(files, file_paths)):
    data = AICSImage(file_path)
    T = data.dims['T'][0]
    time_idx = c.RAW_FILES_GENERALIZE[splitext(file)[0]]
    # Set cutoff to T+1 if there is no cutoff
    # i.e., the file won't be cut off because T is the final
    # timepoint
    # cutoff = cutoffs[file] if cutoffs[file] is not None else T + 1

    Z = data.dims['Z']

    # time_idx = np.random.randint(0, min(cutoff, T), n_timepoints)
    # Dimension order depends on if time_idx is an int or tuple
    order = 'ZYX' if type(time_idx) == int else 'TZYX'
    timepoints = data.get_image_dask_data(
        order, T=time_idx, C=cell_ch).compute()

    # If time_idx is an integer, then we need to create
    # an empty dimension so that we can "iterate" over it
    # in `for timepoint in timepoints`.
    if type(time_idx) == int:
        timepoints = np.expand_dims(timepoints, 0)
    for timepoint in timepoints:
        for slice in timepoint:
            name = f'{idx:05d}'
            save = os.path.join(folder, name)

            slice = cv2.normalize(
                slice, None, alpha=0, beta=255, dtype=cv2.CV_8UC1, norm_type=cv2.NORM_MINMAX)
            slice = cv2.fastNlMeansDenoising(slice, None, 8, 7, 21)

            np.save(save, slice)

            dirs = os.path.dirname(save)
            file = os.path.basename(save)

            idx = idx + 1

            if idx % debug_every == 0:
                # plt.imsave(f'{dirs}/_{file}.{c.IMG_EXT}', slice)
                image = Image.fromarray(slice)
                image.save(f'{dirs}/_{file}.{c.IMG_EXT}')


toc = time()
print(f'make_dataset.py complete after {(toc-tic)/60: .1f} minutes.')
