import os
from os import listdir
from os.path import join, splitext
from time import time
from PIL import Image
import matplotlib.pyplot as plt
from src.data.utils.make_dir import make_dir
import numpy as np
import src.data.utils.utils as utils
import src.data.constants as c
from aicsimageio import AICSImage
import cv2
import pandas as pd

print('make_dataset.py start')

utils.setcwd(__file__)

raw_data_dir = c.RAW_DATA_DIR
files = c.RAW_FILES.keys()
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


cutoffs = c.RAW_CUTOFFS
file_paths = [join(raw_data_dir, file) for file in files]
cell_ch = c.CELL_CHANNEL
mode = 'test'

# Create `IMG_DIR` folder, in case it doesn't exist
folder_name = c.IMG_DIR
# f'../data/interim/{folder_name}'
folder = join(c.DATA_DIR, mode, folder_name)
make_dir(folder)

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
n_timepoints = 5
n_slices = 5
idx = 0
# To store which timepoints and slices
# were randomly chosen for each file:
file_indices = []
for j, (file, file_path) in enumerate(zip(files, file_paths)):
    data = AICSImage(file_path)

    if '.czi' not in file_path:
        T = data.dims['T'][0]
        Z = data.dims['Z'][0]
    else:
        dims = utils.get_czi_dims(data.metadata)
        T = dims['T']
        Z = dims['Z']

    time_ok = c.RAW_FILES_GENERALIZE[splitext(file)[0]]
    # Set cutoff to T+1 if there is no cutoff
    # i.e., the file won't be cut off because T is the final
    # timepoint
    # cutoff = cutoffs[file] if cutoffs[file] is not None else T + 1
    time_ok = T if time_ok is None else time_ok[1]
    # time_idx = np.random.randint(0, min(cutoff, T), n_timepoints)
    # Dimension order depends on if time_idx is an int or tuple
    # order = 'ZXY' if type(time_idx) == int else 'TZXY'
    time_idx = np.random.randint(0, time_ok, n_timepoints)
    timepoints = data.get_image_dask_data(
        'TZXY', T=time_idx, C=cell_ch).compute()

    # If time_idx is an integer, then we need to create
    # an empty dimension so that we can "iterate" over it
    # in `for timepoint in timepoints`.
    # if type(time_idx) == int:
    #     timepoints = np.expand_dims(timepoints, 0)
    indices = []
    for t, timepoint in enumerate(timepoints):
        print('Timepoint', t)
        name = f'{idx:05d}'
        save = os.path.join(folder, name)

        timepoint_mip = np.max(timepoint, axis=0)

        # # z_slice = cv2.normalize(
        #     z_slice, None, alpha=0, beta=255, dtype=cv2.CV_8UC1, norm_type=cv2.NORM_MINMAX)
        # z_slice = cv2.fastNlMeansDenoising(z_slice, None, 8, 7, 21)

        np.save(save, timepoint_mip)
        print('Saved to', save)

        dirs = os.path.dirname(save)
        file = os.path.basename(save)

        if idx % debug_every == 1:
            image = Image.fromarray(timepoint_mip)
            image.save(f'{dirs}/_{file}.{c.IMG_EXT}')

        idx = idx + 1

# Record keeping over which indices were randomly selected
index_record = pd.DataFrame(file_indices)
index_record.to_csv(join(c.DATA_DIR, mode, c.IMG_DIR, 'index_record.csv'))

toc = time()
print(f'make_dataset.py complete after {(toc-tic)/60: .1f} minutes.')
