import os
from os import listdir
from os.path import join
from time import time
import numpy as np
import matplotlib.pyplot as plt
from aicsimageio import AICSImage

import src.data.constants as c

raw_data_dir = c.RAW_DATA_DIR
files = c.RAW_FILES_GENERALIZE[c.START_IDX:]
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
i = 0
for j, (file, file_path) in enumerate(zip(files, file_paths)):
    data = AICSImage(file_path)
    T = data.dims['T'][0]

    # Set cutoff to T+1 if there is no cutoff
    # i.e., the file won't be cut off because T is the final
    # timepoint
    cutoff = cutoffs[file] if cutoffs[file] is not None else T + 1

    Z = data.dims['Z']
    print('Dimensions:')
    print(data.dims, '\n\n')
    print(data.metadata, '\n\n')

    time_idx = np.random.randint(0, min(cutoff, T), n_timepoints)

    timepoints = data.get_image_dask_data(
        'TZYX', T=time_idx, C=cell_ch).compute()
    for timepoint in timepoints:
        slice_idx = np.random.randint(0, Z, n_slices)

        slices = timepoint[slice_idx]

        for slice in slices:

            name = f'{idx:05d}'
            save = os.path.join(folder, name)
            np.save(save, slice)

            dirs = os.path.dirname(save)
            file = os.path.basename(save)

            idx = idx + 1

        if j % debug_every == 0:
            plt.imsave(f'{dirs}/_{file}.{c.IMG_EXT}', slice)



toc = time()
print(f'make_dataset.py complete after {(toc-tic)/60: .1f} minutes.')
