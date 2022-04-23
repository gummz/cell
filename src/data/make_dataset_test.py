import os
from os import listdir
from os.path import join, splitext
import random
from time import time
from PIL import Image
import numpy as np
import src.data.utils.utils as utils
import src.data.constants as c
from aicsimageio import AICSImage
import pandas as pd

mode = 'test'
random.seed(42)

utils.setcwd(__file__)

raw_data_dir = c.RAW_DATA_DIR
files = c.RAW_FILES[mode].keys()

temp_files = utils.add_ext(files)
files = temp_files

file_paths = [join(raw_data_dir, file) for file in files]
cell_ch = c.CELL_CHANNEL

# Create `IMG_DIR` folder, in case it doesn't exist
# f'../data/interim/{folder_name}'
folder = join(c.DATA_DIR, mode, c.IMG_DIR)
utils.make_dir(folder)

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
# were randomly chosen for each file
file_indices = {}
slice_record = []
for j, (file, file_path) in enumerate(zip(files, file_paths)):
    data = AICSImage(file_path)

    if '.czi' not in file_path:
        T = data.dims['T'][0]
        Z = data.dims['Z'][0]
    else:
        dims = utils.get_czi_dims(data.metadata)
        T = dims['T']
        Z = dims['Z']

    # What timepoints it's ok to use, according to excel sheet
    time_ok = c.RAW_FILES[mode][splitext(file)[0]]
    time_ok = T if time_ok is None else time_ok[1]

    time_idx = sorted(random.sample(range(time_ok), n_timepoints))
    indices = []

    for t in time_idx:
        timepoint = data.get_image_dask_data(
            'ZXY', T=t, C=cell_ch).compute()

        slice_idx = sorted(random.sample(range(Z), n_slices))

        # Make record of the indices of T and Z
        record = utils.record_dict(t, slice_idx)
        indices.append(record)

        timepoint_sliced = timepoint[slice_idx]
        for i, z_slice in enumerate(timepoint_sliced):
            name = f'{idx:05d}'
            save = os.path.join(folder, name)

            np.save(save, z_slice)

            if idx % debug_every == 0:
                dirs = os.path.dirname(save)
                filename = os.path.basename(save)
                utils.imsave(f'{dirs}/_{filename}.jpg', z_slice)

            slice_record.append((name, file, t, slice_idx[i]))
            idx = idx + 1

    # ';'.join([f'{key}:{",".join(value)}' for idx in indices for key, value in idx.items()])
    file_indices[file] = indices

# Record keeping over which indices were randomly selected
save = join(c.DATA_DIR, mode, c.IMG_DIR)

# np.savetxt(join(save, 'index_record_np.csv'), file_indices)

index_record = pd.DataFrame(file_indices, columns=list(file_indices.keys()))
index_record.to_csv(join(save, 'index_record.csv'), sep=';')

slices_record = pd.DataFrame(slice_record, columns=[
                             'name', 'file', 'timepoint', 'slice'])
slices_record.to_csv(join(save, 'slice_record.csv'), sep='\t', index=False)

toc = time()
print(f'make_dataset_test.py complete after {(toc-tic)/60: .1f} minutes.')
