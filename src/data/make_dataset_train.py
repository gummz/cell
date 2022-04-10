import os
from os import listdir
from os.path import join, splitext
from time import time
from PIL import Image
import numpy as np
import src.data.utils.utils as utils
import src.data.constants as c
from aicsimageio import AICSImage
import pandas as pd

mode = 'train'

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

    time_idx = np.random.randint(0, time_ok, n_timepoints).tolist()
    timepoints = data.get_image_dask_data(
        'TZXY', T=range(time_ok), C=cell_ch).compute()

    indices = []
    for t, timepoint in enumerate(timepoints):
        print('Timepoint', t)

        for z_slice in timepoint:
            name = f'{idx:05d}'
            save = os.path.join(folder, name)

            np.save(save, z_slice)
            print('---> Saved to', save)

            if idx % debug_every == 1:
                dirs = os.path.dirname(save)
                file = os.path.basename(save)
                image = Image.fromarray(z_slice)
                image.save(f'{dirs}/_{file}.{c.IMG_EXT}')

            idx = idx + 1

    file_indices[file] = indices

# Record keeping over which indices were randomly selected
save = join(c.DATA_DIR, mode, c.IMG_DIR)

np.savetxt(join(save, 'index_record_np.csv', file_indices))

index_record = pd.DataFrame(file_indices, columns=list(file_indices.keys()))
index_record.to_csv(join(save, 'index_record.csv'), sep='\t')

toc = time()
print(f'make_dataset_train.py complete after {(toc-tic)/60: .1f} minutes.')