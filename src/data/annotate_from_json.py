import base64
import io
import json
import os
from os import listdir
from os.path import join
from time import time
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from src.data.utils.make_dir import make_dir
import src.data.constants as c

'''
There will be folders inside masks_test called 'filename_json'.

The algorithm

1. Walk through every subfolder called
'filename_json'
2. Fetch the 'label.png' in each subfolder,
3. Threshold the image.

The manual annotation will be processed: grayscale, threshold.

test/imgs:
_00000.png
00000.npy
etc.

test/masks:
(folder): _00000_json (complete annotation from labelme script)
_00000.png (automatic annotation)

test/masks_full:
00000.npy (thresholded complete annotation)

This algorithm will be applied to the folder `imgs` in the future. 
So: imgs, masks, masks_full.
'''

# Set working directory to script location
c.setcwd()

files = c.RAW_FILES_GENERALIZE
kernel = c.MEDIAN_FILTER_KERNEL
threshold = c.SIMPLE_THRESHOLD
mode = 'test'

img_dir = c.IMG_DIR
mask_dir = c.MASK_DIR
mask_dir_full = c.MASK_DIR_FULL

# Create folder in case it doesn't exist yet
folder_name = c.MASK_DIR
folder = join(c.DATA_DIR, mode, folder_name)
make_dir(folder)
make_dir(join(c.DATA_DIR, mode, mask_dir_full))
make_dir(c.FIG_DIR)

# How often to print out with matplotlib
debug_every = c.DBG_EVERY

# Name of the folder in which the images will reside
imgs_path = join(c.DATA_DIR, mode, c.IMG_DIR)
masks_path = join(c.DATA_DIR, mode, c.MASK_DIR)
# List of filenames of the .npy images
# .jpg files are for visualizing the process
images = [image for image in listdir(imgs_path) if '.json' in image]
masks = [mask for mask in listdir(masks_path) if '.npy' in mask]
# Get full image paths from filename list `images`
image_paths = sorted([join(imgs_path, image) for image in images])

# This is the index we will start on, in case there are already
# data files in there
# So, we are only adding to the existing list of files in /imgs/
# -1 for zero-indexing, +1 because we want to start at the next free index
img_idx = len(masks) - 1 + 1
idx = img_idx if img_idx > 0 else 0  # numbering for images


thresholds = []
idx = 0
tic = time()
manual_labels = []
auto_labels = []
for folders, subfolders, files in os.walk(join(c.DATA_DIR, mode, mask_dir)):
    for file in files:
        # if '.npy' in file:
        #     img_idx = file[0:5]
        #     load_path = join(c.DATA_DIR, mask_dir)
        #     auto_img = np.load(join(load_path, f'{img_idx}.npy'))
        #     if len(auto_img.shape) > 2:
        #         print(file)
        #         print(auto_img.shape)

        if file == 'label.png':
            img_idx = folders.split('_')[-2]
            path = join(folders, file)
            added_img = PIL.Image.open(path)
            added_img = np.array(added_img)
            added_img *= 255

            _, thresh = cv2.threshold(
                added_img, c.SIMPLE_THRESHOLD, 255, cv2.THRESH_BINARY)
            save = join(c.DATA_DIR, mode, mask_dir_full)
            np.save(join(save, img_idx), thresh)

            if idx % debug_every == 0:
                plt.imsave(join(save, f'_{img_idx}.png'), thresh)  # debug
            idx += 1


toc = time()
print(f'annotate_from_json.py complete after {(toc-tic)/60: .1f} minutes.')
