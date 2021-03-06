import cv2
import numpy as np
import os
from os.path import join
from time import time
from os import listdir
import matplotlib.pyplot as plt
from src.data.utils.make_dir import make_dir
import src.data.constants as c

# Set working directory to script location
c.setcwd(__file__)

files = c.RAW_FILES_GENERALIZE
kernel = c.MEDIAN_FILTER_KERNEL
threshold = c.SIMPLE_THRESHOLD
data_dir = c.DATA_DIR
mode = 'train'

# Create folder in case it doesn't exist yet
folder_name = c.MASK_DIR
folder = join(data_dir, mode, folder_name)
make_dir(folder)

# How often to print out with matplotlib
debug_every = c.DBG_EVERY

# Name of the folder in which the images will reside
imgs_path = join(c.DATA_DIR, mode, c.IMG_DIR)
masks_path = join(c.DATA_DIR, mode, c.MASK_DIR)
# List of filenames of the .npy images
# .jpg files are for visualizing the process
images = [image for image in listdir(imgs_path) if '.npy' in image]
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
for i, path in enumerate(image_paths):

    img = np.int16(np.load(path))
    # Convert to PIL image; requirement for the model
    img = cv2.normalize(img, img, alpha=0, beta=255,
                        dtype=cv2.CV_8UC1, norm_type=cv2.NORM_MINMAX)
    name = f'{idx:05d}'
    # Folder to save image to
    save = join(folder, name)
    idx = idx + 1

    filtered = cv2.medianBlur(img, kernel)
    _, thresh = cv2.threshold(filtered,
                              threshold, 255, cv2.THRESH_BINARY)
    # np.save(save, thresh)

    # Save as .jpg for debugging
    # if i % debug_every == 1:
    dirs = os.path.dirname(save)
    file = os.path.basename(save)
    plt.imsave(f'{dirs}/_{file}.{c.IMG_EXT}', thresh, cmap='gray')


toc = time()
print(f'annotate.py complete after {(toc-tic)/60: .1f} minutes.')
