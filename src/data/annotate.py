import cv2
import numpy as np
import os
from os.path import join
from time import time
from os import listdir
import matplotlib.pyplot as plt
import src.data.utils.utils as utils
import src.data.constants as c

mode = 'test'

utils.set_cwd(__file__)

kernel = c.MEDIAN_FILTER_KERNEL
threshold = c.SIMPLE_THRESHOLD
files = c.RAW_FILES[mode]


# Create folder in case it doesn't exist yet
folder = join(c.DATA_DIR, mode, c.MASK_DIR)
utils.make_dir(folder)

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
    img = cv2.normalize(img, img, alpha=0, beta=255,
                        dtype=cv2.CV_8UC1, norm_type=cv2.NORM_MINMAX)
    name = f'{idx:05d}'
    # Folder to save image to
    save = join(folder, name)
    idx = idx + 1

    filtered = cv2.medianBlur(img, kernel)
    _, thresh = cv2.threshold(filtered,
                              threshold, 255, cv2.THRESH_BINARY)

    # Add image and thresholded image together
    # Because we're going to manually annotate over the
    # original image with automatic thresholds marked
    # (to avoid manually annotating unnecessarily)
    added = np.array([img, thresh], dtype=object)
    added_max = np.max(added, axis=0)

    # We don't really have a use for a saved .npy file of the
    # overlayed image only, i.e., the automatic annotations
    # overlayed on the original image. (The purpose of that
    # was that I would know what to annotate)
    # It's therefore best to save the thresholded image only
    # to .npy here.
    # This way, we can use the masks immediately, without
    # any manual annotations.
    np.save(save, thresh)

    # Save as .png for manual annotation
    dirs = os.path.dirname(save)
    file = os.path.basename(save)
    added_max = added_max.astype(np.uint16)
    plt.imsave(f'{dirs}/_{file}.{c.IMG_EXT}', added_max)


toc = time()
print(f'annotate.py on {mode} set complete after {(toc-tic)/60: .1f} minutes.')
