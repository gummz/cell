import cv2
import numpy as np
import os
from os.path import join
from os import mkdir
import matplotlib.pyplot as plt

from src.data.constants import (
    RAW_FILES, START_IDX,
    MEDIAN_FILTER_KERNEL, SIMPLE_THRESHOLD, DATA_DIR,
    IMG_DIR, MASK_DIR, FIG_DIR)

# Set working directory to script location
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Create figures directory in case it doesn't exist
try:
    mkdir(FIG_DIR)
except FileExistsError:
    pass

files = RAW_FILES
kernel = MEDIAN_FILTER_KERNEL
threshold = SIMPLE_THRESHOLD
data_path = DATA_DIR
imgs_dir = IMG_DIR
masks_dir = MASK_DIR
img_name = '00570.npy'
img_path = join(data_path, imgs_dir, img_name)

img = np.int16(np.load(img_path))
img = cv2.normalize(img, img, alpha=0, beta=255,
                    dtype=cv2.CV_8UC1, norm_type=cv2.NORM_MINMAX)

filtered = cv2.medianBlur(img, kernel)
_, thresh = cv2.threshold(filtered,
                          threshold, 255, cv2.THRESH_OTSU)

# Save as .jpg for debugging
plt.imsave(f'{FIG_DIR}/{img_name}.jpg', img)
plt.imsave(f'{FIG_DIR}/{img_name}_thresh.jpg', thresh, cmap='gray')


print('annotate_single.py complete')
