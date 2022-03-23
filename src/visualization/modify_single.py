import os
from os import listdir, makedirs
from os.path import join
import pickle
import sys
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from skimage.filters import threshold_yen
from skimage.exposure import rescale_intensity
from skimage.io import imread, imsave
# from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from src.data.constants import FIG_DIR, DATA_DIR, IMG_DIR, MEDIAN_FILTER_KERNEL, BLOCK_SIZE, C, SIMPLE_THRESHOLD, RAW_DATA_DIR, RAW_FILES, IMG_EXT, C, BLOCK_SIZE

# Set working directory to script location
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


DIR = RAW_DATA_DIR
files = RAW_FILES
KERNEL = MEDIAN_FILTER_KERNEL
imgs_path = join(DATA_DIR, f'{IMG_DIR}')


figures_dir = FIG_DIR
folder = ' modify_single'
modification = ''


images = sorted([image for image in listdir(imgs_path) if '.npy' in image])
# Get full image paths from filename list `images`
image_paths = sorted([join(imgs_path, image) for image in images])
img_idx = 1500
path = image_paths[img_idx]
img_name = images[img_idx].split('.')[0]
save = join(figures_dir, folder, img_name)

# Create image-specific directory
try:
    makedirs(save)
except FileExistsError:
    pass

img = np.int16(np.load(path))
img = cv2.normalize(img, img, alpha=0, beta=255,
                    dtype=cv2.CV_8UC1, norm_type=cv2.NORM_MINMAX)
# hist = cv2.calcHist([img], [0], None, [256], [0, 256])


# Operation
# Median Blur
operation = 'medianblur'
for i in range(1, 21, 2):
    img_blur = cv2.medianBlur(img, i)
    img_blur = np.array(img_blur)
    img_blur = np.where(img_blur > 5, img_blur, 0)
    cv2.imwrite(f'{save}/{operation}_{img_name}_{i}.{IMG_EXT}', img_blur)

cv2.imwrite(join(save, f'img_cv.{IMG_EXT}'), np.load(path))
plt.imsave(join(save, f'img_plt.{IMG_EXT}'), np.int16(np.load(path)))

# Operation
# Denoise
operation = 'denoise'
for i in range(1, 21, 2):
    for j in range(1, 10, 2):
        for k in range(1, 30, 4):
            img_denoise = cv2.fastNlMeansDenoising(img, None, i, j, k)
            cv2.imwrite(
                f'{save}/{operation}_cv2_{img_name}_{i}_{j}_{k}.png', img_denoise)
            plt.imsave(
                f'{save}/_TEST_{operation}_plt_{img_name}_{i}_{j}_{k}.png', img_denoise)
            sys.exit()  

# Operation
# Simple Threshold
operation = 'simple_threshold'
_, thresh = cv2.threshold(img_blur, SIMPLE_THRESHOLD, 255, cv2.THRESH_BINARY)
cv2.imwrite(f'{save}/{operation}_{img_name}.png', thresh)

# # Operation
# # Rescale intensity
# operation = 'rescale_intensity'
# yen_threshold = threshold_yen(img_blur)
# bright = rescale_intensity(img_blur, (0, yen_threshold), (0, 255))
# # bright = Image.fromarray(bright)

# cv2.imwrite(f'{save}/{img_name}_{operation}.jpg', bright)

# # Operation
# # Generate and save histogram
# operation = 'histogram'
# plt.hist(img_blur.ravel(), 256, [0, 256])
# plt.show()
# plt.savefig(f'{save}/{img_name}_{operation}.jpg')

# # Operation
# # Generate and save histogram of intensified image
# operation = 'histogram_intense'
# plt.hist(bright.ravel(), 256, [0, 256])
# plt.show()
# plt.savefig(f'{save}/{img_name}_{operation}.jpg')


print('modify_single.py complete')
