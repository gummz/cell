import os
from os import listdir, makedirs
from os.path import join
import pickle
import sys
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from time import time
from skimage.filters import threshold_yen, frangi
from skimage.exposure import rescale_intensity
from skimage.io import imread, imsave
# from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import src.data.constants as c
import src.data.utils.utils as utils

tic = time()
utils.setcwd(__file__)

DIR = c.RAW_DATA_DIR
ext = c.IMG_EXT
files = c.RAW_FILES
KERNEL = c.MEDIAN_FILTER_KERNEL
mode = 'train'
imgs_path = join('..', c.DATA_DIR, mode, c.IMG_DIR)
filename = os.path.basename(__file__)
filename = os.path.splitext(filename)[0]


figures_dir = c.FIG_DIR
folder = filename
modification = ''


images = sorted([image for image in listdir(imgs_path) if '.npy' in image])
# Get full image paths from filename list `images`
image_paths = sorted([join(imgs_path, image) for image in images])
img_idx = 1000
path = image_paths[img_idx]
img_name = images[img_idx].split('.')[0]
save = join(c.FIG_DIR, mode, img_name)

# Create image-specific directory
utils.make_dir(save)

img = np.int16(np.load(path))
img = cv2.normalize(img, None, alpha=0, beta=255,
                    dtype=cv2.CV_8UC1, norm_type=cv2.NORM_MINMAX)
# hist = cv2.calcHist([img], [0], None, [256], [0, 256])


'''

'''

# Operation
# Median Blur
# operation = 'medianblur'
# for i in range(1, 21, 2):
#     img_blur = cv2.medianBlur(img, i)
#     img_blur = np.array(img_blur)
#     img_blur = np.where(img_blur > 5, img_blur, 0)
#     cv2.imwrite(f'{save}/{operation}_{img_name}_{i}.{c.IMG_EXT}', img_blur)

cv2.imwrite(join(save, f'img_cv.{ext}'), img)
plt.imsave(join(save, f'img_plt.{ext}'), img)

# Operation
# Denoise
# operation = 'denoise'
# for i in range(1, 21, 2):
#     for j in range(1, 10, 2):
#         for k in range(1, 30, 4):
#             img_denoise = cv2.fastNlMeansDenoising(img, None, i, j, k)
#             cv2.imwrite(
#                 f'{save}/{operation}_cv2_{img_name}_{i}_{j}_{k}.{png}', img_denoise)
#             plt.imsave(
#                 f'{save}/_TEST_{operation}_plt_{img_name}_{i}_{j}_{k}.{ext}', img_denoise)

# operation = 'frangi'
# for alpha in np.linspace(0.1, 1, 10):
#     for beta in np.linspace(0.1, 1, 10):
#         for gamma in np.linspace(1, 30, 5):
#             img_frangi = frangi(img, alpha=alpha, beta=beta,
#                                 gamma=gamma, black_ridges=False)
#             name = f'{operation}_plt_{img_name}_{alpha:.2f}_{beta}_{gamma}'
#             plt.imsave(f'{save}/{name}.{ext}', img_frangi)

# TODO: Canny edge detection
# img_canny_norm = cv2.normalize(img, None, alpha=0, beta=255,
#                                dtype=cv2.CV_8UC1, norm_type=cv2.NORM_MINMAX)
operation = 'canny'
for thresh1 in [20, 50, 80, 100, 150, 200][-2:]:
    for thresh2 in [20, 50, 80, 100, 150, 200][-2:]:
        for aperture_size in [3, 5, 7]:
            for L2_gradient in [True, False]:
                img = cv2.fastNlMeansDenoising(img, None, 11, 7, 21)
                img = cv2.normalize(img, None, alpha=0,
                                    beta=1, dtype=cv2.CV_32FC1,
                                    norm_type=cv2.NORM_MINMAX)
                img *= np.where((0.05 < img) & (img < 0.3), img * 3, img)
                img = cv2.normalize(img, None, alpha=0,
                                    beta=255, dtype=cv2.CV_8UC1,
                                    norm_type=cv2.NORM_MINMAX)
                img_canny = cv2.Canny(
                    img, thresh1, thresh2, None,
                    apertureSize=aperture_size, L2gradient=L2_gradient)
                name = (f'canny_{thresh1}_{thresh2}'
                        f'_{aperture_size}_{L2_gradient}')
                utils.imsave(join(save, name), img_canny, 512)

# Operation
# Simple Threshold
# operation = 'simple_threshold'
# _, thresh = cv2.threshold(img_blur, SIMPLE_THRESHOLD, 255, cv2.THRESH_BINARY)
# cv2.imwrite(f'{save}/{operation}_{img_name}.png', thresh)

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

elapsed = utils.time_report(tic, time())
print(f'{filename} complete after {elapsed}.')
