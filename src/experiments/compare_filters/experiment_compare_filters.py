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
from skimage import filters  # threshold_yen, frangi
from skimage.exposure import rescale_intensity
import skimage
import src.data.constants as c
import src.data.utils.utils as utils


def subselect_img(image):
    '''
    Returns a subregion of the image.
    '''
    return image[300:600, 300:600]


def imsave_preproc(path, image):
    '''
    Preprocesses the image before saving it.
    '''
    image = subselect_img(image)
    utils.imsave(path, image, resize=False)


mode = 'train'
img_idx = 1500

tic = time()
utils.set_cwd(__file__)

DIR = c.RAW_DATA_DIR
ext = c.IMG_EXT
files = c.RAW_FILES
KERNEL = c.MEDIAN_FILTER_KERNEL
imgs_path = join('..', c.DATA_DIR, mode, c.IMG_DIR)
filename = os.path.basename(__file__)
filename = os.path.splitext(filename)[0]

images = sorted([image for image in listdir(imgs_path) if '.npy' in image])
# Get full image paths from filename list `images`
image_paths = sorted([join(imgs_path, image) for image in images])
path = image_paths[img_idx]
img_name = images[img_idx].split('.')[0]
save = join(c.FIG_DIR, mode, img_name)

# Create image-specific directory
utils.make_dir(save)

img = np.int16(np.load(path))
img = cv2.normalize(img, None, alpha=0, beta=255,
                    dtype=cv2.CV_8UC1, norm_type=cv2.NORM_MINMAX)
# hist = cv2.calcHist([img], [0], None, [256], [0, 256])

cv2.imwrite(join(save, f'img_cv.{ext}'), img)
plt.imsave(join(save, f'img_plt.{ext}'), img)

# Operation: mean blur
operation = 'meanblur'
utils.make_dir(join(save, operation))
for i in range(1, 21, 2):
    img_blur = cv2.blur(img, (i, i))
    # img_blur = np.array(img_blur)
    # img_blur = np.where(img_blur > 5, img_blur, 0)
    name = f'{operation}_{i}'
    imsave_preproc(join(save, operation, name),
                   img_blur)

# Operation
# Median Blur
operation = 'medianblur'
utils.make_dir(join(save, operation))
for i in range(1, 21, 2):
    name = f'{operation}_{i}'
    if os.path.exists(join(save, operation, name)):
        break
    img_blur = cv2.medianBlur(img, i)
    imsave_preproc(join(save, operation, name),
                   img_blur)


# Operation
# Denoise
operation = 'denoise'
utils.make_dir(join(save, operation))
for i in range(1, 21, 2):
    for j in range(1, 10, 2):
        for k in range(1, 30, 4):
            name = f'{operation}_{i}_{j}_{k}'
            if os.path.exists(join(save, operation, name)):
                break
            img_denoise = cv2.fastNlMeansDenoising(img, None, i, j, k)
            imsave_preproc(join(save, operation, name),
                           img_denoise)


# Operation: Gaussian blur
operation = 'gaussianblur'
utils.make_dir(join(save, operation))
for kernel_size in [1, 5, 9, 15]:
    for sigma_x in [1, 5, 9]:
        for sigma_y in [1, 5, 9]:
            name = f'{operation}_{kernel_size}_{sigma_x}_{sigma_y}'
            if os.path.exists(join(save, operation, name)):
                break

            img_gauss = cv2.GaussianBlur(
                img, (kernel_size, kernel_size),
                sigma_x, sigmaY=sigma_y)

            imsave_preproc(join(save, operation, name),
                           img_gauss)


# Operation: Bilateral filter
operation = 'bilateral'
utils.make_dir(join(save, operation))
for filter_size in [50, 150]:
    for sigma_color in [50, 150]:
        for sigma_space in [5, 9]:
            name = f'{operation}_{filter_size}_{sigma_color}_{sigma_space}'
            if os.path.exists(join(save, operation, name)):
                break
            img_bilateral = cv2.bilateralFilter(
                img, filter_size, sigma_color, sigma_space)

            imsave_preproc(join(save, operation, name),
                           img_bilateral)


operation = 'canny'
utils.make_dir(join(save, operation))
for thresh1 in [20, 50, 80, 100, 150, 200]:
    for thresh2 in [20, 50, 80, 100, 150, 200]:
        for aperture_size in [3, 5, 7]:
            for L2_gradient in [True, False]:
                if os.path.exists(join(save, operation, name)):
                    break
                img = cv2.fastNlMeansDenoising(img, None, 11, 7, 21)
                # img = cv2.normalize(img, None, alpha=0,
                #                     beta=1, dtype=cv2.CV_32FC1,
                #                     norm_type=cv2.NORM_MINMAX)
                # img *= np.where((0.05 < img) & (img < 0.3), img * 3, img)
                # img = cv2.normalize(img, None, alpha=0,
                #                     beta=255, dtype=cv2.CV_8UC1,
                #                     norm_type=cv2.NORM_MINMAX)
                img_canny = cv2.Canny(
                    img, thresh1, thresh2, None,
                    apertureSize=aperture_size, L2gradient=L2_gradient)
                name = (f'canny_{thresh1}_{thresh2}'
                        f'_{aperture_size}_{L2_gradient}')
                imsave_preproc(join(save, operation, name), img_canny)

# Operation
# Simple Threshold
# operation = 'simple_threshold'
# _, thresh = cv2.threshold(img_blur, SIMPLE_THRESHOLD, 255, cv2.THRESH_BINARY)
# cv2.imwrite(f'{save}/{operation}_{img_name}.png', thresh)

# Operation
# Rescale intensity
operation = 'rescale_intensity'
yen_threshold = filters.threshold_yen(img_blur)
for thresh in range(80, 220, 20):
    bright = rescale_intensity(
        img_blur, (0, yen_threshold), (220, 255))
    imsave_preproc(join(save, operation, str(thresh)),
                   bright)

# Operation
# Adaptive Histogram Equalization
operation = 'adaptive_hist_eq'
for kernel_size in range(10, 500, 100):
    for clip_limit in np.linspace(0, 1, 11, endpoint=True):
        bright = skimage.exposure.equalize_adapthist(
            img, kernel_size, clip_limit)
        utils.imsave(
            join(save, operation, f'k_{kernel_size}_c_{clip_limit}'), bright, resize=False)
        bright_thresh = np.where(bright > 50, bright, 0)
        utils.imsave(
            join(save, operation, f'k_{kernel_size}_c_{clip_limit}_thresh'), bright_thresh, resize=False
        )

# bright = Image.fromarray(bright)

# # Operation
# # Generate and save histogram of intensified image
# operation = 'histogram_intense'
# plt.hist(bright.ravel(), 256, [0, 256])
# plt.show()
# plt.savefig(f'{save}/{img_name}_{operation}.jpg')

elapsed = utils.time_report(tic, time())
print(f'{filename} complete after {elapsed}.')
