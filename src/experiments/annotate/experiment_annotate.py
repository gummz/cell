import os
from os import listdir, makedirs
from os.path import join
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
# from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import src.data.constants as c
import src.data.utils.utils as utils




BLOCK_SIZE = 5
C = 14
DIR = c.RAW_DATA_DIR
files = c.RAW_FILES
KERNEL = c.MEDIAN_FILTER_KERNEL
imgs_path = join(c.DATA_DIR, c.IMG_DIR)


def movie():
    '''
    Prints out a movie of images (time-wise).
    '''
    figures_dir = 'figures/annotate_movie'

    # only top two have .pkl files

    file_paths = [os.path.join(DIR, file) for file in files]
    FPS = 10

    # Iterate through all files
    for j, file_path in enumerate(file_paths):
        # Current file name
        name = files[j]
        # Current file's data
        # image format: (page nr., image array)
        images = pickle.load(open(f'data/{name}.pkl', 'rb'))
        # n_img = len(images)
        # Sample image
        # idx = int(0.1*n_img)
        # n_sample = 10
        imgs = images  # [idx:idx+150][::n_sample]
        # n_sample_imgs = len(imgs)
        del images

        imgs_filter = [(img[0], cv2.GaussianBlur(img[1],
                        (11, 11), 13)) for img in imgs]

        # jpg_name = files[j].split('_')[1]

        for j, (i, image) in enumerate(imgs_filter):
            # Draw original with matplotlib
            img_copy = image.copy()

            plt.subplot(1, 2, 1)
            plt.imshow(img_copy)
            plt.axis('off')
            plt.title(f'Page {i}')

            thresh = cv2.adaptiveThreshold(
                img_copy.astype(np.uint8), 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, BLOCK_SIZE, C)

            # Draw thresholded image
            plt.subplot(1, 2, 2)
            plt.imshow(thresh, cmap='gray')
            plt.axis('off')
            plt.title(f'Page {i}, thresholded')
            save = f'series_comparison_{j+1:02d}.jpg'
            plt.savefig(os.path.join(figures_dir, save))

        image_files = [os.path.join(figures_dir, img)
                       for img in os.listdir(figures_dir)
                       if img.endswith(".jpg")]

        clip = ImageSequenceClip(sorted(image_files), fps=FPS)
        save = os.path.join(figures_dir, 'comparison_movie.mp4')
        clip.write_videofile(save)

        break  # only first file

    cv2.destroyAllWindows()

    print('annotate_test.py Movie complete')


def series():
    '''
    Prints out a series of images (time-wise).
    '''
    figures_dir = 'figures/annotate_series'

    # only top two have .pkl files
    file_paths = [os.path.join(DIR, file) for file in files]

    # Iterate through all files
    for j, file_path in enumerate(file_paths):
        # Current file name
        name = files[j]
        # Current file's data
        # image format: (page nr., image array)
        images = pickle.load(open(f'data/{name}.pkl', 'rb'))
        n_img = len(images)
        # Sample image
        idx = int(0.1*n_img)
        n_sample = 10
        imgs = images[idx:idx+150][::n_sample]  # random.choice(images)[1]
        n_imgs_sample = len(imgs)
        del images

        imgs_filter = [(img[0], cv2.GaussianBlur(img[1], (11, 11), 13))
                       for img in imgs]

        #  jpg_name = files[j].split('_')[1]

        k = 1
        m = 2

        plt.figure(figsize=(20, 25))
        for j, (i, image) in enumerate(imgs_filter):
            # Draw original with matplotlib
            img_copy = image.copy()

            print(j, k, m)

            plt.subplot(n_imgs_sample, 2, k)
            plt.imshow(img_copy)
            plt.axis('off')
            plt.title(f'Page {i}')

            thresh = cv2.adaptiveThreshold(
                img_copy.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, BLOCK_SIZE, C)

            plt.subplot(n_imgs_sample, 2, m)
            plt.imshow(thresh, cmap='gray')
            plt.axis('off')
            plt.title(f'Page {i}, thresholded')

            k = m + 1
            m = k + 1

        save = f'series_comparison_pagestart_{idx}.jpg'
        plt.savefig(os.path.join(figures_dir, save))
        # save = f'{jpg_name}_thresh_{i}.jpg'
        # cv2.imwrite(os.path.join(figures_dir,save),thresh)

        break  # only first file

    cv2.destroyAllWindows()

    print('annotate_test.py Series complete')


def test():
    '''
    Prints out the processed result of a number of options.
    For example: Gaussian/Mean preprocess filtering,
                 Gaussian Kernel and Variance, etc.
    '''
    figures_dir = c.FIG_DIR
    folder = 'annotate_gridsearch'

    images = sorted([image for image in listdir(imgs_path) if '.npy' in image])
    # Get full image paths from filename list `images`
    image_paths = sorted([join(imgs_path, image) for image in images])
    n_img = len(image_paths)

    block_sizes = [4*i+1 for i in range(1, 10)]
    Cs = [2*i for i in range(10)]

    # Sample image
    idx = 374
    img_name = images[idx].split('.')[0]

    try:
        makedirs(join(c.FIG_DIR, folder, img_name))
    except FileExistsError:
        pass

    from PIL import Image
    path = image_paths[idx]
    img = np.array(Image.open('_00304.png').convert('L'))
    img = cv2.normalize(img, img, alpha=0, beta=255,
                        dtype=cv2.CV_8UC1, norm_type=cv2.NORM_MINMAX)
    # Define various preprocessing filters
    # Both Gaussian and Mean
    filters_gaus = {f'gaussian_{i}_{k}': cv2.GaussianBlur(
        img, (i, i), k) for i in range(9, 19, 2) for k in range(1, 15, 2)}
    filters_mean = {f'median_{i}': cv2.medianBlur(
        img, i) for i in range(9, 19, 2)}
    filters = {**filters_gaus, **filters_mean}
    # Add unprocessed image to dictionary
    filters['none'] = img
    # filters = {'median_9': cv2.medianBlur(img, KERNEL)}

    #
    #
    #
    # CONTRAST afterwards or before!
    #
    #
    #

    # Draw original with matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt_save = join(figures_dir, folder, img_name,
                    f'Original_plt_{img_name}.jpg')
    plt.savefig(plt_save)

    # Draw original with opencv
    cv_save = join(figures_dir, folder, img_name,
                   f'Original_cv_{img_name}.jpg')
    cv2.imwrite(cv_save, img)

    for block_size in block_sizes:
        for C in Cs:
            for name, image in filters.items():
                # Skip over mean and none versions
                # if 'mean' in name or 'none' in name:
                #     continue
                img_copy = image.copy()

                thresh = cv2.adaptiveThreshold(
                    img_copy.astype(
                        np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, block_size, C)

                save = f'{img_name}_thresh_{block_size}_{C}_{name}.jpg'
                cv2.imwrite(os.path.join(
                    figures_dir, folder, img_name, save), thresh)

    thresholds = range(0, 240, 5)
    for threshold in thresholds:
        for name, image in filters.items():
            img_copy = image.copy()

            _, thresh = cv2.threshold(img_copy,
                                      threshold, 255, cv2.THRESH_BINARY)

            jpg_name = f'{images[idx]}_thresh_simple_{threshold}_{name}.jpg'
            save = join(figures_dir, folder, img_name, jpg_name)
            cv2.imwrite(save, thresh)

    cv2.destroyAllWindows()

    print('annotate_test.py complete')


if __name__ == '__main__':
    utils.setcwd(__file__)
    test()
