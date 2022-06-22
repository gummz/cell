from __future__ import annotations
import os
import random
from matplotlib import pyplot as plt
import pandas as pd
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import src.data.utils.utils as utils
from os.path import join
import src.data.constants as c
from numpy.typing import ArrayLike
import torch
import cv2
import numpy as np


def output_sample(save: str, t: int, timepoint_raw, preds,
                  size: int = 256, device='cpu'):
    utils.make_dir(save)
    # TODO: merge this function with output_pred
    # just by indexing on timepoint_raw

    # Randomly sample 2 slices to debug (per timepoint)
    Z = timepoint_raw.shape[0]
    # debug_idx = np.random.randint(0, Z, 2)
    debug_idx = [Z - int(Z / 2), 61, Z - int(Z / 4)]
    debug_idx = range(Z)

    iterable = zip(debug_idx, timepoint_raw[debug_idx], preds)
    for idx, z_slice, pred in iterable:
        z_slice = np.int16(z_slice)

        boxes = pred['boxes']
        masks = pred['masks']

        if len(masks) == 0:
            figure = plt.figure()
            plt.imshow(z_slice)
            plt.title(f'Number of detections: {len(boxes)}')
            utils.imsave(join(save,
                              f't-{t}_{idx}.jpg'), figure, size)
            continue

        # consolidate masks into one array
        mask = utils.get_mask(masks).unsqueeze(0)
        # overlay masks onto slice

        z_slice = utils.normalize(z_slice, 0, 1, cv2.CV_32FC1, device)
        # temporarily
        z_slice = torch.tensor(z_slice, device=mask.device,
                               dtype=mask.dtype).unsqueeze(0)
        zero = torch.tensor(0, device=mask.device, dtype=mask.dtype)

        assert z_slice.shape == mask.shape, \
            'Shapes of slice and mask must match'
        assert z_slice.dtype == mask.dtype, \
            'Dtypes of slice and mask must match'

        masked_img = torch.where(mask > zero, mask, z_slice)

        bboxed_img = draw_bounding_boxes(masked_img.cpu(), boxes.cpu())[0]

        assert bboxed_img.shape == z_slice.squeeze().shape, \
            'Bounding box image and slice image shapes must match'

        images = (bboxed_img, z_slice.squeeze().cpu())
        titles = ('Prediction', 'Ground Truth')
        grid = (1, 2)
        draw_output(images, titles, grid, save, compare=True)


def prepare_draw(image: torch.Tensor, pred: torch.Tensor):
    '''
    Prepares data for being drawn via the draw_bounding_boxes
    and draw_segmentation_masks functions from PyTorch.
    '''
    # convert grayscale to RGB
    # image = image.repeat(3, 1, 1)
    # image = image.unsqueeze(0)

    boxes = pred['boxes']
    # masks = [mask > 0.5 for mask in pred['masks']]
    masks = pred['masks']

    if len(masks) != 0:
        masks = masks.squeeze(1)
    else:
        masks = torch.empty(0)

    image = utils.normalize(image, 0, 255, cv2.CV_8UC1)

    return image, boxes, masks


def get_colors(
        max_item: ArrayLike,
        colormap: str):
    '''

    Returns color value for integer p.

    If max_array contains floating values, make 

    Inputs:
        p: unique cell identifier.


    Outputs:
        color
    '''
    random.seed(42)

    # find ceiling of color scale
    if type(max_item) == int:
        # max_item is requested number of colors
        scale_max = max_item
    elif isinstance(max_item.dtype, np.dtype('float64')):
        scale_max = len(max_item)

    cmap = plt.get_cmap(colormap)
    colors = [cmap(i) for i in np.linspace(0, 1, scale_max)]

    # shuffle so that
    random.shuffle(colors)
    return colors


def output_pred(mode: str, i: int, inputs: tuple, titles: tuple[str],
                grid: tuple[int], save=None, compare: bool = False,
                dpi: int = None):
    '''
    Outputs images, and their predictions and labels. First two items of `inputs`
    are assumed to be the image array and prediction dictionary, respectively.
    '''
    image, pred = inputs[:2]
    image, bboxes, masks = prepare_draw(image, pred)

    if len(bboxes) != 0:
        bboxed_img = draw_bounding_boxes(image.cpu(), bboxes)
    else:
        bboxed_img = image.squeeze()

    if len(masks) != 0:
        masked_img = utils.get_mask(masks).cpu()
        scalar = torch.tensor(255, dtype=torch.uint8)
        # Threshold 50 comes from function "performance_mask" where threshold
        # is defined as 50 pixels out of 255
        pred_img = torch.where(masked_img > 50, scalar,
                               bboxed_img[0])
    else:
        pred_img = bboxed_img.squeeze()

    if not save:
        save = join(c.PROJECT_DATA_DIR, c.PRED_DIR,
                    'eval', mode)
    # draw_output(image[0].cpu(), pred_img.cpu().squeeze(), save, compare, dpi)
    images = (image.squeeze(), pred_img)
    if len(inputs) > 2:
        rest = inputs[2:]
        rest = (item.cpu() for item in rest)
        images = (*images, *rest)

    draw_output(images, titles, grid, save, i, compare, dpi)


def draw_output(images: tuple[torch.Tensor], titles: tuple[str],
                grid: tuple[int], save: str, idx: int = None, compare: bool = False,
                dpi: int = None):
    utils.make_dir(save)

    if compare:
        len_arr = len(titles)
        fig_size = (8 + len_arr * 2, 5 + len_arr)
        if dpi:
            figure = plt.figure(dpi=dpi, figsize=fig_size)
        else:
            figure = plt.figure(figsize=fig_size)
        x_dim, y_dim = 'X dimension', 'Y dimension'

        iterable = enumerate(zip(images, titles))
        for i, (image, title) in iterable:
            plt.subplot(*grid, i + 1)
            plt.imshow(image.squeeze())
            plt.title(title)
            plt.xlabel(x_dim)
            plt.ylabel(y_dim)

        plt.suptitle(
            f'Prediction for image {idx if idx else None}', fontsize=25)
        plt.tight_layout()
        utils.imsave(save, figure)
        plt.close()

    else:
        utils.imsave(save, images[1], False)
