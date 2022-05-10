from __future__ import annotations
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


def output_sample(save, t, timepoint_raw, pred, Z, size=512, device=None):
    # TODO: merge this function with output_pred
    # just by indexing on timepoint_raw
    # Randomly sample 2 slices to debug (per timepoint)
    debug_idx = np.random.randint(0, Z, 2)
    for i, (idx, z_slice) in enumerate(zip(debug_idx, timepoint_raw[debug_idx])):
        z_slice = np.int16(z_slice)
        z_slice = torch.tensor(z_slice).repeat(3, 1, 1)

        boxes = pred[i]['boxes']
        masks = pred[i]['masks']  # [mask > 0.5 for mask in pred[i]['masks']]

        if len(masks) == 0:
            utils.imsave(join(save,
                              f't-{t}_{idx}.jpg'), z_slice, size)
            continue

        # masks = torch.stack(masks)
        # masks = masks.squeeze(1).to(device)

        z_slice = utils.normalize(z_slice, 0, 255, cv2.CV_8UC1)
        z_slice = torch.tensor(z_slice, device=device).unsqueeze(0)

        # consolidate masks into one array
        mask = utils.get_mask(masks).unsqueeze(0)
        # overlay masks onto slice

        masked_img = torch.where(mask > 0, mask, z_slice[0])

        bboxed_img = draw_bounding_boxes(masked_img.cpu(), boxes.cpu())

        utils.imsave(join(save,
                          f't-{t}_{idx}.jpg'), bboxed_img, 512)


def prepare_draw(image: torch.Tensor, pred: torch.Tensor):
    '''
    Prepares data for being drawn via the draw_bounding_boxes
    and draw_segmentation_masks functions from PyTorch.
    '''
    # convert grayscale to RGB
    # image = image.repeat(3, 1, 1)
    image = image.unsqueeze(0)

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
        max_array: ArrayLike,
        colormap: str):
    '''
    Returns color value for integer p.

    If max_array contains floating values, make 

    Inputs:
        p: unique cell identifier.


    Outputs:
        color
    '''
    # find ceiling of color scale
    scale_max = np.max(max_array)
    if type(scale_max) == np.float64:
        scale_max = len(max_array)
    cmap = plt.get_cmap(colormap)
    colors = [cmap(i) for i in np.linspace(0, 1, scale_max)]

    return colors


def output_pred(mode, i, image, pred, save=None, compare=False):
    image, bboxes, masks = prepare_draw(image, pred)

    if len(bboxes) != 0:
        bboxed_img = draw_bounding_boxes(image.cpu(), bboxes)
    else:
        bboxed_img = image

    if len(masks) != 0:
        masked_img = utils.get_mask(masks)  # .repeat(3, 1, 1)
        # TODO: use torch.where instead of this
        stack = torch.stack([masked_img.cpu(), bboxed_img[0]])
        pred_img = (torch.max(stack, dim=0, keepdim=True)[0]
                    .squeeze().squeeze())

    else:
        pred_img = bboxed_img

    if not save:
        save = join(c.DATA_DIR, c.PRED_DIR, 'eval', mode, str(i))

    draw_output(image[0].cpu(), pred_img, save, compare)


def draw_output(image, pred_img, save, compare):

    if compare:
        figure = plt.figure()
        x_dim, y_dim = 'X dimension', 'Y dimension'

        plt.subplot(121)
        plt.imshow(image)
        plt.title('Original image')
        plt.xlabel(x_dim), plt.ylabel(y_dim)

        plt.subplot(122)
        plt.imshow(pred_img)
        plt.title('Prediction')
        plt.xlabel(x_dim), plt.ylabel(y_dim)

        utils.imsave(save, figure)

    else:
        utils.imsave(save, pred_img, False)
