from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import src.data.utils.utils as utils
from os.path import join
import src.data.constants as c
import torch
import cv2
import numpy as np


def debug_timepoint(save, t, timepoint_raw, pred, Z):
    # Randomly sample 10 slices to debug (per timepoint)
    debug_idx = np.random.randint(0, Z, 2)
    for i, z_slice in enumerate(timepoint_raw[debug_idx]):
        z_slice = np.int16(z_slice)
        z_slice = torch.tensor(z_slice).repeat(3, 1, 1)

        boxes = pred[i]['boxes']
        masks = [mask > 0.5 for mask in pred[i]['masks']]

        if len(masks) == 0:
            utils.imsave(join(save,
                              f't-{t}_{i}.jpg'), z_slice, 512)
            continue

        masks = torch.stack(masks)
        masks = masks.squeeze(1)

        z_slice = utils.normalize(z_slice, 0, 255, cv2.CV_8UC1)
        z_slice = torch.tensor(z_slice)

        masked_img = draw_segmentation_masks(z_slice, masks)
        bboxed_img = draw_bounding_boxes(masked_img, boxes)

        utils.imsave(join(save,
                          f't-{t}_{i}.jpg'), masked_img, 512)


def prepare_draw(image: np.ndarray, pred: torch.Tensor):
    '''
    Prepares data for being drawn via the draw_bounding_boxes
    and draw_segmentation_masks functions from PyTorch.
    '''
    image = image.repeat(3, 1, 1)
    boxes = pred['boxes']

    masks = [mask > 0.5 for mask in pred['masks']]

    if len(masks) != 0:
        masks = torch.stack(masks)
        masks = masks.squeeze(1)
    else:
        masks = torch.empty(0)

    image = utils.normalize(image, 0, 255, cv2.CV_8UC1)
    image = torch.tensor(image)

    return image, boxes, masks


def output_pred(mode, i, image, pred):
    image, bboxes, masks = prepare_draw(image, pred)
    save = join(c.DATA_DIR, c.PRED_DIR, 'eval', mode, str(i))
    draw_output(image, bboxes, masks, save)


def draw_output(image, boxes, masks, save: str):

    if len(boxes) != 0:
        bboxed_img = draw_bounding_boxes(image, boxes)
    else:
        bboxed_img = image

    if len(masks) != 0:
        masked_img = draw_segmentation_masks(bboxed_img, masks)
    else:
        masked_img = image

    utils.imsave(save, masked_img, False)
