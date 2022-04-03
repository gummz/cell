import os
import pickle
from os.path import join
from pprint import pp
from aicsimageio import AICSImage
import cv2
from matplotlib.image import BboxImage
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
import src.data.constants as c
from src.models.utils.model import get_instance_segmentation_model
from src.models.utils.utils import collate_fn
from torch.utils.data import DataLoader, Subset
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

from src.data.utils.make_dir import make_dir

from src.models.BetaCellDataset import BetaCellDataset, get_dataloaders, get_transform


def get_model(time_str):
    # Load model
    load = join(folder, f'model_{time_str}.pkl')
    model = pickle.load(open(load, 'rb'))
    return model


def get_mask(output):
    '''Consolidates the masks into one mask.'''
    if type(output) == dict:  # `output` is direct output of model (from one image)
        mask = output['masks']
    else:
        mask = output  # target is a tensor of masks (from one image)

    mask = torch.squeeze(mask, dim=1)

    if mask.shape[0] != 0:
        mask = torch.max(mask, dim=0).values
    else:
        try:
            mask = torch.zeros((mask.shape[1], mask.shape[2])).to(
                mask.get_device())
        except RuntimeError:  # no GPU
            mask = torch.zeros((mask.shape[1], mask.shape[2]))

    # mask = np.array(mask)
    # mask = np.where(mask > 255, 255, mask)
    # mask = np.where(mask > 200, 255, 0)
    mask = mask.clone().detach().type(torch.uint8)
    # torch.tensor(mask, dtype=torch.uint8)

    return mask


def get_masks(outputs):
    masks = [get_mask(output) for output in outputs]
    masks = torch.stack(masks, dim=0)

    return masks


def get_predictions(model, device, inputs):
    '''
    Input:
        inputs: A tensor or list of inputs.

    Output:
        preds: The raw output of the model.
    '''
    model.eval()

    if type(inputs) == torch.tensor:
        if len(inputs.shape) == 3:
            slices = []
            for z_slice in inputs:
                slices.append(z_slice)
            inputs = slices
        elif len(inputs.shape) == 2:
            raise ValueError('get_predictions is for lists of \
                             images. For a single image, use \
                                 get_prediction.')
    elif type(inputs) == list:
        pass

    inputs = [input.to(device) for input in inputs]

    with torch.no_grad():
        preds = model(inputs)

    return preds


def get_prediction(model, device, input):

    model.eval()

    # Predict the image with batch index `img_idx`
    # with the model
    with torch.no_grad():
        # only choose one image
        # visualize this exact image I'm sending to the model
        input = input.to(device)
        # print(np.unique(img.cpu()))
        # TODO: put target into model to see if it segments
        pred = model([input])

    # pred list only has one item because we only chose one image
    mask = pred[0]['masks']

    # Remove empty color channel dimension (since grayscale)
    # mask = torch.squeeze(mask, dim=1)
    # # Add masks together into one tensor
    # mask = torch.sum(mask, dim=0)
    # print('mask shape', mask.shape, '\n')

    # scores = np.array(pred[0]['scores'].cpu())
    # print('scores max/min', np.max(scores), np.min(scores), '\n')
    # img = cv2.normalize(input.cpu().numpy(), None, alpha=0,
    #                     beta=255, dtype=cv2.CV_8UC1, norm_type=cv2.NORM_MINMAX)
    # img = torch.tensor(img, dtype=torch.uint8)
    # img = img.repeat(3, 1, 1)

    mask = cv2.normalize(mask.cpu().numpy(), None, alpha=0,
                         beta=255, dtype=cv2.CV_8UC1, norm_type=cv2.NORM_MINMAX)

    mask = torch.tensor(mask, dtype=torch.uint8).squeeze(1)
    # mask = np.where(mask > 0, True, False)
    # mask = torch.tensor(mask, dtype=torch.bool)

    return pred[0]


def get_bbox_center(bbox, z):
    '''
    Returns the center of the input bounding box.

    Inputs:
        bbox: list containing xmin, ymin, xmax, ymax
        z: integer, the z-dimension coordinate of the bounding box
    Outut:
        center of bbox in 3D coordinates (X, Y, Z)
    '''
    pass


class CenterLink():
    '333333333333333333333333333333333333333333333333333333333333333333'

    def __init__(self, center: tuple, intensity: float):
        self.next = None
        self.prev = None
        self.center = center
        self.intensity = intensity

    def set_next(self, center):
        self.next = center

    def set_prev(self, center):
        self.prev = center

    def set_intensity(self, intensity):
        self.intensity = intensity

    def get_next(self):
        return self.next

    def get_prev(self):
        return self.prev

    def get_intensity(self):
        return self.intensity


class CenterChain():

    def __init__(self, center: CenterLink):
        links = []
        while True:
            next = center.get_next()
            if next in links:
                raise ValueError('Circular list of links detected')
            if next:
                links.append(next)
                center = next
            else:
                break
        centers = np.array([link.center for link in links])
        intensities = np.array([link.intensity for link in links])

        self.center = np.mean(centers)
        self.intensity = np.mean(intensities)

    def get_center(self):
        return self.center

    def get_intensity(self):
        return self.intensity


def find_closest_center(i: int, center: CenterLink, centers: list, searchrange: int = 1):
    '''
    Finds the next closest center in the slice after slice i.
    The found center must be within `searchrange` distance of `center`.

    Inputs:
        i: the current slice.
        center: the current center.
        centers: centers of next slice (i+1).
        searchrange: the range of which to search for a candidate center.

    Output:
        a center within `searchrange` distance, or None if none is found.
    '''
    pass


def get_chains(timepoint: np.array, preds: list, searchrange: int = 1):
    '''
    Returns the chains of the input timepoint.
    Each chain represents a cell. Each chain is composed of connected links,
    with links to next and/or previous links in the chain.
    Beginning link has no previous link. Last link has no next link.

    Input arguments:
        timepoint: 3D Numpy array of dimensions
            (D, 1024, 1024) where D is the z-dimension length.
        pred: List of model outputs of type dictionary,
            containing boxes, masks, etc. (From Mask R-CNN).
            Each element in this list represents a slice from
            the timepoint.
        searchrange: the tolerance when searching for a center belonging
            to current cell. Default: 1 pixel

    Output:
        centroids: A list for each slice. Each element in the list
            contains the centroids for that slice.
            The centroids are of type CenterLink.
            As before, the average is taken over all occurrences of the
            cell across all slices.

     963 33 2 3
    '''
    # need pixel intensity also
    # fetched from original image
    # for each instance of mask (displayed in a
    # separate tensor) we get the coordinates
    # for which the mask is nonzero.
    # Take from these coordinates, in the original
    # image, the average pixel intensity.

    # Total bounding boxes and masks (each element is for each slice)
    # Each element is also a list; but is now all elements inside the slice;
    # bounding boxes in bbox_tot, and masks in masks_tot.
    bboxes_tot = [pred['boxes'] for pred in preds]
    masks_tot = [pred['masks'] for pred in preds]
    centers_tot = []

    # Get average pixel intensities
    # for z_slice, masks in zip(timepoint, masks_tot):
    #     intensities = []
    #     for mask in masks:
    #         mask_nonzero = np.argwhere(mask)
    #         # Coordinates in original slice
    #         region = z_slice[mask_nonzero]
    #         intensity = np.mean(region)
    #         intensities.append(intensity)
    #     intensities_tot.append(intensities)
    # pred is 3d

    centroids = None
    image_iter = zip(timepoint, masks_tot, bboxes_tot)
    for z, (z_slice, masks, bboxes) in enumerate(image_iter):  # each slice
        centers = []
        for mask, bbox in zip(masks, bboxes):  # each box in slice
            center = get_bbox_center(bbox, z)
            # Coordinates in original slice
            mask_nonzero = np.argwhere(mask)
            # Get cell coordinates in original image (z_slice)
            region = z_slice[mask_nonzero]
            # Calculate average pixel intensity
            intensity = np.mean(region)

            center_link = CenterLink(center, intensity)
            centers.append(center_link)

        centers_tot.append(centers)

    for i, centers in enumerate(centers_tot):  # each slice
        for j, center in enumerate(centers):  # each center in slice
            # search in the direction of z-dimension
            # a total of `searchrange` pixels in all directions
            closest_center = find_closest_center(j, center, centers[i + 1])
            if closest_center is not None:
                closest_center.set_prev(center)
            center.set_next(closest_center)

    # Get only first link of each chain because that's all we need
    # (link is first occurring part of each cell)
    # link = cell occurrence in slice, chain = whole cell
    centroids = [center for center in centers_tot if center.get_prev() is None]
    # Get chains (= whole cells)
    # which contains average pixel intensity and average center
    # for the whole chain (i.e. over all links in the chain)
    chains = [CenterChain(center) for center in centroids]
    return chains


if __name__ == '__main__':
    c.setcwd()
    mode = 'test'
    # Running on CUDA?
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Running on {device}.')

    # Get dataloaders
    size = 1024
    img_idx = 500
    time_str = '12_03_18H_29M_39S'
    folder = f'interim/run_{time_str}'
    # data_tr, data_val = get_dataloaders(resize=size)

    dataset = BetaCellDataset(
        transforms=get_transform(train=False), resize=size, mode=mode)

    model = get_model(time_str)
    model.to(device)

    save = join(c.DATA_DIR, mode, c.PRED_DIR)

    # 1. Choose raw data file
    name = c.RAW_FILES_GENERALIZE.keys()[1]
    path = join(c.RAW_DATA_DIR, name)
    raw_data_file = AICSImage(path)
    # # Choose time range
    time_start = 0
    time_end = raw_data_file.dims['T'][0]
    # # Make directory for this raw data file
    # # i.e. mode/pred/name
    make_dir(join(c.DATA_DIR, mode, c.PRED_DIR, name))
    # 2. Loop over each timepoint at a time
    time_range = range(time_start, time_end)
    for t in time_range:
        timepoint = raw_data_file.get_image_dask_data(
            'ZYX', T=time_range, C=c.CELL_CHANNEL)
        pred = get_predictions(model, device, timepoint.compute())
        centroids = get_chains(timepoint, pred)

        np.save(join(path, f'{t:05d}'), centroids)

    # 3. Get centroids inside the loop
    # - can't store the entire prediction
    #

    # for i, (image, target) in enumerate(dataset):
    #     pred = get_prediction(model, device, image)
    #     # extract_centroids.py uses this file
    #     np.save(save, pred)

    # Get image and target
    input, target = dataset[img_idx]
    # Get mask from target
    target_mask = get_mask(target)

    img, mask = get_prediction(model, device, input)

    # Get segmentations and bounding boxes
    pred_mask = get_mask(mask)
    # Change to RGB image
    pred_mask = pred_mask.repeat(3, 1, 1)
    boxes = torch.tensor(target['boxes'])
    bboxed_image = draw_bounding_boxes(pred_mask, boxes)
    bboxed_image = torch.permute(bboxed_image, (1, 2, 0))
    print(bboxed_image.shape)
    debug_img = np.array(input)[0, :, :]

    # Print mask from model output

    plt.imsave(f'{folder}/pred_debug_mask.jpg',
               bboxed_image.cpu().detach().numpy())
    # Print image that was fed into the model
    plt.imsave(f'{folder}/pred_debug_img.jpg', img[0].cpu())
    # Print target mask from dataset
    plt.imsave(f'{folder}/pred_debug_target_mask.jpg', target_mask)

    print(target['boxes'])

    print('predict_model.py complete')
