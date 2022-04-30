import os
import pickle
from os import listdir
from os.path import join
from pprint import pp

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import src.data.constants as c
import src.data.utils.utils as utils
import src.models.utils.center_link as CL
import src.visualization.utils as viz
import torch
import torchvision
from aicsimageio import AICSImage
from matplotlib.image import BboxImage
from PIL import Image
from src.models.BetaCellDataset import (BetaCellDataset, get_dataloaders,
                                        get_transform)
from src.models.utils.model import get_instance_segmentation_model
from src.models.utils.utils import collate_fn
from torch.utils.data import DataLoader, Subset
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks


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
        if device.type == 'cuda':
            preds = model(inputs)
        else:
            preds = []
            for input in inputs:
                pred = model([input])[0]
                preds.append(pred)
            # preds = [model([input]) for input in inputs]

    for pred in preds:
        bboxes = pred['boxes']
        scores = pred['scores']

        # perform modified NMS algorithm
        idx = remove_boxes(bboxes, scores)
        # select boxes by idx
        pred['boxes'] = pred['boxes'][idx]
        # select masks by idx
        pred['masks'] = pred['masks'][idx]

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

    # mask = cv2.normalize(mask.cpu().numpy(), None, alpha=0,
    #                      beta=255, dtype=cv2.CV_8UC1, norm_type=cv2.NORM_MINMAX)
    # mask = torch.tensor(mask, dtype=torch.uint8).squeeze(1)
    # mask = np.where(mask > 0, True, False)
    # mask = torch.tensor(mask, dtype=torch.bool)

    '''
    NMS
    Indices need to be saved because we need to also remove the corresponding masks
    Apply NMS on all bounding boxes
    The one with the highest score is returned
    '''

    bboxes = pred[0]['boxes']
    masks = pred[0]['masks']
    scores = pred[0]['scores']

    # perform modified NMS algorithm
    idx = remove_boxes(bboxes, scores)
    # select boxes by idx
    pred[0]['boxes'] = pred[0]['boxes'][idx]
    # select masks by idx
    pred[0]['masks'] = pred[0]['masks'][idx]

    return pred[0]


def remove_boxes(bboxes, scores, threshold=0.2):
    if len(bboxes) == 0:
        return torch.tensor([], dtype=torch.int64)

    idx = torchvision.ops.nms(bboxes, scores, iou_threshold=threshold)
    tmp_idx = torch.tensor(range(len(bboxes) - 1))
    bboxes_copy = bboxes.detach().clone()
    scores_copy = scores.detach().clone()

    while len(idx) != len(tmp_idx) - 1:

        bboxes_iter = bboxes_copy[idx]
        scores_iter = scores_copy[idx]
        # need to remove top scoring box, or else the result
        # of next NMS will be identical
        bboxes_iter = bboxes_iter[1:]
        scores_iter = scores_iter[1:]

        tmp_idx = idx
        idx = torchvision.ops.nms(
            bboxes_iter, scores_iter, iou_threshold=threshold)

    # remove all bounding boxes which are inside another
    # bounding box
    bboxes_after = bboxes[tmp_idx]
    bboxes_iter = bboxes[tmp_idx]
    for i, box in enumerate(bboxes_after):
        for box2 in bboxes_after:
            if torch.equal(box, box2):
                continue
            # is box inside box2?
            if bbox_contained(box, box2):
                # delete box
                tmp_idx[i] = -1
                continue

    after_contain_idx = tmp_idx[tmp_idx != -1]

    for i, box in enumerate(bboxes[after_contain_idx]):
        area = calc_bbox_area(box)
        # reject small bounding boxes
        if area <= 4:
            after_contain_idx[i] = -1

    final_idx = after_contain_idx[after_contain_idx != -1]

    return final_idx


def calc_bbox_area(box):
    width = box[2] - box[0]
    height = box[3] - box[1]

    return width * height


def bbox_contained(box1, box2):
    cond0 = box1[0] >= box2[0]
    cond1 = box1[1] >= box2[1]
    cond2 = box1[2] <= box2[2]
    cond3 = box1[3] <= box2[3]

    box1_inside_box2 = all((cond0, cond1, cond2, cond3))

    return box1_inside_box2


def predict_ssh(raw_data_file, time_idx, device, model, save):
    chains_tot = []
    for t in time_idx:
        timepoint_raw = raw_data_file.get_image_dask_data(
            'ZXY', T=t, C=c.CELL_CHANNEL).compute()

        timepoint = prepare_mrcnn_data(timepoint_raw, device)
        pred = get_predictions(model, device, timepoint)

        # Draw bounding boxes on slice for debugging
        # Z = raw_data_file.dims['Z'][0]
        # viz.debug_timepoint(save, t, timepoint_raw, pred, Z)

        chains = CL.get_chains(timepoint, pred, c.SEARCHRANGE)
        chains_tot.append(chains)

    centroids = [(cent.get_center(), cent.get_intensity())
                 for centers in centroids
                 for cent in centers]

    centroids_final = [(centroid[0][0], centroid[0][1],
                        centroid[0][2], centroid[1])
                       for centroid in centroids
                       if not np.isnan(centroid[0]).any()]

    return centroids_final


def prepare_mrcnn_data(timepoint, device):
    ''''
    Prepares data for input to model if BetaCellDataset
    class is not being used.
    '''
    timepoint = cv2.normalize(timepoint, None, alpha=0, beta=255,
                              dtype=cv2.CV_8UC1, norm_type=cv2.NORM_MINMAX)
    z_slices = []
    for z_slice in timepoint:
        # z_slice = cv2.fastNlMeansDenoising(
        #     z_slice, z_slice, 11, 7, 21)
        z_slices.append(z_slice)
    timepoint = np.array(z_slices)
    timepoint = cv2.normalize(timepoint, None, alpha=0, beta=1,
                              dtype=cv2.CV_32F, norm_type=cv2.NORM_MINMAX)
    timepoint = torch.tensor(timepoint).to(device)
    timepoint = torch.unsqueeze(timepoint, 1)
    return timepoint


def predict_local(timepoints, device, model):
    # List to record centroids for all timepoints
    # so each element represents centroids from a
    # certain timepoint
    centroids_list = []
    for t, timepoint in enumerate(timepoints):
        #  timepoint = raw_data_file.get_image_dask_data(
        # 'ZYX', T=t, C=c.CELL_CHANNEL).compute()
        timepoint = prepare_mrcnn_data(timepoint, device)
        pred = get_predictions(model, device, timepoint)
        centroids = CL.get_chains(timepoint, pred)

        np.savetxt(join(c.DATA_DIR, f't_{t:02d}.csv'), centroids)

        centroids_list.append(centroids)

    return centroids_list


if __name__ == '__main__':
    utils.setcwd(__file__)
    mode = 'pred'
    # Running on CUDA?
    device = utils.set_device()
    print(f'Running on {device}.')

    # Get dataloaders
    size = 1024
    time_str = '29_04_21H_43M_43S'
    # data_tr, data_val = get_dataloaders(resize=size)

    model = utils.get_model(time_str, device)
    model.to(device)

    # 1. Choose raw data file
    names = listdir(c.RAW_DATA_DIR)
    names = [name for name in names if '.czi' not in name]
    name = 'LI_2019-02-05_emb5_pos3.lsm'
    # # Make directory for this raw data file
    # # i.e. mode/pred/name
    save = join(c.DATA_DIR, mode, name)
    utils.make_dir(save)

    # either ssh with full data:
    path = join(c.RAW_DATA_DIR, name)
    raw_data_file = AICSImage(path)

    T = raw_data_file.dims['T'][0]
    # ... or local with small debug file:
    # timepoints = np.load(join(c.DATA_DIR, 'sample.npy'))

    # # Choose time range
    time_start = 0
    time_end = 3

    time_range = range(time_start, time_end)
    centroids = predict_ssh(raw_data_file, time_range,
                            device, model, save)

    # pickle.dump(centroids_save, open('temp.pkl', 'wb'))
    # centroids_save = pickle.load(open('temp.pkl', 'rb'))
    # print(centroids_save)
    # center_intens = centroids_save[0][0]

    # average intensity: take only pixels above
    # a certain threshold

    # TODO: investigate nans in center_link

    # print(type(center_intens), type(
    #     center_intens[0]), type(center_intens[0][2]))
    # print([(center, intensity)
    #       for centers in centroids_save
    #       for center, intensity in centers])
    # centers_tot = []
    # for center in centroids_save:
    #     # print(centers)
    #     # print(len(centers))
    #     print(center)
    #     print(center[0][0])

    np.savetxt(
        join(save, f'{name}_{time_start}_{time_end}.csv'), centroids_save)

    # df = pd.DataFrame(centroids_save, columns=['x, y, z, i'])
    # df.to_csv(join(save, f'{name}_{time_start}_{time_end}.csv'), sep=',')

    print('predict_model.py complete')
