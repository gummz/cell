from __future__ import annotations

import os
import pickle
from multiprocessing.spawn import prepare
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
from src.tracking.track_cells import track
from src.visualization import plot
from torch.utils.data import DataLoader, Subset
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks


def get_masks(outputs):
    masks = [get_mask(output) for output in outputs]
    masks = torch.stack(masks, dim=0)

    return masks


def get_predictions(model,
                    device: torch.device,
                    inputs: list[torch.Tensor]) -> list(dict(torch.Tensor)):
    '''
    Input:
        inputs: A tensor or list of inputs.

    Output:
        preds: The raw output of the model.
    '''
    model.eval()

    # if len(inputs.shape) == 3:
    #     pass
    #     # if type(inputs) == np.ndarray:
    #     #     inputs = prepare_model_input(inputs, device)
    #     # elif type(inputs) == list:
    #     #     inputs = torch.stack(inputs)

    #     # inputs = prepare_model_input(inputs, device)

    # elif len(inputs.shape) == 2:
    #     raise ValueError('get_predictions is for lists of \
    #                         images. For a single image, use \
    #                             get_prediction.')
    # else:
    #     inputs = torch.stack(inputs)

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
        pred = model([input])

    bboxes = pred[0]['boxes']
    scores = pred[0]['scores']

    # # perform modified NMS algorithm
    idx = remove_boxes(bboxes, scores)
    # select boxes by idx
    pred[0]['boxes'] = pred[0]['boxes'][idx]
    # select masks by idx
    pred[0]['masks'] = pred[0]['masks'][idx]

    return pred[0]


def remove_boxes(bboxes, scores, threshold=0.3):
    if len(bboxes) == 0:
        return torch.tensor([], dtype=torch.int64)

    select = scores > 0.2
    bboxes = bboxes[select]
    scores = scores[select]

    idx = torchvision.ops.nms(bboxes, scores, iou_threshold=threshold)

    # #TODO: better: find clusters of bounding boxes, and run
    # NMS on them separately
    # KMeans?
    # input to KMeans for each box will be its center

    # tmp_idx = torch.tensor(range(len(bboxes)))
    # bboxes_copy = bboxes.detach().clone()
    # scores_copy = scores.detach().clone()
    # bboxes_iter = None
    # i = 0
    # print('iter start:', len(bboxes), len(idx))
    # while len(idx) != len(tmp_idx) - 1:
    #     i += 1
    #     if i > 100:
    #         print('100 iterations of nms!')
    #         print(bboxes_iter)
    #         print((idx), (tmp_idx))
    #         exit()
    #     print('nms iter')
    #     bboxes_iter = bboxes_copy[idx]
    #     scores_iter = scores_copy[idx]
    #     # need to remove top scoring box, or else the result
    #     # of next NMS will be identical
    #     bboxes_iter = bboxes_iter[1:]
    #     scores_iter = scores_iter[1:]

    #     tmp_idx = idx
    #     idx = torchvision.ops.nms(
    #         bboxes_iter, scores_iter, iou_threshold=threshold)

    # print('iter end', idx)

    # remove all bounding boxes which are inside another
    # bounding box
    bboxes_after = bboxes[idx]
    for i, box in enumerate(bboxes_after):
        for box2 in bboxes_after:
            if torch.equal(box, box2):
                continue
            # is box inside box2?
            if bbox_contained(box, box2):
                # delete box
                idx[i] = -1
                continue

    after_contain_idx = idx[idx != -1]

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


def predict_ssh(raw_data_file: AICSImage,
                time_idx: range,
                device: torch.device,
                model, save: str) -> list(tuple):
    path = join(c.RAW_DATA_DIR, c.PRED_FILE)
    chains_tot = []

    for t in time_idx:
        timepoint_raw = utils.get_raw_array(
            path, t=t).compute()

        # prepare data for model, since we're not using
        # BetaCellDataset class
        timepoint = prepare_model_input(timepoint_raw, device)
        # pred = get_predictions(model, device, timepoint)
        preds = get_predictions(model, device, timepoint)

        # Draw bounding boxes on slice for debugging
        Z = raw_data_file.dims['Z'][0]
        viz.output_sample(join(save, 'debug'), t,
                          timepoint_raw, preds, Z, 1024, device)

        # adjust slices from being model inputs to being
        # inputs to get_chains
        timepoint = [z_slice.squeeze() for z_slice in timepoint]
        # get center and intensity of all cells in timepoint
        chains = CL.get_chains(timepoint, preds, c.SEARCHRANGE)
        chains_tot.append(chains)

    return chains_tot


def prepare_model_input(array, device):
    ''''
    Prepares data for input to model if BetaCellDataset
    class is not being used.

    Inputs:
        array: the data to be prepared for the Mask R-CNN.
        This array can contain individual slices, an entire
        timepoint, or multiple timepoints.
    '''

    # timepoint = np.int16(timepoint)
    # timepoint = cv2.normalize(timepoint, None, alpha=0, beta=255,
    #                           dtype=cv2.CV_8UC1, norm_type=cv2.NORM_MINMAX)
    # z_slices = []
    # for z_slice in timepoint:
    #     # z_slice = cv2.fastNlMeansDenoising(
    #     #     z_slice, z_slice, 11, 7, 21)
    #     z_slices.append(z_slice)
    # timepoint = np.array(z_slices)

    array = utils.normalize(array, 0, 1, cv2.CV_32F, device)
    array = torch.tensor(array, device=device)

    dim_inputs = len(array.shape)
    dim_squeeze = len(array.squeeze())
    if dim_inputs != dim_squeeze:
        array = torch.unsqueeze(array, -3)

    # turn 4d tensor into list of 3d tensors
    # (format which is required by model)
    if len(array.shape) == 4:
        array = [item for item in array]

    return array


def predict_local(timepoints, device, model):
    # List to record centroids for all timepoints
    # so each element represents centroids from a
    # certain timepoint
    centroids_list = []
    for t, timepoint in enumerate(timepoints):
        #  timepoint = raw_data_file.get_image_dask_data(
        # 'ZYX', T=t, C=c.CELL_CHANNEL).compute()
        timepoint = prepare_model_input(timepoint, device)
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

    model = utils.get_model(c.MODEL_STR, device)
    model.to(device)

    # 1. Choose raw data file
    # names = listdir(c.RAW_DATA_DIR)
    # names = [name for name in names if '.czi' not in name]
    name = c.PRED_FILE
    # # Make directory for this raw data file
    # # i.e. mode/pred/name
    save = join(c.DATA_DIR, mode, name)
    utils.make_dir(save)

    # either ssh with full data:
    path = join(c.RAW_DATA_DIR, c.PRED_FILE)
    raw_data_file = AICSImage(path)

    T = raw_data_file.dims['T'][0]
    # ... or local with small debug file:
    # timepoints = np.load(join(c.DATA_DIR, 'sample.npy'))

    # # Choose time range
    time_start = 0
    time_end = 10

    time_range = range(time_start, time_end)
    # centroids = predict_ssh(raw_data_file, time_range,
    #                         device, model, save)
    # pickle.dump(centroids, open(join(save, 'centroids_save.pkl'), 'wb'))
    centroids = pickle.load(open(join(save, 'centroids_save.pkl'), 'rb'))

    centroids_np = [(t, centroid[0], centroid[1],
                     centroid[2], centroid[3])
                    for t, timepoint in enumerate(centroids)
                    for centroid in timepoint]
    np.savetxt(
        join(save, f'{name}_{time_start}_{time_end}.csv'), centroids_np)

    tracked_centroids = track(centroids_np)

    pickle.dump(tracked_centroids,
                open(join(save, 'tracked_centroids.pkl'), 'wb'))

    location = join(save, 'timepoints')
    plot.save_figures(tracked_centroids, location)
    plot.create_movie(location, time_range)

    # df = pd.DataFrame(centroids_save, columns=['x, y, z, i'])
    # df.to_csv(join(save, f'{name}_{time_start}_{time_end}.csv'), sep=',')

    print('predict_model.py complete')
