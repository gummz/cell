from __future__ import annotations

import pickle
import os.path as osp
import cv2
import numpy as np
import src.data.constants as c
import src.data.utils.utils as utils
import src.models.utils.center_link as CL
from torchvision.transforms import functional as F
import src.visualization.utils as viz
import torch
import torchvision
from PIL import Image
from aicsimageio import AICSImage
import src.tracking.track_cells as tracker
from src.visualization import plot
from time import time


def get_predictions(model,
                    device: torch.device,
                    inputs: list[torch.Tensor],
                    accept_range: tuple = (0.5, 1)) -> list(dict(torch.Tensor)):
    '''
    Input:
        model: the model with which to perform the predictions.
        device: the device in which the model resides.
        inputs: list of tensors. Inputs to the model.
        reject_threshold: the threshold at which to reject
            uncertain predictions.

    Output:
        preds: The raw output of the model.
    '''
    model.eval()

    with torch.no_grad():
        if device.type == 'cuda':
            preds = model(inputs)
        else:
            preds = []
            for input in inputs:
                pred = model([input])[0]
                preds.append(pred)

    for pred in preds:
        # perform modified NMS algorithm
        idx = remove_boxes(pred['boxes'], pred['scores'],
                           accept_range)
        # select boxes by idx
        pred['boxes'] = pred['boxes'][idx]
        # select masks by idx
        pred['masks'] = pred['masks'][idx]

        pred['masks'] = threshold_masks(pred['masks'])

    return preds


def get_prediction(model, device, input, accept_range=(0.5, 1)):

    model.eval()

    # Predict the image with batch index `img_idx`
    # with the model
    with torch.no_grad():
        pred = model([input])[0]

    bboxes = pred['boxes']
    scores = pred['scores']

    # perform modified NMS algorithm
    idx = remove_boxes(bboxes, scores, accept_range)
    # select boxes by idx
    pred['boxes'] = pred['boxes'][idx]

    # select masks by idx
    pred['masks'] = pred['masks'][idx]
    # select scores by idx
    pred['scores'] = pred['scores'][idx]

    return pred


def threshold_masks(masks, threshold=0.8):
    '''
    Thresholds masks to make more concise
    segmentations (i.e., model has to be more certain
    in its mask predictions).
    '''
    dtype, device = masks.dtype, masks.device
    threshold = torch.tensor(threshold, dtype=dtype,
                             device=device)
    zero = torch.tensor(0, dtype=dtype,
                        device=device)
    return torch.where(masks > threshold, masks, zero)


def remove_boxes(bboxes, scores,
                 accept_range=(0.5, 1), nms_threshold=0.3):

    if not torch.any(bboxes):
        return torch.tensor([], dtype=torch.int64)

    # select = reject uncertain predictions
    lower, upper = accept_range
    select = torch.logical_and(lower < scores, scores < upper)
    # bboxes = bboxes[select]
    # scores = scores[select]

    idx = torchvision.ops.nms(
        bboxes, scores, iou_threshold=nms_threshold)

    # make idx and select index tensors comparable
    indices = torch.zeros(len(bboxes), dtype=torch.bool, device=bboxes.device)
    indices[idx] = True

    # remove bboxes both using idx and select
    idx = torch.logical_and(indices, select)

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
                idx[i] = False
                continue

    # reject small bounding boxes
    for i, box in enumerate(bboxes[idx]):
        area = calc_bbox_area(box)
        if area <= 4:
            idx[i] = False

    return idx


def calc_bbox_area(box):
    width = box[2] - box[0]
    height = box[3] - box[1]

    return width * height


def bbox_contained(box1, box2):
    '''
    Returns true if box1 is contained in box2.

    Inputs:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
        x1 <= x2
        y1 <= y2

    Output:
        bool
    '''
    cond0 = box1[0] >= box2[0]
    cond1 = box1[1] >= box2[1]
    cond2 = box1[2] <= box2[2]
    cond3 = box1[3] <= box2[3]

    box1_inside_box2 = all((cond0, cond1, cond2, cond3))

    return box1_inside_box2


def predict(data: AICSImage,
            time_range: range, accept_range: tuple,
            device: torch.device,
            model, output_dir: str,
            save_pred: bool = True,
            load_pred: bool = False) -> list(tuple):
    '''
    Returns a list of CenterChains for each timepoint in time_range.

    Inputs:
        data: AICSImage - data to predict on
        time_range: range - range of timepoints to predict on
        device: torch.device - device to predict on
        model: torch.nn.Module - model to predict with
        output_dir: str - directory to save predictions to
        save_pred: bool - whether to save predictions to disk

    Output:
        list(list(CenterChain)) - list of CenterChains
            for each timepoint in time_range
    '''

    # TODO: save predictions to enable rapid dev iteration
    # on local machine
    chains_tot = []
    for t in time_range:
        chains = predict_timepoint(data, device, model, output_dir,
                                   save_pred, load_pred, t, accept_range)
        chains_tot.append(chains)

    return chains_tot


def predict_timepoint(data, device, model, output_dir,
                      save_pred, load_pred, t, accept_range):
    timepoint_raw = utils.get_raw_array(
        data, t=t).compute()

    # prepare data for model, since we're not using
    # BetaCellDataset class
    timepoint = prepare_model_input(timepoint_raw, device)

    preds = handle_predictions(device, model, osp.join(output_dir, 'raw_pred'),
                               save_pred, load_pred, t, timepoint,
                               accept_range=accept_range)

    # draw bounding boxes on slice for debugging
    viz.output_sample(osp.join(output_dir, 'debug'), t,
                      timepoint_raw, preds, 1024, device)

    # adjust slices from being model inputs to being
    # inputs to get_chains
    timepoint = [z_slice.squeeze() for z_slice in timepoint]
    # get center and intensity of all cells in timepoint
    chains = CL.get_chains(timepoint, preds, c.SEARCHRANGE)

    return chains


def handle_predictions(device, model, output_dir, save_pred,
                       load_pred, t, timepoint,
                       accept_range=(0.5, 1.0)):
    '''
    Returns predictions for timepoint.
    Supports loading predictions from disk, and saving predictions to disk.

    Inputs:
        device: torch.device - device to predict on
        model: torch.nn.Module - model to predict with
        output_dir: str - directory to save predictions to
        save_pred: bool - whether to save predictions to disk
        load_pred: bool - whether to load predictions from disk
        t: int - timepoint to predict on
        timepoint: list(torch.Tensor) - list of tensors to predict on

    Output:
        list(torch.Tensor) - list of predictions for timepoint

    '''
    if load_pred:
        # load predictions from disk
        preds = pickle.load(
            open(osp.join(output_dir, f'predictions_t{t}.pkl'), 'rb'))
    else:
        preds = get_predictions(
            model, device, timepoint, accept_range=accept_range)
        # save predictions to file
        if save_pred:
            save_predictions(preds, output_dir, t)
    return preds


def save_predictions(preds, output_dir, t):
    '''
    Saves predictions to file.

    Inputs:
        preds: list(torch.Tensor) - list of predictions
        output_dir: str - directory to save predictions to
        t: int - timepoint to save predictions for

    Outputs:
        None
    '''
    if not osp.exists(output_dir):
        utils.make_dir(output_dir)
    with open(osp.join(output_dir, f'predictions_t{t}.pkl'), 'wb') as f:
        pickle.dump(preds, f)


def prepare_model_input(timepoint, device):
    ''''
    Prepares data for input to model if BetaCellDataset
    class is not being used.

    Inputs:
        timepoint: the data to be prepared for the Mask R-CNN.
        This array can contain individual slices, an entire
        timepoint, or multiple timepoints.
        device: torch.device - device to predict on

    Output:
        list(torch.Tensor) - list of tensors to predict on
    '''

    timepoint = utils.normalize(np.int16(timepoint), 0, 1, cv2.CV_32F, device)

    # timepoint = [skimage.exposure.equalize_adapthist(
    #     z_slice, clip_limit=0.8)
    #     for z_slice in timepoint]
    timepoint = [F.pil_to_tensor(
        Image.fromarray(z_slice)).to(device)
        for z_slice in timepoint]
    timepoint = torch.stack(timepoint)

    # apply filter
    img_filter, args = c.FILTERS['bilateral']
    timepoint = [img_filter(z_slice.cpu().numpy().squeeze(), *args)
                 for z_slice in timepoint]
    timepoint = torch.tensor(np.array(timepoint), device=device)

    return timepoint.unsqueeze(1)


def predict_file(load, device, model,
                 output_dir, name, time_range=None, accept_range=(0, 1),
                 save_pred=True, load_pred=False):

    if load:  # use old (saved) predictions
        centroids = pickle.load(
            open(osp.join(output_dir, 'centroids_save.pkl'), 'rb'))
    else:  # predict
        path = osp.join(c.RAW_DATA_DIR, name)
        data = AICSImage(path)
        if time_range:
            time_start, time_end = min(time_range), max(time_range)
        else:
            time_start, time_end = 0, data.dims['T'][0]
            time_range = range(time_start, time_end)

        centroids = predict(data, time_range, accept_range,
                            device, model, output_dir,
                            save_pred, load_pred)

    centroids_np = [(t, centroid[0], centroid[1],
                     centroid[2], centroid[3], centroid[4].tolist())
                    for t, timepoint in zip(time_range, centroids)
                    for centroid in timepoint]

    tracked_centroids = tracker.track(centroids_np, threshold=5)

    save_tracks(name, output_dir, time_start, time_end,
                centroids, centroids_np, tracked_centroids)


def save_tracks(name, output_dir, time_start, time_end,
                centroids, centroids_np, tracked_centroids):

    pickle.dump(centroids, open(
        osp.join(output_dir, 'centroids_save.pkl'), 'wb'))

    # np.savetxt(
    #     osp.join(output_dir, f'{name}_{time_start}_{time_end}.csv'),
    #     centroids_np)

    pickle.dump(tracked_centroids,
                open(osp.join(output_dir, 'tracked_centroids.pkl'), 'wb'))

    (tracked_centroids
     .sort_values(['frame', 'particle'])
     .to_csv(osp.join(
         output_dir, 'tracked_centroids.csv'), sep=';', index=False))

    location = osp.join(output_dir, 'timepoints')
    time_range = range(time_start, time_end) if time_end else None
    plot.save_figures(tracked_centroids, location)
    utils.png_to_movie(time_range, location)


def start_predict(mode, load, experiment_name, accept_range, model_id,
                  save_pred, load_pred, time_start, time_end):
    utils.set_cwd(__file__)
    # Running on CUDA?
    device = utils.set_device()
    print(f'Running on {device}.')

    model = utils.get_model(model_id, device)
    model.to(device)

    time_range = range(time_start, time_end) if time_end else None

    # 1. Choose raw data file(s)
    # names = listdir(c.RAW_DATA_DIR)
    # names = [name for name in names if '.czi' not in name]
    files = c.RAW_FILES[mode].keys()
    files = utils.add_ext(files)
    # development purposes:
    files = [file for file in files if '2019-02-05_emb5_pos4' in file]
    
    len_files = len(files)

    for i, name in enumerate(files):
        print('Predicting', name,
              f'(file {i + 1}/{len_files})...', end='')
        output_dir = osp.join(c.DATA_DIR, c.PRED_DIR,
                              experiment_name, osp.splitext(name)[0])
        utils.make_dir(output_dir)
        predict_file(load, device,
                     model, output_dir, name,
                     time_range, accept_range,
                     save_pred, load_pred)
        print('done.')


if __name__ == '__main__':
    mode = 'test'  # for which embryos to predict
    load = False  # load complete tracks? useful for reproducing figures and movie
    experiment_name = 'pred_2'  # name of folder to save predictions to
    accept_range = (0.91, 1)  # how certain the model should be
    model_id = c.MODEL_STR  # model to use

    # whether to save or load predictions from disk
    # different from variable `load` above;
    # `save_pred` and `load_pred` only load the raw mask predictions,
    # and not the complete cell trajectories, which the `load` variable
    # above does
    save_pred = False
    load_pred = False

    # choose time range
    # endpoint excluded
    time_start = 0
    time_end = None  # Set to None to predict from time_start to final timepoint

    tic = time()
    start_predict(mode, load, experiment_name, accept_range,
                  model_id, save_pred, load_pred, time_start, time_end)

    print(f'predict_model.py complete in {utils.time_report(tic, time())}.')
