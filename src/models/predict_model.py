from __future__ import annotations

import pickle
from os.path import join
import cv2
import numpy as np
import skimage
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
    # print('len of boxes before rejection', len(pred['boxes']))
    pred['boxes'] = pred['boxes'][idx]
    # print('len of boxes after rejection', len(pred['boxes']))

    # select masks by idx
    pred['masks'] = pred['masks'][idx]
    # select scores by idx
    # print('accept range', accept_range)
    # print(pred['scores'])
    # print('avg score before rejection', np.mean(pred['scores'].cpu().numpy()))
    pred['scores'] = pred['scores'][idx]
    # print('avg score after rejection', np.mean(
    # pred['scores'].cpu().numpy()), '\n')

    pred['masks'] = threshold_masks(pred['masks'])

    return pred


def threshold_masks(masks, threshold=0.5):
    '''
    Thresholds masks to make more concise
    segmentations (i.e., model has to be more certain
    in its predictions).
    '''
    dtype, device = masks.dtype, masks.device
    threshold = torch.tensor(threshold, dtype=dtype,
                             device=device)
    zero = torch.tensor(0, dtype=dtype,
                        device=device)
    return torch.where(masks > threshold, masks, zero)


def remove_boxes(bboxes, scores,
                 accept_range=(0.5, 1), nms_threshold=0.3):
    if len(bboxes) == 0:
        return torch.tensor([], dtype=torch.int64)

    # reject uncertain predictions
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
    # print(scores)
    # print(indices, select)
    # exit()

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

    for i, box in enumerate(bboxes[idx]):
        area = calc_bbox_area(box)
        # reject small bounding boxes
        if area <= 4:
            idx[i] = False

    return idx


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


def predict(data: AICSImage,
            time_range: range,
            device: torch.device,
            model, save: str) -> list(tuple):

    chains_tot = []
    for t in time_range:
        timepoint_raw = utils.get_raw_array(
            data, t=t).compute()

        # prepare data for model, since we're not using
        # BetaCellDataset class
        timepoint = prepare_model_input(timepoint_raw, device)

        preds = get_predictions(
            model, device, timepoint, accept_range=(0.5, 1))

        # draw bounding boxes on slice for debugging
        viz.output_sample(join(save, 'debug'), t,
                          timepoint_raw, preds, 1024, device)

        # adjust slices from being model inputs to being
        # inputs to get_chains
        timepoint = [z_slice.squeeze() for z_slice in timepoint]
        # get center and intensity of all cells in timepoint
        chains = CL.get_chains(timepoint, preds, c.SEARCHRANGE)
        chains_tot.append(chains)

        del timepoint_raw

    return chains_tot


def prepare_model_input(timepoint, device):
    ''''
    Prepares data for input to model if BetaCellDataset
    class is not being used.

    Inputs:
        array: the data to be prepared for the Mask R-CNN.
        This array can contain individual slices, an entire
        timepoint, or multiple timepoints.
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
    timepoint = torch.tensor(timepoint, device=device)

    return timepoint.unsqueeze(1)


if __name__ == '__main__':

    utils.setcwd(__file__)
    mode = 'pred'
    load = False

    # Running on CUDA?
    device = utils.set_device()
    print(f'Running on {device}.')

    model = utils.get_model(c.MODEL_STR, device)
    model.to(device)

    # Choose time range
    # time_start = 80
    # time_end = 130  # endpoint included
    # time_range = range(time_start, time_end)

    # 1. Choose raw data file
    # names = listdir(c.RAW_DATA_DIR)
    # names = [name for name in names if '.czi' not in name]
    files = tuple(c.RAW_FILES['test'].keys())
    files = utils.add_ext(files)
    # files = ('LI_2019-08-30_emb2_pos1.lsm',)
    len_files = len(files)
    for i, name in enumerate(files):
        print('Predicting file', name, f'(file {i + 1}/{len_files})')
        # # Make directory for this raw data file
        # # i.e. mode/pred/name
        save = join(c.PROJECT_DATA_DIR, mode, name)
        utils.make_dir(save)

        if load:  # use old (saved) predictions
            centroids = pickle.load(
                open(join(save, 'centroids_save.pkl'), 'rb'))
        else:  # predict
            path = join(c.RAW_DATA_DIR, name)
            data = AICSImage(path)
            time_start, time_end = 0, data.dims['T'][0]
            time_range = range(time_start, time_end)
            centroids = predict(data, time_range,
                                device, model, save)
            try:
                data.close()
                print(f'File {name} closed successfully.')
            except AttributeError as e:
                print(f'File {name} raised error upon closing:\n{e}')

            pickle.dump(centroids, open(
                join(save, 'centroids_save.pkl'), 'wb'))

        centroids_np = [(t, centroid[0], centroid[1],
                        centroid[2], centroid[3])
                        for t, timepoint in zip(time_range, centroids)
                        for centroid in timepoint]
        np.savetxt(
            join(save, f'{name}_{time_start}_{time_end}.csv'), centroids_np)

        tracked_centroids = tracker.track(centroids_np, threshold=20)

        pickle.dump(tracked_centroids,
                    open(join(save, 'tracked_centroids.pkl'), 'wb'))
        tracked_centroids.to_csv(join(save, 'tracked_centroids.csv'))

        location = join(save, 'timepoints')
        plot.save_figures(tracked_centroids, location)
        # plot.create_movie(location, time_range)

        # df = pd.DataFrame(centroids_save, columns=['x, y, z, i'])
        # df.to_csv(join(save, f'{name}_{time_start}_{time_end}.csv'), sep=',')

    print('predict_model.py complete')
