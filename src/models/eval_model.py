from time import time
from typing import Union
from matplotlib import pyplot as plt

import numpy as np
import src.data.utils.utils as utils
import torch
import torchvision
from src.models.BetaCellDataset import BetaCellDataset, get_transform
from src.models.predict_model import get_prediction
from src.models.utils.utils import collate_fn
from torch.cuda.amp import autocast
import scipy.optimize as sciopt
from torch.utils.data import DataLoader
import os.path as osp
import seaborn as sns
import src.models.train_model as train
import torchmetrics.functional as F
import src.models.BetaCellDataset as bcd
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import src.data.constants as c
from os.path import join
import src.visualization.utils as viz


def eval_model(model, dataset, mode, device, save=None,
               accept_range=c.ACCEPT_RANGE,
               match_threshold=c.MATCH_THRESHOLD):
    '''
    Calculates IOU for chosen set for the input model.
    '''
    len_dataset = len(dataset)
    scores_mask = np.zeros(len_dataset)
    scores_bbox = np.zeros(len_dataset)
    cm_mask_tot = np.zeros((2, 2))
    cm_bbox_tot = np.zeros((2, 2))
    certainty = []  # np.zeros(len_dataset)

    model.eval()
    for i, (image, target) in enumerate(dataset):
        image = image.to(device)
        target['masks'] = target['masks'].to(device)
        target['boxes'] = target['boxes'].to(device)

        with autocast():
            pred = get_prediction(model, device, image, accept_range)

        if i % 10 == 0:
            save_path = osp.join(save, f'{i:05d}')
            target_mask = utils.get_mask(target['masks']) * 255
            inputs = (image.cpu(), pred, target_mask.cpu())
            titles = ('Original', 'Prediction', 'Ground Truth')
            grid = (1, 3)
            viz.output_pred(mode, i, inputs, titles,
                            grid, save_path, True, 300)

        cm_mask, score_mask = performance_mask(pred, target, match_threshold)
        scores_mask[i] = np.mean(score_mask)
        cm_mask_tot += cm_mask

        cm_bbox, score_bbox = performance_bbox(pred, target, match_threshold)
        scores_bbox[i] = np.mean(score_bbox)
        cm_bbox_tot += cm_bbox

        if any(pred['scores']):
            certainty.append(np.mean(pred['scores'].cpu().numpy()))

    avg_score_mask = np.round(np.mean(scores_mask), 2)
    avg_score_bbox = np.round(np.mean(scores_bbox), 2)

    avg_certainty = np.round(np.mean(certainty), 2)

    return (cm_bbox_tot, avg_score_bbox,
            cm_mask_tot, avg_score_mask,
            avg_certainty)


def performance_bbox(pred, target, match_threshold=0.3):
    '''
    Returns mean IOU score of all bounding box detections
    in a slice.
    '''
    pred_bboxes = pred['boxes']
    len_pred = len(pred_bboxes)
    target_bboxes = target['boxes']

    # edge cases
    if len(pred_bboxes) == 0 and len(target_bboxes) == 0:
        confusion_matrix = create_cm(tp=0, fn=0, fp=0)
        return confusion_matrix, (1,)
    elif len(pred_bboxes) != 0 and len(target_bboxes) == 0:
        fp = len(pred_bboxes)
        confusion_matrix = create_cm(tp=0, fn=0, fp=fp)
        return confusion_matrix, (0,)
    elif len(pred_bboxes) == 0 and len(target_bboxes) != 0:
        fn = len(target_bboxes)
        confusion_matrix = create_cm(tp=0, fn=fn, fp=0)
        return confusion_matrix, (0,)

    scores = np.zeros(len(target_bboxes))
    tp = 0  # true postives
    fn = 0  # false negatives
    taken = []

    for i, target_bbox in enumerate(target_bboxes):
        iou_scores = np.zeros(len_pred)
        for j, pred_bbox in enumerate(pred_bboxes):
            iou_scores[j] = calc_iou_bbox(pred_bbox, target_bbox)

        max_arg = np.argmax(iou_scores)
        if iou_scores[max_arg] < match_threshold:
            fn += 1
        elif max_arg not in taken:
            taken.append(max_arg)
            tp += 1
            scores[i] = iou_scores[max_arg]
    # cost_matrix = torchvision.ops.boxes.box_iou(pred_bboxes,
    #                                             target_bboxes)
    # col_select = torch.argwhere(torch.max(cost_matrix, dim=1) > match_threshold)
    # cost_matrix = cost_matrix[:, col_select]
    # cost_rows, cost_cols = sciopt.linear_sum_assignment(cost_matrix, maximize=True)
    # cost_matrix_opt = cost_matrix[cost_rows, cost_cols]
    # opt_rows, opt_cols = cost_matrix_opt.shape
    # print(opt_rows, opt_cols)
    # if cost_rows == cost_cols:
    #     pass  # no false positives, no false negatives
    # elif cost_rows > cost_cols:
    #     fp = cost_rows - cost_cols
    #     fp += torch.sum(cost_matrix_opt < 0.3)
    # elif cost_rows < cost_cols:
    #     fn = cost_cols - cost_rows
    # tp = torch.sum(cost_matrix_opt > 0.3)

    # no matches; all predictions are false positives,
    # and all targets are false negatives
    if sum(scores) == 0:
        fp = len(pred_bboxes)
        fn = len(target_bboxes)
        confusion_matrix = create_cm(tp=0, fn=fn, fp=fp)
        return confusion_matrix, (0,)

    # didn't yield a true positive
    fp = len_pred - tp
    confusion_matrix = create_cm(tp=tp, fn=fn, fp=fp)

    # count remaining false positives as score 0
    fps = tuple(0 for item in range(fp))
    scores = tuple(scores) + fps

    return confusion_matrix, scores


def calc_iou_bbox(pred_bbox, target_bbox):
    return torchvision.ops.boxes.box_iou(
        pred_bbox.unsqueeze(0), target_bbox.unsqueeze(0))


def performance_mask(pred, target, match_threshold=0.2):
    # match instance masks between prediction and target:
    # pred_mask == target_mask, sum, the one with the highest
    # sum value is deemed to be matched with this instance
    # of the prediction
    pred_masks = pred['masks']
    target_masks = target['masks']

    # edge cases
    if len(pred_masks) == 0 and len(target_masks) == 0:
        confusion_matrix = create_cm(tp=0, fn=0, fp=0)
        return confusion_matrix, (1,)
    elif len(pred_masks) != 0 and len(target_masks) == 0:
        fp = len(pred_masks)
        confusion_matrix = create_cm(tp=0, fn=0, fp=fp)
        return confusion_matrix, (0,)
    elif len(pred_masks) == 0 and len(target_masks) != 0:
        fn = len(target_masks)
        confusion_matrix = create_cm(tp=0, fn=fn, fp=0)
        return confusion_matrix, (0,)

    scores = np.zeros(len(target_masks))
    tp = 0  # true positives
    fn = 0  # false negatives

    # if (len(pred_masks_nonzero) == 0 and len(target_masks_nonzero) != 0) or False:
    #     return scores
    taken = []
    # find the most similar mask among all target masks
    for i, target_mask in enumerate(target_masks > 0):
        max_iou = 0
        idx = 0
        for j, pred_mask in enumerate(pred_masks > 0):
            iou = calc_iou_mask(pred_mask, target_mask)

            if iou > max_iou:
                max_iou = iou
                idx = j

        if max_iou < match_threshold:  # 25**2 out of 255**2
            # we don't bother with marking this as a match
            fn += 1
        elif idx not in taken:
            # here, the most suitable predicted mask has been found

            # take the mask out of consideration
            taken.append(idx)

            # suitable_prd = pred_masks_nonzero[idx]
            tp += 1
            scores[i] = max_iou

    # no matches; all predictions are false positives,
    # and all targets are false negatives
    if sum(scores) == 0:
        fp = len(pred_masks)
        fn = len(target_masks)
        confusion_matrix = create_cm(tp=0, fn=fn, fp=fp)
        return confusion_matrix, (0,)

    # false positives are all the predictions which
    # didn't yield a true positive
    fp = len(pred_masks) - tp
    confusion_matrix = create_cm(tp=tp, fn=fn, fp=fp)

    # count remaining false positives as score 0
    fps = tuple(0 for item in range(fp))
    scores = tuple(scores) + fps

    return confusion_matrix, scores


def calc_iou_mask(pred_mask, target_mask):
    inter_cond = torch.logical_and(pred_mask, target_mask)
    area_inter = torch.sum(inter_cond.flatten())

    union_cond = torch.logical_or(pred_mask, target_mask)
    area_union = torch.sum(union_cond.flatten())

    iou = area_inter / area_union

    return iou


def create_cm(tp: int, fn: int, fp: int):
    confusion_matrix = np.array([[tp, fn], [fp, 0]],
                                dtype=np.int32)
    return confusion_matrix


def score_report(metrics):
    (tp, fn), (fp, _) = metrics[0]
    n_pred = int(tp + fp)
    print('\nNumber of bbox predictions:', n_pred)
    print('Bbox CM\n', metrics[0], '\n')
    print('Bbox IOU\n', metrics[1], '\n')
    print('Mask CM\n', metrics[2], '\n')
    print('Mask IOU\n', metrics[3], '\n')
    print('Average certainty\n', metrics[4], '\n')


if __name__ == '__main__':
    mode = 'test'

    tic = time()
    utils.set_cwd(__file__)
    device = utils.set_device()
    save = osp.join(c.PROJECT_DATA_DIR, c.PRED_DIR, 'eval',
                    'seg_2d', f'model_{c.MODEL_STR}', mode)
    print('Outputting images to', save)

    model = utils.get_model(c.MODEL_STR, device)
    model = model.to(device)
    dataset = bcd.get_dataset(mode=mode)

    metrics = eval_model(model, dataset, mode, device, save,
                         accept_range=(0.91, 1), match_threshold=0.3)
    score_report(metrics)
    viz.save_cm(metrics[0], save)
    print(f'Evaluation complete after {utils.time_report(tic, time())}.')
