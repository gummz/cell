from time import time

import numpy as np
import src.data.utils.utils as utils
import torch
import torchvision
from src.models.BetaCellDataset import BetaCellDataset, get_transform
from src.models.predict_model import get_prediction
from src.models.utils.utils import collate_fn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
import src.models.train_model as train
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import src.data.constants as c
from os.path import join
import src.visualization.utils as viz


def eval_model(model, dataloader, mode, device, save=None):
    '''Calculates IOU for chosen set for the input model.
    TODO: Use with TensorBoard (add_scalar)
            (With Optuna in grid search, not in here)
    '''

    scores_mask = np.zeros(len(dataloader))
    scores_bbox = np.zeros(len(dataloader))
    cm_mask_tot = np.zeros((2, 2))
    cm_bbox_tot = np.zeros((2, 2))

    model.eval()
    for i, (images, targets) in enumerate(dataloader):
        # batch size of 1
        image = images[0]
        image = image.to(device)
        target = targets[0]

        # move target to device
        target['masks'] = target['masks'].to(device)
        target['boxes'] = target['boxes'].to(device)
        with autocast():
            pred = get_prediction(model, device, image)

        if i % 40 == 0:
            save_path = join(save, f'{i:05d}')
            target_mask = utils.get_mask(target['masks']) * 255
            inputs = (image.cpu(), pred, target_mask.cpu())
            titles = ('Original', 'Prediction', 'Ground Truth')
            grid = (1, 3)
            viz.output_pred(mode, i, inputs, titles, grid, save_path, True, 300)

        cm_mask, score_mask = performance_mask(pred, target)
        scores_mask[i] = np.mean(score_mask)
        cm_mask_tot += cm_mask

        cm_bbox, score_bbox = performance_bbox(pred, target)
        scores_bbox[i] = np.mean(score_bbox)
        cm_bbox_tot += cm_bbox

    avg_score_mask = np.round(np.mean(scores_mask), 2)
    avg_score_bbox = np.round(np.mean(scores_bbox), 2)

    return (cm_mask_tot, avg_score_mask,
            cm_bbox_tot, avg_score_bbox)


def performance_bbox(pred, target):
    '''
    Returns mean IOU score of all bounding box detections
    in a slice.
    '''
    pred_bboxes = pred['boxes']
    target_bboxes = target['boxes']

    # edge cases
    if len(pred_bboxes) == 0 and len(target_bboxes) == 0:
        confusion_matrix = np.array([[0, 0], [0, np.nan]])
        return confusion_matrix, np.array([1])
    elif len(pred_bboxes) != 0 and len(target_bboxes) == 0:
        fp = len(pred_bboxes)
        confusion_matrix = np.array([[0, 0], [fp, np.nan]])
        return confusion_matrix, np.array([0])
    elif len(pred_bboxes) == 0 and len(target_bboxes) != 0:
        fn = len(target_bboxes)
        confusion_matrix = np.array([[0, fn], [0, np.nan]])
        return confusion_matrix, np.array([0])

    scores = np.zeros(len(target_bboxes))
    tp = 0  # true postives
    fn = 0  # false negatives

    for i, target_bbox in enumerate(target_bboxes):
        iou_scores = np.zeros(len(pred_bboxes))
        for j, pred_bbox in enumerate(pred_bboxes):
            iou_scores[j] = calc_iou_bbox(pred_bbox, target_bbox)

        max_arg = np.argmax(iou_scores)
        if iou_scores[max_arg] < 0.1:
            fn += 1
            scores[i] = 0
        else:
            tp += 1
            # pred_copy[max_arg] = 0
            # target_copy[i] = 0
            scores[i] = iou_scores[max_arg]

    # didn't yield a true positive
    fp = len(pred_bboxes) - tp
    confusion_matrix = np.array([[tp, fn], [fp, np.nan]])

    # no matches; all predictions are false positives,
    # and all targets are false negatives
    if len(scores) == 0:
        fp = len(pred_bboxes)
        fn = len(target_bboxes)
        confusion_matrix = np.array([[0, fn], [fp, np.nan]])
        return confusion_matrix, np.array([0])

    # count remaining false positives as score 0
    fps = tuple(0 for item in range(fp))
    scores = tuple(scores) + fps

    return confusion_matrix, scores


def calc_iou_bbox(pred_bbox, target_bbox):
    return torchvision.ops.boxes.box_iou(
        pred_bbox.unsqueeze(0), target_bbox.unsqueeze(0))


def performance_mask(pred, target):
    # match instance masks between prediction and target:
    # pred_mask == target_mask, sum, the one with the highest
    # sum value is deemed to be matched with this instance
    # of the prediction
    thresh = 50
    pred_masks = pred['masks']
    target_masks = target['masks']

    # edge cases
    if len(pred_masks) == 0 and len(target_masks) == 0:
        confusion_matrix = np.array([[0, 0], [0, np.nan]])
        return confusion_matrix, np.array([1])
    elif len(pred_masks) != 0 and len(target_masks) == 0:
        fp = len(pred_masks)
        confusion_matrix = np.array([[0, 0], [fp, np.nan]])
        return confusion_matrix, np.array([0])
    elif len(pred_masks) == 0 and len(target_masks) != 0:
        fn = len(target_masks)
        confusion_matrix = np.array([[0, fn], [0, np.nan]])
        return confusion_matrix, np.array([0])

    pred_masks_nonzero = [mask > thresh for mask in pred_masks]
    target_masks_nonzero = [mask != 0 for mask in target_masks]
    scores = np.zeros(len(target_masks))
    tp = 0  # true positives
    fn = 0  # false negatives

    # if (len(pred_masks_nonzero) == 0 and len(target_masks_nonzero) != 0) or False:
    #     return scores

    # find the most similar mask among all target masks
    for i, target_mask in enumerate(target_masks_nonzero):
        max_sum = 0
        for j, pred_mask in enumerate(pred_masks_nonzero):
            same = pred_mask == target_mask
            summed = torch.sum(same.flatten())

            if summed > max_sum:
                max_sum = summed
                idx = j

        if max_sum < 625:  # 25**2 out of 255**2
            # we don't bother with marking this as a match
            fn += 1
            scores[i] = 0
        else:
            # here, the most suitable predicted mask has been found
            suitable_prd = pred_masks[idx]
            tp += 1

            scores[i] = calc_iou_mask(suitable_prd, target_mask)

    # false positives are all the predictions which
    # didn't yield a true positive
    fp = len(pred_masks) - tp
    confusion_matrix = np.array([[tp, fn], [fp, np.nan]])

    # no matches; all predictions are false positives,
    # and all targets are false negatives
    if len(scores) == 0:
        fp = len(pred_masks)
        fn = len(target_masks)
        confusion_matrix = np.array([[0, fn], [fp, np.nan]])
        return confusion_matrix, np.array([0])

    # count remaining false positives as score 0
    fps = tuple(0 for item in range(fp))
    scores = tuple(scores) + fps

    return confusion_matrix, scores


def calc_iou_mask(pred_mask, target_mask):
    inter_cond = (pred_mask == 1) == (target_mask == 1)
    area_inter = torch.sum(inter_cond.flatten())
    union_cond = (pred_mask == 1) == (target_mask == 1)
    area_union = torch.sum(union_cond.flatten())

    iou = area_inter / area_union

    return iou


if __name__ == '__main__':
    tic = time()
    utils.setcwd(__file__)
    device = utils.set_device()
    model = utils.get_model(c.MODEL_STR, device)
    dataset = BetaCellDataset(
        transforms=get_transform(train=False), mode='val',
        n_img_select=1, manual_select=1)
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=1, collate_fn=collate_fn)

    ious = eval_model(model, dataloader, device)
    print('\nMask CM\n', ious[0], '\n')
    print('Mask IOU\n', ious[1], '\n')
    print('Bbox CM\n', ious[2], '\n')
    print('Bbox IOU\n', ious[3], '\n')

    print(f'Evaluation complete after {utils.time_report(tic, time())}.')
