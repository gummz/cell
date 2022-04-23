import numpy as np
from src.models.BetaCellDataset import BetaCellDataset, get_transform
from src.models.predict_model import get_prediction
import torch
import torchvision
from torch.utils.data import DataLoader
from src.models.utils.utils import collate_fn
import src.data.utils.utils as utils


def eval_model(model, mode):
    '''Calculates IOU for chosen set for the input model.
    TODO: Use with TensorBoard (add_scalar)
            (With Optuna in grid search, not in here)
    '''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on {device}.')

    dataset = BetaCellDataset(
        transforms=get_transform(train=False), mode=mode)
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=True, num_workers=2, collate_fn=collate_fn)
    scores_mask = np.zeros(len(dataloader))
    scores_bbox = np.zeros(len(dataloader))

    model.eval()
    for i, (image, target) in enumerate(dataloader):
        print(image)
        pred = get_prediction(model, device, image)
        score_mask = calc_score_mask(pred, target)
        scores_mask[i] = np.mean(score_mask)

        score_bbox = calc_score_bbox(pred, target)
        scores_bbox[i] = np.mean(score_bbox)

    avg_score_mask = np.mean(scores_mask)
    avg_score_bbox = np.mean(scores_bbox)

    return avg_score_mask, avg_score_bbox


def calc_score_bbox(pred, target):
    pred_bboxes = pred['boxes']
    target_bboxes = target['boxes']
    scores = np.zeros(len(pred_bboxes))

    for i, pred_bbox in enumerate(pred_bboxes):
        iou_scores = np.zeros(len(pred_bboxes))
        for j, target_bbox in enumerate(target_bboxes):
            iou_scores[j] = calc_iou_bbox(pred_bbox, target_bbox)
        max_score = np.max(iou_scores)
        if max_score < 10:
            max_score = 0
        scores[i] = max_score

    return scores


def calc_iou_bbox(pred_bbox, target_bbox):
    return torchvision.ops.boxes.box_iou(
        pred_bbox, target_bbox)


def calc_score_mask(pred, target):
    # match instance masks between prediction and target:
    # pred_mask == target_mask, sum, the one with the highest
    # sum value is deemed to be matched with this instance
    # of the prediction
    thresh = 50
    pred_masks = pred['masks']
    target_masks = target['masks']
    pred_masks_nonzero = [mask > thresh for mask in pred_masks]
    target_masks_nonzero = [mask != 0 for mask in target_masks]
    scores = np.zeros(len(pred_masks))

    # find the most similar mask among all target masks
    for i, pred_mask in enumerate(pred_masks_nonzero):
        max_sum = 0
        for j, target_mask in enumerate(target_masks_nonzero):
            same = pred_mask == target_mask
            summed = sum(same)
            if summed > max_sum:
                max_sum = summed
                idx = j

        if max_sum < 10:
            # we don't bother with marking this as a match
            scores[i] = 0
        else:
            # here, the most suitable target mask has been found
            suitable_tgt = target_masks[idx]
            scores[i] = calc_iou_mask(pred_mask, suitable_tgt)


def calc_iou_mask(pred_mask, target_mask):
    inter_cond = pred_mask == 1 and target_mask == 1
    area_inter = torch.sum(inter_cond)
    union_cond = pred_mask == 1 or target_mask == 1
    area_union = torch.sum(union_cond)

    iou = area_inter / area_union

    return iou


if __name__ == '__main__':
    utils.setcwd(__file__)
    time_str = '12_03_18H_29M_39S'
    folder = f'interim/run_{time_str}'
    model = utils.get_model(folder, time_str, utils.set_device())
    ious = eval_model(model, 'val')
    print(ious)
