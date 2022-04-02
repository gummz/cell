import numpy as np
from src.models.BetaCellDataset import BetaCellDataset, get_transform
from src.models.predict_model import get_prediction
import torch


def eval_model(model, manual=0):
    '''Calculates IOU for test set for the input model.
    TODO: Use with TensorBoard (add_scalar)

    manual: how many manual annotations to include.'''

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Running on {device}.')

    dataset = BetaCellDataset(
        transforms=get_transform(train=False), mode='test')
    scores = np.zeros(len(dataset))

    model.eval()
    for i, (image, target) in enumerate(dataset):
        pred = get_prediction(model, device, image)
        score = calc_iou(pred, target)
        scores[i] = score

    average_score = np.mean(scores)

    return(average_score)


def calc_iou(pred, target):
    # match instance masks between prediction and target:
    # pred_mask == target_mask, sum, the one with the highest
    # sum value is deemed to be matched with this instance
    # of the prediction
    pred_masks = pred['masks']
    target_masks = target['masks']
    pred_masks_nonzero = [mask != 0 for mask in pred_masks]
    target_masks_nonzero = [mask != 0 for mask in target_masks]

    for mask in pred_masks_nonzero:
        pass
