import os
import pickle
from os.path import join

import cv2
from matplotlib.image import BboxImage
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
from src.data.constants import DATA_DIR, TIMEPOINTS
from src.models.utils.model import get_instance_segmentation_model
from src.models.utils.utils import collate_fn
from torch.utils.data import DataLoader, Subset
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

from BetaCellDataset import BetaCellDataset, get_dataloaders, get_transform

# Set working directory to file location
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Running on CUDA?
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Running on {device}.')

# Get dataloaders
size = 1024
img_idx = 1700
time_str = '12_03_18H_29M_39S'
folder = f'interim/run_{time_str}'
# data_tr, data_val = get_dataloaders(resize=size)
dataset = BetaCellDataset(transforms=get_transform(train=False), resize=size)


def get_model(time_str):
    # Load model
    load = join(folder, f'model_{time_str}.pkl')
    model = pickle.load(open(load, 'rb'))
    return model


def get_mask(target):
    '''Consolidates the masks into one mask.'''
    if type(target) == dict:  # target is direct output of model
        mask = target['masks']
    else:  # target is a tensor of masks
        mask = target
    mask = torch.squeeze(mask, dim=1)
    mask = torch.sum(mask, dim=0)
    mask = np.array(mask)
    # mask = np.where(mask > 255, 255, mask)
    # mask = np.where(mask > 200, 255, 0)
    mask = torch.tensor(mask, dtype=torch.uint8)

    return mask


def get_prediction(model, device, image):

    model.eval()

    # Predict the image with batch index `img_idx`
    # with the model
    with torch.no_grad():
        # only choose one image
        # visualize this exact image I'm sending to the model
        img = image.to(device)
        # print(np.unique(img.cpu()))
        # TODO: put target into model to see if it segments
        pred = model([img])

    # pred list only has one item because we only chose one image
    mask = pred[0]['masks']

    # Remove empty color channel dimension (since grayscale)
    # mask = torch.squeeze(mask, dim=1)
    # # Add masks together into one tensor
    # mask = torch.sum(mask, dim=0)
    # print('mask shape', mask.shape, '\n')

    # scores = np.array(pred[0]['scores'].cpu())
    # print('scores max/min', np.max(scores), np.min(scores), '\n')
    img = cv2.normalize(img.cpu().numpy(), None, alpha=0,
                        beta=255, dtype=cv2.CV_8UC1, norm_type=cv2.NORM_MINMAX)
    img = torch.tensor(img, dtype=torch.uint8)
    img = img.repeat(3, 1, 1)

    mask = cv2.normalize(mask.cpu().numpy(), None, alpha=0,
                         beta=255, dtype=cv2.CV_8UC1, norm_type=cv2.NORM_MINMAX)

    mask = torch.tensor(mask, dtype=torch.uint8).squeeze(1)
    # mask = np.where(mask > 0, True, False)
    # mask = torch.tensor(mask, dtype=torch.bool)

    return img, mask


model = get_model(time_str)
model.to(device)

# Get image and target
image, target = dataset[img_idx]
# Get mask from target
target_mask = get_mask(target)


img, mask = get_prediction(model, device, image)

# Get segmentations and bounding boxes
pred_mask = get_mask(mask)
# Change to RGB image
pred_mask = pred_mask.repeat(3, 1, 1)
boxes = torch.tensor(target['boxes'])
bboxed_image = draw_bounding_boxes(pred_mask, boxes)
bboxed_image = torch.permute(bboxed_image, (1, 2, 0))
print(bboxed_image.shape)
debug_img = np.array(image)[0, :, :]

# Print mask from model output

plt.imsave(f'{folder}/pred_debug_mask.jpg',
           bboxed_image.cpu().detach().numpy())
# Print image that was fed into the model
plt.imsave(f'{folder}/pred_debug_img.jpg', img[0].cpu())
# Print target mask from dataset
plt.imsave(f'{folder}/pred_debug_target_mask.jpg', target_mask)

print(target['boxes'])

print('predict_model.py complete')
