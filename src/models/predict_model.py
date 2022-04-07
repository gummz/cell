import os
import pickle
from os.path import join
from pprint import pp

import cv2
import matplotlib.pyplot as plt
import numpy as np
import src.data.constants as c
import src.models.utils.center_link as CL
import torch
import torchvision
from aicsimageio import AICSImage
from matplotlib.image import BboxImage
from PIL import Image
import src.data.utils.utils as utils
from src.models.BetaCellDataset import (BetaCellDataset, get_dataloaders,
                                        get_transform)
from src.models.utils.model import get_instance_segmentation_model
from src.models.utils.utils import collate_fn
from torch.utils.data import DataLoader, Subset
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks


def get_model(folder, time_str: str, device: torch.device):
    # Load model
    load_path = join(folder, f'model_{time_str}.pkl')
    if device.type == 'cuda':
        model = pickle.load(open(load_path, 'rb'))
    else:
        # model = pickle.load(open(load, 'rb'))
        model = utils.CPU_unpickler(open(load_path, 'rb')).load()
        '''Attempting to deserialize object on a CUDA device
        but torch.cuda.is_available() is False.
        If you are running on a CPU-only machine, please use
        torch.load with map_location=torch.device('cpu') to map your storages to the CPU.'''

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
        if device.type == 'cuda':
            preds = model(inputs)
        else:
            preds = []
            for input in inputs:
                pred = model([input])[0]
                preds.append(pred)
            # preds = [model([input]) for input in inputs]

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


def predict_ssh(raw_data_file, time_idx, device, model):
    centroids_tot = []
    for t in time_idx:
        timepoint = raw_data_file.get_image_dask_data(
            'ZXY', T=t, C=c.CELL_CHANNEL).compute()
        slice_idx = utils.active_slices(timepoint)
        timepoint_sliced = timepoint[slice_idx]
        
        # debug
        save_dir = join(c.DATA_DIR, 'test', c.PRED_DIR)
        for idx, slice in zip(slice_idx, timepoint_sliced):
            plt.imsave(join(save_dir, f'active_{idx}.png'))
        for i, slice in enumerate(timepoint):
            plt.imsave(join(save_dir, f'orig_{i}.png'))

        timepoint = prepare_mrcnn_data(timepoint, device)
        pred = get_predictions(model, device, timepoint)
        centroids = CL.get_chains(timepoint, pred)
        centroids_tot.append(centroids)

    return centroids_tot


def prepare_mrcnn_data(timepoint, device):
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
    mode = 'test'
    # Running on CUDA?
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    print(f'Running on {device}.')

    # Get dataloaders
    size = 1024
    img_idx = 500
    time_str = '12_03_18H_29M_39S'
    folder = f'interim/run_{time_str}'
    # data_tr, data_val = get_dataloaders(resize=size)

    dataset = BetaCellDataset(
        transforms=get_transform(train=False), resize=size, mode=mode)

    model = get_model(folder, time_str, device)
    model.to(device)

    # 1. Choose raw data file
    name = list(c.RAW_FILES_GENERALIZE.keys())[1]
    name = utils.add_ext([name])
    # # Make directory for this raw data file
    # # i.e. mode/pred/name
    save = join(c.DATA_DIR, mode, c.PRED_DIR)
    utils.make_dir(join(save, name))

    # either ssh with full data:
    path = join(c.RAW_DATA_DIR, name)
    raw_data_file = AICSImage(path)
    # ... or local with small debug file:
    # timepoints = np.load(join(c.DATA_DIR, 'sample.npy'))

    # # Choose time range
    time_start = 0
    time_end = 1

    # 2. Loop over each timepoint at a time
    time_range = range(time_start, time_end)
    centroids = predict_ssh(raw_data_file, [10, 100],
                            device, model)
    centroids_save = [(cent.get_center(), cent.get_intensity())
                      for centers in centroids for cent in centers]
    np.savetxt(
        join(save, name, f'{name}_{time_start}_{time_end}.csv'), centroids_save)
    # np.save(join(path, f'{t:05d}'), centroids)
