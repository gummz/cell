# from skimage.io import imread
import datetime
import os
import pickle
import sys
import math
from os import mkdir
# from torchsummary import summary
from os.path import join
from time import time
from memory_profiler import profile

import cv2
import matplotlib.pyplot as plt
import numpy as np
from src.data.utils.utils import get_mask
import src.data.utils.utils as utils
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import torch.nn.functional as F
import torch.utils.data
from torchvision.utils import make_grid
import torchvision
from PIL import Image
from src.data.constants import (CV2_CONNECTED_ALGORITHM, DATA_DIR, IMG_DIR,
                                MASK_DIR, MEDIAN_FILTER_KERNEL,
                                NUMBER_CONNECTIVITY, SIMPLE_THRESHOLD)
from src.models.utils import transforms as T
from src.models.utils.model import get_instance_segmentation_model
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

from src.models.BetaCellDataset import BetaCellDataset, get_dataloaders

# import torch.optim as optim
# import torchvision


def train(model, device, opt, epochs, data_tr, data_val, time_str, hparam_dict, writer, save=False, write=False):
    '''Train'''
    torch.backends.cudnn.benchmark = True
    print(f'Training has begun for model: {time_str}')

    size = hparam_dict['image_size']
    batch_size = hparam_dict['batch_size']

    # TODO: send HPC support email
    # regarding why it runs out of memory but shows only
    # 2 gb used

    scheduler = ReduceLROnPlateau(opt, threshold=0.01, verbose=True)
    log_every = 1  # How often to print out losses
    save_every = 10  # How often to save model
    scaler = GradScaler()
    loss_list = hparam_dict['losses'].split(';')
    # loss_classifier, loss_objectness
    # TODO: remove loss_classifier from loss_list
    # and also objectness? if we don't care about detecting
    # objects, maybe the faint ones will be caught as well.

    # Transforms
    scale_jitter = T.ScaleJitter((size / 2, size / 2), scale_range=[0.7, 1.5])
    transforms_list = [T.RandomIoUCrop(), scale_jitter]
    transforms = T.Compose(transforms_list)

    tot_train_losses = []
    tot_val_losses = []

    # print(x_val[0].unsqueeze(0).shape)
    # writer.add_graph(model, [x_val[0].to(device)])

    for i, epoch in enumerate(range(epochs)):
        model.train()  # train mode
        tic = time()
        print(f'\n* Epoch {epoch+1}/{epochs}')

        x_val, y_val = next(iter(data_val))
        train_loss = 0
        for j, (x_batch, y_batch) in enumerate(data_tr):

            with autocast():
                x_batch = [x.to(device) for x in x_batch]

                y_batch = [{k: v.to(device) for k, v in t.items()}
                           for t in y_batch]

                # TODO: Invalid box coordinates
                # Transforms that can't run parallel in dataloader
                # need to be performed here
                # for (x, y) in zip(x_batch, y_batch):
                #     x, y = transforms_list[0](x.squeeze(0), y)
                #     print(np.unique(y['boxes'].cpu()), 'after crop')
                #     x, y = transforms_list[1](x.squeeze(0), y)
                #     print(np.unique(y['boxes'].cpu()), 'after scale jitter')

                x_batch = torch.stack(x_batch)
                x_batch.to(device)

                # set parameter gradients to zero
                opt.zero_grad(set_to_none=True)

                # forward pass
                Y_pred = model(x_batch, y_batch)

                # print(x_batch.shape, len(y_batch))
                # print(np.unique(x_batch.cpu()), '\n' * 5)
                # print(np.unique(y_batch.cpu()), '\n' * 7)

                # Select only losses of interest
                losses = [value for loss, value in Y_pred.items()
                          if loss in loss_list]

                losses = sum(losses)

                # End of training loop for mini-batch

            scaler.scale(losses).backward()
            scaler.step(opt)
            scaler.update()

            # calculate metrics to show the user
            train_loss += float(losses / len(data_tr))

            # End training loop for epoch

        tot_train_losses.append(train_loss)
        writer.add_scalar('Training Loss', train_loss, epoch)

        # Validation
        val_losses = 0
        for x_val, y_val in data_val:
            with torch.no_grad(), autocast():
                model.train()
                # x_val, y_val = to_device([x_val, y_val], device)
                x_val = [x.to(device) for x in x_val]
                y_val = [{k: v.to(device) for k, v in t.items()}
                         for t in y_val]

                val_losses += get_loss(model, loss_list, x_val, y_val)

                # TODO: make sure scheduler works
                # by printing out the learning rate each epoch
                # if write:

                # if i == save_every:
                #     debug_opencv_mask()

                model.eval()
                y_hat = model(x_val)

                # Convert ys to masks
                # yhat_boxes = [y['boxes']] .....
                # Convert y_hat to CUDA
                # y_hat = to_device([y_hat], device)
                # y_hat = [{k: v.to(device) for k, v in t.items()}
                #          for t in y_hat]
                # Strip everything except masks
                y_hat, y_val = y_to_mask([y_hat, y_val])
                # Consolidate masks in batch
                y_hat = [get_mask(y) for y in y_hat]
                y_val = [get_mask(y) for y in y_val]

            # y_hat = torch.cat(y_hat, dim=0)  # .detach().cpu()
            x_val = [x.squeeze() for x in x_val]

            # end validation loop

        val_loss = float(val_losses) / len(data_val)
        tot_val_losses.append(val_loss)
        writer.add_scalar('Validation Loss',
                          val_loss, epoch)

        if write:
            image_grid = create_grid(x_val, y_val, y_hat,
                                     batch_size)

            writer.add_image(f'epoch_{epoch}', image_grid,
                             epoch, dataformats='NCHW')

            # writer.add_hparams(
            #     hparam_dict, {'hparam/loss': val_losses.item()}, run_name=f'runs/{time_str}')

        scheduler.step(val_losses)

        if i % log_every == 0:
            # loss is nan; cancel training
            if math.isnan(float(train_loss)):
                print('training loss is nan\n')
                return train_loss, np.nan

            print(f'Training loss: {train_loss:.3f}')
            print(f'Validation loss: {val_loss:.3f}')

            elapsed = utils.time_report(tic, time())
            print('Time:', elapsed)

        # Save progress every `save_every` epochs
        if (i + 1) % save_every == 0 and save:
            dump_model(model, time_str)

        # early stopping
        patience = 15  # number of epochs to wait for validation loss to improve
        if i >= 5 and i % 5 == 0:
            worsen_ratio = 1.2
            val_prev = tot_val_losses[i-patience]
            val_now = val_loss
            # if validation loss has worsened this much
            # in the last 5 iterations, then stop
            if val_now / val_prev > worsen_ratio:
                print('Early stopping activated; stopping training.')
                model = model_prev
                break

        # end epoch

    # select random images and their target indices
    # images, labels = select_n_random()

    # # get the class labels for each image
    # # class_labels = [classes[lab] for lab in labels]

    # # log embeddings
    # # features = images.view(-1, 28 * 28)
    # writer.add_embedding(images,
    #                         label_img=images.unsqueeze(1))

    return tot_train_losses, tot_val_losses, model


def create_grid(x_val, y_val, y_hat, batch_size):
    image_grid = make_grid(
        [*x_val, *y_hat, *y_val], nrow=batch_size, pad_value=220, padding=30)
    image_grid = image_grid.squeeze().unsqueeze(1)
    image_grid = (image_grid * 255).type(torch.uint8)
    return image_grid


def to_device(tensor_list, device):
    '''
    Moves data onto device.
    '''
    main_list = []
    for batch in tensor_list:
        if type(batch) == dict:
            batch = [{k: v.to(device) for k, v in t.items()}
                     for t in batch]
        elif type(batch) == torch.Tensor:
            batch = [x.to(device) for x in batch]
        main_list.append(batch)
    return main_list


def y_to_mask(ys):
    if type(ys) == dict:
        ys = [item['masks'] for item in ys]
        ys = [item.squeeze(1) for item in ys]
        return ys
    for y in ys:
        y = [item['masks'] for item in y]
        y = [item.squeeze(1) for item in y]
    return ys


def get_loss(model, loss_list, x_val, y_val):
    output = model(x_val, y_val)  # losses
    # float(sum(loss for loss in output.values()))
    losses = [value
              for loss, value in output.items()
              if loss in loss_list]

    losses = sum(losses)
    return losses


def debug_opencv_mask():
    # Visualize the masks generated by opencv
    # for debugging purposes
    dataset = BetaCellDataset(DATA_DIR)
    img, target = dataset[500]
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='viridis')
    plt.subplot(1, 2, 2)
    # Values in target['masks'] are either 0 or 1
    # so multiply by 255 for image pixel values
    plotted = torch.sum(target['masks'], dim=0) * 255
    plt.imshow(plotted, cmap='gray')
    plt.savefig(join(save, 'opencv_mask.jpg'))


def dump_model(model, time_str):
    # Make folder unique to this run in order to save model and loss
    utils.make_dir(save)

    pickle.dump(model, open(join(save, f'model_{time_str}.pkl'), 'wb'))


def predict(model, data):
    '''Predict'''
    model.eval()  # testing mode
    Y_pred = [F.sigmoid(model(X_batch.to(device))) for X_batch, _ in data]
    return np.array(Y_pred)

# helper function


def select_n_random(n=100):
    '''
    Selects n random datapoints and their corresponding labels from a dataset
    source:
    https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
    '''
    assert len(data) == len(labels)

    data = BetaCellDataset()

    perm = torch.randperm(len(data))
    return data[perm][:n]


def bce_loss(y_real, y_pred):
    '''bce_loss'''
    return torch.mean(y_pred - y_real * y_pred +
                      torch.log(1 + torch.exp(-y_pred)))


if __name__ == '__main__':
    tic = time()

    # Environment variable for memory management
    alloc_conf = 'PYTORCH_CUDA_ALLOC_CONF'
    try:
        print(alloc_conf, os.environ[alloc_conf])
    except KeyError:
        print(alloc_conf, 'not found')

    conn = NUMBER_CONNECTIVITY
    algo = CV2_CONNECTED_ALGORITHM
    kernel = MEDIAN_FILTER_KERNEL
    threshold = SIMPLE_THRESHOLD

    device = utils.set_device()
    print(f'Running on {device}.')

    utils.set_cwd(__file__)

    # our dataset has two classes only - background and person
    num_classes = 2

    # hyperparameters
    size = 1024
    batch_size = 8  # 2
    pretrained = True
    num_epochs = 30  # 500
    lr = 3.418507038460298e-06
    wd = 1.2957404400334042e-08
    beta1 = 0.2438598958001344
    beta2 = 0.9849760264270886
    n_img_select = 1101
    manual_select = 1
    img_filter = 'bilateral'

    data_tr, data_val = get_dataloaders(
        batch_size=batch_size, num_workers=4, resize=size,
        n_img_select=(n_img_select, 1), manual_select=(manual_select, 1), img_filter=img_filter)

    # get the model using our helper function
    model = get_instance_segmentation_model(pretrained=pretrained)
    model.to(device)

    # Unique identifier for newly saved objects
    now = datetime.datetime.now()
    time_str = f'{now.day:02d}_{now.month:02d}_{now.hour}H_{now.minute}M_{now.second}S'
    save = f'interim/run_{time_str}'

    params = [p for p in model.parameters() if p.requires_grad]
    opt = optim.Adam(params, lr=lr, weight_decay=wd, betas=[beta1, beta2])

    loss_list = ['loss_mask', 'loss_rpn_box_reg', 'loss_box_reg',
                 'loss_classifier', 'loss_objectness']
    # loss_list = ['loss_mask', 'loss_rpn_box_reg']

    hparam_dict = {
        'learning_rate': lr,
        'weight_decay': wd,
        'num_epochs': num_epochs,
        'optimizer': f'{opt}',
        'losses': ';'.join(loss_list),
        'image_size': size if size else 1024,
        'batch_size': batch_size,
        'pretrained': pretrained
    }
    # TODO: add "how many weakly annotated"
    # TODO: add /pred/ folder in addition to /runs/
    # so make a writer in predict_model which saves images
    # add_video í SummaryWriter

    description = f'''{time_str}\n
                        Learning rate: {lr}\n
                        Weight decay: {wd}\n
                        Optimizer: {opt}\n
                        Losses: {loss_list}
                        '''

    with SummaryWriter(f'runs/{time_str}') as w:
        train_loss, val_loss, model = train(model, device, opt, num_epochs,
                                            data_tr, data_val, time_str, hparam_dict, w, save=True, write=True)
        w.add_text('description', description)
    save = join('interim', f'run_{time_str}')

    utils.make_dir(save)
    pickle.dump(model, open(
        join('interim', f'run_{time_str}', f'model_{time_str}.pkl'), 'wb'))

    np.savetxt(join(save, 'train_loss.csv'), train_loss,
               delimiter=';')
    np.savetxt(join(save, 'val_loss.csv'), val_loss,
               delimiter=';')

    elapsed = utils.time_report(tic, time())
    print('train_model finished after', elapsed)
    
