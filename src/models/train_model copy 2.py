# from skimage.io import imread
import datetime
import os
import pickle
import sys
from os import mkdir
# from torchsummary import summary
from os.path import join
from time import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import src.models.utils.utils as utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
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

from BetaCellDataset import BetaCellDataset, get_dataloaders, print_unique

# import torch.optim as optim
# import torchvision

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

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Running on {device}.')

# Set working directory to file location
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


# our dataset has two classes only - background and person
num_classes = 2

# get the model using our helper function
model = get_instance_segmentation_model(pretrained=True)
# move model to the right device
model.to(device)

# Get data
# Size of image
size = 1024
batch_size = 8  # 1024, 8; 128, 16
data_tr, data_val = get_dataloaders(
    batch_size=batch_size, num_workers=2, resize=size)
# TODO: switch to batch_size=2, size=1024


# Unique identifier for newly saved objects
now = datetime.datetime.now()
time_str = f'{now.day:02d}_{now.month:02d}_{now.hour}H_{now.minute}M_{now.second}S'
save = f'interim/run_{time_str}'


def train(model, opt, epochs, data_tr, data_val, time_str):
    '''Train'''
    print(f'Training has begun for model: {time_str}')

    # TODO: send HPC support email
    # regarding why it runs out of memory but shows only
    # 2 gb used

    scheduler = ReduceLROnPlateau(opt, threshold=0.01, verbose=True)
    log_every = 1  # How often to print out losses
    save_every = 10  # How often to save model
    scaler = GradScaler()
    loss_list = ['loss_mask', 'loss_rpn_box_reg',
                 'loss_classifier', 'loss_objectness']
    # loss_classifier, loss_objectness
    # TODO: remove loss_classifier from loss_list
    # and also objectness? if we don't care about detecting
    # objects, maybe the faint ones will be caught as well.

    x_val, y_val = next(iter(data_val))

    # Transforms
    scale_jitter = T.ScaleJitter((size / 2, size / 2), scale_range=[0.7, 1.5])
    transforms_list = [T.RandomIoUCrop(), scale_jitter]
    transforms = T.Compose(transforms_list)

    tot_train_losses = []
    tot_val_losses = []

    for i, epoch in enumerate(range(epochs)):
        tic = time()
        print(f'\n* Epoch {epoch+1}/{epochs}')

        train_loss = 0

        model.train()  # train mode
        for j, (x_batch, y_batch) in enumerate(data_tr):

            with autocast():
                print_unique(x_batch[0].cpu(), 'before to(device)')
                x_batch = [x.to(device) for x in x_batch]
                print_unique(x_batch[0].cpu(), 'after to(device)')

                y_batch = [{k: v.to(device) for k, v in t.items()}
                           for t in y_batch]

                # TODO: Invalid box coordinates
                # "Found invalid box [68.75, 7.03125, 68.75, 7.8125]"
                # print_unique(x_batch[0].cpu(), 'before transforms')
                # Transforms that can't run parallel in dataloader
                # need to be performed here
                for (x, y) in zip(x_batch, y_batch):
                    x, y = transforms_list[0](x.squeeze(0), y)
                    print(np.unique(y['boxes'].cpu()), 'after crop')
                    x, y = transforms_list[1](x.squeeze(0), y)
                    print(np.unique(y['boxes'].cpu()), 'after scale jitter')

                x_batch = torch.stack(x_batch)
                print_unique(x_batch[0].cpu(), 'after transforms')

                x_batch.to(device)

                # print(x_batch.shape)

                # set parameter gradients to zero
                opt.zero_grad(set_to_none=True)

                # forward
                # for i, image in enumerate(y_batch):
                #     print(f'Image {i}:')
                #     print(image['boxes'], '\n\n')
                print_unique(x_batch[0].cpu(), 'before forward pass')

                Y_pred = model(x_batch, y_batch)

                # {'loss_classifier': tensor(0.8181, device='cuda:0', grad_fn=<NllLossBackward0>), 'loss_box_reg': tensor(0.0933, device='cuda:0', grad_fn=<DivBackward0>), 'loss_mask': tensor(1.2863, device='cuda:0',
                #    grad_fn=<BinaryCrossEntropyWithLogitsBackward0>), 'loss_objectness': tensor(8.1862, device='cuda:0',
                #    grad_fn=<BinaryCrossEntropyWithLogitsBackward0>), 'loss_rpn_box_reg': tensor(0.9159, device='cuda:0', grad_fn=<DivBackward0>)}
                # print(Y_pred)
                # sys.exit()
                # sum(loss for loss in Y_pred.values())
                # print(Y_pred)
                # print(type(Y_pred.values()))

                # Select only losses of interest
                losses = [value for loss, value in Y_pred.items()
                          if loss in loss_list]

                losses = sum(losses)
            # losses = Y_pred['loss_mask']
            # print(losses)
            # losses.backward()
            # opt.step()
            scaler.scale(losses).backward()
            scaler.step(opt)
            scaler.update()

            # calculate metrics to show the user
            train_loss += float(losses / len(data_tr))

        toc = time()
        tot_train_losses.append(train_loss)

        time_print = f'''Training loss: {train_loss: .3f};
                        Time: {(toc-tic)/60: .1f} minutes'''
        if i % log_every == 0:
            print(time_print)

        # Validation
        with torch.no_grad(), autocast():
            x_val = [x.to(device) for x in x_val]
            y_val = [{k: v.to(device) for k, v in t.items()}
                     for t in y_val]

            # print(model(X_val), '\n'*5)
            output = model(x_val, y_val)  # losses
            # float(sum(loss for loss in output.values()))
            val_losses = [value for loss, value
                          in output.items() if loss in loss_list]

            val_losses = sum(val_losses)
            # val_losses = output['loss_mask']
            scheduler.step(val_losses)
            # TODO: make sure scheduler works
            # by printing out the learning rate each epoch

            # Get current learning rate
            if i % log_every == 0:
                # print(f'Current learning rate: {scheduler}')
                print(f'Validation loss: {val_losses:.3f}')

            tot_val_losses.append(float(val_losses))

        # Save progress every 50 epochs
        if i % save_every == 0:
            # Make folder unique to this run in order to save model and loss
            try:
                mkdir(save)
            except FileExistsError:
                pass

            pickle.dump(model, open(join(save, f'model_{time_str}.pkl'), 'wb'))

        if i == save_every:
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

        # model.eval()
        # y_hat = model(x_val)

        # y_hat = [y['masks'] for y in y_hat]
        # y_hat = [item.squeeze(1) for item in y_hat]

        # y_hat = torch.cat(y_hat, dim=0)  # .detach().cpu()
        # print('yhat', y_hat.shape)
        # for x in x_val:
        #     print('first', x.shape)
        #     x = np.array(x.cpu())
        #     x = cv2.normalize(x, x, alpha=0, beta=255,
        #                       dtype=cv2.CV_8UC1, norm_type=cv2.NORM_MINMAX)
        #     print('after normalize', x.shape)
        #     # x = np.expand_dims(x, 0)
        #     # print('after expand', x.shape)

        #     x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
        #     print('after cvtcolor', x.shape)

        #     x = torch.tensor(x)
        #     print(x.shape)
        #     print(x)
        #     y_hat = draw_segmentation_masks(x, y_hat)
        # print('yhat_val', y_hat.shape)

        # #  y_batch = [{k: v.to(device) for k, v in t.items()}
        # # for t in y_batch]

        # for k in range(batch_size):
        #     plt.subplot(2, batch_size, k+1)
        #     plt.imshow(np.rollaxis(x_val[k].numpy(), 0, 3), cmap='gray')
        #     plt.title('Real')
        #     plt.axis('off')

        #     plt.subplot(2, batch_size, k+batch_size+1)
        #     plt.imshow(y_hat[k], cmap='gray')
        #     plt.title('Output')
        #     plt.axis('off')
        # plt.suptitle('%d / %d - loss: %f' % (epoch+1, epochs, val_losses))
        # plt.show()
        # plt.savefig(join(save, 'training_{i}.jpg'))

        # pickle.dump([tot_train_losses, tot_val_losses], open('loss.pkl', 'wb'))

    return tot_train_losses, tot_val_losses


def predict(model, data):
    '''Predict'''
    model.eval()  # testing mode
    Y_pred = [F.sigmoid(model(X_batch.to(device))) for X_batch, _ in data]
    return np.array(Y_pred)


def bce_loss(y_real, y_pred):
    '''bce_loss'''
    return torch.mean(y_pred - y_real * y_pred
                      + torch.log(1 + torch.exp(-y_pred)))


# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
#                                                step_size=3,
#                                                gamma=0.1)

params = [p for p in model.parameters() if p.requires_grad]

# Start with learning rate of just 1
# due to decreasing learning rate on plateau
# num_epochs = 3
# learning_rates = [1]  # [1e-8, 1e-6, 1e-4, 1e-2, 0.1, 1]
# optimizers = ['SGD', 'Adam']
# weight_decays = [1e-8, 1e-2, 0.1, 1]

# def grid_search(model, params, data_tr, data_val, train, num_epochs, learning_rates, optimizers, weight_decays):
#     '''Grid search for hyperparameters'''
#     losses = []

#     for learning_rate in learning_rates:
#         for weight_decay in weight_decays:
#             for name in optimizers:
#                 if name == 'SGD':
#                     opt = torch.optim.SGD(params, lr=learning_rate,
#                                           momentum=0.9, weight_decay=weight_decay)
#                 else:
#                     opt = optim.Adam(params, lr=learning_rate,
#                                      weight_decay=weight_decay)
#                 print('____________________________________')
#                 print('Training:')
#                 grid = f'lr={learning_rate}, wd={weight_decay}, opt={name}'
#                 print(grid)
#                 loss = train(model, opt, num_epochs, data_tr, data_val)

#                 losses.append([loss, grid])
#     return np.array(losses)

# losses = grid_search(model, params, data_tr, data_val, train, num_epochs, learning_rates, optimizers, weight_decays)

# losses = grid_search(model, params, data_tr, data_val, train,
#                      num_epochs, learning_rates, optimizers, weight_decays)

num_epochs = 50  # 500
lr = 0.00001  # 0.0001
wd = 0.001  # 0.001
opt = optim.Adam(params, lr=lr, weight_decay=wd, betas=[0.9, 0.99])

losses = train(model, opt, num_epochs, data_tr, data_val, time_str)
losses = np.array(losses).T

description = f'''{time_str}\n
                Learning rate: {lr}\n
                Weight decay: {wd}\n
                Optimizer: {opt}\n
               '''

# Special note that is saved as the name of a file with 
# a name which is the value of the string `special_mark`
special_mark = ''
if special_mark:
    np.savetxt(join(save, f'{special_mark}_{time_str}.txt'), special_mark)

np.savetxt(join(save, f'descr_{time_str}.txt'), description)
np.savetxt(join(save, f'losses_{time_str}.csv'), losses)
pickle.dump(model, open(join(save, f'model_{time_str}.pkl'), 'wb'))
pickle.dump(np.array([]), open(join(save, f'loss_{losses[1][-1]:.3f}.pkl')))

plt.subplot(121)
plt.plot(losses[0])
plt.title('Training loss')
plt.xlabel('Epoch')
plt.ylabel('Total loss')
plt.subplot(122)
plt.plot(losses[1])
plt.title('Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Total loss')

# TODO: autocast, optuna
# optuna: what losses to include?


# class EncDec(nn.Module):
#     def __init__(self):
#         super().__init__()

#         # encoder (downsampling)
#         self.enc_conv0 = nn.Conv2d(1, 64, 3, padding=1)
#         self.pool0 = nn.MaxPool2d(2, 2, padding=0)  # 256 -> 128
#         self.enc_conv1 = nn.Conv2d(64, 64, 3, padding=1)
#         self.pool1 = nn.MaxPool2d(2, 2, padding=0)  # 128 -> 64
#         self.enc_conv2 = nn.Conv2d(64, 64, 3, padding=1)
#         self.pool2 = nn.MaxPool2d(2, 2, padding=0)  # 64 -> 32
#         self.enc_conv3 = nn.Conv2d(64, 64, 3, padding=1)
#         self.pool3 = nn.MaxPool2d(2, 2, padding=0)  # 32 -> 16

#         # bottleneck
#         self.bottleneck_conv = nn.Conv2d(64, 64, 3, padding=1)

#         # decoder (upsampling)
#         self.upsample0 = nn.Upsample(32)  # 16 -> 32
#         self.dec_conv0 = nn.Conv2d(64, 64, 3, padding=1)
#         self.upsample1 = nn.Upsample(64)  # 32 -> 64
#         self.dec_conv1 = nn.Conv2d(64, 64, 3, padding=1)
#         self.upsample2 = nn.Upsample(128)  # 64 -> 128
#         self.dec_conv2 = nn.Conv2d(64, 64, 3, padding=1)
#         self.upsample3 = nn.Upsample(256)  # 128 -> 256
#         self.dec_conv3 = nn.Conv2d(64, 1, 3, padding=1)

#         self.upsample4 = nn.Upsample(512)  # 256 -> 512
#         self.dec_conv4 = nn.Conv2d(64, 64, 3, padding=1)
#         self.upsample5 = nn.Upsample(1024)  # 512 -> 1024
#         self.dec_conv5 = nn.Conv2d(64, 1, 3, padding=1)

#     def forward(self, x):
#         # encoder
#         e0 = self.pool0(F.relu(self.enc_conv0(x)))
#         e1 = self.pool1(F.relu(self.enc_conv1(e0)))
#         e2 = self.pool2(F.relu(self.enc_conv2(e1)))
#         e3 = self.pool3(F.relu(self.enc_conv3(e2)))

#         # bottleneck
#         b = F.relu(self.bottleneck_conv(e3))

#         # decoder
#         d0 = F.relu(self.dec_conv0(self.upsample0(b)))
#         d1 = F.relu(self.dec_conv1(self.upsample1(d0)))
#         d2 = F.relu(self.dec_conv2(self.upsample2(d1)))
#         d3 = F.relu(self.dec_conv3(self.upsample3(d2)))
#         d4 = F.relu(self.dec_conv4(self.upsample4(d3)))
#         d5 = self.dec_conv5(self.upsample5(d4))  # no activation
#         return d5


# class UNet(nn.Module):
#     def __init__(self):
#         super().__init__()

#         # encoder (downsampling)
#         self.enc_conv0 = nn.Conv2d(1, 64, 3, padding=1)
#         self.pool0 = nn.MaxPool2d(2, 2, padding=0)  # 128 -> 64
#         self.enc_conv1 = nn.Conv2d(64, 64, 3, padding=1)
#         self.pool1 = nn.MaxPool2d(2, 2, padding=0)  # 64 -> 32
#         self.enc_conv2 = nn.Conv2d(64, 64, 3, padding=1)
#         self.pool2 = nn.MaxPool2d(2, 2, padding=0)  # 32 -> 16

#         # bottleneck
#         self.bottleneck_conv = nn.Conv2d(64, 64, 3, padding=1)

#         # decoder (upsampling)
#         self.upsample0 = nn.Upsample(32)  # 16 -> 32
#         self.dec_conv0 = nn.Conv2d(128, 64, 3, padding=1)
#         self.upsample1 = nn.Upsample(64)  # 32 -> 64
#         self.dec_conv1 = nn.Conv2d(128, 64, 3, padding=1)
#         self.upsample2 = nn.Upsample(128)  # 64 -> 128
#         self.dec_conv2 = nn.Conv2d(128, 1, 3, padding=1)

#     def forward(self, x):
#         # encoder
#         e0 = self.pool0(F.relu(self.enc_conv0(x)))
#         e1 = self.pool1(F.relu(self.enc_conv1(e0)))
#         e2 = self.pool2(F.relu(self.enc_conv2(e1)))

#         # bottleneck
#         b = F.relu(self.bottleneck_conv(e2))

#         # decoder
#         skip0 = torch.cat([self.upsample0(b), F.relu(self.enc_conv2(e1))], 1)
#         d0 = F.relu(self.dec_conv0(skip0))
#         skip1 = torch.cat([self.upsample1(d0), F.relu(self.enc_conv1(e0))], 1)
#         d1 = F.relu(self.dec_conv1(skip1))
#         skip2 = torch.cat([self.upsample2(d1), F.relu(self.enc_conv0(x))], 1)
#         d2 = self.dec_conv2(skip2)
#         return d2
