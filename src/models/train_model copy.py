# from skimage.io import imread
import os
import pickle
# from torchsummary import summary
from os.path import join

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
from src.models.utils.engine import evaluate, train_one_epoch
from src.models.utils.model import get_instance_segmentation_model
from torch import optim
from torch.optim.lr_scheduler import StepLR

from BetaCellDataset import BetaCellDataset, get_dataloaders

# import torch.optim as optim
# import torchvision


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


# Visualize the masks generated by opencv
dataset = BetaCellDataset(DATA_DIR)
img, target = dataset[500]
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='viridis')
plt.subplot(1, 2, 2)
# Values in target['masks'] are either 0 or 1
# so multiply by 255 for image pixel values
plotted = torch.sum(target['masks'], dim=0)*255
plt.imshow(plotted, cmap='gray')
plt.savefig('train_debug.jpg')


# our dataset has two classes only - background and person
num_classes = 2

# get the model using our helper function
model = get_instance_segmentation_model(num_classes)
# move model to the right device
model.to(device)
data_tr, data_val = get_dataloaders()


# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)


# let's train it for 10 epochs
num_epochs = 10


def train(model, opt, loss_fn, epochs, data_tr, data_val):
    X_val, Y_val = next(iter(data_val))

    for i, epoch in enumerate(range(epochs)):
        # tic = time()
        print('* Epoch %d/%d' % (epoch+1, epochs))

        avg_loss = 0
        model.train()  # train mode
        for X_batch, Y_batch in data_tr:
            # print(X_batch[0].shape)
            # print(X_batch)
            # X_batch = X_batch.to(device)
            # Y_batch = Y_batch.to(device)
            X_batch = [x.to(device) for x in X_batch]
            Y_batch = [{k: v.to(device) for k, v in t.items()}
                       for t in Y_batch]
            # set parameter gradients to zero
            opt.zero_grad()

            # forward
            Y_pred = model(X_batch, Y_batch)
            losses = sum(loss for loss in Y_pred.values())
            losses.backward()
            opt.step()
            # loss = loss_fn(Y_batch, Y_pred)  # forward-pass
            # loss.backward()  # backward-pass
            opt.step()  # update weights

            # calculate metrics to show the user
            avg_loss += losses / len(data_tr)
        # toc = time()
        print(' - loss: %f' % avg_loss)

        # show intermediate results
        model.eval()  # testing mode
        Y_hat = F.sigmoid(model(X_val.to(device))).detach().cpu()

        for k in range(6):
            plt.subplot(2, 6, k+1)
            plt.imshow(np.rollaxis(X_val[k].numpy(), 0, 3), cmap='gray')
            plt.title('Real')
            plt.axis('off')

            plt.subplot(2, 6, k+7)
            plt.imshow(Y_hat[k, 0], cmap='gray')
            plt.title('Output')
            plt.axis('off')
        plt.suptitle('%d / %d - loss: %f' % (epoch+1, epochs, avg_loss))
        plt.show()
        plt.savefig(f'interim/training_{i}.jpg')


def predict(model, data):
    model.eval()  # testing mode
    Y_pred = [F.sigmoid(model(X_batch.to(device))) for X_batch, _ in data]
    return np.array(Y_pred)


def bce_loss(y_real, y_pred):
    return torch.mean(y_pred - y_real*y_pred
                      + torch.log(1 + torch.exp(-y_pred)))


train(model, optim.Adam(model.parameters()),
      bce_loss, num_epochs, data_tr, data_val)

pickle.dump(model, 'model_custom.pkl')


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
