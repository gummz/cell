from os import listdir
from os.path import join

import cv2
import numpy as np
import torch
from PIL import Image
import src.data.constants as c
from src.models.utils import transforms as T
from src.models.utils.utils import collate_fn
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F

conn = c.NUMBER_CONNECTIVITY
algo = c.CV2_CONNECTED_ALGORITHM
kernel = c.MEDIAN_FILTER_KERNEL
threshold = c.SIMPLE_THRESHOLD


def print_unique(image, pre=''):
    # print(pre, np.unique(image))
    pass


class BetaCellDataset(torch.utils.data.Dataset):
    '''Dataset class for beta cell data'''

    def __init__(self, root=c.DATA_DIR, transforms=None, resize=1024, mode='train'):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        imgs = [image for image in listdir(
            join(root, mode, c.IMG_DIR)) if '.npy' in image]
        masks = [image for image in listdir(
            join(root, mode, c.MASK_DIR)) if '.npy' in image]

        self.imgs = list(sorted(imgs))
        self.masks = list(sorted(masks))
        self.resize = resize

    def __getitem__(self, idx):
        # load images and mask
        img_path = join(self.root, c.IMG_DIR, self.imgs[idx])
        mask_path = join(self.root, c.MASK_DIR, self.masks[idx])

        img = np.int16(np.load(img_path))
        # PREPROCESSING
        img = cv2.normalize(img, None, alpha=0, beta=255,
                            dtype=cv2.CV_8UC1, norm_type=cv2.NORM_MINMAX)
        img = cv2.fastNlMeansDenoising(
            img, None, 11, 7, 21)
        # END PREPROCESSING

        mask = np.load(mask_path)
        mask = np.array(mask)

        if self.resize != 1024:
            size = (self.resize, self.resize)
            img = cv2.resize(img, size, cv2.INTER_AREA)
            mask = cv2.resize(mask, size, cv2.INTER_AREA)

        output = cv2.connectedComponentsWithStatsWithAlgorithm(
            mask, conn, cv2.CV_32S, algo)

        # labels: labeled segmentations
        # so each cell is marked with a unique label

        # Name elements of output for ease of use
        # lower number of labels by 1 to account for
        # background removal
        numLabels = output[0] - 1
        # TODO: preprocess inside __getitem__
        # don't need to create a new dataset every time!
        # easier to experiment this way.
        # TODO: annotate inside __GETITEM__!!!! yeah!

        # `labels_out` denotes
        labels_out = np.array(output[1])
        # remove background label from stats
        stats = np.array(output[2][1:])
        # remove background label from centroids

        # NOTE!: I can use this for the tracking.
        centroids = torch.as_tensor(np.array(output[3][1:]), dtype=torch.uint8)

        # Create variables for Mask R-CNN dataloader ###
        ################################
        # Select only x, y, width, height (not area)
        boxes = stats[:, :4]
        boxes = torch.tensor(boxes, dtype=torch.int16)

        # Bounding boxes from opencv's connectedComponent function
        # have the format (x,y,width,height) but we want (xmin,ymin,xmax,ymax)
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        object_ids = np.unique(labels_out)
        # Remove background label
        object_ids = object_ids[1:]
        num_objects = numLabels
        # `labels` denotes the classes of the objects;
        # there is only one (a beta cell),
        # so every object is marked with a 1
        labels = torch.ones((num_objects,), dtype=torch.int64)
        # Not unique, images between files will have the same ids
        # because the id is the page number
        image_id = torch.tensor([idx])
        area = torch.tensor(stats[:, -1], dtype=torch.int16)
        iscrowd = torch.zeros((num_objects,), dtype=torch.int8)
        masks = labels_out == object_ids[:, None, None]
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["centroids"] = centroids

        if self.transforms is not None:
            # img, target = self.transforms(img, target)
            img = cv2.normalize(img, img, alpha=0, beta=1,
                                dtype=cv2.CV_32F, norm_type=cv2.NORM_MINMAX)
            img = F.pil_to_tensor(Image.fromarray(img))

        # TODO:
        # https://stackoverflow.com/questions/66370250/how-does-pytorch-dataloader-interact-with-a-pytorch-dataset-to-transform-batches
        # the transform does not seem to be applied to the entire batch
        # but rather one at a time.
        # because the images in the batch have randomly different dimensions now.
        # issue isolated to ScaleJitter

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_dataloaders(batch_size=4, num_workers=2, resize=1024, manual_annot=0):
    '''Get dataloaders.
        resize: resize image in __getitem__ method of dataset class.

        manual_annot: how many manually annotated images to include. Default: None'''
    # use our dataset and defined transformations
    dataset = BetaCellDataset(
        c.DATA_DIR, get_transform(train=True), resize=resize)
    dataset_val = BetaCellDataset(
        c.DATA_DIR, get_transform(train=False), resize=resize)

    # split the dataset in train and test set
    torch.manual_seed(1)

    # indices = torch.randperm(len(dataset)).tolist()

    # val_idx = int(0.1 * len(dataset))
    indices = range(len(dataset))
    val_tp = c.TIMEPOINTS[-1]
    dataset = torch.utils.data.Subset(dataset, indices[:-val_tp])
    dataset_val = torch.utils.data.Subset(dataset_val, indices[-val_tp:])

    # define training and validation data loaders
    data_tr = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=collate_fn)

    data_val = DataLoader(
        dataset_val, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=collate_fn)

    return data_tr, data_val


def get_transform(train):
    '''Gets transforms based on if train=True or train=False.'''
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        pass
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        # transforms.append(T.RandomHorizontalFlip(0.5))
        # transforms.append(T.RandomIoUCrop())
        # transforms.append(T.RandomZoomOut())
        # transforms.append(T.RandomPhotometricDistort())
        # transforms.append(T.ScaleJitter((128, 128)))

    return T.Compose(transforms)
