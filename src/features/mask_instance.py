import pickle
from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
import torch
import pandas as pd

from constants import (NUMBER_CONNECTIVITY,
                       CV2_CONNECTED_ALGORITHM,
                       RAW_FILES)

conn = NUMBER_CONNECTIVITY
algo = CV2_CONNECTED_ALGORITHM

data_path = 'data'
label_ext = '_labels.pkl'
files = [join(data_path, f) for f in listdir(data_path)
         if isfile(join(data_path, f)) and f.endswith(label_ext)]
# Take only one file at a time due to memory constraints
files = [files[0]]

# List of the segments for each file in `files`
segments_list = [pickle.load(open(labels, 'rb')) for labels in files]
# List of the outputs for each file in `files`
outputs_list = []
features = ['boxes', 'labels', 'image_id', 'area', 'iscrowd', 'masks']

for k, segments in enumerate(segments_list):
    # List of image outputs containing instance masks
    # for each image in `outputs_list`
    # corresponding to an image in `segments`
    shape = (len(segments), 6)
    output_df = pd.DataFrame(0, index=np.arange(len(segments)),
                             columns=features)
    # label: an image with segmented cells
    # obtained via thresholding
    for j, (i, segment) in enumerate(segments):
        # output is a tuple and contains the following items
        # ('numLabels', 'labels', 'stats', 'centroids')
        output = list(cv2.connectedComponentsWithStatsWithAlgorithm(
            segment, conn, cv2.CV_32S, algo))
        # labels: labeled segmentations
        # so each cell is marked with a unique label

        # remove background label from stats
        output[2] = output[2][1:]
        # remove background label from centroids
        output[3] = output[3][1:]
        # lower number of labels by 1 to account for
        # background removal
        output[0] = output[0] - 1

        # Keys for returned dictionary
        # names = ['boxes', 'labels', 'image_id', 'area', 'iscrowd', 'masks']

        # Name elements of output for ease of use
        numLabels = output[0]
        # `labels_out` denotes
        labels_out = np.array(output[1])
        stats = np.array(output[2])
        centroids = np.array(output[3])

        # Create variables for Mask R-CNN dataloader ###
        ################################
        boxes = stats
        object_ids = np.unique(labels_out)
        # Remove background label
        object_ids = object_ids[1:]
        num_objects = len(object_ids)
        # `labels` denotes the classes of the objects;
        # there is only one (a beta cell),
        # so every object is marked with a 1
        labels = torch.ones((num_objects,), dtype=torch.int64)
        # Not unique, images between files will have the same ids
        # because the id is the page number
        image_id = torch.tensor([i])
        area = torch.tensor([item[-1] for item in stats],
                            dtype=torch.float32)
        iscrowd = torch.zeros((num_objects,), dtype=torch.int64)
        masks = labels_out == object_ids[:, None, None]
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        #################################
        print(type(boxes), type(labels), type(image_id),
              type(area), type(iscrowd), type(masks))
        print(boxes.shape, labels.shape, image_id.shape, area.shape,
              iscrowd.shape, masks.shape)
        out_list = [boxes, labels, image_id, area, iscrowd, masks]

        # out_dict = dict(zip(names, out_list))
        # i is index of image
        # output_df.append((i, out_dict))
        output_df.iloc[j] = out_list

    outputs_list.append(output_df)

print('Finished loop')

pickle.dump(outputs_list, open(f'{data_path}/outputs_list.pkl', 'wb'))
print('Finished masking')

# # SAVE TO DATA FOLDER ###
# for i in range(2):
#     pickle.dump(outputs_list[i], open(
#         f'data/{RAW_FILES[i]}_targets.pkl', 'wb'))
#     outputs_list[i] = []

# print('mask_instance.py complete')
