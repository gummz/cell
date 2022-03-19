import pickle
from os.path import join
import torch
import matplotlib.pyplot as plt


# TESTS ###
# Double-check by comparing instance masks with the thresholded images
# check if, for every value in an instance mask image,
# there is correspondingly a value in the thresholded image
data_path = 'data'

outputs_load = join(data_path, 'outputs_list.pkl')
segments_load = join(data_path, 'segments_list.pkl')
outputs_list = pickle.load(open(outputs_load))
segments_list = pickle.load(open(segments_load))

# segments: segmentations of images from a raw data file
# outputs: labeled masks of those segmentations
#           still from the same raw data file
for segments, outputs in zip(segments_list, outputs_list):
    for (i, segment), output in zip(segments, outputs):
        print(i)
        segment = torch.tensor(segment)
        masks = output[1]['masks']
        # torch.sum(masks):
        # Not the right form of masks; it will only have 1
        # when it should be labeled according to component
        # so 1, 2, 3, ... etc.
        # But this doesn't matter because only nonzero
        # is being tested below.

        # TODO: something wrong in the != 1 case
        # no::: the algorithm detected a blob that wans't there.
        # tolerance for erroneous detections?

        # masks is binary and has an extra dimension for each detected object
        # so if there are two detected objects, the shape will be
        # 2 x 1024 x 1024 instead of 1024 x 1024
        # with both masks on one image
        if masks.shape[0] != 1:
            masks = torch.stack([masks])
            masks = torch.sum(masks, dim=0)
        else:
            masks = masks.squeeze()
            masks = torch.sum(masks, dim=0)
        masks[:, [0, 1]] = masks[:, [1, 0]]
        # print('mask', torch.nonzero(masks))

        segment_where = torch.nonzero(segment)
        mask_where = torch.nonzero(masks)
        # print('Printing second time')
        # print(segment_where.shape)
        # print(mask_where.shape)
        # print('segment', segment_where)
        # print('mask', mask_where)
        try:
            eq = torch.eq(segment_where, mask_where)
        except RuntimeError:
            if masks.shape[0] > 1:
                print(masks.shape)
                # for i, (segment, mask) in enumerate(zip(segments, masks)):
                #     plt.subplot(2, n, i+1)
                #     plt.imshow(mask, cmap='gray')
                #     plt.axis('off')
                #     plt.savefig(f'figures/mask_debug_mask{i}.jpg')
            else:
                plt.imshow(masks, cmap='gray')
                plt.axis('off')
                plt.savefig('figures/mask_debug.jpg')

        eq_sum = torch.sum(eq)
        eq_n = torch.numel(eq)
        try:
            assert eq_sum/eq_n > 0.999
        except AssertionError:
            plt.imshow(masks, cmap='gray')
            plt.axis('off')
            plt.savefig('figures/mask_debug.jpg')
        # Test that background label has been removed
        # It starts at 0 and ends at 1024 for both
        # width and height
        boxes = output[1]['boxes']
        assert 0 not in boxes[0] and 1024 not in boxes[0]
