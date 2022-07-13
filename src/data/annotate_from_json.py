from labelme import utils as labelme_utils
from labelme.logger import logger
import PIL.Image
import imgviz
import os.path as osp
from os import listdir
from os.path import join
import os
import json
import base64
import src.data.constants as c
import src.data.utils.utils as utils
import base64
import io
import json
import os
from os import listdir
from os.path import join
from time import time
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import src.data.utils.utils as utils
import src.data.constants as c

'labelme_json_to_dataset _00400.json -o _00400_json'


def annotate_from_json(mode):
    # logger.warning(
    #     "This script is aimed to demonstrate how to convert the "
    #     "JSON file to a single image dataset."
    # )
    # logger.warning(
    #     "It won't handle multiple JSON files to generate a "
    #     "real-use dataset."
    # )

    # parser = argparse.ArgumentParser()
    # parser.add_argument("json_file")
    # parser.add_argument("-o", "--out", default=None)
    # args = parser.parse_args()

    # if args.out is None:
    #     out_dir = osp.basename(json_file).replace(".", "_")
    #     out_dir = osp.join(osp.dirname(json_file), out_dir)
    # else:
    #     out_dir = args.out
    # if not osp.exists(out_dir):
    #     os.mkdir(out_dir)
    '''
    STEP 1:
    Inside test/masks, this algorithm will generate a json folder
    (with label.png inside) for every image (png) in masks_test.

    For training: this algorithm will generate a json folder 
    (with label.png inside) for every image (png) in masks.

    STEP 2:
    There will be folders inside masks_test called 'filename_json'.

    The algorithm

    1. Walk through every subfolder called
    'filename_json'
    2. Fetch the 'label.png' in each subfolder,
    3. Threshold the image.

    The manual annotation will be processed: grayscale, threshold.

    test/imgs:
    _00000.png
    00000.npy
    etc.

    test/masks:
    (folder): _00000_json (complete annotation from labelme script)
    _00000.png (automatic annotation)

    test/masks_full:
    00000.npy (thresholded complete annotation)

    This algorithm will be applied to the folder `imgs` in the future.
    So: imgs, masks, masks_full.
    '''

    # STEP 1
    generate_label(mode)

    # STEP 2
    annotate_from_label(mode)


def generate_label(mode):
    mask_dir = join(c.DATA_DIR, mode, c.MASK_DIR)
    files = listdir(mask_dir)
    json_files = [file for file in files if '.json' in file]

    for json_file in json_files:
        out_dir = json_file.replace('.', '_')
        out_dir = join(mask_dir, out_dir)
        utils.make_dir(out_dir)
        json_path = join(mask_dir, json_file)
        data = json.load(open(json_path))
        imageData = data.get("imageData")

        if not imageData:
            imagePath = os.path.join(
                os.path.dirname(json_path), data["imagePath"])
            with open(imagePath, "rb") as f:
                imageData = f.read()
                imageData = base64.b64encode(imageData).decode("utf-8")
        img = labelme_utils.img_b64_to_arr(imageData)

        label_name_to_value = {"_background_": 0}
        for shape in sorted(data["shapes"], key=lambda x: x["label"]):
            label_name = shape["label"]
            if label_name in label_name_to_value:
                label_value = label_name_to_value[label_name]
            else:
                label_value = len(label_name_to_value)
                label_name_to_value[label_name] = label_value
        lbl, _ = labelme_utils.shapes_to_label(
            img.shape, data["shapes"], label_name_to_value
        )

        label_names = [None] * (max(label_name_to_value.values()) + 1)
        for name, value in label_name_to_value.items():
            label_names[value] = name

        lbl_viz = imgviz.label2rgb(
            lbl, imgviz.asgray(img), label_names=label_names, loc="rb"
        )

        PIL.Image.fromarray(img).save(osp.join(out_dir, "img.png"))
        labelme_utils.lblsave(osp.join(out_dir, "label.png"), lbl)
        PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, "label_viz.png"))

        with open(osp.join(out_dir, "label_names.txt"), "w") as f:
            for lbl_name in label_names:
                f.write(lbl_name + "\n")

        # logger.info("Saved to: {}".format(out_dir))


def annotate_from_label(mode):

    files = c.RAW_FILES[mode]
    kernel = c.MEDIAN_FILTER_KERNEL
    threshold = c.SIMPLE_THRESHOLD

    img_dir = c.IMG_DIR
    mask_dir = c.MASK_DIR
    mask_dir_full = c.MASK_DIR_FULL

    # Create folder in case it doesn't exist yet
    folder_name = c.MASK_DIR
    folder = join(c.DATA_DIR, mode, folder_name)
    utils.make_dir(folder)
    utils.make_dir(join(c.DATA_DIR, mode, img_dir))
    utils.make_dir(join(c.DATA_DIR, mode, mask_dir_full))
    utils.make_dir(c.FIG_DIR)

    # How often to print out with matplotlib
    debug_every = c.DBG_EVERY

    # Name of the folder in which the images will reside
    imgs_path = join(c.DATA_DIR, mode, c.IMG_DIR)
    masks_path = join(c.DATA_DIR, mode, c.MASK_DIR)
    # List of filenames of the .npy images
    # .jpg files are for visualizing the process
    images = [image for image in listdir(imgs_path) if '.json' in image]
    masks = [mask for mask in listdir(masks_path) if '.npy' in mask]
    # Get full image paths from filename list `images`
    image_paths = sorted([join(imgs_path, image) for image in images])

    # This is the index we will start on, in case there are already
    # data files in there
    # So, we are only adding to the existing list of files in /imgs/
    # -1 for zero-indexing, +1 because we want to start at the next free index
    img_idx = len(masks) - 1 + 1
    idx = img_idx if img_idx > 0 else 0  # numbering for images

    thresholds = []
    idx = 0
    tic = time()
    manual_labels = []
    auto_labels = []
    for folders, subfolders, files in os.walk(join(c.DATA_DIR, mode, mask_dir)):
        for file in files:
            # if '.npy' in file:
            #     img_idx = file[0:5]
            #     load_path = join(c.DATA_DIR, mask_dir)
            #     auto_img = np.load(join(load_path, f'{img_idx}.npy'))
            #     if len(auto_img.shape) > 2:
            #         print(file)
            #         print(auto_img.shape)
            if file == 'label.png':
                img_idx = folders.split('_')[-2]
                path = join(folders, file)
                added_img = PIL.Image.open(path)
                added_img = np.array(added_img)
                added_img *= 255

                _, thresh = cv2.threshold(
                    added_img, c.SIMPLE_THRESHOLD, 255, cv2.THRESH_BINARY)
                save = join(c.DATA_DIR, mode, mask_dir_full)
                np.save(join(save, img_idx), thresh)

                if idx % debug_every == 0:
                    utils.imsave(
                        join(save, f'_{img_idx}.jpg'), thresh, 512)  # debug

                idx += 1


if __name__ == "__main__":
    tic = time()
    utils.set_cwd(__file__)
    # for mode in ['train', 'val', 'test']:
    annotate_from_json(mode='train')
    elapsed = utils.time_report(tic, time())
    print(f'annotate_from_json completed after {elapsed}.')
