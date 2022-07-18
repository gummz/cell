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
from os.path import join
from time import time
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import src.data.utils.utils as utils
import src.data.constants as c

'labelme_json_to_dataset _00400.json -o _00400_json'


def annotate_from_json(db_version, mode):
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
    generate_label(db_version, mode)

    # STEP 2
    annotate_from_label(db_version, mode)


def generate_label(db_version, mode):
    data_dir = c.DATA_DIR if osp.exists(c.DATA_DIR) else c.PROJECT_DATA_DIR
    process_dir = osp.join(data_dir, 'db_versions',
                           db_version, mode)
    files = os.listdir(osp.join(process_dir, c.IMG_DIR))
    json_files = sorted((file for file in files if '.json' in file))
    image_files = [image for image in files if '.png' in image]
    # take only json files which have corresponding image files
    json_files = (file for file in json_files
                  if file.split('.')[0] + '.png' in image_files)
    for json_file in json_files:
        out_name = json_file.replace('.', '_')
        out_dir = osp.join(process_dir, c.MASK_DIR, out_name)
        utils.make_dir(out_dir)
        json_path = osp.join(process_dir, c.IMG_DIR, json_file)
        data = json.load(open(json_path))
        imageData = data.get("imageData")

        if not imageData:
            imagePath = osp.join(
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
        PIL.Image.fromarray(lbl_viz).save(
            osp.join(out_dir, "label_viz.png"))

        with open(osp.join(out_dir, "label_names.txt"), "w") as f:
            for lbl_name in label_names:
                f.write(lbl_name + "\n")

        # logger.info("Saved to: {}".format(out_dir))


def annotate_from_label(db_version, mode):

    files = c.RAW_FILES[mode]

    data_dir = c.DATA_DIR if osp.exists(c.DATA_DIR) else c.PROJECT_DATA_DIR
    img_dir = c.IMG_DIR
    mask_dir_full = c.MASK_DIR_FULL

    # Create folder in case it doesn't exist yet
    folder = osp.join(data_dir, 'db_versions',
                      db_version, mode)
    utils.make_dir(osp.join(folder, img_dir))
    utils.make_dir(osp.join(folder, mask_dir_full))

    # How often to print out with matplotlib
    debug_every = c.DBG_EVERY

    idx = 0

    sequence = os.walk(osp.join(data_dir, 'db_versions',
                                db_version, mode, img_dir))
    for folders, subfolders, files in sequence:
        for file in files:
            # if '.npy' in file:
            #     img_idx = file[0:5]
            #     load_path = osp.join(c.DATA_DIR, mask_dir)
            #     auto_img = np.load(osp.join(load_path, f'{img_idx}.npy'))
            #     if len(auto_img.shape) > 2:
            #         print(file)
            #         print(auto_img.shape)
            if file == 'label.png':
                img_idx = osp.basename(folders)[:5]
                path = osp.join(folders, file)
                added_img = PIL.Image.open(path)
                added_img = np.array(added_img)
                added_img *= 255

                _, thresh = cv2.threshold(
                    added_img, c.SIMPLE_THRESHOLD, 255, cv2.THRESH_BINARY)
                save = osp.join(folder, mask_dir_full)
                np.save(osp.join(save, img_idx), thresh)

                if idx % debug_every == 0:
                    utils.imsave(
                        osp.join(save, f'_{img_idx}.jpg'), thresh, 512)  # debug

                idx += 1


if __name__ == "__main__":
    tic = time()
    utils.set_cwd(__file__)
    db_version = 'hist_eq'
    for mode in ('train', 'val', 'test'):
        annotate_from_json('hist_eq', mode=mode)

    elapsed = utils.time_report(tic, time())
    print(f'annotate_from_json completed after {elapsed}.')
