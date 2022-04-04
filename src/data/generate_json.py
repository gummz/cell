from labelme import utils
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
from src.data.utils.make_dir import make_dir
'labelme_json_to_dataset _00400.json -o _00400_json'

'''
Inside test/masks, this algorithm will generate a json folder
 (with label.png inside) for every image (png) in masks_test.

For training: this algorithm will generate a json folder 
(with label.png inside) for every image (png) in masks.
'''


def main():
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
    c.setcwd(__file__)
    mode = 'test'
    mask_dir = join(c.DATA_DIR, mode, c.MASK_DIR)
    files = listdir(mask_dir)
    json_files = [file for file in files if '.json' in file]

    for json_file in json_files:
        out_dir = json_file.replace('.', '_')
        out_dir = join(mask_dir, out_dir)
        make_dir(out_dir)
        data = json.load(open(join(mask_dir, json_file)))
        imageData = data.get("imageData")

        if not imageData:
            imagePath = os.path.join(
                os.path.dirname(json_file), data["imagePath"])
            with open(imagePath, "rb") as f:
                imageData = f.read()
                imageData = base64.b64encode(imageData).decode("utf-8")
        img = utils.img_b64_to_arr(imageData)

        label_name_to_value = {"_background_": 0}
        for shape in sorted(data["shapes"], key=lambda x: x["label"]):
            label_name = shape["label"]
            if label_name in label_name_to_value:
                label_value = label_name_to_value[label_name]
            else:
                label_value = len(label_name_to_value)
                label_name_to_value[label_name] = label_value
        lbl, _ = utils.shapes_to_label(
            img.shape, data["shapes"], label_name_to_value
        )

        label_names = [None] * (max(label_name_to_value.values()) + 1)
        for name, value in label_name_to_value.items():
            label_names[value] = name

        lbl_viz = imgviz.label2rgb(
            lbl, imgviz.asgray(img), label_names=label_names, loc="rb"
        )

        PIL.Image.fromarray(img).save(osp.join(out_dir, "img.png"))
        utils.lblsave(osp.join(out_dir, "label.png"), lbl)
        PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, "label_viz.png"))

        with open(osp.join(out_dir, "label_names.txt"), "w") as f:
            for lbl_name in label_names:
                f.write(lbl_name + "\n")

        logger.info("Saved to: {}".format(out_dir))


if __name__ == "__main__":
    main()
