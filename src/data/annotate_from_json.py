import base64
import io
import json
import os
from os import listdir
from os.path import join
from time import time

import src.data.constants as c

# Set working directory to script location
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

files = c.RAW_FILES_GENERALIZE
kernel = c.MEDIAN_FILTER_KERNEL
threshold = c.SIMPLE_THRESHOLD

# Create folder in case it doesn't exist yet
folder_name = c.MASK_DIR_TEST
folder = join(c.DATA_DIR, folder_name)
try:
    os.mkdir(folder)
except FileExistsError:
    pass
try:
    os.mkdir(FIG_DIR)
except FileExistsError:
    pass

# How often to print out with matplotlib
debug_every = c.DBG_EVERY

# Name of the folder in which the images will reside
imgs_path = join(c.DATA_DIR, c.IMG_DIR_TEST)
masks_path = join(c.DATA_DIR, c.MASK_DIR_TEST)
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
for i, path in enumerate(image_paths):
    label_file = LabelFile(path)
    # img = img_b64_to_arr(label_file.imageData)
    # img_data = base64.b64decode(label_file.imageData)
    img_data = base64.decodebytes(label_file.imageData)
    # print(type(img_data))
    f = io.BytesIO()
    f.write(img_data)
    q = np.frombuffer(img_data, dtype=np.float32)
    print(q.shape)

    q = q.reshape((1024,1024,38))
    # print(len(img_data))
    # img_pil = Image.open(f)
    # print(img_pil.shape)
    exit()
    img = np.moveaxis(img, 2, 0)
 
    for i in range(4):
        plt.subplot(4, 1, i+1)
        print(img[i].shape)
        plt.plot(img[i])
    plt.savefig(join(FIG_DIR, 'test.png'))

    # Convert to PIL image; requirement for the model
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(img.shape)
    img=cv2.normalize(img, None, alpha=0, beta=255,
                        dtype=cv2.CV_8UC1, norm_type=cv2.NORM_MINMAX)
    name=f'{idx:05d}'
    # Folder to save image to
    save=join(folder, name)
    idx=idx + 1

    filtered=cv2.medianBlur(img, kernel)
    _, thresh=cv2.threshold(filtered,
                              threshold, 255, cv2.THRESH_BINARY)
    np.save(save, thresh)

    # Save as .jpg for debugging
    if i % debug_every == 0:
        dirs=os.path.dirname(save)
        file=os.path.basename(save)
        plt.imsave(f'{dirs}/_{file}.jpg', thresh, cmap='gray')


toc=time()
print(f'annotate_from_json.py complete after {(toc-tic)/60: .1f} minutes.')
