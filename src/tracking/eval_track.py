from os import listdir

from matplotlib import pyplot as plt
import numpy as np
import src.data.constants as c
import pickle
from os.path import join
import src.data.utils.utils as utils
from aicsimageio import AICSImage


def eval_track(tracked_centroids, filename):
    # use `tracked_centroids`'s frame, x, y, p
    # to overlay the points onto MIP of raw image

    # if there are images already, don't create new ones
    # folder is: pred/eval/track/
    load = join(c.DATA_DIR, c.PRED_DIR, 'eval', 'track', filename)
    utils.make_dir(load)
    images = [image for image in listdir(load) if '.png' in image]
    if len(images) > 0:
        images = [plt.imread(join(load, image)) for image in images]
    else:
        # create new raw MIP images
        path = join(c.RAW_DATA_DIR, filename)
        data = AICSImage(path)
        T = data.dims['T'][0]
        images = np.empty((T, 1024, 1024))
        for t in range(T):
            array = utils.get_raw_array(path, 'timepoint', t)
            array_mip = np.max(array, axis=0)

            tubes = utils.get_raw_array(path, 'timepoint', t, channel=1)
            tubes_mip = np.max(tubes, axis=0)

            images[t] = np.max(
                np.array([array_mip, tubes_mip]), axis=0)

            plt.imsave(join(load, f'{t}.png'), images[t])

    # overlay points in each frame on corresponding
    # raw MIP image
    for frame, image in zip(tracked_centroids, images):
        x, y, p = frame['x', 'y', 'p']
        plt.plot(x, y, color=p)
        plt.imshow(image)
        plt.savefig(join(load, f'{frame}.png'))
        plt.close()


if __name__ == '__main__':
    # output 2D movie
    mode = 'pred'  # we are in test mode but files are in pred
    name = c.PRED_NAME
    tracked_centroids = pickle.load(open(join(c.DATA_DIR, mode, name)))
    eval_track(tracked_centroids)
