import numpy as np
import src.data.utils.utils as utils
import src.data.constants as c
from os import listdir
from os.path import join
from aicsimageio import AICSImage

if __name__ == '__main__':
    utils.set_cwd(__file__)
    raw_files = listdir(c.RAW_DATA_DIR)
    raw_files = [file for file in raw_files if '.czi' not in file]
    file_hists = np.empty
    bins = 10

    for file in raw_files:
        path = join(c.RAW_DATA_DIR, file)
        data = AICSImage(path)
        T = data.dims['T'][0]
        del data

        mean_hists = np.empty((T, bins))
        for t in range(T):
            timepoint = utils.get_raw_array(path, t)
            hist = np.histogram(timepoint, bins)
            mean_hist = np.mean(hist)
            mean_hists[t] = mean_hist
