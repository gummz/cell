import cv2
import inspect
import os
from os import makedirs, mkdir, listdir
import numpy as np
import src.data.constants as c
from aicsimageio import AICSImage
import pickle
import torch
import io


class CPU_unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def active_slices(timepoint: np.array):
    max_val = np.max(timepoint)
    min_val = np.min(timepoint)
    diff = np.abs(max_val - min_val)
    thresh = diff * c.ACTIVE_SLICES_RATIO

    timepoint = np.where(timepoint > thresh, 1, 0)
    sums = np.sum(timepoint, axis=(1, 2))
    active_idx = np.argpartition(sums, -3)[-3:]

    return active_idx


def add_ext(files):
    raw_files = listdir(c.RAW_DATA_DIR)
    temp_files = []
    for file in files:
        if f'{file}.lsm' in raw_files:
            tmp = f'{file}.lsm'
        elif f'{file}.czi' in raw_files:
            tmp = f'{file}.czi'
        elif f'{file}.ims' in raw_files:
            tmp = f'{file}.ims'
        temp_files.append(tmp)

    if len(temp_files) == 1:
        return temp_files[0]

    return temp_files


def get_czi_dims(metadata):
    search_T = './Metadata/Information/Image/SizeT'
    search_Z = './Metadata/Information/Image/SizeZ'
    search_X = './Metadata/Information/Image/SizeX'
    search_Y = './Metadata/Information/Image/SizeY'
    search_strings = [search_T, search_Z, search_X, search_Y]
    names = ['T', 'Z', 'X', 'Y']
    dims = {}
    for search_string, name in zip(search_strings, names):
        element = metadata.findall(search_string)
        attributes = inspect.getmembers(
            element[0], lambda a: not(inspect.isroutine(a)))
        special_attributes = dict([a for a in attributes if not(
            a[0].startswith('__') and a[0].endswith('__'))])

        dims[name] = int(special_attributes['text'])

    return dims


def get_raw_array(file_path, index):
    '''
    Returns an array of the raw data, i.e.,
    without Maximal Intensity Projection.
    Output is 4D (timepoints and xyz spatial dimensions)
    '''
    raw_data = AICSImage(file_path)

    if '.czi' not in file_path:
        if type(index) == tuple:
            data = raw_data.get_image_dask_data(
                'TZXY', T=index, C=c.CELL_CHANNEL)
        elif type(index) == int:
            data = raw_data.get_image_dask_data(
                'ZXY', T=index, C=c.CELL_CHANNEL)
    else:
        dims = get_czi_dims(raw_data.metadata)

        if type(index) == tuple:
            data = raw_data.get_image_dask_data(
                'TZXY', T=index, C=c.CELL_CHANNEL)
        elif type(index) == int:
            data = raw_data.get_image_dask_data(
                'ZXY', T=index, C=c.CELL_CHANNEL)

    return data

    # if train:
    #     timepoints = c.TIMEPOINTS[index]
    #     # z-dimension
    #     D = c.RAW_FILE_DIMENSIONS[index]
    #     file = c.RAW_FILES[index]
    # else:
    #     timepoints = c.TIMEPOINTS_TEST[index]
    #     D = c.RAW_FILE_DIMENSIONS_TEST[index]
    #     file = c.RAW_FILES_GENERALIZE[index]

    # file_path = join(c.RAW_DATA_DIR, file)
    # num_samples = int(timepoints * sample)

    # with TiffFile(file_path) as f:
    #     # If only every second image contains beta cells
    #     if every_second:
    #         pages = f.pages[::2]
    #     else:
    #         pages = f.pages

    #     images = [page.asarray()[0, :, :] for page in pages]
    #     images = torch.tensor(images, dtype=torch.uint8)
    #     images = images.view((timepoints, D))
    #     idx = torch.multinomial(num_samples=num_samples,
    #                             replacement=False)
    #     images = images[idx]

    #     return images


def make_dir(path):
    if '/' in path:
        try:
            makedirs(path)
        except FileExistsError:
            pass
    else:
        try:
            mkdir(path)
        except FileExistsError:
            pass


def record_dict(t, slice_idx):
    slice_idx = [str(idx) for idx in slice_idx]
    record = {t: ','.join(slice_idx)}
    return record


def setcwd(file_path):
    '''Set working directory to script location'''
    abspath = os.path.abspath(file_path)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
