import cv2
import inspect
import os
from os import makedirs, mkdir, listdir
from os.path import join
import numpy as np
import src.data.constants as c
from aicsimageio import AICSImage
import pickle
import torch
import matplotlib.pyplot as plt
import io


class CPU_unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def active_slices(timepoint: np.array, ratio=None):
    max_val = np.max(timepoint)
    min_val = np.min(timepoint)
    diff = np.abs(max_val - min_val)
    ratio = ratio if ratio else c.ACTIVE_SLICES_RATIO
    thresh = diff * ratio

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
        else:
            raise RuntimeError(f'File not found with extension: {file}')
        temp_files.append(tmp)

    if len(temp_files) == 1:
        return temp_files[0]

    return temp_files


def del_multiple(list_object, indices):
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)


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


def set_device():
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    return device


def get_model(folder, time_str: str, device: torch.device):
    # Load model
    load_path = join(folder, f'model_{time_str}.pkl')
    if device.type == 'cuda':
        model = pickle.load(open(load_path, 'rb'))
    else:
        # model = pickle.load(open(load, 'rb'))
        model = CPU_unpickler(open(load_path, 'rb')).load()
        '''Attempting to deserialize object on a CUDA device
        but torch.cuda.is_available() is False.
        If you are running on a CPU-only machine, please use
        torch.load with map_location=torch.device('cpu') to map your storages to the CPU.'''

    return model


def get_raw_array(file_path, which, idx):
    '''
    Returns an array of the raw data, i.e.,
    without Maximal Intensity Projection.
    Output is 4D (timepoints and xyz spatial dimensions)
    '''
    raw_data = AICSImage(file_path)

    if which == 'timepoint':
        if '.czi' not in file_path:
            if type(idx) == tuple:
                data = raw_data.get_image_dask_data(
                    'TZXY', T=idx, C=c.CELL_CHANNEL)
            elif type(idx) == int:
                data = raw_data.get_image_dask_data(
                    'ZXY', T=idx, C=c.CELL_CHANNEL)
        else:
            dims = get_czi_dims(raw_data.metadata)

            if type(idx) == tuple:
                data = raw_data.get_image_dask_data(
                    'TZXY', T=idx, C=c.CELL_CHANNEL)
            elif type(idx) == int:
                data = raw_data.get_image_dask_data(
                    'ZXY', T=idx, C=c.CELL_CHANNEL)
    elif which == 'slice':
        if '.czi' not in file_path:
            if type(idx) == tuple:
                data = raw_data.get_image_dask_data(
                    'TZXY', Z=idx, C=c.CELL_CHANNEL)
            elif type(idx) == int:
                data = raw_data.get_image_dask_data(
                    'TXY', Z=idx, C=c.CELL_CHANNEL)
        else:
            dims = get_czi_dims(raw_data.metadata)

            if type(idx) == tuple:
                data = raw_data.get_image_dask_data(
                    'TZXY', Z=idx, C=c.CELL_CHANNEL)
            elif type(idx) == int:
                data = raw_data.get_image_dask_data(
                    'TXY', Z=idx, C=c.CELL_CHANNEL)

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


def imsave(path, img, resize=512):
    dirs = os.path.dirname(path)
    make_dir(dirs)
    if resize:
        if type(img) != np.ndarray:
            img = np.array(img)
        if len(img.shape) > 2:
            img = img[0]
        img = cv2.resize(img, (resize, resize), cv2.INTER_AREA)

    if path[-4:] not in ['.png', '.jpg']:
        path += '.jpg'

    plt.imsave(path, img)


def make_dir(path):
    dirs = os.path.dirname(path)
    if len(dirs) > 1:  # more than one directory
        makedirs(path, exist_ok=True)
    else:
        try:
            mkdir(path)
        except FileExistsError:
            pass


def normalize(img, alpha, beta, out):
    if type(img) != np.ndarray:
        img = np.array(img)
    return cv2.normalize(img, None, alpha=alpha, beta=beta, norm_type=cv2.NORM_MINMAX, dtype=out)


def record_dict(t, slice_idx):
    # print('slice idx', slice_idx)
    # slice_idx = [str(idx) for idx in slice_idx]
    record = {t: slice_idx}
    return record


def setcwd(file_path):
    '''Set working directory to script location'''
    abspath = os.path.abspath(file_path)
    dname = os.path.dirname(abspath)
    os.chdir(dname)


def time_report(path, tic, toc):
    elapsed = max(tic, toc) - min(tic, toc)
    file = os.path.basename(path)
    if elapsed / 60 < 1:
        print(f'{file} complete after {elapsed:.1f} seconds.')
    else:
        print(f'{file} complete after {elapsed / 60:.1f} minutes.')

