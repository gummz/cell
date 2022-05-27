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
import matplotlib as mplt
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


def calc_sensitivity(confusion_matrix):
    tp, fn = confusion_matrix[0][0], confusion_matrix[0][1]
    sensitivity = tp / (fn + tp)
    return sensitivity


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


def get_mask(output):
    '''Consolidates the masks into one mask.'''
    if type(output) == dict:  # `output` is direct output of model (from one image)
        mask = output['masks']
    else:
        mask = output  # target is a tensor of masks (from one image)
    mask = torch.squeeze(mask, dim=1)

    if mask.shape[0] != 0:
        mask = torch.max(mask, dim=0).values
    else:
        try:
            mask = torch.zeros((mask.shape[1], mask.shape[2])).to(
                mask.get_device())
        except RuntimeError:  # no GPU
            mask = torch.zeros((mask.shape[1], mask.shape[2]))

    # mask = np.array(mask)
    # mask = np.where(mask > 255, 255, mask)
    # mask = np.where(mask > 200, 255, 0)
    mask *= 255
    mask = mask.clone().detach().type(torch.uint8)
    # torch.tensor(mask, dtype=torch.uint8)

    return mask


def get_model(time_str: str, device: torch.device):

    folder = f'../models/interim/run_{time_str}'

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


def get_raw_array(file_path, t=None, z=None,
                  ch=c.CELL_CHANNEL):
    '''
    Returns an array of the raw data, i.e.,
    without Maximal Intensity Projection.
    Output is 4D (timepoints and xyz spatial dimensions)
    '''

    if t is None and z is None:
        return None

    if os.path.exists(file_path):
        raw_data = AICSImage(file_path)
    else:
        raw_data = AICSImage(c.SAMPLE_PATH)
        print((
            f'WARNING: Could not access {file_path}.\n'
            f'Using sample dataset from {c.SAMPLE_PATH}.'
        ))

    t_tuple = 'T' if has_len(t) or t is None else ''
    z_tuple = 'Z' if has_len(z) != int or z is None else ''
    channel_tuple = 'C' if type(ch) != int or ch is None else ''
    order = f'{t_tuple}{z_tuple}XY{channel_tuple}'

    data = None

    if t is not None and z is None:
        data = raw_data.get_image_dask_data(
            order, T=t, C=ch)

    elif t is None and z is not None:
        data = raw_data.get_image_dask_data(
            order, Z=z, C=ch)

    elif t is not None and z is not None:
        data = raw_data.get_image_dask_data(
            order, T=t, Z=z, C=ch)

    return data


def has_len(obj):
    obj_method = getattr(obj, 'len', None)
    return callable(obj_method)


def is_int(number):
    types = [np.int32, np.int16, np.int64, int]
    return type(number) in types


def imsave(path, img, resize=512, cmap=None):
    dirs = os.path.dirname(path)
    make_dir(dirs)

    if path[-4:] not in ['.png', '.jpg']:
        path += '.jpg'

    if type(img) == mplt.figure.Figure:
        img.savefig(path)
        return
    else:
        if type(img) == torch.Tensor:
            img = np.array(img.cpu())
        elif type(img) == list:
            img = np.array(img)

        if len(img.shape) > 2:
            print('Unintended: image has shape', img.shape)
        if resize:
            img = cv2.resize(img, (resize, resize), cv2.INTER_AREA)

    if cmap:
        plt.imsave(path, img, cmap=cmap)
    else:
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


def normalize(image, alpha, beta, out, device=None):
    if type(image) == torch.Tensor:
        image = image.cpu().numpy()
        normalized = cv2.normalize(image, None, alpha=alpha,
                                   beta=beta, norm_type=cv2.NORM_MINMAX, dtype=out)
        return torch.tensor(normalized,
                            device=device
                            if device else torch.device('cpu'))
    return cv2.normalize(image, None, alpha=alpha, beta=beta, norm_type=cv2.NORM_MINMAX, dtype=out)


def record_dict(t, slice_idx):
    # print('slice idx', slice_idx)
    # slice_idx = [str(idx) for idx in slice_idx]
    record = {t: slice_idx}
    return record


def set_device(device_str: str = None):
    if device_str:
        device = torch.device(device_str)
        return device

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    return device


def setcwd(file_path):
    '''Set working directory to script location'''
    abspath = os.path.abspath(file_path)
    dname = os.path.dirname(abspath)
    os.chdir(dname)


def time_report(tic, toc):
    elapsed = max(tic, toc) - min(tic, toc)
    if elapsed / 60 < 1:
        return f'{elapsed:.1f} seconds'
    else:
        return f'{elapsed / 60:.1f} minutes'
