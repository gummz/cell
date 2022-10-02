import cv2
import inspect
import os
from os import makedirs, mkdir, listdir
import os.path as osp
import imageio
import numpy as np
import pandas as pd
import src.data.constants as c
from aicsimageio import AICSImage
import pickle
from PIL import Image
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
    if osp.exists(c.RAW_DATA_DIR):
        raw_files = listdir(c.RAW_DATA_DIR)
    else:
        raw_files = pd.read_csv(osp.join(c.PROJECT_DATA_DIR, 'raw', 'raw_files.csv'))[
            'file'].values
    temp_files = []
    if not isinstance(files, str):
        for file in files:  # `files` is ArrayLike
            file_with_ext = add_ext_single(file, raw_files)
            temp_files.append(file_with_ext)
        if len(temp_files) == 1:
            return temp_files[0]
    else:  # `files` is a string
        file = os.path.basename(files)
        return add_ext_single(file, raw_files)

    return temp_files


def add_ext_single(file, raw_files):
    if f'{file}.lsm' in raw_files:
        return f'{file}.lsm'
    elif f'{file}.czi' in raw_files:
        return f'{file}.czi'
    elif f'{file}.ims' in raw_files:
        return f'{file}.ims'
    else:
        raise FileNotFoundError(f'File not found with extension: {file}')


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


def get_data_dir():
    if osp.exists(c.DATA_DIR):
        return c.DATA_DIR
    else:
        if osp.exists(c.PROJECT_DATA_DIR):
            return c.PROJECT_DATA_DIR
        else:
            return osp.join('..', c.PROJECT_DATA_DIR)


def get_dir_size(path='.', recursive=False):
    total = 0
    if not osp.exists(path):
        return total
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir() and recursive:
                total += get_dir_size(entry.path)
    return total


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


def get_model(time_str: str, device: torch.device, ):

    folder = osp.join(c.PROJECT_DIR, 'src', 'models',
                      'interim', f'run_{time_str}')
    if not osp.exists(folder):
        folder = osp.join(c.HPC_PROJECT, 'src', 'models',
                          'interim', f'run_{time_str}')

    # Load model
    load_path = osp.join(folder, f'model_{time_str}.pkl')
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


def get_raw_array(data, t=None, z=None,
                  ch=c.CELL_CHANNEL):
    '''
    Returns an array of the raw data, i.e.,
    without Maximal Intensity Projection.
    Output is 4D (timepoints and xyz spatial dimensions)

    If get_raw_array needs to be used multiple times in the same script,
    drawing from the same raw data file, it's better to pass an AICSImage
    as `input`.
    '''

    if isinstance(data, str):  # `data` is path to file, not file itself
        if os.path.exists(data) and os.path.isfile(data):
            raw_data = AICSImage(data)
        else:
            raw_data = np.load(c.SAMPLE_PATH)
            print((
                f'WARNING: Could not locate {data}.\n'
                f'Using sample dataset from {c.SAMPLE_PATH}.'
            ))
    elif isinstance(data, AICSImage):  # `data` is file itself
        raw_data = data
    else:
        raise ValueError('Unknown data type for `data`.')

    t_tuple = 'T' if has_len(t) or t is None else ''
    z_tuple = 'Z' if has_len(z) or z is None else ''
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

    elif t is None and z is None:
        data = raw_data.get_image_dask_data(
            order, C=ch)

    return data


def has_len(obj):
    try:
        len(obj)
        return True
    except TypeError:
        return False


def is_int(number):
    types = [np.int32, np.int16, np.int64, int]
    return type(number) in types


def imsave(path, img, resize=512, cmap=None):
    dirs = os.path.dirname(path)
    make_dir(dirs)

    if osp.splitext(path)[1] not in ('.png', '.jpg'):
        path += '.jpg'

    if type(img) == mplt.figure.Figure:
        img.savefig(path)
        plt.close()
        return
    else:
        if type(img) == torch.Tensor:
            img = np.array(img.cpu())
        elif type(img) == list:
            img = np.array(img)

        # if len(img.shape) > 2:
        #     print('Unintended: image has shape', img.shape)
        if resize:
            img = cv2.resize(img, (resize, resize), cv2.INTER_AREA)

    if cmap:
        plt.imsave(path, img, cmap=cmap)
    else:
        plt.imsave(path, img)


def make_dir(path):
    if not path:
        return
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


def png_to_movie(time_range, location):
    '''Saves a movie from a list of images.
    Saves the movie in the specified location,
    where it is assumed the images are (in `location`).
    '''
    make_dir(osp.dirname(location))
    # append each image to the movie frames list
    movie_frames = []
    iterable = sorted(os.listdir(location))
    time_range = time_range if time_range else range(len(iterable))
    for image in iterable:
        name, ext = osp.splitext(image)
        image_path = osp.join(location, image)
        if ext == '.png' and int(name) in time_range:
            movie_frames.append(Image.open(image_path))

    raw_data_name, _ = osp.splitext(osp.basename(
        osp.dirname(osp.dirname(location))))
    start, stop = time_range[0], time_range[-1]
    movie_name = f'{raw_data_name}_movie_t{start}-{stop}.mp4'
    save_loc = osp.join(location, movie_name)
    imageio.mimsave(save_loc, movie_frames, fps=1)


def record_dict(t, slice_idx):
    record = {t: slice_idx}
    return record


def set_device(device_str: str = None):
    if device_str:
        device = torch.device(device_str)
        return device

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    return device


def set_cwd(file_path):
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


def to_csv(df, path, sep):
    # Prepend dtypes to the top of df (from https://stackoverflow.com/a/43408736/7607701)
    # df.loc[-1] = df.dtypes
    # df.index = df.index + 1
    # df.sort_index(inplace=True)
    # Then save it to a csv
    df.to_csv(path, sep=sep, index=False)


def read_csv(path):
    # Read types first line of csv
    dtypes = pd.read_csv(path, nrows=1).iloc[0].to_dict()
    # Read the rest of the lines with the types from above
    return pd.read_csv(path, dtype=dtypes, skiprows=[1])
