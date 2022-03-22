import torch
from tifffile.tifffile import TiffFile
from src.data.constants import RAW_DATA_DIR, TIMEPOINTS, RAW_FILE_DIMENSIONS, TIMEPOINTS_TEST, RAW_FILE_DIMENSIONS_TEST, RAW_FILES, RAW_FILES_GENERALIZE
from os.path import join


def get_raw_array(file_path, index, train=True, every_second=True, sample=0.01):
    '''
    Returns an array of the raw data, i.e.,
    without Maximal Intensity Projection.
    '''
    if train:
        timepoints = TIMEPOINTS[index]
        # z-dimension
        D = RAW_FILE_DIMENSIONS[index]
        file = RAW_FILES[index]
    else:
        timepoints = TIMEPOINTS_TEST[index]
        D = RAW_FILE_DIMENSIONS_TEST[index]
        file = RAW_FILES_GENERALIZE[index]

    file_path = join(RAW_DATA_DIR, file)
    num_samples = int(timepoints * sample)

    with TiffFile(file_path) as f:
        # If only every second image contains beta cells
        if every_second:
            pages = f.pages[::2]
        else:
            pages = f.pages

        images = [page.asarray()[0, :, :] for page in pages]
        images = torch.tensor(images, dtype=torch.uint8)
        images = images.view((timepoints, D))
        idx = torch.multinomial(num_samples=num_samples,
                                replacement=False)
        images = images[idx]

        return images
