import torch.tensor
import torch.multinomial
from tifffile.tifffile import TiffFile
import src.data.constants as c
from os.path import join


def get_raw_array(file_path, index, train=True, every_second=True, sample=0.01):
    '''
    Returns an array of the raw data, i.e.,
    without Maximal Intensity Projection.
    Output is 4D (timepoints and xyz spatial dimensions)
    '''
    if train:
        timepoints = c.TIMEPOINTS[index]
        # z-dimension
        D = c.RAW_FILE_DIMENSIONS[index]
        file = c.RAW_FILES[index]
    else:
        timepoints = c.TIMEPOINTS_TEST[index]
        D = c.RAW_FILE_DIMENSIONS_TEST[index]
        file = c.RAW_FILES_GENERALIZE[index]

    file_path = join(c.RAW_DATA_DIR, file)
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
