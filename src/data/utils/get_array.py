import torch
from tifffile.tifffile import TiffFile
import numpy as np


def get_array(file_path, dimensions, every_second=True, sample=0.01):
    '''
    Returns an array of the full data, i.e., 
    without Maximal Intensity Projection.
    '''
    timepoints = dimensions[0]
    # z-dimension
    D = dimensions[1]
    num_samples = int(timepoints*sample)

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
