import numpy as np
import src.data.constants as c

from aicsimageio import AICSImage
from os import listdir
from os.path import join
import os

# Set working directory to script location
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

beta_ch = c.CELL_CHANNEL

# Get raw files and filepaths
files = listdir(c.RAW_DATA_DIR)
filepaths = [join(c.RAW_DATA_DIR, file) for file in files]

# Make folder for files under this script's name
script = os.path.basename(__file__)
script = os.path.splitext(script)[0]
script_file_path = join(c.FILE_DIR, script)
try:
    os.makedirs(script_file_path)
except FileExistsError:
    pass

average_tot = {}
for (file, filepath) in zip(files, filepaths):
    data = AICSImage(filepath)
    average = []
    T = data.dims['T'][0]

    for t in range(T):
        timepoint = data.get_image_dask_data('ZYX', T=t, C=beta_ch)
        average.append(np.round(np.mean(timepoint), 3))

    average_tot[file] = np.mean(average)

np.savetxt(join(script_file_path, 'avg_pixel_intensity.csv'))

print('avg_pixel_intensity.py complete')
