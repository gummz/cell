import numpy as np
import src.data.constants as c

from aicsimageio import AICSImage
from os import listdir
from os.path import join
import os
import numba

print('Script start')


@numba.jit(nopython=True, parallel=True)
def mean(array):
    return np.mean(array)


# Set working directory to script location
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

beta_ch = c.CELL_CHANNEL

# Get raw files and filepaths
files = listdir(c.RAW_DATA_DIR)
filepaths = [join(c.RAW_DATA_DIR, file) for file in files]

print('s')
# Make folder for files under this script's name
script = os.path.basename(__file__)
script = os.path.splitext(script)[0]
script_file_path = join(c.FILE_DIR, script)
try:
    os.makedirs(script_file_path)
except FileExistsError:
    pass

average_tot = {}
print('For loop start')
for (file, filepath) in zip(files, filepaths):
    data = AICSImage(filepath)
    average = []
    T = data.dims['T'][0]
    average = np.zeros(T, int)

    print('Inner for loop start')
    for t in range(T):
        timepoint = data.get_image_dask_data('ZXY', T=t, C=beta_ch).compute()
        average[t] = np.round(mean(timepoint), 3)
        print(f'Timepoint {t} done')

    average_tot[file] = mean(average)
    print(f'File {file} done\n\n\n')

np.savetxt(join(script_file_path, 'avg_pixel_intensity.csv'))

print('avg_pixel_intensity.py complete')
