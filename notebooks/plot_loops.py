from aicsimageio import AICSImage
import src.data.constants as c
import src.data.utils.utils as utils
import os.path as osp
from matplotlib import pyplot as plt
import numpy as np
import dask
import warnings
np.random.seed(42)
from time import time
warnings.filterwarnings('ignore', '.*PerformanceWarning.*')

tic = time()
utils.set_cwd(__file__)
data_dir = osp.join('/', 'dtu-compute', 'tubes', 'results')

file = 'LI_2019-02-05_emb5_pos4'
folder = osp.join('test', file)
tif_file = 'cyctpy15-pred-0.7-semi-40_2019-02-05_emb5_pos4.tif'
tif_load_path = osp.join(data_dir, folder, tif_file)

tif_data = AICSImage(tif_load_path)
tif_array = np.moveaxis(tif_data.get_image_dask_data()[0, :, :, :, :, 0], 1, 0)
tif_array_thresh = dask.array.where(tif_array > 0, 255, 0)

print('tif_data.dims', tif_data.dims, '\n')

plt.figure(figsize=(20, 20), dpi=200)
t_idx = int(np.random.randint(0, tif_array.shape[0]-3, 1))
z_idx = int(np.random.randint(0, tif_array.shape[1]-3, 1))
print('t_idx', t_idx, 'z_idx', z_idx, '\n')

count = 0
for i, array_i in enumerate(tif_array[t_idx:(t_idx+3), :, :]):
    for j, array_j in enumerate(array_i[z_idx:(z_idx+3), :, :]):
        plt.subplot(3, 3, count + 1)
        count += 1
        plt.imshow(array_j)
        print(np.unique(array_j.compute(), return_counts=True))
plt.savefig('9_output.png')

print(f'plot_loops.py done in {utils.time_report(tic, time())}.')