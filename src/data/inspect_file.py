from tifffile import TiffFile
import numpy as np

file_path = '/dtu-compute/tubes/raw_data/LI_2019-11-21_emb6_pos3.lsm'

with TiffFile(file_path) as f:
    print(len(f.pages))
    print(np.array(f.pages).shape)
    print(len(f.pages)/29)
