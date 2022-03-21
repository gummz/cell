from pydantic import FilePath
from tifffile import TiffFile

file_path = '/dtu-compute/tubes/raw_data/LI_2019-11-21_emb6_pos3.lsm'

with TiffFile(file_path) as f:
    print(len(f.pages))
    print(f.pages.asarray()[0, :, :].shape)
    print(len(f.pages)/29)
