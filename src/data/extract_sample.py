from pydantic import FilePath
from src.data.utils.get_array import get_raw_array
from src.data.constants import RAW_DATA_DIR, TIMEPOINTS_TEST, RAW_FILE_DIMENSIONS_TEST
from os.path import join

file = 'LI_2019-01-17_emb7_pos3.lsm'
file_path = join(RAW_DATA_DIR, file)
array = get_raw_array(0, False)
print(array.shape)
