from pydantic import FilePath
from src.data.utils.get_array import get_raw_array
from src.data.constants import RAW_DATA_DIR, TIMEPOINTS_TEST, RAW_FILE_DIMENSIONS_TEST
from os.path import join

array = get_raw_array(0, train=False)
print(array.shape)
