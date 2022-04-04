from os.path import join
from pydantic import FilePath
from src.data.utils.get_array import get_raw_array
import src.data.constants as c
from os.path import join
import numpy as np

print('start')
c.setcwd(__file__)
# file = list(c.RAW_FILES_GENERALIZE.keys())[2]
file = 'LI_2020-12-10_emb3_pos4.czi'
# timepoints = c.RAW_FILES_GENERALIZE[file]
file_path = join(c.RAW_DATA_DIR, file)

array = get_raw_array(file_path, 150)
save = join(c.DATA_DIR, 'sample.npy')
np.save(save, array)
print('Saved to', save)
print(array.shape)
