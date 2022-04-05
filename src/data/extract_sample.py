from os.path import join
from pydantic import FilePath
import src.data.utils.utils as utils
import src.data.constants as c
from os.path import join
import numpy as np

print('start')
utils.setcwd(__file__)
# file = list(c.RAW_FILES_GENERALIZE.keys())[2]
file = 'LI_2020-12-10_emb3_pos4.czi'
# timepoints = c.RAW_FILES_GENERALIZE[file]
file_path = join(c.RAW_DATA_DIR, file)

array = utils.get_raw_array(file_path, 150)
save = join(c.DATA_DIR, 'sample.npy')
np.save(save, array)
print('Saved to', save)
print(array.shape)
