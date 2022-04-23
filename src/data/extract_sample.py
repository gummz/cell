from os.path import join

import pandas as pd
import src.data.constants as c
import src.data.utils.utils as utils

index = 45  # '00023'
mode = 'train'
custom = False
# which:
# Should the extract be one timepoint across slices,
# or one slice across timepoints?
which = 'slice'  # or which = 'slice'

utils.setcwd(__file__)

record_path = join(c.DATA_DIR, mode, c.IMG_DIR, 'slice_record.csv')
slice_record = pd.read_csv(
    record_path, sep='\t', header=0, dtype={'name': str})
name, file, t, z = slice_record.iloc[index].values
if custom:
    t = 42
raw_file_path = join(c.RAW_DATA_DIR, file)

idx = int(t) if which == 'timepoint' else int(z)
data = utils.get_raw_array(raw_file_path, which, idx)
folder = f'{which}_{idx}'
output_path = join(c.DATA_DIR, c.EXTRACT_DIR, file, folder)
utils.make_dir(output_path)

for i, array in enumerate(data):
    name_str = f'{t:02d}_{i:02d}.jpg' if which == 'timepoint' \
        else f'{i:02d}_{z:02d}.jpg'
    utils.imsave(join(output_path, name_str), array)


print('Extracted from\n\t', file, '\nthe', which, '\n\t', idx, '\nTimepoint\n\t', t, '\nSlice\n\t', z)
