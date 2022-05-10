from os.path import join

import pandas as pd
import src.data.constants as c
import src.data.utils.utils as utils


def get_slice_record(index, mode):
    record_path = join(c.DATA_DIR, mode, c.IMG_DIR, 'slice_record.csv')
    slice_record = pd.read_csv(
        record_path, sep='\t', header=0, dtype={'name': str})
    rec = slice_record.iloc[index]
    name, file, t, z = (rec[col] for col in rec.columns)
    return name, file, t, z


if __name__ == '__main__':
    index = 88
    mode = 'train'
    # whether the user wants a specific timepoint
    # of his choosing regardless of slice index above
    custom = False
    resize = False
    # which:
    # Should the extract be one timepoint across slices,
    # or one slice across timepoints?
    which = 'timepoint'  # or which = 'slice'

    utils.setcwd(__file__)

    name, file, t, z = get_slice_record(index, mode)
    t = t if which == 'timepoint' else None
    z = z if t is None else None

    if custom:
        t = 9
        z = 11

    idx = int(t) if which == 'timepoint' else int(z)
    folder = f'{which}_{idx}'
    output_path = join(c.DATA_DIR, c.EXTRACT_DIR, file, folder)
    utils.make_dir(output_path)

    raw_file_path = join(c.RAW_DATA_DIR, file)
    data = utils.get_raw_array(raw_file_path, t, z)

    for i, array in enumerate(data):
        name_str = f'{t:02d}_{i:02d}.jpg' if which == 'timepoint' \
            else f'{i:02d}_{z:02d}.jpg'
        utils.imsave(join(output_path, name_str), array, resize=resize)

    print('Extracted index', index, 'from\n\t', file, '\nthe',
          which, '\n\t', idx, '\nTimepoint\n\t', t, '\nSlice\n\t', z)
