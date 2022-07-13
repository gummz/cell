from time import time
import numpy as np
import torch
import src.data.utils.utils as utils
import src.data.constants as c
from src.models.BetaCellDataset import BetaCellDataset, get_dataloaders
from src.data.extract_cross_sample import get_slice_record
from os.path import join
from src.models.predict_model import get_prediction, get_predictions
import src.visualization.utils as viz

if __name__ == '__main__':
    tic = time()
    utils.set_cwd(__file__)
    device = utils.set_device()
    model = utils.get_model(c.MODEL_STR, device)
    mode = 'val'
    save = join(c.DATA_DIR, c.PRED_DIR, mode, 'compare')
    utils.make_dir(save)

    indices = (f'{idx:05d}' for idx in range(100, 104))
    names, files, ts, zs = get_slice_record(indices, mode)

    data_val = BetaCellDataset(transforms=1, mode=mode,
                               manual_select=1)

    iterable = zip(names, files, ts, zs)
    for name, file, t, z in iterable:
        i = int(name)

        # get prediction straight from the raw dataset
        z_slice = utils.get_raw_array(
            join(c.RAW_DATA_DIR, file), t, z).compute()
        # need to convert z_slice from np.uint16 to np.int16
        # for tensor conversion
        z_slice = torch.tensor(np.int16(z_slice),
                               device=device, dtype=torch.int16)
        pred = get_prediction(model, device, z_slice)

        # get prediction from dataset in data/interim/mode/
        z_slice_v, _ = data_val[i]
        pred_v = get_prediction(model, device, z_slice_v)

        # compare the two predictions
        viz.output_pred(mode, i, z_slice, pred, save, compare=True)
        # if same index i is used, the latter call to output_pred
        # will overwrite the former call
        viz.output_pred(mode, i + len(names), z_slice_v,
                        pred_v, save, compare=True)

    elapsed = utils.time_report(tic, time())
    print(f'compare_dataset_raw for {mode} completed in {elapsed}.')
