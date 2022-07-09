import pandas as pd
import src.data.utils.utils as utils
import src.data.constants as c
from time import time
import src.models.BetaCellDataset as bcd
import os.path as osp
import src.models.eval_model as eval
import numpy as np
import src.experiments.eval_gridsearch.eval_gridsearch as eval_grid


def perform_study(device, save, mode, model, dataset, accept_ranges, match_thresholds):
    metrics_tot = eval_grid.perform_study(device, save,
                                          mode, model, dataset,
                                          accept_ranges,
                                          match_thresholds)

    # # reject no predictions - compare with rejection threshold
    # # to calibrate
    # ious = eval.eval_model(model, dataloader, mode, device,
    #                        save, reject_threshold=0,
    #                        match_threshold=0.2)

    # fetch rejected positives (i.e., false negatives)
    # get cols 2 and 5 (i.e. confusion matrix and certainty)
    columns = (2, 5)
    subframe = metrics_tot.loc[:, columns]
    confusion_matrices, avg_certainty = (subframe[c].values
                                         for c in columns)

    positive_ratio = (matrix[0][0] / (matrix[0][0] + matrix[1][0])
                      for matrix in confusion_matrices)
    return avg_certainty, positive_ratio, confusion_matrices


if __name__ == '__main__':
    mode = 'val'
    model_id = '27_06_17H_16M_53S'
    img_mode = 'auto'

    tic = time()
    utils.set_cwd(__file__)
    device = utils.set_device()
    save = osp.join('..', c.PROJECT_DATA_DIR, c.PRED_DIR, 'eval',
                    'seg_2d', f'model_{model_id}',
                    mode, 'model_calibration')

    model = utils.get_model(model_id, device)
    model = model.to(device)
    manual_select = 1 if img_mode == 'manual' else 0
    dataset = bcd.get_dataset(mode=mode, manual_select=manual_select)

    # grid search variables
    int_lower = np.round(np.linspace(0, 0.9, 10), 2)
    int_upper = np.round(np.linspace(0.1, 1, 10), 2)
    accept_ranges = tuple((lower, upper)
                          for lower, upper in zip(int_lower, int_upper))
    match_thresholds = (0.1,)

    avg_certainty, positive_ratio, cm = perform_study(
        device, save, mode, model, dataset,
        accept_ranges, match_thresholds)

    study = pd.DataFrame(
        {'accept_ranges': accept_ranges,
         'positive_ratio': positive_ratio,
         'avg_certainty': avg_certainty,
         'confusion_matrix': cm})
    study.to_csv('model_calibration.csv', sep=';')

    print(
        f'Model calibration complete after {utils.time_report(tic, time())}.')
