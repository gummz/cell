import pandas as pd
import src.data.utils.utils as utils
import src.data.constants as c
from time import time
import src.models.BetaCellDataset as bcd
import os.path as osp
import src.models.eval_model as eval
import numpy as np
import src.experiments.eval_gridsearch.eval_gridsearch as eval_grid


def perform_study(device, save, model, dataset, accept_ranges, match_thresholds):
    metrics_tot = eval_grid.perform_study(device, save,
                                          model, dataset,
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

    positive_ratio = (matrix[0][0] / (matrix[0][0] + matrix[0][1])
                      for matrix in confusion_matrices)
    return avg_certainty, positive_ratio


if __name__ == '__main__':
    tic = time()
    utils.setcwd(__file__)
    mode = 'val'
    device = utils.set_device()

    save = osp.join('..', c.PROJECT_DATA_DIR, c.PRED_DIR, 'eval',
                    'seg_2d', f'model_{c.MODEL_STR}',
                    mode, 'model_calibration')

    model = utils.get_model(c.MODEL_STR, device)
    model = model.to(device)
    dataset = bcd.get_dataset(mode=mode)

    # grid search variables
    int_lower = np.round(np.linspace(0, 0.9, 10), 2)
    int_upper = np.round(np.linspace(0.1, 1, 10), 2)
    accept_ranges = tuple((lower, upper)
                          for lower, upper in zip(int_lower, int_upper))
    match_thresholds = (0.1,)

    avg_certainty, positive_ratio = perform_study(
        device, save, model, dataset, accept_ranges, match_thresholds)

    study = pd.DataFrame(
        {'accept_ranges': accept_ranges,
         'positive_ratio': positive_ratio,
         'avg_certainty': avg_certainty})
    study.to_csv('model_calibration.csv', sep=';')

    study.plot()

    print(
        f'Model calibration complete after {utils.time_report(tic, time())}.')
