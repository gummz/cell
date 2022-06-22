import src.data.utils.utils as utils
import src.data.constants as c
from time import time
import src.models.BetaCellDataset as bcd
import src.models.eval_model as eval
import os.path as osp
import pandas as pd
import numpy as np


def perform_study(device, save, model, dataset,
                  accept_ranges, match_thresholds):

    n_items = 6
    ious_tot_len = len(accept_ranges) * len(match_thresholds)
    metrics_tot = np.empty((ious_tot_len, n_items), dtype=object)
    metrics_tot = pd.DataFrame(metrics_tot, dtype=object)

    for i, accept_range in enumerate(accept_ranges):
        for j, match_threshold in enumerate(match_thresholds):
            save_experiment = osp.join(
                save, f'{accept_range}_{match_threshold}')
            metrics = eval.eval_model(model, dataset,
                                      'val', device,
                                      save_experiment, accept_range=accept_range,
                                      match_threshold=match_threshold)

            (tp, fn), (fp, _) = metrics[0]
            precision = tp / (fp + tp)
            recall = tp / (fn + tp)
            if precision + recall > 0:
                f1_score = 2 * (precision * recall) / (precision + recall)
            else:
                f1_score = np.nan

            # confusion matrices
            matrices = metrics[0], metrics[2]
            avg_certainty = metrics[4]

            metrics_tot.iloc[i + j] = (accept_range,
                                       match_threshold,
                                       *matrices,
                                       round(f1_score, 3),
                                       round(avg_certainty, 3))

            print('=' * 30 + '\n', '=' * 30, '\n')
            print(
                f'Results for accept range ({accept_range[0]:.2f} ' +
                f'{accept_range[1]:.2f})' +
                f' and {match_threshold=:.3f}:')
            eval.score_report(metrics)

    return metrics_tot


if __name__ == '__main__':
    tic = time()
    utils.setcwd(__file__)
    mode = 'val'
    device = utils.set_device()

    save = osp.join('..', c.PROJECT_DATA_DIR, c.PRED_DIR, 'eval',
                    'seg_2d', f'model_{c.MODEL_STR}',
                    mode, 'eval_gridsearch')

    model = utils.get_model(c.MODEL_STR, device)
    model = model.to(device)
    dataset = bcd.get_dataset(mode='val')

    # grid search variables
    int_lower = np.round(np.linspace(0, 0.9, 10), 2)
    int_upper = np.round(np.linspace(0.1, 1, 10), 2)
    accept_ranges = tuple((lower, 1)
                          for lower, upper in zip(int_lower, int_upper))
    match_thresholds = np.linspace(0, 1, 11)

    metrics_tot = perform_study(device, save, model, dataset,
                                accept_ranges, match_thresholds)

    columns = ('accept_range', 'match_threshold', 'bbox_cm', 'bbox_iou',
               'f1_score', 'avg_certainty')
    study = metrics_tot
    study.columns = columns
    study.to_csv('eval_gridsearch_study.csv', sep=';')

    print(
        f'Evaluation gridsearch complete after {utils.time_report(tic, time())}.')
