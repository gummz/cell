import datetime
import math
from time import time
import cv2

import numpy as np
import optuna
from os.path import join
import src.data.constants as c
import src.data.utils.utils as utils
import pickle
from src.models.BetaCellDataset import get_dataloaders
from src.models.train_model import train
from src.models.utils.model import get_instance_segmentation_model
from torch import optim
from torch.utils.tensorboard import SummaryWriter


def objective(trial):

    device = utils.set_device()

    # hyperparameters
    # pretrained = trial.suggest_categorical('pretrained', [True, False])
    amsgrad = trial.suggest_categorical('amsgrad', [True, False])
    batch_sizes = (32, 64, 128, 256)
    batch_size = trial.suggest_categorical('batch_size', batch_sizes)
    beta1 = trial.suggest_float('beta1', 0, 1)
    beta2 = trial.suggest_float('beta2', 0, 1)
    # the parameters of these filters have been optimized
    # with experiments
    filters = {
        'none': None,
        'mean': (cv2.blur, [(5, 5)]),
        'gaussian': (cv2.GaussianBlur, [(5, 5), 0]),
        'median': (cv2.medianBlur, [5]),
        'bilateral': (cv2.bilateralFilter, [9, 50, 50]),
        'nlmeans': (cv2.fastNlMeansDenoising, [None, 11, 7, 21]),
        'canny': (utils.canny_filter, [20, 20, 3, False])
    }
    filter_optim = trial.suggest_categorical('filters', list(filters.keys()))
    img_filter = filters[filter_optim]

    image_size = 128  # trial.suggest_int('image_size', 28, 512)
    loss_list = [
        'loss_mask', 'loss_rpn_box_reg', 'loss_box_reg',
        'loss_classifier', 'loss_objectness'
    ]
    # loss_selection = [
    #     [loss_list[:-1], loss_list[:-2], loss_list[:-3]],
    #     [loss_list[:-4], loss_list[0], loss_list[1]],
    #     [loss_list[2], loss_list[3], loss_list[4]],
    #     [loss_list[1:3]]
    # ]
    # losses = trial.suggest_categorical('losses', loss_selection)
    lr = trial.suggest_float('learning_rate', 1e-10, 1e-3, log=True)
    manual_select = 1  # trial.suggest_int('manual_ratio', 2, 27)
    weight_decay = trial.suggest_float('weight_decay', 1e-8, 1e-2)
    # end hyperparameters

    data_tr, data_val = get_dataloaders(
        root=join('..', c.DATA_DIR),
        batch_size=batch_size, num_workers=4, resize=image_size, n_img_select=200, manual_select=manual_select, img_filter=img_filter)

    model = get_instance_segmentation_model(pretrained=True)
    model.to(device)
    # dict so train_model can access hyperparameters
    hparam_dict = {
        'batch_size': batch_size,
        'losses': ';'.join(loss_list),
        'image_size': image_size,
        'size': image_size
    }
    params = [p for p in model.parameters() if p.requires_grad]
    opt = optim.Adam(params, lr=lr,
                     weight_decay=weight_decay,
                     betas=[beta1, beta2], amsgrad=amsgrad)

    # unique identifier for newly saved objects
    now = datetime.datetime.now()
    time_str = f'{now.day:02d}_{now.month:02d}_{now.hour}H_{now.minute}M_{now.second}S'

    with SummaryWriter(f'runs/gridsearch/{trial.number}') as w:
        print('\n', lr, beta1, beta2, weight_decay,
              batch_size, image_size, manual_select, '\n')
        train_loss, val_loss = train(
            model, device, opt, 15, data_tr, data_val, time_str, hparam_dict, w)

    # IOU results
    # if not math.isnan(train_loss):
    #     results = eval_model(model, data_val, 'val')
    if math.isnan(train_loss):
        return np.nan

    # calculate average sensitivity between mask and bboxes
    # tp, fp = results[0][0][0], results[0][1][0]
    # sensitivity_bbox = tp / (fp + tp)
    # tp, fp = results[2][0][0], results[2][1][0]
    # sensitivity_mask = tp / (fp + tp)
    # avg_sensitivity = (sensitivity_bbox + sensitivity_mask) / 2

    # print(results)

    return val_loss


if __name__ == '__main__':
    tic = time()
    optuna.logging.set_verbosity(optuna.logging.DEBUG)
    utils.setcwd(__file__)

    # create study
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=300)

    # save study
    df = study.trials_dataframe()
    df.to_csv('train_study.csv')
    pickle.dump(study, open('train_study.pkl', 'wb'))

    # print elapsed time of script
    elapsed = utils.time_report(tic, time())
    print(f'Training grid search complete in {elapsed}.')
