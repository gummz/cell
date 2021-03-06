import datetime
import math
from time import time
import cv2

import numpy as np
import torch
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
    amsgrad = True  # trial.suggest_categorical('amsgrad', [True, False])
    batch_sizes = (4, 8, 32)
    batch_size = 8  # trial.suggest_categorical('batch_size', batch_sizes)
    beta1 = trial.suggest_float('beta1', 0.05, 0.25)
    beta2 = trial.suggest_float('beta2', 0.95, 0.99)
    # the parameters of these filters have been optimized
    # with experiments
    filters = list(c.FILTERS.keys())
    filter_optim = trial.suggest_categorical('filters', ['bilateral', 'mean'])
    img_filter = filter_optim  # filters[filter_optim]

    image_size = 1024  # trial.suggest_int('image_size', 28, 512)
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
    lr = trial.suggest_float('learning_rate', 1e-6,
                             9e-5, log=True)  # 1e-10, 1e-4
    manual_select = 1  # trial.suggest_int('manual_ratio', 2, 27)
    weight_decay = trial.suggest_float('weight_decay', 1e-8, 1e-2, log=True)
    # end hyperparameters
    n_img_select = 1100
    data_tr, data_val = get_dataloaders(
        root=join('..', c.DATA_DIR),
        batch_size=batch_size, num_workers=4, resize=image_size, n_img_select=(n_img_select, 1), manual_select=(manual_select, 1), img_filter=img_filter)

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
        train_losses, val_losses = train(
            model, device, opt, 30, data_tr, data_val, time_str, hparam_dict, w)

    # IOU results
    # if not math.isnan(train_loss):
    #     results = eval_model(model, data_val, 'val')
    if math.isnan(np.mean(train_losses)):
        return np.nan

    # calculate average sensitivity between mask and bboxes
    # tp, fp = results[0][0][0], results[0][1][0]
    # sensitivity_bbox = tp / (fp + tp)
    # tp, fp = results[2][0][0], results[2][1][0]
    # sensitivity_mask = tp / (fp + tp)
    # avg_sensitivity = (sensitivity_bbox + sensitivity_mask) / 2

    # print(results)

    return val_losses[-1]


if __name__ == '__main__':
    tic = time()
    optuna.logging.set_verbosity(optuna.logging.DEBUG)
    utils.set_cwd(__file__)

    # create study
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)

    # save study
    df = study.trials_dataframe()
    df.to_csv('train_study.csv', sep=';')
    pickle.dump(study, open('train_study.pkl', 'wb'))

    # print elapsed time of script
    elapsed = utils.time_report(tic, time())
    print(f'Training grid search complete in {elapsed}.')
