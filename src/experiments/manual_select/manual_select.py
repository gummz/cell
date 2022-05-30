import datetime
from os import listdir
from time import time
import pandas as pd
import numpy as np

from os.path import join
import src.data.constants as c
import src.data.utils.utils as utils
from pprint import pprint
from torch.utils.data import DataLoader
import pickle
from src.models.BetaCellDataset import BetaCellDataset, get_dataloaders, get_transform
from src.models.eval_model import eval_model
from src.models.train_model import train
from src.models.utils.model import get_instance_segmentation_model
from torch import optim
from torch.utils.tensorboard import SummaryWriter


def objective(n_img_select, manual_select, n_epochs, device):

    # hyperparameters
    # pretrained = trial.suggest_categorical('pretrained', [True, False])
    amsgrad = 0
    batch_size = 4
    beta1 = 0.51929
    beta2 = 0.61231
    image_size = 512
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
    lr = 6.02006e-5
    # 1100  # = manual_select later
    # n_img_select = 1100  # manual_select if manual_select > 0 else 200
    weight_decay = 0.0007618
    # end hyperparameters

    num_workers = 4
    root = join('..', c.DATA_DIR)
    data_tr, data_val = get_dataloaders(
        root=root, batch_size=batch_size, num_workers=num_workers,
        resize=image_size, n_img_select=(n_img_select, 1),
        manual_select=(manual_select, 1))

    model = get_instance_segmentation_model(pretrained=True)
    model.to(device)
    # dict so train_model can access hyperparameters
    hparam_dict = {
        'batch_size': batch_size,
        'image_size': image_size,
        'losses': ';'.join(loss_list),
        'size': image_size

    }
    params = [p for p in model.parameters() if p.requires_grad]
    opt = optim.Adam(params, lr=lr,
                     weight_decay=weight_decay,
                     betas=[beta1, beta2], amsgrad=amsgrad)

    # unique identifier for newly saved objects
    now = datetime.datetime.now()
    time_str = f'{now.day:02d}_{now.month:02d}_{now.hour}H_{now.minute}M_{now.second}S'

    with SummaryWriter(f'runs/manual_{manual_select}') as w:
        print(batch_size, image_size,
              manual_select, '\n')
        train_loss, val_loss = train(
            model, device, opt, n_epochs, data_tr, data_val, time_str, hparam_dict, w, save=False, write=True)

    # reload validation dataloaders with batch size of 1
    _, data_val = get_dataloaders(
        root=root, batch_size=1, num_workers=num_workers,
        resize=image_size, n_img_select=(1, 1),
        manual_select=(1, 1), shuffle=False)
    _, data_val_no_manual = get_dataloaders(
        root=root, batch_size=1, num_workers=num_workers,
        resize=image_size, n_img_select=(1, 1), manual_select=(1, 0),
        shuffle=False)

    # need to eval model twice for every trial
    # once where validation set has no manual labels,
    # and once where validation set has all manual labels
    # (to get a better feeling for how well it does)
    save = join('..', c.PROJECT_DATA_DIR, c.PRED_DIR,
                'eval', 'manual_select', 'val',
                f'man_{manual_select}_{time_str}')
    results = eval_model(model, data_val, 'val', device, save)
    results_no_manual = eval_model(
        model, data_val_no_manual, 'val', device, join(save, 'sans'))

    cm_bbox = results[0]
    cm_mask = results[2]
    cm_sans_bbox = results_no_manual[0]
    cm_sans_mask = results_no_manual[2]
    sens_bbox = utils.calc_sensitivity(results[0])
    sens_mask = utils.calc_sensitivity(results[2])
    sens_sans_bbox = utils.calc_sensitivity(results_no_manual[0])
    sens_sans_mask = utils.calc_sensitivity(results_no_manual[2])
    bbox_score, mask_score = results[1], results[3]
    sans_bbox_score, sans_mask_score = results_no_manual[1], results_no_manual[3]

    # TODO: separate into manual and automatic annotations
    # in this function

    return (cm_bbox, cm_mask, cm_sans_bbox, cm_sans_mask,
            sens_bbox, bbox_score, sens_mask, mask_score,
            sens_sans_bbox, sans_bbox_score, sens_sans_mask,
            sans_mask_score, train_loss, val_loss)


def score_report(result: tuple):
    sens = f'bbox: {result[4]:.2f}, mask: {result[6]:.2f}'
    sens_sans = f'bbox: {result[8]:.2f}, mask: {result[10]:.2f}'
    print(f'\nSensitivity with manuals: {sens}')
    print(f'Sensitivity without manuals: {sens_sans}')
    score = f'bbox: { result[5]:.2f}, mask: {result[7]:.2f}'
    score_sans = f'bbox: {result[9]:.2f}, mask: {result[11]:.2f}'
    print(f'Average score: {score}')
    print(f'Average sans score: {score_sans}')
    print()


if __name__ == '__main__':
    tic = time()
    device = utils.set_device()
    utils.setcwd(__file__)
    debug = False

    n_labels = 1100
    n_epochs = 1 if debug else 30
    increment = 1100 if debug else 200
    n_trials = int(n_labels / increment) + 1

    iterable = range(0, n_labels, increment)
    if n_labels not in iterable:
        iterable = (*iterable, n_labels)
        n_trials += 1

    # shape of study array:
    # 9: results from objective function
    # n_epochs * 2: train and validation loss for all epochs
    # 2: average train and val loss over epochs
    study = np.empty((n_trials, 16))
    tr_col = (f'tr_loss_{i}' for i in range(n_epochs))
    val_col = (f'val_loss_{i}' for i in range(n_epochs))
    # print(len(tr_col), len(val_col))
    columns = ['n_img', 'n_manual', 'cm_bbox', 'cm_mask', 'cm_sans_bbox',
               'cm_sans_mask', 'sens_bbox', 'bbox_score', 'sens_mask',
               'mask_score', 'sens_sans_bbox', 'sans_bbox_score',
               'sens_sans_mask', 'sans_mask_score',
               'mean_tr_loss', 'mean_val_loss']
    # print(study.shape, len(columns))
    # for i, column in enumerate(columns):
    #     print(i, column)
    study = pd.DataFrame(study, columns=columns, dtype=object)

    # perform study
    for i, manual_select in enumerate(iterable):
        print('=' * 60, '\n' + '=' * 60)
        print('Number of manually labeled images:', manual_select)
        n_img_select = 1100  # manual_select if manual_select > 0 else 200
        result = objective(n_img_select, manual_select, n_epochs, device)
        train_loss, val_loss = result[-2], result[-1]
        mean_tr = np.mean(train_loss)
        mean_val = np.mean(val_loss)

        # if early stopping activated, fill rest of epochs with zeros
        if len(train_loss) < n_epochs:
            pad = (0, n_epochs - len(train_loss))
            train_loss = np.pad(train_loss, pad)
            val_loss = np.pad(val_loss, pad)

        study.iloc[i] = (n_img_select, manual_select, *result[:-2],
                         mean_tr, mean_val)

        # output results of current iteration
        score_report(result)

    # save study
    study.to_csv('manual_select_study.csv')
    pickle.dump(study, open('manual_select_study.pkl', 'wb'))

    # print elapsed time of script
    print('manual_select.py complete after', utils.time_report(tic, time()))
