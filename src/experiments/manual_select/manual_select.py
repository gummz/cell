import datetime
from time import time
import pandas as pd
import numpy as np

from os.path import join
import src.data.constants as c
import src.data.utils.utils as utils
import pickle
import src.models.BetaCellDataset as bcd
from src.models.eval_model import eval_model
from src.models.train_model import train
from src.models.utils.model import get_instance_segmentation_model
from torch import optim
from torch.utils.tensorboard import SummaryWriter


def objective(n_img_select, manual_select, n_epochs, device):

    # hyperparameters
    # pretrained = trial.suggest_categorical('pretrained', [True, False])
    amsgrad = 0
    batch_size = 8
    beta1 = 0.244
    beta2 = 0.985
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
    lr = 3.419e-6
    # 1100  # = manual_select later
    # n_img_select = 1100  # manual_select if manual_select > 0 else 200
    weight_decay = 1.296e-8
    # end hyperparameters

    num_workers = 4
    root = join('..', c.DATA_DIR)
    data_tr, data_val = bcd.get_dataloaders(
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
        train_loss, val_loss, _ = train(
            model, device, opt, n_epochs, data_tr, data_val, time_str, hparam_dict, w, save=False, write=False)

    # load validation datasets
    data_val = bcd.get_dataset(
        root=root, mode='val', resize=image_size,
        n_img_select=1, manual_select=1)
    data_val_no_manual = bcd.get_dataset(
        root=root, mode='val', resize=image_size,
        n_img_select=1, manual_select=0)

    save = join('..', c.PROJECT_DATA_DIR, c.PRED_DIR,
                'eval', 'manual_select', 'val',
                f'man_{manual_select}_{time_str}')

    # need to eval model twice for every trial
    # once where validation set has no manual labels,
    # and once where validation set has all manual labels
    # (to get a better feeling for how well it does)
    results = eval_model(model, data_val, 'val', device, save)
    # model trained only with automatic labels
    #  needs different accept range
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
    print(f'Confusion matrix:\n{result[0]}\n')


def perform_study(device, img_mode, debug):

    n_labels = 1100
    n_epochs = 1 if debug else 30
    increment = 1100 if debug else 200
    n_trials = int(n_labels / increment) + 1

    columns = ['n_img', 'n_manual', 'cm_bbox', 'cm_mask', 'cm_sans_bbox',
               'cm_sans_mask', 'sens_bbox', 'bbox_score', 'sens_mask',
               'mask_score', 'sens_sans_bbox', 'sans_bbox_score',
               'sens_sans_mask', 'sans_mask_score',
               'mean_tr_loss', 'mean_val_loss']

    study = []

    iterable = range(0, n_labels, increment)
    if n_labels not in iterable:
        iterable = (*iterable, n_labels)
        n_trials += 1

    for i, manual_select in enumerate(iterable):
        print('=' * 60, '\n' + '=' * 60, '\n')
        print('Number of manually labeled images:', manual_select)

        if img_mode == 'auto':
            n_img_select = manual_select if manual_select > 0 else 200
        else:
            n_img_select = 1100

        print('Number of images in training set:', n_img_select)

        result = objective(n_img_select, manual_select, n_epochs, device)
        train_loss, val_loss = result[-2], result[-1]
        mean_tr = np.mean(train_loss)
        mean_val = np.mean(val_loss)

        # if early stopping activated, fill rest of epochs with zeros
        # if len(train_loss) < n_epochs:
        #     pad = (0, n_epochs - len(train_loss))
        #     train_loss = np.pad(train_loss, pad)
        #     val_loss = np.pad(val_loss, pad)
        study.append((n_img_select, manual_select, *result[:-2],
                     mean_tr, mean_val))

        # output results of current iteration
        score_report(result)

    return pd.DataFrame(study, columns=columns, dtype=object)


if __name__ == '__main__':
    debug = False

    tic = time()
    device = utils.set_device()
    utils.set_cwd(__file__)
    img_modes = ('auto', 'manual')
    for img_mode in img_modes:
        # perform study
        study = perform_study(device, img_mode, debug)

        # save study
        name = f'manual_select_study_{img_mode}{"_debug" if debug else ""}'
        study.to_csv(f'{name}.csv', sep=';')
        pickle.dump(study, open(f'{name}.pkl', 'wb'))

        # print elapsed time of script
        print('manual_select.py complete after',
              utils.time_report(tic, time()))
