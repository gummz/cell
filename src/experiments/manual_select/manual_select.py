import datetime
from os import listdir
from time import time

import optuna
from os.path import join
import src.data.constants as c
import src.data.utils.utils as utils
import pickle
from src.models.BetaCellDataset import get_dataloaders
from src.models.eval_model import eval_model
from src.models.train_model import train
from src.models.utils.model import get_instance_segmentation_model
from torch import optim
from torch.utils.tensorboard import SummaryWriter


def objective(trial):

    device = utils.set_device()

    # hyperparameters
    # pretrained = trial.suggest_categorical('pretrained', [True, False])
    amsgrad = 0
    batch_size = 2
    beta1 = 0.51929
    beta2 = 0.61231
    image_size = 1024
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
    n_masks_full = len(
        sorted(listdir(join('..', c.DATA_DIR, 'train', c.MASK_DIR_FULL))))
    manual_select = trial.suggest_int('manual_ratio', 2, n_masks_full)
    weight_decay = 0.0007618
    # end hyperparameters

    data_tr, data_val = get_dataloaders(
        root=join('..', c.DATA_DIR),
        batch_size=batch_size, num_workers=1, resize=image_size, n_img_ratio=0.1, manual_select=manual_select)

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

    with SummaryWriter(f'runs/manual_200/{time_str}') as w:
        print(batch_size, image_size,
              manual_select, '\n')
        train_loss, val_loss = train(
            model, device, opt, 50, data_tr, data_val, time_str, hparam_dict, w)

    results = eval_model(model, data_val, 'val')
    tp, fp = results[0][0][0], results[0][1][0]
    sensitivity = tp / (fp + tp)
    # return val_loss

    # TODO: separate into manual and automatic annotations
    # in this function

    return sensitivity


if __name__ == '__main__':
    tic = time()
    optuna.logging.set_verbosity(optuna.logging.DEBUG)

    utils.setcwd(__file__)
    # unique identifier for newly saved objects
    now = datetime.datetime.now()
    time_str = f'{now.day:02d}_{now.month:02d}_{now.hour}H_{now.minute}M_{now.second}S'

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)
    trial = study.best_trial
    print(trial)

    # save study
    df = study.trials_dataframe()
    df.to_csv('manual_select_study.csv')
    pickle.dump(study, open('manual_select_study.pkl', 'wb'))
    # print elapsed time of script
    utils.time_report(__file__, tic, time())
