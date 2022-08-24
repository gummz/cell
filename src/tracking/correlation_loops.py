from aicsimageio import AICSImage
import src.data.constants as c
import src.data.utils.utils as utils
from time import time
import os.path as osp
import os.listdir

if __name__ == '__main__':
    mode = 'test'
    experiment_name = 'pred_1'
    tic = time()
    root_dir = c.DATA_DIR if osp.exists(
        c.DATA_DIR) else c.PROJECT_DATA_DIR_FULL
    loops_path = osp.join(root_dir, 'loops', 'test')
    loop_dirs = sorted(os.listdir(loops_path))
    pred_path = osp.join(root_dir, 'pred', experiment_name)
    pred_dirs = os.listdir(pred_path)
