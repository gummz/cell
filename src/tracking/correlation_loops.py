from aicsimageio import AICSImage
import src.data.constants as c
import src.data.utils.utils as utils
from time import time
import os.path as osp
import os
import src.tracking.eval_track as evtr


def get_loop_files(loops_path, pred_dirs, draft_file,
                   filter_prefix, file_ext):
    pred_dirs_noext = [osp.splitext(file)[0]
                       for file in pred_dirs]
    loop_dirs = sorted(osp.join(loops_path, file)
                       for file in os.listdir(loops_path))
    loop_files = {}
    for loop_dir in loop_dirs:
        # only fetch tif files for which there exist
        # corresponding predictions
        loop_dir_base = osp.basename(loop_dir)
        if loop_dir_base in pred_dirs_noext and loop_dir_base == draft_file:
            loop_file = [file for file in os.listdir(loop_dir)
                         if filter_prefix in file
                         and file.endswith(file_ext)][0]
            loop_files[loop_dir_base] = osp.join(loop_dir, loop_file)

    return loop_files


if __name__ == '__main__':
    mode = 'test'
    experiment_name = 'pred_1'
    draft_file = 'LI_2019-02-05_emb5_pos4'

    filter_prefix = 'cyctpy15'
    file_ext = 'tif'
    tic = time()
    root_dir = c.DATA_DIR if osp.exists(
        c.DATA_DIR) else c.PROJECT_DATA_DIR_FULL

    pred_path = osp.join(root_dir, 'pred', experiment_name)
    pred_dirs = sorted(os.listdir(pred_path))

    loops_path = osp.join(root_dir, 'loops', 'test')
    loop_files = get_loop_files(loops_path, pred_dirs, draft_file,
                                filter_prefix, file_ext)

    for pred_dir in pred_dirs:
        if osp.splitext(pred_dir)[0] == draft_file:
            pred_dir_path = osp.join(pred_path, pred_dir)
            n_timepoints = len(os.listdir(pred_dir_path))
            half_tp = n_timepoints // 2
            time_range = range(half_tp, half_tp + 10 + 1)
            
            pr_trajs = evtr.get_pr_trajs(pred_dir_path, time_range)
            
            loop_file = loop_files[osp.splitext(pred_dir)[0]]
            loops = utils.get_raw_array(loop_file)
            pass

    # I can include segmentations in the dataframe that trackpy
    # receives as input
    pass
