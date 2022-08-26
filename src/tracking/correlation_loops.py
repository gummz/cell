from aicsimageio import AICSImage
import src.data.constants as c
from time import time
import os.path as osp
import os
import src.tracking.eval_track as evtr
import numpy as np
import pandas as pd


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


def matrix_to_terse(frames, loops):
    '''
    Condense loops into a terse representation.
    '''
    loops_terse = []
    loops_slice = []
    for frame in frames:
        loops_slice = []
        for j, z_slice in enumerate(loops[frame]):
            unique = np.unique(z_slice.compute())[1:]
            if any(unique):
                loop_idx = np.argwhere(
                    np.max(z_slice.compute(), axis=2) == unique[:, None, None])
                for val1, val2 in zip(np.unique(loop_idx[:, 0]), unique):
                    loop_idx[loop_idx[:, 0] == val1, 0] = val2

                loop_final = np.zeros((len(loop_idx), 5), dtype=np.float32)
                loop_final[:, 0] = frame
                loop_final[:, 1:4] = loop_idx
                loop_final[:, 4] = j

                loops_slice.append(loop_final)

        loops_terse.append(np.row_stack(loops_slice))

    return np.row_stack(loops_terse)


def get_loops(loop_path, frames, save=False, load=False):
    if load:
        return pd.read_csv(osp.join(loop_path, 'loops.csv'))
    else:
        loop_file = AICSImage(loop_path)
        # get raw loops
        loops = np.moveaxis(loop_file.get_image_dask_data('CZYXS'), 1, 0)
        columns = ['timepoint', 'loop_id', 'x', 'y', 'z']
        # condense loops into terse representation
        loops = pd.DataFrame(matrix_to_terse(frames, loops), columns=columns)

        if save:
            orig_name = osp.splitext(osp.basename(loop_path))[0]
            fr_min, fr_max = min(frames), max(frames)
            name = f'{orig_name}_t{fr_min}-{fr_max}.csv'
            loops.to_csv(osp.join(osp.dirname(loop_path), name))
    return loops


def loop_analysis(loops):
    pass


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

    results_tot = {}
    for pred_dir in pred_dirs:
        if osp.splitext(pred_dir)[0] == draft_file:
            pred_dir_path = osp.join(pred_path, pred_dir)
            pr_trajs = evtr.get_pr_trajs(pred_dir_path, groupby=None)
            frames = [item[0] for item in pr_trajs.groupby('frame')]
            n_timepoints = len(frames)
            half_tp = n_timepoints // 2
            time_range = range(half_tp, half_tp + 10 + 1)

            loop_path = loop_files[osp.splitext(pred_dir)[0]]
            loops = get_loops(loop_path, frames, save=True)

            results = loop_analysis(loops)
            results_tot[pred_dir] = results

    # I can include segmentations in the dataframe that trackpy
    # receives as input
