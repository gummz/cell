from aicsimageio import AICSImage
import src.data.constants as c
from time import time
import os.path as osp
import os
import src.tracking.eval_track as evtr
import numpy as np
import pandas as pd
from tqdm import tqdm
import src.data.utils.utils as utils
from scipy.spatial import Delaunay

'''
This script:
1. Loads the loops from disk.
2. Computes a terse representation of the loops.
    - Saves to disk.
    - Option to save or load terse representation from disk.
3. Fills in loops so that a direct, pixel-by-pixel comparison
    between cells and loops can be made (i.e. cell == loop)
4. Adds a column to trajectory (pr_trajs) file that indicates
    whether a cell is in a loop or not.    
    - Saves modified version to disk.
5. Performs intensity analysis on cell trajectories and determines
    whether a cell has turned on at a given timepoint or not.
    - Adds a column to trajectory (pr_trajs) file that indicates
    whether a cell is on or off at a given timepoint.
'''


def set_paths(experiment_name):
    if osp.exists(c.RAW_DATA_DIR):  # on DTU HPC cluster
        root_dir = osp.join(c.RAW_DATA_DIR, '..')
        pred_path = osp.join(c.PROJECT_DATA_DIR, 'pred', experiment_name)
        loops_path = osp.join(root_dir, 'results')
    else:  # on local machine
        root_dir = c.PROJECT_DATA_DIR
        pred_path = osp.join(root_dir, 'pred', experiment_name)
        loops_path = osp.join(root_dir, 'loops')
    return pred_path, loops_path


def get_loop_files(loops_path, pred_dirs, draft_file,
                   filter_prefix, file_ext):
    pred_dirs_noext = [osp.splitext(file)[0]
                       for file in pred_dirs]

    loop_files = {}

    for folders, subfolders, files in os.walk(loops_path):
        # only fetch tif files for which there exist
        # corresponding predictions
        dir_folders = [folder for folder in subfolders
                       if 'LI' in folder]
        for folder in dir_folders:
            if (folder in pred_dirs_noext
                    and folder == draft_file):
                loop_file = [file
                             for file in os.listdir(osp.join(folders, folder))
                             if filter_prefix in file
                             and file.endswith(file_ext)][0]
                loop_files[folder] = osp.join(folders, folder, loop_file)
    return loop_files


def matrix_to_terse(frames, loops):
    '''
    Condense loops into a terse representation.
    '''

    loops_terse = []
    for frame in tqdm(frames, desc='Condensing loops: frame'):
        loops_slice = []
        for j, z_slice in tqdm(enumerate(loops[frame]), desc='Z-slice'):
            z_slice_comp = z_slice.compute()
            unique, counts = np.unique(z_slice_comp, return_counts=True)
            unique, counts = unique[1:], counts[1:]
            if any(unique):
                # loop_idx = np.argwhere(
                #     z_slice == unique[:, None, None, None])[:, :3].compute()

                nonzero = np.nonzero(z_slice_comp)
                loop_id = z_slice_comp[nonzero]
                # for val1, val2 in zip(np.unique(loop_idx[:, 0]), unique):
                #     loop_idx[loop_idx[:, 0] == val1, 0] = val2

                loop_final = np.zeros((len(loop_id), 5), dtype=np.float32)
                loop_final[:, 0] = frame
                loop_final[:, 1] = loop_id
                loop_final[:, 2:4] = np.row_stack(nonzero[:2]).T
                loop_final[:, 4] = j

                argwhere_raw = np.argwhere(z_slice_comp).shape[0]
                argwhere_final = loop_final.shape[0]
                assert argwhere_raw == argwhere_final, \
                    f'{argwhere_raw} != {argwhere_final}'

                unique_final, unique_counts_final = np.unique(loop_final[:, 1],
                                                              return_counts=True)
                assert np.all(unique == unique_final), \
                    f'{unique} != {unique_final}'
                assert np.all(counts == unique_counts_final), \
                    f'{counts} != {unique_counts_final}'

                loops_slice.append(loop_final)

        loops_terse.append(np.row_stack(loops_slice))

    return np.row_stack(loops_terse)


def get_loops(loop_path, pred_path, frames, load=True):
    if load:
        raw_file = osp.splitext(osp.basename(pred_path))[0]
        fr_min, fr_max = min(frames), max(frames)
        name = f'loops_{raw_file}_t{fr_min}-{fr_max}.csv'
        loops = pd.read_csv(osp.join(osp.dirname(pred_path), raw_file, name))
    else:
        loop_file = AICSImage(loop_path)
        # get raw loops
        loops = np.moveaxis(loop_file.get_image_dask_data('CZYXS'), 1, 0)
        columns = ['timepoint', 'loop_id', 'x', 'y', 'z']
        # condense loops into terse representation
        loops = pd.DataFrame(matrix_to_terse(frames, loops), columns=columns)

    return loops


def loops_to_disk(loop_path, frames, loops):
    orig_name = osp.splitext(osp.basename(loop_path))[0]
    fr_min, fr_max = min(frames), max(frames)
    name = f'loops_{orig_name}_t{fr_min}-{fr_max}.csv'
    loops.astype(int).to_csv(osp.join(osp.dirname(loop_path), name),
                             index=False)


def loop_analysis(frames, pr_trajs, loops):
    return inside_loop(frames, pr_trajs, loops)


def inside_loop(frames, pr_trajs, loops):
    '''
    Determine if a particle is inside a loop.

    Input
    ----------
    pr_trajs : pandas.DataFrame
        Particle trajectories.
    loops : pandas.DataFrame
        Loops.

    Output
    ----------
    np.ndarray
    '''
    dist_loop = np.zeros((len(pr_trajs), 2))
    excluded_p = []  # particles found inside a previous loop are excluded
    # iterate over all loops
    tp_tmp = np.inf  # keep track of when current timepoint changes
    for (tp, _), loop in loops.groupby(['timepoint', 'loop_id']):
        # only filter trajectories if timepoint changes
        if tp > tp_tmp or tp_tmp == np.inf:
            trajs_frame = pr_trajs[pr_trajs.frame == tp]
            tp_tmp = tp
        for _, row in trajs_frame.iterrows():
            particle = row[['x', 'y', 'z', 'particle']]

            # find particle's index in pr_trajs
            # (for accurate assignment to dist_loop)
            eq_tp = pr_trajs.frame == tp
            eq_p = pr_trajs.particle == particle.particle
            p_select = int(pr_trajs.index[eq_tp & eq_p].values)

            # find if particle is inside loop
            if (particle.particle not in excluded_p
                    and p_in_loop(particle, loop)):

                dist_loop[p_select] = True, dist_to_loop(particle, loop)
                excluded_p.append(particle.particle)
            else:
                dist_loop[p_select] = False, dist_to_loop(particle,
                                                          nearest_loop(particle, loops))
    return dist_loop


def fill_loops(loops):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html#scipy.interpolate.griddata
    # ^ right approach
    interp = []
    for (tp, _), loop in loops:
        interp_loop = interpolate_3d(loop)
        interp.append(interp_loop)
    return pd.DataFrame(np.row_stack(interp), columns=loops.columns)


def interpolate_3d(loop):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html#scipy.interpolate.griddata
    # ndimage.binary_fill_holes(loop[['x', 'y', 'z']])
    # .find_simplex(loop[['x', 'y', 'z']])
    coord_cols = ['x', 'y', 'z']
    exclude_coord = np.zeros((len(loop), 3))
    first_coord, sec_coord = get_first_coords(loop, coord_cols)
    exclude_coord[0] = first_coord
    exclude_coord[1] = sec_coord
    # loop[['x', 'y', 'z']] = Delaunay(loop[['x', 'y', 'z']])

    return loop


def get_first_coords(loop, coord_cols):
    coord_idx = np.random.choice(range(len(loop)))
    first_coord = loop.iloc[coord_idx][coord_cols]
    norms = np.argsort(np.linalg.norm(
        loop[coord_cols] - first_coord, axis=1))
    sec_coord = loop.iloc[norms[1]][coord_cols]
    return first_coord, sec_coord


def p_in_loop(row, loop):
    # sample, interpolate, and calculate intersection
    # of interpolated line segment with particle
    pass


def dist_to_loop(row, loop):
    '''
    Calculates the distance to a loop.
    Loop can be indexed to select a subset of
    the loop's coordinates.
    '''
    pass


def nearest_loop(row, loops):
    '''
    Find the nearest loop to a particle.

    Algorithm:
    1. Sample one coordinate from all loops.
    2. Keep the four loops with the smallest
        distance to the sample.
    3. Take many coordinate samples and return the loop
        with the smallest distance to the particle.
    '''
    pass


if __name__ == '__main__':
    mode = 'test'
    experiment_name = 'pred_1'
    draft_file = 'LI_2019-02-05_emb5_pos4'
    save_loops = False  # save condensed loops to disk
    load_loops = True  # load condensed loops from disk
    # redundant to save loaded loops to disk
    save_loops = False if load_loops else save_loops

    np.random.seed(42)
    utils.set_cwd(__file__)
    filter_prefix = 'cyctpy15'
    file_ext = 'tif'
    tic = time()
    pred_path, loops_path = set_paths(experiment_name)

    # can choose only predicted dirs, or all files in /raw_data/
    select_dirs = sorted(os.listdir(pred_path))
    loop_files = get_loop_files(loops_path, select_dirs, draft_file,
                                filter_prefix, file_ext)

    results_tot = {}
    for pred_dir in tqdm(select_dirs, desc='Prediction'):
        if osp.splitext(pred_dir)[0] == draft_file:
            pred_dir_path = osp.join(pred_path, pred_dir)
            pr_trajs = evtr.get_pr_trajs(pred_dir_path, groupby=None)
            frames = [item[0] for item in pr_trajs.groupby('frame')]
            n_timepoints = len(frames)
            half_tp = n_timepoints // 2
            time_range = range(half_tp, half_tp + 10 + 1)

            loop_path = loop_files[osp.splitext(pred_dir)[0]]
            loops = get_loops(loop_path, pred_dir_path, frames, load_loops)
            if save_loops:
                utils.make_dir(pred_dir_path)
                loops_to_disk(pred_dir_path, frames, loops)

            # enable comparison between cells and loop interiors
            # by filling in each loop;
            # so we can see if cell is inside loop with a simple 'cell == loop'
            loops = fill_loops(loops.groupby(['timepoint', 'loop_id']))

            # pr_trajs[['in_loop', 'dist_loop']] = loop_analysis(frames,
            #                                                    pr_trajs, loops)

            csv_path = osp.join(pred_dir_path,
                                'tracked_centroids_loops.csv')
            pr_trajs.to_csv(csv_path, index=False)

    # I can include segmentations in the dataframe that trackpy
    # receives as input
    # It can also be terse like loops.csv
