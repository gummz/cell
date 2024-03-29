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
import cc3d
import scipy
import skimage as ski


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
    '''
    Set paths to data.

    Parameters
    ----------
    experiment_name : str
        Name of experiment.

    Returns
    -------
    pred_path : str
        Path to directory containing prediction files.
    loops_path : str
        Path to directory containing loop files.
    '''
    if osp.exists(c.RAW_DATA_DIR):  # on DTU HPC cluster
        root_dir = osp.join(c.RAW_DATA_DIR, '..')
        pred_path = osp.join(c.DATA_DIR, 'pred', experiment_name)
        loops_path = osp.join(root_dir, 'results')
    else:  # on local machine
        root_dir = c.PROJECT_DATA_DIR
        pred_path = osp.join(root_dir, 'pred', experiment_name)
        loops_path = osp.join(root_dir, 'loops')
    return pred_path, loops_path


def get_loop_files(loops_path, pred_dirs,
                   filter_prefix, file_ext):
    '''
    Get loop files from directory.

    Parameters
    ----------
    loops_path : str
        Path to directory containing loop files.
    pred_dirs : list
        List of prediction directories.
    filter_prefix : str
        Prefix to filter loop files by.
    file_ext : str
        File extension to filter loop files by.

    Returns
    -------
    loop_files : list
        List of loop files.
    '''
    pred_dirs_noext = [osp.splitext(file)[0]
                       for file in pred_dirs]

    loop_files = {}

    for folders, subfolders, files in os.walk(loops_path):
        # only fetch tif files for which there exist
        # corresponding predictions
        dir_folders = [folder for folder in subfolders
                       if 'LI' in folder]
        for folder in dir_folders:
            if folder in pred_dirs_noext:
                loop_file = [file
                             for file in os.listdir(osp.join(folders, folder))
                             if filter_prefix in file
                             and file.endswith(file_ext)][0]
                loop_files[folder] = osp.join(folders, folder, loop_file)
    return loop_files


def matrix_to_terse(frames, loops):
    '''
    Condense loops into a terse representation.

    Parameters
    ----------
    frames : np.ndarray
        1D array of frame indices.
    loops : np.ndarray
        5D array of loop matrices.

    Returns
    -------
    terse_loops : np.ndarray
        2D array of loop information.
    '''
    loops_terse = []

    for frame in tqdm(frames, desc='Condensing loops: frame'):
        loops_comp = loops[frame, :, :, :, 0].compute()

        unique, counts = np.unique(loops_comp, return_counts=True)
        unique, counts = unique[1:], counts[1:]
        if any(unique):
            labels_out = cc3d.dust(cc3d.connected_components(loops_comp),
                                   threshold=5)

            nonzero = np.nonzero(labels_out)
            loop_id = labels_out[nonzero]
            len_ids = len(loop_id)
            len_un_ids = len(np.unique(loop_id))

            loops_frame = np.zeros((len_ids, 5), dtype=np.int16)
            loops_frame[:, 0] = frame

            add = 50 if len_un_ids <= 205 else 255 - len_un_ids
            loops_frame[:, 1] = loop_id + add

            # assert 255 - len_un_ids > 0, \
            #     'Too many loops to assign ids from 0 to 255'

            loops_frame[:, 2:] = np.column_stack(nonzero)

        loops_terse.append(loops_frame)

    return np.row_stack(loops_terse)


def shape_check(z_slice_comp, unique, counts, loop_final):
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


def get_loops(loop_path, name,
              frames=None, load=True):
    if load:
        # raw_file = osp.splitext(osp.basename(pred_path))[0]
        # fr_min, fr_max = min(frames), max(frames)
        # name = f'loops_{raw_file}_t{fr_min}-{fr_max}.csv'
        # print()
        if osp.exists(name):
            loops = pd.read_csv(name)
            print(f'\tLoaded loops file from\n\t\t{name}.')
        else:
            print(
                f'\tLoop file not found:\n\t\t{name}\n\t\t(check for typos).\n\tGenerating new loops file.')
            loops, frames = get_loops(loop_path, name,
                                      frames, load=False)
        if frames is not None:
            loops = loops[loops.timepoint.isin(frames)]
    else:
        loop_file = AICSImage(loop_path)
        # get raw loops
        loops = np.moveaxis(loop_file.get_image_dask_data('CZYXS'), 1, 0)
        columns = ['timepoint', 'loop_id', 'z', 'y', 'x']
        frames = np.arange(loops.shape[0], dtype=np.int16)
        # condense loops into terse representation
        loops = pd.DataFrame(matrix_to_terse(frames, loops), columns=columns)

    return loops, frames


def terse_to_disk(save_path, frames, loops):
    (loops.astype(int)
     .sort_values(['timepoint', 'loop_id'], ignore_index=True)
     .to_csv(save_path, index=False))


def process_loops(frames, pr_trajs, loops):
    in_loop = inside_loop(frames, pr_trajs, loops)
    dist_loop = dist_to_loop(frames, pr_trajs, loops)
    is_beta = is_beta_cell(frames, pr_trajs)
    return (np.round(in_loop, 4), np.round(dist_loop, 4),
            is_beta)


def inside_loop(frames, pr_trajs, loops):
    '''
    Determine if a list of particles is inside a loop.

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
    coord_col = ['x', 'y', 'z']
    in_loop = np.zeros(len(pr_trajs), dtype=np.float32)
    in_loop2 = np.zeros(len(pr_trajs), dtype=np.float32)

    for (tp, _), loop in tqdm(loops.groupby(['timepoint', 'loop_id']),
                              desc='Add in_loop column'):

        loop_coord = loop[coord_col].values

        # check if loop is 3-dimensional
        if not loop_3d_check(loop_coord):
            # loop is not 3-dimensional since for at least one of
            # x, y, or z, there is only one unique value
            # so we cannot create a convex hull

            # so we perform a 2D-to-3D stretch of the loop
            loop_coord = stretch_to_3d(loop_coord)


        # loop is 3-dimensional
        # this can be further vectorized
        # by splitting the in_loop array correctly (row-wise)
        # and then dividing by the number of mask coordinates for each cell
        # so only one call to overlap_hull would be needed,
        # for the entire pr_trajs array
        hull = scipy.spatial.ConvexHull(loop_coord, qhull_options='QJ')

        for idx, cell in pr_trajs[pr_trajs.frame == tp].iterrows():
            overlap = overlap_hull(cell['mask'], hull)
            # in the unlikely event of a cell being inside two hulls,
            # we take the one with the largest overlap
            if overlap > in_loop[idx]:
                in_loop[idx] = round(overlap, 4)

        # TODO: alternative, faster, 
        # completely vectorized version
        # (UNFINISHED)
        # cells_tp = pr_trajs[pr_trajs.frame == tp]
        # masks = cells_tp['mask']
        # overlap = overlap_hull3(np.row_stack(masks.values), hull)
        # mask_lens = masks.apply(len)
        # cumsum = np.cumsum(mask_lens)  # indices to separate `overlap` vector
        # vectors = np.split(overlap, cumsum[:-1])
        # overlaps = vectors / mask_lens
        # print('pr_trajs', pr_trajs.shape)
        # print('masks', masks.shape)
        # print('mask_lens', mask_lens.shape)
        # print('cumsum', cumsum.shape)
        # print('vectors', len(vectors))
        # print('overlaps', len(overlaps))
        # print('cells_tp.index', cells_tp.index.values)
        # print('in_loop[cells_tp.index]', in_loop[cells_tp.index.values].shape)

        # overlaps_gt = np.greater(overlaps, in_loop[cells_tp.index.values])
        # in_loop2[cells_tp.index[overlaps_gt].values] = overlaps
        # assert not np.sum(in_loop != in_loop2), 'in_loop != in_loop2'
        # print('assert passed')
        # print(overlaps.shape)
        # print(overlaps)
        # exit()

    return in_loop


def stretch_to_3d(loop_coord):
    '''
    Turn a 2D loop into a 3D loop by stretching the missing dimension.
    I.e. if the loop only has x and y dimension (z is constant),
    then we copy the shape for a few values of z.
    Circle becomes cylinder, square becomes cube, etc.

    Input
    ----------
    loop_coord : np.ndarray
        Loop coordinates.

    Output
    ----------
    np.ndarray
        Stretched loop coordinates.
    '''
    # number of copies to make of the loop
    # along the missing dimension
    # i.e.: if the loop is a 2D circle, then make
    #  n_stretch  number of copies of it to make a cylinder
    # which has depth of  n_stretch .
    # loop is stretched by n_stretch * 2
    # in all dimensions which are missing
    n_stretch = (100, 100, 5)
    dim_range = range(3)
    for dim in dim_range:
        n_coord = len(loop_coord)
        unique = np.unique(loop_coord[:, dim])
        if len(unique) == 1:
            select = [x for x in dim_range if x != dim]
            copies = np.repeat(
                loop_coord[:, select], n_stretch[dim] * 2, axis=0)
            low, high = unique + np.array([-1, 1]) * n_stretch[dim]
            if low < 0:
                high = -low + high
                low = 0

            stretch = np.repeat(np.arange(low, high), n_coord)
            loop_coord = np.column_stack((copies, stretch))

            if dim == 0:
                loop_coord = loop_coord[:, (2, 0, 1)]
            elif dim == 1:
                loop_coord = loop_coord[:, (0, 2, 1)]

    return loop_coord


def loop_3d_check(loop):
    '''
    Check if a loop is 3-dimensional.

    Input
    ----------
    loop : pandas.DataFrame
        A loop.

    Output
    ----------
    bool
        If true, loop is 3-dimensional.
        If false, loop is not 3-dimensional.
    '''
    return all(len(np.unique(loop[:, d])) > 1 for d in range(3))


def overlap_hull(points, hull, tol=1e-12):
    '''
    Returns the ratio of overlap between a cell and
    the convex hull of a loop boundary.

    Input
    ----------
    points : np.ndarray
        Cell mask.
    hull : scipy.spatial.ConvexHull
        Convex hull of loop boundary.

    Output
    ----------
    float
        Overlap between cell mask and loop's convex hull.
        Between 0 and 1 (inclusive).
    '''
    return np.sum(np.all(hull.equations[:, :-1] @ points.T + np.repeat(hull.equations[:, -1][None, :], len(points), axis=0).T <= tol, 0)) / len(points)


def dist_to_loop(frames, pr_trajs, loops):
    '''
    Calculates the distance matrix between cell centers and loop coordinates.
    Every loop coordinate is used, while only the center is used for a cell.
    Loop can be indexed to select a subset of
    the loop's coordinates.

    Inputs
    ----------
    frames : pandas.DataFrame
        Frame data.
    pr_trajs : pandas.DataFrame
        Trajectory data.
    loops : pandas.DataFrame
        Loop data.

    Outputs
    ----------
    loop_dist : np.ndarray
        Distance array between cell centers and nearest loop coordinate.
    '''
    coord_col = ['x', 'y', 'z']
    loop_dist = np.zeros(len(pr_trajs))
    for tp in tqdm(frames, desc='Add dist_loop column'):
        cells_tp = pr_trajs[pr_trajs.frame == tp]
        loops_tp = loops[loops.timepoint == tp]
        if cells_tp.empty or loops_tp.empty:  # no data for this timepoint
            continue
        dist = scipy.spatial.distance.cdist(
            cells_tp[coord_col].values, loops_tp[coord_col].values)
        loop_dist[cells_tp.index] = np.min(dist, axis=1)

    return loop_dist


def is_beta_cell(frames, pr_trajs):
    '''
    Returns whether a cell is a beta cell.

    Inputs
    ----------
    frames : pandas.DataFrame
        Frame data.
    pr_trajs : pandas.DataFrame
        Trajectory data.

    Outputs
    ----------
    is_beta : np.ndarray
        Whether a cell is a beta cell.
    '''
    threshold = 0.07
    grad_thresh = 0.55
    risen_thresh = 0.5
    ids = [item[1] for item in pr_trajs.groupby('id')]

    dim_to_bright = np.full(len(ids), False)
    for i, cell in tqdm(enumerate(ids), desc='Add "is beta cell?" column'):
        intensity = cell.mean_intensity.values

        # skip tracks that were present at the start
        if np.all(frames[:3] == range(3)) and np.all(intensity[:3] > threshold / 2):
            continue

        grad = np.gradient(intensity)
        ascending = np.sum(grad > 0) / len(grad) > grad_thresh
        risen = np.sum(intensity > threshold) / len(intensity) > risen_thresh
        reach = np.max(intensity) > threshold
        if any((ascending, risen and reach)):
            dim_to_bright[i] = True

    dim_to_bright_id = [np.unique(cell.id)[0]
                        for cell, idxbool in zip(ids, dim_to_bright)
                        if idxbool]

    return pr_trajs.id.isin(dim_to_bright_id)


def prep_file(save_loops, load_loops,
              pred_path, loop_files, pred_dir):
    '''
    Prepares the data for the analysis.

    Inputs
    ----------
    save_loops : bool
        If true, save the loops to a file.
    load_loops : bool
        If true, load the loops from a file.
    pred_path : str
        Path to the prediction file.
    loop_files : list of str
        List of paths to the loop files.
    pred_dir : str
        Path to the directory containing the predictions.

    Outputs
    ----------
    pr_trajs : pandas.DataFrame
        Predictions.
    loops : pandas.DataFrame
        Loops.
    '''
    # check if loop file is available in /dtu-compute/tubes/results
    # LI_2015-07-12_emb5_pos1 is at least not available
    if pred_dir not in loop_files:
        print(f'Loop file {pred_dir} not available in /dtu-compute/tubes/results.',
              'Skipping.\n')
        return

    # load predictions
    pred_dir_path = osp.join(pred_path, pred_dir)
    pr_trajs = evtr.get_pr_trajs(pred_dir_path, groupby=None, mask=True)
    frames = np.unique(pr_trajs.frame.values)
    fr_min, fr_max = min(frames), max(frames)

    name_loops = f'loops_{pred_dir}_t{fr_min}-{fr_max}.csv'
    saved_path_loops = osp.join(pred_dir_path, '../..', 'loops', name_loops)
    # check for existing loops and filled loops
    # disabled while debugging
    # if osp.exists(saved_path_loops):
    #     continue

    # get loop boundaries
    loop_path = loop_files[osp.splitext(pred_dir)[0]]
    loops, frames = get_loops(loop_path, saved_path_loops, frames, load_loops)
    if save_loops:
        terse_to_disk(saved_path_loops, frames, loops)

    pr_trajs = pr_trajs[pr_trajs.frame.isin(frames)]
    in_loop, dist_loop, beta = process_loops(
        frames, pr_trajs, loops)
    pr_trajs['in_loop'] = in_loop
    pr_trajs['dist_loop'] = dist_loop
    pr_trajs['beta'] = beta

    return pr_trajs


def prep_cells_w_loops(experiment_name, save_loops, load_loops):
    filter_prefix = 'cyctpy15'
    file_ext = 'tif'
    pred_path, loops_path = set_paths(experiment_name)
    utils.make_dir(osp.join(pred_path, '..', '..', 'loops'))

    # can choose only predicted dirs, or all files in /raw_data/
    select_dirs = sorted(os.listdir(pred_path))
    # skip incomplete or missing predictions
    select_dirs = [folder for folder in select_dirs
                   if utils.get_dir_size(osp.join(pred_path,
                                                  osp.splitext(folder)[0])) > 50000]

    loop_files = get_loop_files(loops_path, select_dirs,
                                filter_prefix, file_ext)

    for pred_dir in tqdm(select_dirs, desc='Prediction'):
        print(f'\n{pred_dir} is being processed.')
        csv_path = osp.join(pred_path, pred_dir,
                            'tracked_centroids_loops.csv')
        if osp.exists(csv_path):
            print(
                f'File {csv_path} already exists.\nSkipping loop processing.\n')
            continue

        pr_trajs = prep_file(save_loops, load_loops,
                             pred_path, loop_files, pred_dir)

        # re-save the file if either in_loop or dist_loop
        # has been populated with values
        if (pr_trajs is not None and
                (np.sum(pr_trajs.in_loop) or np.sum(pr_trajs.dist_loop))):
            (pr_trajs
             .drop(columns=['mask'], errors='ignore')
             .to_csv(csv_path, index=False, sep=';'))


if __name__ == '__main__':
    experiment_name = 'nouse'

    save_loops = True  # save condensed loops to disk
    load_loops = True  # load condensed loops from disk

    # redundant to save loaded loops to disk
    # save_loops = False if load_loops else save_loops

    tic = time()
    np.random.seed(42)
    utils.set_cwd(__file__)
    pr_trajs = prep_cells_w_loops(
        experiment_name, save_loops, load_loops)

    print(f'prep_loop_corr: Time elapsed: {utils.time_report(tic, time())}.')
