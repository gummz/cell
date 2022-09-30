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
    if osp.exists(c.RAW_DATA_DIR):  # on DTU HPC cluster
        root_dir = osp.join(c.RAW_DATA_DIR, '..')
        pred_path = osp.join(c.DATA_DIR, 'pred', experiment_name)
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

        loops_comp = loops[frame, :, :, :, 0].compute()

        unique, counts = np.unique(loops_comp, return_counts=True)
        unique, counts = unique[1:], counts[1:]
        if any(unique):
            labels_out = cc3d.dust(
                cc3d.connected_components(loops_comp), threshold=30)

            nonzero = np.nonzero(labels_out)
            loop_id = labels_out[nonzero]
            len_ids = len(loop_id)
            len_un_ids = len(np.unique(loop_id))

            loops_frame = np.zeros((len_ids, 5), dtype=np.int16)
            loops_frame[:, 0] = frame

            add = 50 if len_un_ids <= 205 else 255 - len_un_ids
            loops_frame[:, 1] = loop_id + add
            # print('frame', frame,
            #       'len_ids', len_ids,
            #       'len_un_ids', len_un_ids,
            #       'add', add)
            assert 255 - len_un_ids > 0, \
                'Too many loops to assign ids from 0 to 255'

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
        loops = pd.read_csv(name)
        if frames:
            loops = loops[loops.timepoint.isin(frames)]
    else:
        loop_file = AICSImage(loop_path)
        # get raw loops
        loops = np.moveaxis(loop_file.get_image_dask_data('CZYXS'), 1, 0)
        columns = ['timepoint', 'loop_id', 'z', 'y', 'x']
        # condense loops into terse representation
        loops = pd.DataFrame(matrix_to_terse(frames, loops), columns=columns)

    return loops


def terse_to_disk(save_path, frames, loops):
    (loops.astype(int)
     .sort_values(['timepoint', 'loop_id'], ignore_index=True)
     .to_csv(save_path, index=False))


def process_loops(frames, pr_trajs, loops):
    in_loop = inside_loop(frames, pr_trajs, loops)
    dist_loop = dist_to_loop(frames, pr_trajs, loops)
    return in_loop, dist_loop


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

    for (tp, _), loop in tqdm(loops.groupby(['timepoint', 'loop_id']),
                              desc='Analysis per loop'):
        loop_coord = loop[coord_col].values

        # check if loop is 3-dimensional
        if not loop_3d_check(loop):
            # loop is not 3-dimensional since for at least one of
            # x, y, or z, there is only one unique value
            # so we cannot create a convex hull'

            # so we perform a 3D dilation
            loop_coord = dilate_3d(loop_coord)

        # loop is 3-dimensional
        # this can be further vectorized
        # by splitting the in_loop array correctly (row-wise)
        # and then dividing by the number of mask coordinates for each cell
        for idx, cell in tqdm(pr_trajs[pr_trajs.frame == tp].iterrows(),
                              desc='Analysis per cell'):
            hull = scipy.spatial.ConvexHull(loop_coord)
            overlap = overlap_hull(cell['mask'], hull)
            # in the unlikely event of a cell being inside two hulls,
            # we take the one with the largest overlap
            if overlap > in_loop.iloc[idx]:
                in_loop.iloc[idx] = round(overlap, 4)

    return in_loop


def dilate_3d(loop_coord):
    '''
    Dilate a 2D loop to a 3D loop.
    '''
    loop_img = np.zeros((1024, 1024, 1024), dtype=np.bool)
    loop_img[loop_coord[:, 0], loop_coord[:, 1], loop_coord[:, 2]] = 1
    dilated = ski.morphology.binary_dilation(
        loop_img, ski.morphology.ball(radius=30)
    )
    loop_coord = np.argwhere(dilated)
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
    coord_col = ['x', 'y', 'z']
    return all(len(loop[c].unique()) > 1 for c in coord_col)


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
    return np.sum(hull.equations[:, :-1] @ points.T + np.repeat(hull.equations[:, -1][None, :], len(points), axis=0).T <= tol, 0) / len(points)


def overlap_hull2(points, hull, tol=1e-12):
    '''
    Returns the ratio of overlap between a cell and 
    the convex hull of a loop boundary.

    Input
    ----------
    points : np.ndarray
        Cell mask.
    hull : scipy.spatial.ConvexHull
        Convex hull of loop boundary.
    tol : float
        Tolerance for numerical errors.

    Output
    ----------
    float
        Ratio of overlap between cell and convex hull.
    '''
    # check if cell is completely inside loop
    if hull.find_simplex(points, tol=tol) >= 0:
        return 1

    # check if cell is completely outside loop
    if hull.find_simplex(points[0], tol=tol) < 0:
        return 0

    # cell overlaps with loop
    # get the convex hull of the cell
    cell_hull = scipy.spatial.ConvexHull(points)

    # get the volume of the convex hull
    cell_vol = cell_hull.volume

    # get the volume of the intersection of the cell and loop
    # the intersection is the convex hull of the intersection of the points
    # of the cell and the points of the loop
    pts = np.concatenate([points, hull.points])
    inter_hull = scipy.spatial.ConvexHull(pts)
    inter_vol = inter_hull.volume

    return inter_vol / cell_vol


def dist_to_loop(frames, pr_trajs, loops):
    '''
    Calculates the distance matrix between cell centers and loop coordinates.
    Every loop coordinate is used, while only the center is used for a cell.
    Loop can be indexed to select a subset of
    the loop's coordinates.
    '''
    dist = scipy.spatial.distance_matrix(pr_trajs[['x', 'y', 'z']].values,
                                         loops[['x', 'y', 'z']].values)
    return dist.min(axis=1)


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


def prep_file(draft_file, save_loops, load_loops,
              pred_path, loop_files, pred_dir):
    if osp.splitext(pred_dir)[0] == draft_file:
        pred_dir_path = osp.join(pred_path, pred_dir)
        pr_trajs = evtr.get_pr_trajs(pred_dir_path, groupby=None)
        frames = [item[0] for item in pr_trajs.groupby('frame')]
        fr_min, fr_max = min(frames), max(frames)

        name_loops = f'loops_{pred_dir}_t{fr_min}-{fr_max}.csv'
        saved_path_loops = osp.join(pred_dir_path, name_loops)
        # check for existing loops and filled loops
        # disabled while debugging
        # if osp.exists(saved_path_loops):
        #     continue

        # get loop boundaries
        loop_path = loop_files[osp.splitext(pred_dir)[0]]
        loops = get_loops(loop_path, saved_path_loops, frames, load_loops)
        if save_loops:
            utils.make_dir(pred_dir_path)
            terse_to_disk(saved_path_loops, frames, loops)

        pr_trajs[['in_loop', 'dist_loop']] = process_loops(
            frames, pr_trajs, loops)

        return pr_trajs


def prep_cells_w_loops(experiment_name, draft_file, save_loops, load_loops):
    filter_prefix = 'cyctpy15'
    file_ext = 'tif'
    pred_path, loops_path = set_paths(experiment_name)

    # can choose only predicted dirs, or all files in /raw_data/
    select_dirs = sorted(os.listdir(pred_path))
    loop_files = get_loop_files(loops_path, select_dirs, draft_file,
                                filter_prefix, file_ext)

    for pred_dir in tqdm(select_dirs, desc='Prediction'):
        pr_trajs = prep_file(draft_file, save_loops, load_loops,
                             pred_path, loop_files, pred_dir)

        if pr_trajs:
            csv_path = osp.join(pred_path, pred_dir,
                                'tracked_centroids_loops.csv')
            pr_trajs.to_csv(csv_path, index=False)


if __name__ == '__main__':
    experiment_name = 'pred_2'
    draft_file = 'LI_2019-02-05_emb5_pos4'

    save_loops = False  # save condensed loops to disk
    load_loops = True  # load condensed loops from disk

    # redundant to save loaded loops to disk
    save_loops = False if load_loops else save_loops

    tic = time()
    np.random.seed(42)
    utils.set_cwd(__file__)
    pr_trajs = prep_cells_w_loops(
        experiment_name, draft_file, save_loops, load_loops)

    print(f'prep_loop_corr: Time elapsed: {utils.time_report(tic, time())}.')
