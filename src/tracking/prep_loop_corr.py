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
import matplotlib.pyplot as plt
import scipy

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
        loops_slice = []
        for j, z_slice in tqdm(enumerate(loops[frame, :, :, :, 0]), desc='Z-slice'):
            z_slice_comp = z_slice.compute()
            unique, counts = np.unique(z_slice_comp, return_counts=True)
            unique, counts = unique[1:], counts[1:]
            if any(unique):
                nonzero = np.nonzero(z_slice_comp)
                loop_id = z_slice_comp[nonzero]

                loop_final = np.zeros((len(loop_id), 5), dtype=np.int16)
                loop_final[:, 0] = frame
                loop_final[:, 1] = loop_id
                loop_final[:, 2:4] = np.row_stack(nonzero[:2]).T
                loop_final[:, 4] = j

                # shape_check(z_slice_comp, unique, counts, loop_final)

                loops_slice.append(loop_final)

        loops_terse.append(np.row_stack(loops_slice))

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


def get_loops(loop_path, pred_path, frames, load=True):
    if load:
        raw_file = osp.splitext(osp.basename(pred_path))[0]
        fr_min, fr_max = min(frames), max(frames)
        name = f'loops_{raw_file}_t{fr_min}-{fr_max}.csv'
        loops = pd.read_csv(osp.join(osp.dirname(pred_path), raw_file, name))
        loops = loops[loops.timepoint.isin(frames)]
    else:
        loop_file = AICSImage(loop_path)
        # get raw loops
        loops = np.moveaxis(loop_file.get_image_dask_data('CZYXS'), 1, 0)
        columns = ['timepoint', 'loop_id', 'x', 'y', 'z']
        # condense loops into terse representation
        loops = pd.DataFrame(matrix_to_terse(frames, loops), columns=columns)

    return loops


def get_filled(pred_path, frames, loops=None, load=True):
    if load:
        raw_file = osp.splitext(osp.basename(pred_path))[0]
        fr_min, fr_max = min(frames), max(frames)
        name = f'loops_filled_{raw_file}_t{fr_min}-{fr_max}.csv'
        filled = pd.read_csv(osp.join(osp.dirname(pred_path), raw_file, name))
        filled = filled[filled.timepoint.isin(frames)]
    else:
        # enable comparison between cells and loop interiors
        # by filling in each loop;
        # so we can see if cell is inside loop with a simple 'cell == loop'
        filled = fill_loops(loops.groupby(['timepoint', 'loop_id']))

    return filled


def terse_to_disk(save_path, frames, loops):
    # orig_name = osp.splitext(osp.basename(loop_path))[0]
    # fr_min, fr_max = min(frames), max(frames)
    # name = f'loops_{orig_name}_t{fr_min}-{fr_max}.csv'
    loops.astype(int).to_csv(save_path, index=False)


def filled_to_disk(save_path, frames, filled):
    # orig_name = osp.splitext(osp.basename(loop_path))[0]
    # fr_min, fr_max = min(frames), max(frames)
    # name = f'loops_filled_{orig_name}_t{fr_min}-{fr_max}.csv'
    filled.astype(int).to_csv(save_path, index=False)


def loop_analysis(frames, pr_trajs, loops):
    return inside_loop(frames, pr_trajs, loops)


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
    filled = []
    for (tp, _), loop in tqdm(loops, desc='Filling loops'):
        # if tp != 106 or np.all(loop.loop_id != 94):
        #     continue
        filled_loop = fill_loop(loop)
        len_loop = len(filled_loop)
        filled.append(np.column_stack(((tp,)*len_loop,
                                      (loop.loop_id.iloc[0],)*len_loop,
                                      filled_loop)))

    return pd.DataFrame(np.row_stack(filled), columns=loops.obj.columns)


def fill_loop(loop):
    # coord_cols = ['x', 'y', 'z']

    # coord_idx = np.random.choice(range(len(loop)))
    # first_coord = loop.iloc[coord_idx][coord_cols]
    # norms = np.linalg.norm(
    #     loop[coord_cols] - first_coord, axis=1)
    # if len(norms) < 3:  # check for degenerate loop
    #     return loop[coord_cols]
    # argnorms = np.argsort(norms)
    # sec_coord = loop.iloc[argnorms[1]][coord_cols]
    # first_set = loop.iloc[argnorms[2::2]]
    # sec_set = loop.iloc[argnorms[3::2]]
    # line_segments = get_line_segments(coord_cols, first_set, sec_set)
    # filled_loop = floodfill(loop, line_segments)
    image = np.zeros((1024, 1024, 1024), dtype=bool)
    image[loop.x, loop.y, loop.z] = 1
    img_filled = fill_hull(image)
    filled_loop = np.argwhere(img_filled)
    # if len(filled_loop) > 1000:
    #     print('filled_loop > 1000')
    #     print(filled_loop.shape)
    #     plt.imshow(np.max(img_filled, axis=0))
    #     plt.savefig('convex_hull0.png')
    #     plt.close()
    #     plt.imshow(np.max(img_filled, axis=1))
    #     plt.savefig('convex_hull1.png')
    #     plt.close()
    #     plt.imshow(np.max(img_filled, axis=2))
    #     plt.savefig('convex_hull2.png')
    #     plt.close()
    #     exit()

    return filled_loop


def fill_hull(image):
    """
    Compute the convex hull of the given binary image and
    return a mask of the filled hull.
    
    Adapted from:
    https://stackoverflow.com/a/46314485/162094
    This version is slightly (~40%) faster for 3D volumes,
    by being a little more stingy with RAM.

    Taken from:
    https://gist.github.com/stuarteberg/8982d8e0419bea308326933860ecce30
    """
    # (The variable names below assume 3D input,
    # but this would still work in 4D, etc.)

    if (np.array(image.shape) > np.iinfo(np.int16).max).all():
        raise ValueError(f"This function assumes your image is smaller than {2**15} in each dimension")

    points = np.argwhere(image).astype(np.int16)
    hull = scipy.spatial.ConvexHull(points)
    deln = scipy.spatial.Delaunay(points[hull.vertices])

    # Instead of allocating a giant array for all indices in the volume,
    # just iterate over the slices one at a time.
    idx_2d = np.indices(image.shape[1:], np.int16)
    idx_2d = np.moveaxis(idx_2d, 0, -1)

    idx_3d = np.zeros((*image.shape[1:], image.ndim), np.int16)
    idx_3d[:, :, 1:] = idx_2d

    mask = np.zeros_like(image, dtype=bool)
    for z in range(len(image)):
        idx_3d[:, :, 0] = z
        s = deln.find_simplex(idx_3d)
        mask[z, (s != -1)] = 1

    return mask


def get_line_segments(coord_cols, first_set, sec_set):
    len_first, len_sec = len(first_set), len(sec_set)
    line_segments = np.zeros(
        (max(len_first, len_sec), 10), dtype=np.int16)
    line_segments[:len_first, :3] = first_set[coord_cols]
    line_segments[:len_sec, 3:6] = sec_set[coord_cols]
    line_segments[:, 6:9] = line_segments[:, 3:6] - line_segments[:, :3]
    seg_norms = np.linalg.norm(line_segments[:, 6:9], axis=1)
    line_segments[:, 9] = np.ceil(seg_norms)
    return line_segments


def floodfill(line_segments):
    '''Performs the floodfill algorithm for a 3D system.'''
    filled_loop = []

    for coord in line_segments:
        coord_zero = np.all((coord[:3] == 0) | (coord[3:6] == 0))
        if coord[9] == 1 or coord_zero:
            continue
        # size of increment needs to be such that we will fill
        # all boxes along the way
        increment = round(1 / coord[9], 4)
        filled_seg = []
        # fill in the first coordinate along the segment
        fill = np.round(coord[:3] + increment * coord[6:9])

        mult = 1
        # as long as we're not at the end of the segment
        while np.any(fill != coord[3:6]):
            filled_seg.append(fill)
            t = round(increment * mult, 4)
            # avoid situation in which the line segment
            # narrowly misses the mark (i.e. misses coord[3:6])
            # and continues to infinity
            if t > 1:
                t = 1
                fill = np.round(coord[:3] + t * coord[6:9])
                filled_seg.append(fill)
                break

            fill = np.round(coord[:3] + t * coord[6:9])
            mult += 1

        if filled_seg:
            filled_loop.append(filled_seg)

    if not filled_loop:  # no filling was performed
        return_loop = np.row_stack(np.split(line_segments, (3, 6), axis=1)[:2])
        rows_without_zeros = return_loop[~np.all(return_loop == 0, axis=1)]
        return np.int16(rows_without_zeros)

    filled_loop = np.row_stack(filled_loop)
    filled_loop = np.int16(np.row_stack((line_segments[:, :3],
                                         line_segments[:, 3:6],
                                         filled_loop)))
    # img = np.zeros((1024, 1024))
    # img[filled_loop[:, 0], filled_loop[:, 1]] = 1
    # plt.imshow(img)
    # plt.title(f't{loop.timepoint.iloc[0]}, id{loop.loop_id.iloc[0]}\n z: {np.unique(loop.z)}')
    # plt.savefig(f'debug_{loop.timepoint.iloc[0]}_{loop.loop_id.iloc[0]}.png')
    # plt.close()
    return np.unique(filled_loop, axis=0)


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
    # mode = 'test'
    experiment_name = 'pred_2'
    draft_file = 'LI_2019-02-05_emb5_pos4'

    save_loops = False  # save condensed loops to disk
    load_loops = True  # load condensed loops from disk
    save_filled = True  # save filled loops to disk
    load_filled = False  # load filled loops from disk

    # redundant to save loaded loops to disk
    save_loops = False if load_loops else save_loops
    save_filled = False if load_filled else save_filled

    tic = time()
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
            fr_min, fr_max = min(frames), max(frames)

            name_loops = f'loops_{pred_dir}_t{fr_min}-{fr_max}.csv'
            name_filled = f'loops_filled_{pred_dir}_t{fr_min}-{fr_max}.csv'
            saved_path_loops = osp.join(pred_dir_path, name_loops)
            saved_path_fill = osp.join(pred_dir_path, name_filled)
            # check for existing loops and filled loops
            # disabled while debugging
            # if osp.exists(saved_path_loops) and osp.exists(saved_path_fill):
            #     continue

            if not load_filled:
                # get loop boundaries
                loop_path = loop_files[osp.splitext(pred_dir)[0]]
                loops = get_loops(loop_path, pred_dir_path, frames, load_loops)
                if save_loops:
                    utils.make_dir(pred_dir_path)
                    terse_to_disk(saved_path_loops, frames, loops)

                # get filled loops
                filled = get_filled(pred_dir_path, frames, loops, load_filled)
            else:
                filled = get_filled(pred_dir_path, frames, None, load_filled)

            if save_filled:
                filled_to_disk(saved_path_fill, frames, filled)

            pr_trajs[['in_loop', 'dist_loop']] = loop_analysis(frames,
                                                               pr_trajs, filled)

            # csv_path = osp.join(pred_dir_path,
            #                     'tracked_centroids_loops.csv')
            # pr_trajs.to_csv(csv_path, index=False)

    # I can include segmentations in the dataframe that trackpy
    # receives as input
    # It can also be terse like loops.csv

    print(f'prep_loop_corr: Time elapsed: {utils.time_report(tic, time())}.')
