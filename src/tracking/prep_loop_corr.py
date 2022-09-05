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
import cc3d
import scipy
import random

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

        # print(loops_comp[33, 656, 446])
        # print(labels_out[33, 656, 446])
        # maybe different ids is good enough
        # TODO: check if algorithm works correctly
        unique, counts = np.unique(loops_comp, return_counts=True)
        unique, counts = unique[1:], counts[1:]
        if any(unique):
            labels_out = cc3d.dust(
                cc3d.connected_components(loops_comp), threshold=5)
            # loops_argw = np.argwhere(loops)
            # labels_argw = np.argwhere(labels_out)
            # set_diff = np.setdiff1d(loops_argw, labels_argw)
            # print('loops_argw before', loops_argw.shape)
            # loops[set_diff] = 0
            # print('loops_argw after', np.argwhere(loops).shape)
            # loops_argw = labels_argw = set_diff = None

            # u_loops, c_loops = np.unique(loops_comp, return_counts=True)
            # u_loops, c_loops = u_loops[1:], c_loops[1:]
            # u_lab, c_lab = np.unique(labels_out, return_counts=True)
            # u_lab, c_lab = u_lab[1:], c_lab[1:]

            # print('unique loops', u_loops, len(u_loops))
            # print('loops count', c_loops)
            # print('unique labels', u_lab, len(u_lab))
            # print('labels count', c_lab)
            # print('loops in labels', np.isin(c_loops, c_lab))
            # print('labels in loops', np.isin(c_lab, c_loops))

            nonzero = np.nonzero(labels_out)
            loop_id = labels_out[nonzero]
            len_ids = len(loop_id)
            len_un_ids = len(np.unique(loop_id))
            assert 255 - len_un_ids > 0, \
                'Too many loops to assign ids from 0 to 255'

            loops_frame = np.zeros((len_ids, 5), dtype=np.int16)
            loops_frame[:, 0] = frame

            add = 50 if len_un_ids <= 205 else 255 - len_un_ids
            loops_frame[:, 1] = loop_id + add
            print('frame', frame,
                  'len_ids', len_ids,
                  'len_un_ids', len_un_ids,
                  'add', add)

            loops_frame[:, 2:] = np.column_stack(nonzero)

        #     nonzero = loop_id = loops_comp = None

        #     # iterable = zip(loops_final)
        #     # loop_ids = np.unique(loops_final[:, 1])
        #     print('for loop')
        #     for label, image in cc3d.each(labels_out, binary=False, in_place=True):
        #         print(label, np.argwhere(image).shape)
        #         label_argw = np.argwhere(label)
        #         label_nonzero = np.nonzero(image)
        #         lab_u, _ = np.sort(np.unique(
        #             loops[label_nonzero], return_counts=True), axis=1)
        #         loop_id = lab_u[-1]

        #         label_nonzero = lab_u = None

        #         loop_argw = np.argwhere(
        #             loops_final[loops_final[:, 1] == loop_id][:, 2:])
        #         label_in_loop = np.isin(label_argw, loop_argw)
        #         loop_in_label = np.isin(loop_argw, label_argw)
        #         print('before np.all and not np.all')
        #         if np.all(label_in_loop) and not np.all(loop_in_label):
        #             loop_in_label = None
        #             print('label in loop')
        #             # loop_sans_label = np.delete(
        #             #     loop_argw, np.argwhere(loop_in_label), axis=0)

        #             # double-check this statement... wrong?
        #             label_sans_loop = np.delete(
        #                 label_argw, loop_in_label, axis=0)

        #             new_label = random.choice(
        #                 list(set([x for x in range(50, 255)]) - set(unique)))
        #             print('loops_final labels before', unique)
        #             loops_final[label_argw][:, 1] = new_label  # was label_sans_loop instead of label_argw
        #             print('loops_final labels after',
        #                   np.unique(loops_final[:, 1]))
        # print(f'Time elapsed: {utils.time_elapsed(tic, time())}')
        # print('\nProgram exited\n')
        # exit()

        # for j, z_slice in tqdm(enumerate(loops[frame, :, :, :, 0]), desc='Z-slice'):
        #     z_slice_comp = z_slice.compute()
        #     unique, counts = np.unique(z_slice_comp, return_counts=True)
        #     unique, counts = unique[1:], counts[1:]
        #     if any(unique):
        #         nonzero = np.nonzero(z_slice_comp)
        #         loop_id = z_slice_comp[nonzero]

        #         loop_final = np.zeros((len(loop_id), 5), dtype=np.int16)
        #         loop_final[:, 0] = frame
        #         loop_final[:, 1] = loop_id
        #         loop_final[:, 2] = j
        #         loop_final[:, 3:] = np.row_stack(nonzero[:2]).T

        #         # shape_check(z_slice_comp, unique, counts, loop_final)

        #         loops_slices.append(loop_final)

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
        columns = ['timepoint', 'loop_id', 'z', 'y', 'x']
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
    (loops.astype(int)
     .sort_values(['timepoint', 'loop_id'], ignore_index=True)
     .to_csv(save_path, index=False))


def filled_to_disk(save_path, frames, filled):
    # orig_name = osp.splitext(osp.basename(loop_path))[0]
    # fr_min, fr_max = min(frames), max(frames)
    # name = f'loops_filled_{orig_name}_t{fr_min}-{fr_max}.csv'
    (filled.astype(int)
     .sort_values(['timepoint', 'loop_id'], ignore_index=True)
     .to_csv(save_path, index=False))


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
    # dist_loop = np.zeros((len(pr_trajs), 2))
    # excluded_p = []  # particles found inside a previous loop are excluded
    # # iterate over all loops
    # tp_tmp = np.inf  # keep track of when current timepoint changes
    in_loop = np.zeros(len(pr_trajs), dtype=np.bool)

    for (tp, _), loop in tqdm(loops.groupby(['timepoint', 'loop_id']), desc='Analysis per loop'):
        # only filter trajectories if timepoint changes
        # if tp > tp_tmp or tp_tmp == np.inf:
        select_tp = pr_trajs.frame == tp
        masks_frame = pr_trajs[select_tp].mask
        # tp_tmp = tp

        # += so I can check afterwards if some columns have more than 1 ...
        # then something doesn't work correctly
        # but: each entry gets a score for how much of it is in a loop
        # (i.e., how many mask coordinates of that cell are in a loop)
        in_loop[select_tp] += points_in_hull(masks_frame, loop)

        # need to be a bit careful... how should the returned data format be?

        # for _, row in trajs_frame.iterrows():
        #     particle = row[['x', 'y', 'z', 'particle']]

        #     # find particle's index in pr_trajs
        #     # (for accurate assignment to dist_loop)
        #     eq_tp = pr_trajs.frame == tp
        #     eq_p = pr_trajs.particle == particle.particle
        #     p_select = int(pr_trajs.index[eq_tp & eq_p].values)

        #     # find if particle is inside loop
        #     if (particle.particle not in excluded_p
        #             and p_in_loop(particle, loop)):

        #         dist_loop[p_select] = True, dist_to_loop(particle, loop)
        #         excluded_p.append(particle.particle)
        #     else:
        #         dist_loop[p_select] = False, dist_to_loop(particle,
        #                                                   nearest_loop(particle, loops))
    return in_loop


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

    return filled_loop


def points_in_hull(p, hull, tol=1e-12):
    return np.all(hull.equations[:, :-1] @ p.T + np.repeat(hull.equations[:, -1][None, :], len(p), axis=0).T <= tol, 0)


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
        raise ValueError(
            f"This function assumes your image is smaller than {2**15} in each dimension")

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


def prep_file(draft_file, save_loops, load_loops,
              save_filled, load_filled, pred_path,
              loop_files, pred_dir):
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
        #     filled = get_filled(pred_dir_path, frames, loops, load_filled)
        # else:
        #     filled = get_filled(pred_dir_path, frames, None, load_filled)

        # if save_filled:
        #     filled_to_disk(saved_path_fill, frames, filled)

        # pr_trajs[['in_loop', 'dist_loop']] = loop_analysis(frames,
        #                                                    pr_trajs, filled)


def prepare_loops(experiment_name, draft_file, save_loops, load_loops,
                  save_filled, load_filled):
    filter_prefix = 'cyctpy15'
    file_ext = 'tif'
    pred_path, loops_path = set_paths(experiment_name)

    # can choose only predicted dirs, or all files in /raw_data/
    select_dirs = sorted(os.listdir(pred_path))
    loop_files = get_loop_files(loops_path, select_dirs, draft_file,
                                filter_prefix, file_ext)

    results_tot = {}
    for pred_dir in tqdm(select_dirs, desc='Prediction'):
        prep_file(draft_file, save_loops, load_loops,
                  save_filled, load_filled, pred_path,
                  loop_files, pred_dir)


if __name__ == '__main__':
    # mode = 'test'
    experiment_name = 'pred_2'
    draft_file = 'LI_2019-02-05_emb5_pos4'

    save_loops = True  # save condensed loops to disk
    load_loops = False  # load condensed loops from disk
    save_filled = False  # save filled loops to disk
    load_filled = False  # load filled loops from disk

    # redundant to save loaded loops to disk
    save_loops = False if load_loops else save_loops
    save_filled = False if load_filled else save_filled

    tic = time()
    np.random.seed(42)
    utils.set_cwd(__file__)
    prepare_loops(experiment_name, draft_file, save_loops, load_loops,
                  save_filled, load_filled)

    # csv_path = osp.join(pred_dir_path,
    #                     'tracked_centroids_loops.csv')
    # pr_trajs.to_csv(csv_path, index=False)

    # I can include segmentations in the dataframe that trackpy
    # receives as input
    # It can also be terse like loops.csv

    print(f'prep_loop_corr: Time elapsed: {utils.time_report(tic, time())}.')
