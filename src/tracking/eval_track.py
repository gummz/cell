from calendar import c
import os.path as osp
import src.data.constants as c
import src.visualization.utils as viz
import src.data.utils.utils as utils
import os
import json
import pandas as pd
import numpy as np
import scipy.optimize as sciopt
import glob
from time import time


def find_json_files(gt_dir_path):
    iterable = os.walk(osp.join(gt_dir_path))
    for folders, subfolders, files in iterable:
        json_files = sorted([file for file in files
                             if file.endswith('.json')])
        if json_files:
            json_files = [osp.join(folders, file) for file in json_files]
            cell_images = 'cell_images' in folders
            break
    else:
        cell_images = False
    return json_files, cell_images


def find_png_files(gt_dir_path):
    iterable = os.walk(osp.join(gt_dir_path))
    for folders, subfolders, files in iterable:
        png_files = sorted([file for file in files
                            if file.endswith('.png')])
        json_files = sorted([file for file in files
                            if file.endswith('.json')])
        if png_files and json_files:
            break
    return png_files


def find_files(gt_dir_path):
    json_files, cell_images = find_json_files(gt_dir_path)
    png_files = find_png_files(gt_dir_path)
    return json_files, cell_images, png_files


def get_timepoints(png_files, cell_images):
    if png_files:
        timepoints = []
        for file in png_files:
            if cell_images:
                timepoint = int(osp.splitext(
                    osp.basename(file).split('_')[0])[0])
            else:
                timepoint = int(osp.splitext(osp.basename(file))[0])
            timepoints.append(timepoint)
    else:
        pass
    return timepoints


def evaluate(pred_dir_path, json_files, cell_images,
             timepoints, threshold=80):
    pred_trajs = get_pr_trajs(pred_dir_path, timepoints)
    gt_trajs = get_gt_trajs(json_files if json_files else None,
                            cell_images)
    # TODO: do not return traj_score_matrix from match_trajs,
    # and do not pass it to compute_score
    row_ind, col_ind, traj_score_matrix, traj_matrix = match_trajs(
        timepoints, pred_trajs, gt_trajs, threshold)

    # NB: sum of traj_score_matrix[row_ind, col_ind]
    # are the AssA scores
    score = compute_score(row_ind, col_ind, traj_score_matrix,
                          traj_matrix, pred_trajs, gt_trajs,
                          threshold)

    # precision

    # recall

    return score


def get_pr_trajs(pred_dir_path, timepoints=None, groupby='particle'):
    track_path = osp.join(pred_dir_path, 'tracked_centroids.csv')
    tracked_centroids = pd.read_csv(track_path, header=0)
    if timepoints:
        time_range = range(min(timepoints), max(timepoints) + 1)
        tc_filter = tracked_centroids[tracked_centroids['frame'].isin(
            time_range)]
    else:
        tc_filter = tracked_centroids
    return tc_filter.groupby(groupby) if groupby else tc_filter


def get_gt_trajs(json_files, cell_images):
    json_data = load_json_data(json_files, cell_images)
    df_json = pd.DataFrame(json_data, columns=['frame', 'x', 'y', 'particle'])
    if not cell_images:
        # need to adjust coordinates if the big
        # six-figure image was used ('with_tracking' folder)
        df_json[['x', 'y']] -= c.BIGFIG_OFFSET

    return df_json.groupby('particle')


def load_json_data(json_files, cell_images):
    if json_files:
        json_data = (json.load(open(json_file, 'rb'))['shapes']
                     for json_file in json_files)

        json_traj = []
        for json_entry, frame in zip(json_data, json_files):
            for item in json_entry:
                if cell_images:
                    frame_int = osp.basename(frame).split('_')[0]
                else:
                    frame_int = osp.basename(frame)

                item_frame = int(osp.splitext(frame_int)[0])
                item_x = item['points'][0][0]
                item_y = item['points'][0][1]
                item_label = int(item['label'])
                json_traj.append((item_frame, item_x, item_y, item_label))
    else:
        json_traj = ()

    return json_traj


def compute_traj_matrix(frames, pr_trajs, gt_trajs, threshold=80):
    '''
    Compute similarity between predicted and ground truth trajectories
    '''
    columns = ['frame', 'x', 'y']
    traj_score_matrix = np.zeros((len(pr_trajs), len(gt_trajs)))
    traj_matrix = pd.DataFrame(index=range(len(pr_trajs.groups)),
                               columns=range(len(gt_trajs.groups)),
                               dtype=object)
    for i, pr_label in enumerate(pr_trajs.groups):
        pr_traj = pr_trajs.get_group(pr_label)[columns]
        for j, gt_label in enumerate(gt_trajs.groups):

            gt_traj = gt_trajs.get_group(gt_label)[columns]

            similarity, arrays = compute_similarity(
                frames, pr_traj, gt_traj, threshold)

            traj_score_matrix[i, j] = similarity
            traj_matrix.iloc[i, j] = arrays

    return traj_score_matrix, traj_matrix


def compute_similarity(frames, pr_traj, gt_traj, threshold,
                       return_vectors=True):
    '''
    Compute similarity between predicted and ground truth trajectories
    '''
    args = (frames, pr_traj, gt_traj, threshold)
    tp_idx = get_traj_tp(*args)
    fn_idx = get_traj_fn(*args)
    fp_idx = get_traj_fp(*args)
    tp = np.sum(tp_idx)
    fn = np.sum(fn_idx)
    fp = np.sum(fp_idx)
    similarity = tp / (tp + fn + fp)

    if return_vectors:
        return similarity, np.array((tp_idx, fn_idx, fp_idx))
    else:
        return similarity


def get_traj_fp(frames, pr_traj, gt_traj, threshold):
    in_gt = pr_traj['frame'].isin(gt_traj['frame'])
    in_pr = gt_traj['frame'].isin(pr_traj['frame'])
    coord = ['x', 'y']

    pr_in_gt = pr_traj[in_gt][coord].values
    gt_in_pr = gt_traj[in_pr][coord].values
    norms = np.linalg.norm(pr_in_gt - gt_in_pr, axis=1)
    exceed_threshold = norms > threshold

    pred_gt = np.zeros((len(frames), 3), dtype=np.int32)
    pred_gt[:, 0] = np.isin(frames, pr_traj.frame)
    pred_gt[:, 1] = np.isin(frames, gt_traj.frame)
    fp_idx = np.logical_and((pred_gt[:, 0] == 1),
                            (pred_gt[:, 1] == 0))
    eq_idx = np.logical_and(pred_gt[:, 0], pred_gt[:, 1])

    pred_gt[eq_idx, 2] = exceed_threshold
    fp_idx = np.logical_or(pred_gt[:, 2], fp_idx)

    return fp_idx


def get_traj_fn(frames, pr_traj, gt_traj, threshold):
    '''
    If a prediction is present in a frame, but it exceeds
    the threshold, then it is counted as a false positive,
    not a false negative.
    '''
    in_gt = pr_traj['frame'].isin(gt_traj['frame'])
    in_pr = gt_traj['frame'].isin(pr_traj['frame'])
    coord = ['x', 'y']

    pr_in_gt = pr_traj[in_gt][coord].values
    gt_in_pr = gt_traj[in_pr][coord].values
    norms = np.linalg.norm(pr_in_gt - gt_in_pr, axis=1)
    exceed_threshold = norms > threshold

    pred_gt = np.zeros((len(frames), 3), dtype=np.int32)
    pred_gt[:, 0] = np.isin(frames, pr_traj.frame)
    pred_gt[:, 1] = np.isin(frames, gt_traj.frame)
    fn_idx = np.logical_and((pred_gt[:, 0] == 0),
                            (pred_gt[:, 1] == 1))
    eq_idx = np.logical_and(pred_gt[:, 0], pred_gt[:, 1])

    pred_gt[eq_idx, 2] = exceed_threshold
    fn_idx = np.logical_or(pred_gt[:, 2], fn_idx)

    return fn_idx


def get_traj_tp(frames, pr_traj, gt_traj, threshold):
    in_gt = pr_traj['frame'].isin(gt_traj['frame'])
    in_pr = gt_traj['frame'].isin(pr_traj['frame'])
    coord = ['x', 'y']
    pr_in_gt = pr_traj[in_gt][coord].values
    gt_in_pr = gt_traj[in_pr][coord].values
    norms = np.linalg.norm(pr_in_gt - gt_in_pr, axis=1)
    within_threshold = norms <= threshold

    pred_gt = np.zeros((len(frames), 3), dtype=np.int32)
    pred_gt[:, 0] = np.isin(frames, pr_traj.frame)
    pred_gt[:, 1] = np.isin(frames, gt_traj.frame)
    tp_idx = np.logical_and((pred_gt[:, 0] == 1),
                            (pred_gt[:, 1] == 1))
    eq_idx = np.logical_and(pred_gt[:, 0], pred_gt[:, 1])

    pred_gt[eq_idx, 2] = within_threshold
    tp_idx = np.logical_and(pred_gt[:, 2], tp_idx)

    return tp_idx


def match_trajs(frames, pr_trajs, gt_trajs, threshold=80):
    '''
    Predicted trajectories (PrTraj) are matched with
    ground truth trajectories (GtTraj)
    '''
    traj_score_matrix, traj_matrix = compute_traj_matrix(
        frames, pr_trajs, gt_trajs, threshold)

    # linear sum assignment problem
    row_ind, col_ind = sciopt.linear_sum_assignment(
        traj_score_matrix, maximize=True)

    return row_ind, col_ind, traj_score_matrix, traj_matrix


def compute_score(row_ind, col_ind, traj_score_matrix,
                  traj_matrix, pred_trajs, gt_trajs,
                  threshold):
    '''
    Compute HOTA score.
    '''
    # associative score
    AssA = 0
    for i, j in zip(row_ind, col_ind):
        tp_idx, fn_idx, fp_idx = traj_matrix.iloc[i, j]
        TPA = np.sum(tp_idx)
        FNA = np.sum(fn_idx)
        FPA = np.sum(fp_idx)
        AssA += TPA / (TPA + FNA + FPA)

    # detection score
    pred_frames = pred_trajs.obj.groupby('frame')
    gt_frames = gt_trajs.obj.groupby('frame')
    DetA = compute_DetA(pred_frames, gt_frames, threshold)

    return np.sqrt(AssA * DetA)


def compute_DetA(pred_frames, gt_frames, threshold):
    tp_tot = 0
    fn_tot = 0
    fp_tot = 0
    coord = ['x', 'y']
    if pred_frames and gt_frames:
        for pred_frame, gt_frame in zip(pred_frames, gt_frames):
            dist_matrix = compute_dist_matrix(
                pred_frame[1][coord], gt_frame[1][coord])
            row_ind, col_ind = sciopt.linear_sum_assignment(dist_matrix)
            matrix_thresh = np.where(dist_matrix[row_ind, col_ind] < threshold,
                                     1, 0)
            tp_tot += np.sum(matrix_thresh == 1)

            fn_tot += dist_matrix.shape[1] - len(col_ind)
            fn_tot += np.sum(matrix_thresh == 0)

            fp_tot += dist_matrix.shape[0] - len(row_ind)
            fp_tot += np.sum(matrix_thresh == 0)

        DetA = 1 / (tp_tot + fn_tot + fp_tot)
    elif not pred_frames and gt_frames:
        DetA = 1 / len(gt_frames)
    elif pred_frames and not gt_frames:
        DetA = 1 / len(pred_frames)

    return DetA


def compute_dist_matrix(pred_frame, gt_frame):
    dist_matrix = np.zeros((len(pred_frame), len(gt_frame)))
    for i, particle in enumerate(pred_frame.iterrows()):
        for j, obj in enumerate(gt_frame.iterrows()):
            dist_matrix[i, j] = np.linalg.norm(particle[1] - obj[1])
    return dist_matrix


if __name__ == '__main__':
    mode = 'test'
    experiment_name = 'pred_1'

    tic = time()
    root_dir = c.DATA_DIR if osp.exists(
        c.DATA_DIR) else c.PROJECT_DATA_DIR_FULL
    groundtruth_path = osp.join(root_dir, 'pred', 'eval', 'track_2D', mode)
    gt_dirs = sorted(os.listdir(groundtruth_path))
    pred_path = osp.join(root_dir, 'pred', experiment_name)
    pred_dirs = os.listdir(pred_path)

    score_tot = {}
    # approximate the HOTA integral (Luiten et al. p. 8)
    # by averaging over different localization thresholds
    for alpha in (80,):  # np.linspace(5, 100, 5):
        threshold = alpha
        score_alpha = {}
        for gt_dir in gt_dirs:
            gt_dir_path = osp.join(groundtruth_path, gt_dir)
            pred_dir_path = osp.join(pred_path, gt_dir)

            # find json and png files
            json_files, cell_images, png_files = find_files(gt_dir_path)

            timepoints = get_timepoints(png_files, cell_images)

            score = evaluate(pred_dir_path, json_files, cell_images,
                             timepoints, threshold)

            score_alpha[gt_dir] = score

        score_tot[np.round(alpha, 2)] = score_alpha

    print(f'eval_track completed in {utils.time_report(tic, time())}.')
    print('Scores:\n')
    print(*score_tot)
