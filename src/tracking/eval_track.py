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


def get_pr_trajs(pred_dir_path, timepoints, groupby='particle'):
    time_range = range(min(timepoints), max(timepoints) + 1)
    track_path = osp.join(pred_dir_path, 'tracked_centroids.csv')
    tracked_centroids = pd.read_csv(track_path, header=0, sep=';')
    if timepoints:
        tc_filter = tracked_centroids[tracked_centroids['frame'].isin(time_range)]
    else:
        tc_filter = tracked_centroids
    return tc_filter.groupby(groupby) if groupby else tc_filter


def get_gt_trajs(json_files, cell_images):
    json_data = (json.load(open(json_file, 'rb'))['shapes']
                 for json_file in json_files)
    json_data = [(int(osp.splitext(osp.basename(frame))[0]),
                  item['points'][0][0], item['points'][0][1], int(item['label']))
                 for json_entry, frame in zip(json_data, json_files)
                 for item in json_entry]
    df_json = pd.DataFrame(json_data, columns=['frame', 'x', 'y', 'particle'])
    if not cell_images:
        # need to adjust coordinates if the big
        # six-figure image was used ('with_tracking' folder)
        df_json[['x', 'y']] -= c.BIGFIG_OFFSET

    return df_json.groupby('particle')


def compute_traj_matrix(frames, pr_trajs, gt_trajs, threshold=50):
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


def compute_similarity(frames, pr_traj, gt_traj, threshold):
    '''
    Compute similarity between predicted and ground truth trajectories
    '''
    args = (frames, pr_traj, gt_traj, threshold)
    tp_idx = get_traj_tp(*args)
    # fn_idx = get_traj_fn(*args)
    fp_idx = get_traj_fp(*args)
    tp = np.sum(tp_idx)
    # fn = np.sum(fn_idx)
    fp = np.sum(fp_idx)
    similarity = tp / (tp + fn + fp)
    return similarity, np.array((tp_idx, fn_idx, fp_idx))


def get_traj_fp(frames, pr_traj, gt_traj, threshold):
    in_gt = pr_traj['frame'].isin(gt_traj['frame'])
    in_pr = gt_traj['frame'].isin(pr_traj['frame'])
    coord = ['x', 'y']

    pr_in_gt = pr_traj[in_gt][coord].values
    gt_in_pr = gt_traj[in_pr][coord].values
    norms = np.linalg.norm(pr_in_gt - gt_in_pr, axis=1)
    exceed_threshold = norms > threshold

    exceed_idx = np.zeros((len(frames), 2), dtype=np.int32)
    exceed_idx[:, 0] = frames
    # TODO: ~in_gt is wrong
    # it will raise an error when number of frames in pred 
    # is less than the total number of frames
    gt_idx = np.zeros(len(frames))
    gt_idx[~in_gt] = 1
    pr_frames = pr_traj['frame'].values
    gt_frames = gt_traj['frame'].values
    intersect = np.intersect1d(pr_frames, gt_frames)
    exceed_idx[exceed_idx[:, 0].isin(intersect), 1] = exceed_threshold
    fp_idx = np.logical_or(exceed_idx[:, 1], gt_idx)

    return fp_idx


def get_traj_fn(frames, pr_traj, gt_traj, threshold):
    '''
    If a prediction is present in a frame, but it exceeds
    the threshold, then it is counted as a false positive,
    not a false negative.
    '''
    in_pr = np.in1d(gt_traj['frame'], pr_traj['frame'])

    in_both = np.intersect1d(pr_traj['frame'], gt_traj['frame'])
    idx = np.zeros((len(frames), 2), dtype=np.int32)
    idx[:, 0] = frames
    fn_idx = np.zeros(len(frames))
    fn_idx[~in_pr] = 1

    return fn_idx


def get_traj_tp(frames, pr_traj, gt_traj, threshold):
    in_gt = pr_traj['frame'].isin(gt_traj['frame'])
    in_pr = gt_traj['frame'].isin(pr_traj['frame'])
    coord = ['x', 'y']
    pr_in_gt = pr_traj[in_gt][coord].values
    gt_in_pr = gt_traj[in_pr][coord].values
    norms = np.linalg.norm(pr_in_gt - gt_in_pr, axis=1)
    within_threshold = norms <= threshold

    within_idx = np.zeros((len(frames), 2), dtype=np.int32)
    within_idx[:, 0] = frames
    pr_frames = pr_traj[in_gt]['frame'].values
    gt_frames = gt_traj[in_pr]['frame'].values
    intersect = np.intersect1d(pr_frames, gt_frames)
    within_idx[np.in1d(within_idx[:, 0], intersect), 1] = within_threshold
    tp_idx = within_idx[:, 1]
    return tp_idx


def match_trajs(frames, pr_trajs, gt_trajs, threshold=80):
    '''
    Predicted trajectories (PrTraj) are matched with
    ground truth trajectories (GtTraj)

    '''
    traj_score_matrix, traj_matrix = compute_traj_matrix(
        frames, pr_trajs, gt_trajs, threshold)

    # linear sum assignment problem
    row_ind, col_ind = sciopt.linear_sum_assignment(traj_score_matrix)

    return row_ind, col_ind, traj_matrix


def compute_score(row_ind, col_ind, traj_matrix, gt_trajs):
    '''
    Compute HOTA score.
    '''
    for i, j in zip(row_ind, col_ind):
        tp, fn, fp = traj_matrix.iloc[i, j]
        gt_traj = gt_trajs.get_group(j)
        # calculate associative scores
        pass

    score = np.sum(traj_matrix[row_ind, col_ind])


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


def evaluate(pred_dir_path, json_files, cell_images, timepoints):
    pred_trajs = get_pr_trajs(pred_dir_path, timepoints)
    gt_trajs = get_gt_trajs(json_files if json_files else None,
                            cell_images)
    row_ind, col_ind, traj_matrix = match_trajs(
        timepoints, pred_trajs, gt_trajs, 80)
    score = compute_score(row_ind, col_ind, traj_matrix, gt_trajs)
    return score


if __name__ == '__main__':
    mode = 'test'
    experiment_name = 'pred_1'
    root_dir = c.DATA_DIR if osp.exists(
        c.DATA_DIR) else c.PROJECT_DATA_DIR_FULL
    groundtruth_path = osp.join(root_dir, 'pred', 'eval', 'track_2D', mode)
    gt_dirs = sorted(os.listdir(groundtruth_path))
    pred_path = osp.join(root_dir, 'pred', experiment_name)
    pred_dirs = os.listdir(pred_path)

    score_tot = {}
    for gt_dir in gt_dirs:
        gt_dir_path = osp.join(groundtruth_path, gt_dir)
        pred_dir_path = osp.join(pred_path, gt_dir)

        # find json and png files
        json_files, cell_images, png_files = find_files(gt_dir_path)

        timepoints = get_timepoints(png_files, cell_images)

        score = evaluate(pred_dir_path, json_files, cell_images, timepoints)

        score_tot[gt_dir] = score
