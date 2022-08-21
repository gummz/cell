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
    time_range = range(min(timepoints), max(timepoints) + 2)
    track_path = osp.join(pred_dir_path, 'tracked_centroids.csv')
    tracked_centroids = pd.read_csv(track_path, header=0, sep=';')
    tc_filter = tracked_centroids[tracked_centroids['frame'].isin(time_range)]
    return tc_filter.groupby(groupby)


def get_groundtruth(json_found, json_file):
    if json_found:  # ground truth is available
        json_data = json.load(open(json_file, 'rb'))
        gt_labels = pd.DataFrame(json_data['shapes'])[['points', 'label']]
        gt_labels['x'] = gt_labels['points'].apply(
            lambda x: x[0][0])
        gt_labels['y'] = gt_labels['points'].apply(
            lambda x: x[0][1])
        gt_labels['particle'] = gt_labels['label']
        gt_labels = gt_labels[['x', 'y', 'particle']]
    else:  # no json files, so empty ground truth
        # in the case where there were no beta cells
        # on the image
        gt_labels = pd.DataFrame(
            {'x': [], 'y': [], 'particle': []})

    return gt_labels


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


def compute_distance_matrix(pred_labels, gt_labels):
    distance_matrix = np.zeros((len(pred_labels), len(gt_labels)))
    for i, pred_label in enumerate(pred_labels.iterrows()):
        for j, gt_label in enumerate(gt_labels.iterrows()):
            coord_gt = gt_label[1][['x', 'y']]
            coord_pred = pred_label[1][['x', 'y']]
            distance_matrix[i, j] = np.linalg.norm(
                coord_pred - coord_gt)

    return distance_matrix


def compute_traj_matrix(frames, pr_trajs, gt_trajs, threshold=50):
    '''
    Compute similarity between predicted and ground truth trajectories
    '''
    columns = ['frame', 'x', 'y']
    traj_matrix = np.zeros((len(pr_trajs.groups), len(gt_trajs.groups)))
    for i, pr_label in enumerate(pr_trajs.groups):
        pr_traj = pr_trajs.get_group(pr_label)[columns]
        for j, gt_label in enumerate(gt_trajs.groups):
            gt_traj = gt_trajs.get_group(gt_label)[columns]
            traj_matrix[i, j] = compute_similarity(
                frames, pr_traj, gt_traj, threshold)

    return traj_matrix


def compute_similarity(frames, pr_traj, gt_traj, threshold):
    '''
    Compute similarity between predicted and ground truth trajectories
    '''
    args = (frames, pr_traj, gt_traj, threshold)
    fp = get_traj_fp(*args)
    fn = get_traj_fn(*args)
    tp = get_traj_tp(*args)
    similarity = tp / (tp + fn + fp)
    return similarity


def get_traj_fp(frames, pr_traj, gt_traj, threshold):
    in_gt = pr_traj['frame'].isin(gt_traj['frame'])
    in_pr = gt_traj['frame'].isin(pr_traj['frame'])
    fp = np.sum(~in_gt)
    coord = ['x', 'y']

    pr_in_gt = pr_traj[in_gt][coord].values
    gt_in_pr = gt_traj[in_pr][coord].values
    norms = np.linalg.norm(pr_in_gt - gt_in_pr, axis=1)
    exceed_threshold = norms > threshold
    fp += np.sum(exceed_threshold)

    exceed_idx = np.zeros((len(frames), 2))
    exceed_idx[:, 0] = frames
    pr_frames = pr_traj[in_gt]['frame'].values
    gt_frames = gt_traj[in_pr]['frame'].values
    intersect = np.intersect1d(pr_frames, gt_frames,
                               assume_unique=True,
                               return_indices=True)
    exceed_idx[(exceed_idx[:, 0] == intersect), 1] = exceed_threshold
    fp_idx = np.logical_and(exceed_idx[~in_gt], exceed_idx)

    return fp, fp_idx


def get_traj_fn(frames, pr_traj, gt_traj, threshold):
    '''
    If a prediction is present in a frame, but it exceeds
    the threshold, then it is counted as a false positive,
    not a false negative.
    '''
    in_pr = gt_traj['frame'].isin(pr_traj['frame'])
    fn = np.sum(~in_pr)
    fn_idx = np.zeros(frames)

    return fn, fn_idx


def get_traj_tp(frames, pr_traj, gt_traj, threshold):
    in_gt = pr_traj['frame'].isin(gt_traj['frame'])
    in_pr = gt_traj['frame'].isin(pr_traj['frame'])
    tp = 0
    coord = ['x', 'y']
    pr_in_gt = pr_traj[in_gt][coord].values
    gt_in_pr = gt_traj[in_pr][coord].values
    norms = np.linalg.norm(pr_in_gt - gt_in_pr, axis=1)
    within_threshold = norms <= threshold
    tp += np.sum(within_threshold)
    tp_idx = in_gt & in_pr & ()

    return tp


def match_trajs(frames, pr_trajs, gt_trajs, threshold=80):
    '''
    Predicted trajectories (PrTraj) are matched with
    ground truth trajectories (GtTraj)

    '''
    traj_matrix = compute_traj_matrix(frames, pr_trajs, gt_trajs, threshold)

    # linear sum assignment problem
    row_ind, col_ind = sciopt.linear_sum_assignment(traj_matrix)

    return row_ind, col_ind, traj_matrix


def compute_score(row_ind, col_ind, traj_matrix,
                  pr_trajs, gt_trajs):
    '''
    Compute HOTA score.
    '''
    for i, j in zip(row_ind, col_ind):
        pr_traj = pr_trajs.get_group(i)
        gt_traj = gt_trajs.get_group(j)

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

        # find json files
        json_files, cell_images = find_json_files(gt_dir_path)
        png_files = find_png_files(gt_dir_path)

        timepoints = get_timepoints(png_files, cell_images)

        pred_trajs = get_pr_trajs(pred_dir_path, timepoints)
        gt_trajs = get_gt_trajs(json_files if json_files else None,
                                cell_images)
        row_ind, col_ind, traj_mat = match_trajs(
            timepoints, pred_trajs, gt_trajs, 80)
        matches = (row_ind, col_ind)
        score = compute_score(matches, traj_mat, pred_trajs, gt_trajs)

        score_tot[gt_dir] = score
