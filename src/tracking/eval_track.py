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


def get_timepoints(json_files, cell_images):
    if json_files:
        timepoints = []
        for file in json_files:
            if cell_images:
                timepoint = int(osp.splitext(
                    osp.basename(file).split('_')[0])[0])
            else:
                timepoint = int(osp.splitext(osp.basename(file))[0])
            timepoints.append(timepoint)
    else:
        pass
    return timepoints


def get_frames(pred_dir_path, timepoints, groupby='frame'):
    time_range = range(min(timepoints), max(timepoints) + 2)
    track_path = osp.join(pred_dir_path, 'tracked_centroids.csv')
    tracked_centroids = pd.read_csv(track_path, header=0, sep=';')
    if groupby == 'frame':
        track_frames = [frame
                        for frame in tracked_centroids.groupby(groupby)
                        if frame[0] in time_range]

        return track_frames
    else:
        return tracked_centroids


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


def compute_distance_matrix(pred_labels, gt_labels):
    distance_matrix = np.zeros((len(pred_labels), len(gt_labels)))
    for i, pred_label in enumerate(pred_labels.iterrows()):
        for j, gt_label in enumerate(gt_labels.iterrows()):
            coord_gt = gt_label[1][['x', 'y']]
            coord_pred = pred_label[1][['x', 'y']]
            distance_matrix[i, j] = np.linalg.norm(
                coord_pred - coord_gt)

    return distance_matrix


def matches_file(json_files, json_found, track_frames, cell_images):
    '''
    Predicted trajectories (PrTraj) are matched with
    ground truth trajectories (GtTraj)
    
    '''
    iterable = zip(track_frames,
                   json_files if json_files else range(len(track_frames)))
    matches = []
    for track_frame, json_file in iterable:
        pred_labels = track_frame[1][['x', 'y', 'particle']]
        gt_labels = get_groundtruth(json_found, json_file)
        if not cell_images:
            # need to adjust coordinates if the big
            # six-figure image was used ('with_tracking' folder)
            gt_labels[['x', 'y']] -= c.BIGFIG_OFFSET

        # compute distance matrix
        dist_matrix = compute_distance_matrix(pred_labels, gt_labels)

        # linear sum assignment problem
        row_ind, col_ind = sciopt.linear_sum_assignment(dist_matrix)
        # TODO: draw predictions on image to debug (use output_tracks.py)
        matches.append((pred_labels.particle.values,
                        col_ind))

    return matches


if __name__ == '__main__':
    mode = 'test'
    experiment_name = 'pred_1'
    root_dir = c.DATA_DIR if osp.exists(
        c.DATA_DIR) else c.PROJECT_DATA_DIR_FULL
    groundtruth_path = osp.join(root_dir, 'pred', 'eval', 'track_2D', mode)
    gt_dirs = sorted(os.listdir(groundtruth_path))
    pred_path = osp.join(root_dir, 'pred', experiment_name)
    pred_dirs = os.listdir(pred_path)

    matches_files = {}
    for gt_dir in gt_dirs:
        gt_dir_path = osp.join(groundtruth_path, gt_dir)
        pred_dir_path = osp.join(pred_path, gt_dir)

        # find json files
        json_files, cell_images = find_json_files(gt_dir_path)
        json_found = json_files is not None

        timepoints = get_timepoints(json_files, cell_images)

        track_frames = get_frames(pred_dir_path, timepoints)

        matches = matches_file(json_files, json_found,
                               track_frames, cell_images)
        matches_files[gt_dir] = matches
