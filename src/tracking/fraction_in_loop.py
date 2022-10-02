import os.path as osp

import numpy as np
import src.data.constants as c
import glob
import pandas as pd
import src.data.utils.utils as utils


def born_in_loop(iterable, first):
    start = np.array(
            [cell[1].iloc[:first].in_loop
            for cell in iterable]
        )

    cells_inloop = np.sum(np.any(start > 0, axis=1)) / len(start)
    cells_outloop = np.sum(np.all(start == 0, axis=1)) / len(start)

    return cells_inloop, cells_outloop


def compare_dist(iterable1, iterable0, 
first):
    dist_beta = np.array(
            [cell[1].iloc[:first].dist_loop
                for cell in iterable1]
        )
    dist_nonbeta = np.array(
            [cell[1].iloc[:first].dist_loop
            for cell in iterable0]
        )

    mean_beta = dist_beta.mean(axis=1).mean()
    mean_nonbeta = dist_nonbeta.mean(axis=1).mean()

    return mean_beta, mean_nonbeta


if __name__ == '__main__':
    experiment_name = 'pred_2'
    pred_dir = '/home/gummz/cell/data/interim/pred/pred_2'

    utils.set_cwd(__file__)
    pattern = osp.join(pred_dir, '*', '*.csv')
    pred_files = glob.glob(pattern, recursive=True)
    dev_check = ('/home/gummz/cell/data/interim/pred/pred_2/LI_2018-12-07_emb5_pos2/tracked_centroids_loops.csv',
    '/home/gummz/cell/data/interim/pred/pred_2/LI_2018-12-07_emb6_pos3/tracked_centroids_loops.csv',
    '/home/gummz/cell/data/interim/pred/pred_2/LI_2018-12-18_emb4_pos4/tracked_centroids_loops.csv',
    '/home/gummz/cell/data/interim/pred/pred_2/LI_2020-05-06_emb7_pos4/tracked_centroids_loops.csv')
    pred_files = [file for file in pred_files if file in dev_check]
    
    names = pd.Series([osp.basename(osp.dirname(pred_path))
            for pred_path in pred_files])
    values = pd.DataFrame(np.empty((len(pred_files), 12), dtype=np.float32))
    columns = [
    'filename', 
    'cells_inloop0', 'cells_outloop0',
    'cells_inloop5', 'cells_outloop5',
    'mean_beta01', 'mean_nonbeta01',
    'mean_beta51', 'mean_nonbeta51',
    'mean_beta02', 'mean_nonbeta02',
    'mean_beta52', 'mean_nonbeta52'
    ]
    for i, (name, pred_path) in enumerate(zip(names, pred_files)):
        pred_file = pd.read_csv(pred_path, sep=';')
        iterable1 = pred_file[pred_file.beta==1].groupby('particle')
        iterable0 = pred_file[pred_file.beta==0].groupby('particle')

        cells_inloop0, cells_outloop0 = born_in_loop(iterable1, 1)
        cells_inloop5, cells_outloop5 = born_in_loop(iterable1, 5)

        mean_beta01, mean_nonbeta01 = compare_dist(iterable1, iterable0, 1)
        mean_beta51, mean_nonbeta51 = compare_dist(iterable1, iterable0, 5)

        pred_file.dist_loop = np.where(
        ((pred_file.in_loop > 0) & (pred_file.beta == 1)), 
        pred_file.dist_loop, 0) # reverse pred_file.dist_loop and 0?
        iterable1 = pred_file[pred_file.beta==1].groupby('particle')
        iterable0 = pred_file[pred_file.beta==0].groupby('particle')
        mean_beta02, mean_nonbeta02 = compare_dist(iterable1, iterable0, 1)
        mean_beta52, mean_nonbeta52 = compare_dist(iterable1, iterable0, 5)

        entry = np.round((
                cells_inloop0, cells_outloop0,
                cells_inloop5, cells_outloop5,
                mean_beta01, mean_nonbeta01,
                mean_beta51, mean_nonbeta51,
                mean_beta02, mean_nonbeta02,
                mean_beta52, mean_nonbeta52), 3)

        values.iloc[i] = entry

    output = pd.concat((names, values), axis=1)
    output.columns = columns
    output.reset_index()
    output.to_csv('saved.csv')