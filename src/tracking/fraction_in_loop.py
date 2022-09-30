import os.path as osp

import numpy as np
import src.data.constants as c
import glob
import pandas as pd


def born_in_loop(iterable, first):
    start = np.array(
            [cell[1].iloc[:first].in_loop
            for cell in iterable]
        )

    cells_inloop = np.sum(np.any(start > 0, axis=1)) / len(start)
    cells_outloop = np.sum(np.all(start == 0, axis=1)) / len(start)

    return (np.round(cells_inloop, 3), 
    np.round(cells_outloop, 3))


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
    pattern = osp.join(pred_dir, '*', '*.csv')
    pred_files = glob.glob(pattern, recursive=True)
    dev_check = ('/home/gummz/cell/data/interim/pred/pred_2/LI_2018-12-07_emb5_pos2/tracked_centroids_loops.csv',
    '/home/gummz/cell/data/interim/pred/pred_2/LI_2018-12-07_emb6_pos3/tracked_centroids_loops.csv',
    '/home/gummz/cell/data/interim/pred/pred_2/LI_2018-12-18_emb4_pos4/tracked_centroids_loops.csv',
    '/home/gummz/cell/data/interim/pred/pred_2/LI_2020-05-06_emb7_pos4/tracked_centroids_loops.csv')
    pred_files = [file for file in pred_files if file in dev_check]
    
    names = [osp.splitext(osp.basename(pred_path))[0]
            for pred_path in pred_files]
    values = np.empty((len(pred_files), 12), dtype=np.float32)
    for i, (name, pred_path) in enumerate(zip(names, pred_files)):
        name = osp.splitext(osp.basename(pred_path))[0]
        pred_file = pd.read_csv(pred_path, sep=';')
        iterable1 = pred_file[pred_file.beta==1].groupby('particle')
        iterable0 = pred_file[pred_file.beta==0].groupby('particle')

        cells_inloop0, cells_outloop0 = born_in_loop(iterable1, 1)
        cells_inloop5, cells_outloop5 = born_in_loop(iterable1, 5)

        mean_beta01, mean_nonbeta01 = compare_dist(iterable1, iterable0, 1)
        mean_beta51, mean_nonbeta51 = compare_dist(iterable1, iterable0, 5)

        pred_file['dist_loop'][pred_file.beta == 1] = np.where(
        pred_file[pred_file.beta == 1]['in_loop'] > 0, pred_file[pred_file.beta == 1]['dist_loop'], 0)
        iterable1 = pred_file[pred_file.beta==1].groupby('particle')
        iterable0 = pred_file[pred_file.beta==0].groupby('particle')
        mean_beta20, mean_nonbeta20 = compare_dist(iterable1, iterable0, 1)
        mean_beta25, mean_nonbeta25 = compare_dist(iterable1, iterable0, 5)

        entry = (cells_inloop0, cells_outloop0,
                cells_inloop5, cells_outloop5,
                mean_beta01, mean_nonbeta01,
                mean_beta51, mean_nonbeta51,
                mean_beta20, mean_nonbeta20,
                mean_beta25, mean_nonbeta25)

        values[i] = entry

    output = np.column_stack((names, values))