import numpy as np
import src.tracking.track_cells as tracker
import pickle
import os.path as osp
import pandas as pd
import src.data.utils.utils as utils
import src.data.constants as c
from time import time

if __name__ == '__main__':
    tic = time()
    utils.setcwd(__file__)
    name = c.PRED_FILE
    save = osp.join('..', c.PROJECT_DATA_DIR, 'pred', name)
    utils.make_dir(save)

    centroids = pickle.load(open(osp.join(save, 'centroids_save.pkl'), 'rb'))

    centroids_np = [(t, centroid[0], centroid[1],
                     centroid[2], centroid[3])
                    for t, timepoint in enumerate(centroids)
                    for centroid in timepoint]
    
    iter_radius = enumerate(range(5, 105, 5))
    len_radius = len(tuple(iter_radius))
    iter_memory = enumerate(range(0, 6))
    len_memory = len(tuple(iter_memory))
    iter_step = enumerate(np.linspace(0.9, 1, 5, endpoint=True))
    len_step = len(tuple(iter_step))
    radius_track = []

    for i, track_radius in enumerate(range(10, 110, 30)):
        for j, memory in enumerate(range(0, 3)):
            for k, thresh in enumerate(np.linspace(0, 19, 5, endpoint=True)):
                tracked_centroids = tracker.track(centroids_np, track_radius,
                                                  memory, thresh, 0.8)
                frames = tracked_centroids.groupby('frame')
                radius_track.append(to_append)

                np.savetxt('track_radius.csv', radius_track,
                           delimiter=';',
                           header='track_radius;memory;threshold;n_particles;mean_len_traj;mean_diff;track_metric')

    elapsed = utils.time_report(tic, time())
    print(f'track_radius completed after {elapsed}.')
