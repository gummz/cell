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
    
    radius_track = np.empty((8, 2))
    for i, track_radius in enumerate(range(5, 105, 5)):
        tracked_centroids = tracker.track(centroids_np, track_radius)
        n_particles = len(pd.unique(tracked_centroids['particle']))
        radius_track[i] = (track_radius, n_particles)

    np.savetxt('radius_track.csv', radius_track, delimiter=',', header='track_radius,n_particles')

    elapsed = utils.time_report(tic, time())
    print(f'track_radius completed after {elapsed}.')
