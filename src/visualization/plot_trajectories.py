import src.data.utils.utils as utils
import src.data.constants as c
import os.path as osp
import pickle
import trackpy
import matplotlib.pyplot as plt

if __name__ == '__main__':

    utils.set_cwd(__file__)
    files = c.RAW_FILES['test']
    files = tuple(files.keys())[:4]
    files = utils.add_ext(files)
    # tracked_centroids_tot = []

    for file in files:
        print('Printing trajectory for file:', file)
        load_loc = osp.join(c.PROJECT_DATA_DIR, 'pred', 'reject_not', file)
        file_loc = osp.join(load_loc, 'tracked_centroids.pkl')
        tracked_centroids = pickle.load(open(file_loc, 'rb'))
        # tracked_centroids_tot.append(tracked_centroids)

        trackpy.plot_traj3d(tracked_centroids)
        save_name = f'tracked_centroids_{file}.png'
        plt.savefig(osp.join(load_loc, save_name))
        plt.close()

    print('plot_trajectories.py complete')
