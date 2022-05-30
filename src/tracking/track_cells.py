import numpy as np
import pandas as pd
import src.data.utils.utils as utils
import src.data.constants as c
import trackpy
from os.path import join
import pickle


def track(chains: list, track_radius=c.TRACK_RADIUS) -> pd.DataFrame:
    columns = ['frame', 'x', 'y', 'z', 'intensity']
    df_chains = pd.DataFrame(
        chains, columns=columns, dtype=float
    )

    tr = trackpy.link(df_chains[columns], track_radius)

    # tr_filter = list(tr.groupby('particle'))
    # # filter out short tracks
    # particles_after = [item[1].particle for item in tr_filter
    #                    if len(item[1]) > 5]

    # tr_final = tr[tr.particle.isin(particles_after)]
    # print(type(tr), type(tr_final))
    return tr
    # list(tr_final.groupby('frame'))
    """
    Need to find the constant velocity of the tissue first,
    by looking at the embryo in question over timepoints.
    """


def calc_turnon(tracked_centroids: pd.DataFrame) -> dict:
    '''
    For each cell in the dataframe, returns the timepoint 
    at which the cell is considered to have "turned on."
    '''
    particles = tracked_centroids.groupby('particle')
    for _, particle in particles:
        I = particle['intensity']
        # calculation needs to be proportional to
        # maximum and minimum intensity
        max_val, min_val = np.max(I), np.min(I)
        diff = max_val - min_val
        

if __name__ == "__main__":
    pass
