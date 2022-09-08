import numpy as np
import pandas as pd
import src.data.constants as c
import trackpy


def track(chains: list, track_radius: int = c.TRACK_RADIUS,
          memory: int = 5,
          threshold: int = 0,
          adaptive_step: float = 0.8) -> pd.DataFrame:
    columns = ['frame', 'x', 'y', 'z', 'intensity']
    df_chains = pd.DataFrame(
        chains, columns=columns, dtype=float
    )

    tr = trackpy.link(df_chains[columns], track_radius,
                      memory=memory, adaptive_step=adaptive_step,
                      adaptive_stop=10)
    filtered = trackpy.filtering.filter_stubs(tr, threshold=threshold)
    filtered.reset_index(drop=True, inplace=True)

    return filtered


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
