import numpy as np
import pandas as pd
import src.data.utils.utils as utils
import src.data.constants as c
import trackpy
from os.path import join
import pickle


def track(chains: list):
    print(len(chains), len(chains[0]))

    df_chains = pd.DataFrame(chains, columns=['t', 'x', 'y', 'z', 'intensity'])
    tr = trackpy.link(df_chains[['t, x, y, z']], 5)

    return list(tr.groupby('t'))
    '''
    Need to find the constant velocity of the tissue first,
    by looking at the embryo in question over timepoints.
    '''


if __name__ == '__main__':
    pass
