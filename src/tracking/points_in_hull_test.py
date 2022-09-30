import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
# from pyhull.convex_hull import ConvexHull
import pandas as pd

def points_in_hull(p, hull, tol=1e-12):
    return np.all(hull.equations[:, :-1] @ p.T + np.repeat(hull.equations[:, -1][None, :], len(p), axis=0).T <= tol, 0)


if __name__ == '__main__':
    loops = pd.read_csv('/home/gummz/cell/data/interim/pred/pred_2/LI_2019-02-05_emb5_pos4/loops_LI_2019-02-05_emb5_pos4_t0-258.csv')
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    x = 10 * np.outer(np.cos(u), np.sin(v))
    y = 10 * np.outer(np.sin(u), np.sin(v))
    z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))
    x1 = 7 + 10 * np.outer(np.cos(u), np.sin(v))
    y1 = 7 + 10 * np.outer(np.sin(u), np.sin(v))
    z1 = 7 + 10 * np.outer(np.ones(np.size(u)), np.cos(v))
    n_hull = 300
    n_p = 500
    H = np.random.rand(n_hull, 3)
    H = np.array([x, y, z]).T
    P = np.random.rand(n_p, 3)

    hull = ConvexHull(H)
    in_hull = points_in_hull(P, hull)
    ax.scatter(H[:, 0], H[:, 1], H[:, 2], c='b')
    ax.scatter(P[in_hull, 0], P[in_hull, 1], P[in_hull, 2], c='g')
    ax.scatter(P[~in_hull, 0], P[~in_hull, 1], P[~in_hull, 2], c='r')
    plt.savefig('points_in_hull_test.png', dpi=300, bbox_inches='tight')
