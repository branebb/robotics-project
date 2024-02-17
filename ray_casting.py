import numpy as np
# https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection


def ray_casting(lines1, lines2):

    indices1 = np.arange(lines1.shape[0]).repeat(lines2.shape[0])
    indices2 = np.tile(np.arange(lines2.shape[0]), lines1.shape[0])

    x1 = lines1[indices1, 0, 0]
    y1 = lines1[indices1, 0, 1]
    x2 = lines1[indices1, 1, 0]
    y2 = lines1[indices1, 1, 1]

    x3 = lines2[indices2, 0, 0]
    y3 = lines2[indices2, 0, 1]
    x4 = lines2[indices2, 1, 0]
    y4 = lines2[indices2, 1, 1]

    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    den[den == 0] = np.finfo(float).eps

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    u = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / den

    P = np.vstack([x1 + t * (x2 - x1), y1 + t * (y2 - y1)]).T

    P = P.reshape(lines1.shape[0], lines2.shape[0], 2)
    O = np.tile(lines2[:, 0], (lines1.shape[0], 1, 1))
    D = np.sqrt((P[:, :, 0] - O[:, :, 0]) ** 2 + (P[:, :, 1] - O[:, :, 1]) ** 2)

    valid_indices = ((0 <= t) & (t <= 1) & (0 <= u) & (u <= 1)).reshape(lines1.shape[0], lines2.shape[0])
    D[~valid_indices] = np.inf

    min_values = np.min(D, axis=0)
    min_indices = np.argmin(D, axis=0)
    min_pair_indices = np.vstack([min_indices, np.array(range(lines2.shape[0]))]).T
    mpi_valid = min_pair_indices[~np.isinf(min_values)]


    output = lines2.copy()
    output[~np.isinf(min_values), 1, :] = P[mpi_valid[:, 0], mpi_valid[:, 1]]
    return output


