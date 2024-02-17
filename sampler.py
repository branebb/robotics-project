import numpy as np


def sampler(weights):
    indices = np.zeros(weights.shape, dtype=int)
    M = weights.shape[0]
    index = int(np.random.random() * M)
    beta = 0
    weight_max = np.max(weights)
    for i in range(M):
        beta += np.random.random() * 2 * weight_max
        while weights[index] < beta:
            beta -= weights[index]
            index = (index + 1) % M
        indices[i] = index
    return indices

