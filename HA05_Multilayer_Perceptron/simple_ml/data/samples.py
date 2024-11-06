import numpy as np


def generate_2d_classification_circle(N = 500, r_0 = 2.3, r_1 = 3.5, r_2 = 5, noise = 0, label_in = 1, label_out = -1, seed = None):
    if seed is not None:
        np.random.seed(seed)
    
    N_in = N // 2

    angles = np.random.uniform(-np.pi, np.pi, (N, 1))
    dists = np.vstack((
        np.random.uniform(0, r_0, (N_in, 1)),
        np.random.uniform(r_1, r_2, (N - N_in, 1))
    ))
    coords = np.hstack((np.cos(angles), np.sin(angles))) * dists

    labels = np.ones((N, 1))
    labels[:N_in] *= label_in
    labels[N_in:] *= label_out

    return np.hstack((coords, labels))

