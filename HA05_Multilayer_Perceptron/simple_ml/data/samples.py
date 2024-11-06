import numpy as np


def label_split_for_2d_classification_dataset(data: np.ndarray):
    return data[:, :-1], data[:, -1].reshape(-1, 1) # [x_1, x_2], t


def generate_2d_classification_circle(N = 500, r_0 = 2.3, r_1 = 3.5, r_2 = 5, noise = 0, label_in = 1, label_out = -1, seed = None):
    if seed is not None:
        np.random.seed(seed)
    
    N_in = N // 2

    angles = np.random.uniform(-np.pi, np.pi, (N, 1))
    dists = np.r_[
        np.random.uniform(0, r_0, (N_in, 1)),
        np.random.uniform(r_1, r_2, (N - N_in, 1))
    ]
    coords = np.c_[np.cos(angles), np.sin(angles)] * dists

    labels = np.ones((N, 1))
    labels[:N_in] *= label_in
    labels[N_in:] *= label_out

    return np.c_[coords, labels]


def generate_2d_classification_exclusive_or(N = 500, l = 5.2, pad = 0.3, noise = 0, label_low = -1, label_high = 1, seed = None):
    if seed is not None:
        np.random.seed(seed)

    N_q = N // 4
    N_half = N_q * 2
    N_tq = N_q * 3

    coords = np.random.uniform(pad, l, (N, 2))
    coords[N_q:N_half, :] *= -1 # --
    coords[N_half:N_tq, 0] *= -1 # -+
    coords[N_tq:, 1] *= -1 # +-

    labels = np.ones((N, 1))
    labels[:N_half] *= label_high
    labels[N_half:] *= label_low

    return np.c_[coords, labels]

