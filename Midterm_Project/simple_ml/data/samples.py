import numpy as np


def label_split_for_single_output_dataset(data: np.ndarray):
    return [data[:, :-1], data[:, -1].reshape(-1, 1)] # [x_1, x_2], t


def generate_2d_classification_circle(N = 500, r_0 = 2.3, r_1 = 3.5, r_2 = 5, noise = 0, label_in = 1, label_out = -1, seed = None, **kwargs):
    if seed is not None:
        np.random.seed(seed)
    
    N_in = N // 2

    angles = np.random.uniform(-np.pi, np.pi, (N, 1))
    dists = np.r_[
        np.random.uniform(0, r_0, (N_in, 1)) + np.random.normal(0, noise, (N_in, 1)),
        np.random.uniform(r_1, r_2, (N - N_in, 1)) + np.random.normal(0, noise, (N - N_in, 1))
    ]
    coords = np.c_[np.cos(angles), np.sin(angles)] * dists

    labels = np.ones((N, 1))
    labels[:N_in] *= label_in
    labels[N_in:] *= label_out

    return np.c_[coords, labels]


def generate_2d_classification_exclusive_or(N = 500, l = 5.2, pad = 0.3, noise = 0, label_low = -1, label_high = 1, seed = None, **kwargs):
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


def generate_2d_classification_gaussians(gaussian_configs, seed = None, **kwargs):
    if seed is not None:
        np.random.seed(seed)

    data = np.zeros((0, 3))
    for mean, cov, N, label in gaussian_configs:
        data = np.r_[
            data,
            np.c_[np.random.multivariate_normal(mean, cov, N), np.ones((N, 1)) * label]
        ]

    return data


def generate_2d_classification_example_scatter(**kwargs):
    return np.array([
        [-3.2, 4.5, 1],
        [-2.6, 4.7, 1],
        [0.1, 4.2, 1],
        [0.3, 2.1, 1],
        [2.2, 3.2, 1],
        [4.6, 2.8, 1],
        [3.8, 1.4, 1],
        [4.9, 0.4, 1],
        [0.2, -0.05, 1],
        [1.8, -0.05, 1],
        [-0.3, -3.2, 1],
        [4, -1, 1],
        [5.3, -0.9, 1],
        [0.4, -2.95, 1],
        [2.4, -3, 1],
        [4.1, -3.1, 1],
        [1.6, -5, 1],
        [-0.4, 4.5, -1],
        [-1.8, 3.1, -1],
        [-3.2, 2, -1],
        [-3.35, 0.45, -1],
        [-2.1, 1.4, -1],
        [-0.1, 1.4, -1],
        [1.7, 2, -1],
        [0.05, -1.8, -1],
        [2.05, -1.6, -1],
        [1, -4.2, -1],
        [1.95, -3.3, -1],
        [-1.75, -0.3, -1],
        [-2.8, -0.38, -1],
        [-2, -2.1, -1],
        [-3.8, -2, -1]
    ])


def generate_1d_regression_with_function(N = 500, f = lambda x: np.cos(x) + np.exp(-x ** 2) + x ** 3 / 233, x_range = (-10, 10), noise = 1, seed = None):
    assert f is not None

    if seed is not None:
        np.random.seed(seed)

    x = np.random.uniform(*x_range, (N, 1))
    y = f(x) + np.random.normal(0, noise, (N, 1))

    return np.c_[x, y]

