import os
import random
from typing import Union

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from ..data.types import Variable


class Dataset:
    def __init__(self, **kwargs):
        self.data_raw: np.ndarray = kwargs.get('data', None) # the whole table
        self.file_path: str = kwargs.get('file_path', None)
        assert (self.data_raw is not None and self.file_path is None) or (self.data_raw is None and self.file_path is not None) # only one of them should be provided

        self.preprocess_func = kwargs.get('preprocess_func', None)
        if self.preprocess_func is None:
            self.preprocess_func = self.default_preprocess_func

        if self.file_path is not None:
            # from disk
            file_type = kwargs.get('file_type', 'csv')
            if file_type == 'csv':
                # csv
                header = kwargs.get('header', None) # no header in default
                self.data_raw = pd.read_csv(self.file_path, header = header).to_numpy() # TODO: header
            else:
                pass # TODO: other file types
        
        self.datas: Union[np.ndarray, list[np.ndarray]] = self.preprocess_func(self.data_raw)
            # np.ndarray for the case of only one data source, list[np.ndarray] for the case of multiple data sources
    
    """ @Override """
    def default_preprocess_func(self, data: np.ndarray):
        pass # modify/fill data (the reference to self.data_raw)
        return data # return a list of spilt tables (each of them is a view sliced from self.data_raw, so self.data_raw is not needed to be deleted)
            # only one element, no list wrapping

    def __len__(self):
        return self.data_raw.shape[0]

    def __getitem__(self, index):
        if isinstance(self.datas, np.ndarray):
            return self.datas[index]
        elif isinstance(self.datas, list) or isinstance(self.datas, tuple):
            return [d[index] for d in self.datas]


class DataIterator:
    def __init__(self, dataset, batch_size = 1, shuffle = False, cyclic = False):
        self.dataset: Dataset = dataset
        self.batch_size: int = min(batch_size, len(dataset)) if batch_size is not None else len(dataset)
        self.shuffle: bool = shuffle
        self.cyclic: bool = cyclic

        self.indices: list = list(range(len(dataset)))
        self.next_idx: int = batch_size

        if shuffle:
            random.shuffle(self.indices)
    
    def reset(self):
        self.next_idx = self.batch_size
        if self.shuffle:
            random.shuffle(self.indices)

    def __iter__(self):
        return self

    def __next__(self):
        exceed = self.next_idx - len(self.dataset)
        if exceed >= 0:
            if exceed >= self.batch_size: # did not cycle to the beginning and exceed (StopIteration)
                self.reset()
                raise StopIteration
            batch_indices = self.indices[(self.next_idx - self.batch_size):] # collect left samples
            if self.cyclic: # cyclic, complete the batch
                self.next_idx = exceed
                if self.shuffle:
                    random.shuffle(self.indices)
                batch_indices.extend(self.indices[:exceed])
        else:
            batch_indices = self.indices[(self.next_idx - self.batch_size):self.next_idx]
        
        self.next_idx += self.batch_size
        batch_data = self.dataset[batch_indices]
        return batch_data # return in np.ndarray


def merge_datasets(datasets):
    data_raw = np.concatenate([dataset.data_raw for dataset in datasets], axis = 0)
    preprocess_func = datasets[0].preprocess_func
    return Dataset(data = data_raw, preprocess_func = preprocess_func)


def split_train_and_test_dataset(dataset, test_ratio = 0.2, seed = None):
    data_raw = dataset.data_raw.copy()
    preprocess_func = dataset.preprocess_func
    del dataset

    N = data_raw.shape[0]

    if seed is not None:
        np.random.seed(seed)
    indices = np.arange(N)
    np.random.shuffle(indices)

    split_index = int(N * (1 - test_ratio))
    train_indices, test_indices = indices[:split_index], indices[split_index:]

    train_dataset = Dataset(data = data_raw[train_indices], preprocess_func = preprocess_func)
    test_dataset = Dataset(data = data_raw[test_indices], preprocess_func = preprocess_func)

    return train_dataset, test_dataset


def split_train_and_test_dataset_with_equal_binary_label(dataset, test_ratio = 0.2, seed = None):
    data_raw = dataset.data_raw.copy()
    preprocess_func = dataset.preprocess_func
    del dataset

    label_0_data = data_raw[data_raw[:, -1] == 0]
    label_1_data = data_raw[data_raw[:, -1] == 1]

    if seed is not None:
        np.random.seed(seed)

    np.random.shuffle(label_0_data)
    np.random.shuffle(label_1_data)

    split_index_0 = int(label_0_data.shape[0] * (1 - test_ratio))
    split_index_1 = int(label_1_data.shape[0] * (1 - test_ratio))

    train_data_0, test_data_0 = label_0_data[:split_index_0], label_0_data[split_index_0:]
    train_data_1, test_data_1 = label_1_data[:split_index_1], label_1_data[split_index_1:]

    train_data = np.vstack([train_data_0, train_data_1])
    test_data = np.vstack([test_data_0, test_data_1])

    np.random.shuffle(train_data)
    np.random.shuffle(test_data)

    train_dataset = Dataset(data = train_data, preprocess_func = preprocess_func)
    test_dataset = Dataset(data = test_data, preprocess_func = preprocess_func)

    return train_dataset, test_dataset


def split_k_fold_cross_validation_dataset(dataset, k = 5, seed = None):
    data_raw = dataset.data_raw.copy()
    preprocess_func = dataset.preprocess_func
    del dataset

    N = data_raw.shape[0]

    if seed is not None:
        np.random.seed(seed)
    indices = np.arange(N)
    np.random.shuffle(indices)

    fold_size = N // k
    fold_indices = [indices[i * fold_size:(i + 1) * fold_size] for i in range(k - 1)]
    fold_indices.append(indices[(k - 1) * fold_size:])

    return [Dataset(data = data_raw[fold_indices[i]], preprocess_func = preprocess_func) for i in range(k)]


def balance_binary_labels_random_repeat(dataset, label_pos = 1, label_neg = 0):
    """
        Balance the number of positive and negative samples by randomly repeating samples.
    """
    data_raw = dataset.data_raw.copy()
    preprocess_func = dataset.preprocess_func

    pos_samples = data_raw[data_raw[:, -1] == label_pos]
    neg_samples = data_raw[data_raw[:, -1] == label_neg]

    num_pos = pos_samples.shape[0]
    num_neg = neg_samples.shape[0]

    if num_pos == num_neg:
        return dataset

    if num_pos < num_neg:
        oversampled_pos = pos_samples[np.random.choice(num_pos, num_neg, replace = True)]
        balanced_data = np.vstack((oversampled_pos, neg_samples))
    else:
        oversampled_neg = neg_samples[np.random.choice(num_neg, num_pos, replace = True)]
        balanced_data = np.vstack((pos_samples, oversampled_neg))

    np.random.shuffle(balanced_data)

    balanced_dataset = Dataset(data = balanced_data, preprocess_func = preprocess_func)

    return balanced_dataset


def balance_binary_labels_random_remove(dataset, label_pos = 1, label_neg = 0):
    """
        Balance the number of positive and negative samples by randomly removing samples.
    """
    data_raw = dataset.data_raw.copy()
    preprocess_func = dataset.preprocess_func

    pos_samples = data_raw[data_raw[:, -1] == label_pos]
    neg_samples = data_raw[data_raw[:, -1] == label_neg]

    num_pos = pos_samples.shape[0]
    num_neg = neg_samples.shape[0]

    if num_pos == num_neg:
        return dataset

    if num_pos > num_neg:
        sampled_pos = pos_samples[np.random.choice(num_pos, num_neg, replace = False)]
        balanced_data = np.vstack((sampled_pos, neg_samples))
    else:
        sampled_neg = neg_samples[np.random.choice(num_neg, num_pos, replace = False)]
        balanced_data = np.vstack((pos_samples, sampled_neg))

    np.random.shuffle(balanced_data)

    balanced_dataset = Dataset(data = balanced_data, preprocess_func = preprocess_func)

    return balanced_dataset


def balance_binary_labels_cluster_downsample(dataset, label_pos = 1, label_neg = 0, n_clusters = None):
    """
        Balance the number of positive and negative samples by clustering and downsampling.
    """
    data_raw = dataset.data_raw.copy()
    preprocess_func = dataset.preprocess_func

    pos_samples = data_raw[data_raw[:, -1] == label_pos]
    neg_samples = data_raw[data_raw[:, -1] == label_neg]

    num_pos = pos_samples.shape[0]
    num_neg = neg_samples.shape[0]

    if num_pos == num_neg:
        return dataset

    target_count = min(num_pos, num_neg) if n_clusters is None else n_clusters

    def cluster_downsample(samples, target_count):
        if samples.shape[0] <= target_count:
            return samples

        features = samples[:, :-1]
        kmeans = KMeans(n_clusters=target_count, random_state=42)
        cluster_labels = kmeans.fit_predict(features)

        downsampled_samples = []
        for cluster_idx in range(target_count):
            cluster_points = samples[cluster_labels == cluster_idx]
            cluster_center = kmeans.cluster_centers_[cluster_idx]
            closest_point_idx = np.argmin(np.linalg.norm(cluster_points[:, :-1] - cluster_center, axis=1))
            downsampled_samples.append(cluster_points[closest_point_idx])

        return np.array(downsampled_samples)

    if num_pos > num_neg:
        pos_samples = cluster_downsample(pos_samples, target_count)
    else:
        neg_samples = cluster_downsample(neg_samples, target_count)

    balanced_data = np.vstack((pos_samples, neg_samples))
    np.random.shuffle(balanced_data)

    balanced_dataset = Dataset(data=balanced_data, preprocess_func=preprocess_func)

    return balanced_dataset


def balance_binary_labels_combined(dataset, label_pos = 1, label_neg = 0, n_clusters = None, oversample_factor = 10):
    """
        Balance the number of positive and negative samples by combining random oversampling and cluster-based downsampling.
    """
    data_raw = dataset.data_raw.copy()
    preprocess_func = dataset.preprocess_func

    pos_samples = data_raw[data_raw[:, -1] == label_pos]
    neg_samples = data_raw[data_raw[:, -1] == label_neg]

    num_pos = pos_samples.shape[0]
    num_neg = neg_samples.shape[0]

    if num_pos == num_neg:
        return dataset

    # random oversampling
    oversampled_pos_samples = []
    for _ in range(oversample_factor):
        indices = np.random.choice(num_pos, num_pos, replace=True)
        oversampled_pos_samples.append(pos_samples[indices])
    pos_samples = np.vstack(oversampled_pos_samples)

    # cluster-based downsampling
    target_neg_count = pos_samples.shape[0]
    if n_clusters is None:
        n_clusters = target_neg_count

    def cluster_downsample(samples, target_count):
        if samples.shape[0] <= target_count:
            return samples

        features = samples[:, :-1]
        kmeans = KMeans(n_clusters=target_count, random_state=42)
        cluster_labels = kmeans.fit_predict(features)

        downsampled_samples = []
        for cluster_idx in range(target_count):
            cluster_points = samples[cluster_labels == cluster_idx]
            cluster_center = kmeans.cluster_centers_[cluster_idx]
            closest_point_idx = np.argmin(np.linalg.norm(cluster_points[:, :-1] - cluster_center, axis=1))
            downsampled_samples.append(cluster_points[closest_point_idx])

        return np.array(downsampled_samples)

    neg_samples = cluster_downsample(neg_samples, n_clusters)

    balanced_data = np.vstack((pos_samples, neg_samples))
    np.random.shuffle(balanced_data)

    balanced_dataset = Dataset(data=balanced_data, preprocess_func=preprocess_func)

    return balanced_dataset


def balance_binary_labels_smote(dataset, label_pos = 1, label_neg = 0, k_neighbors = 5, target_ratio = 1.0):
    """
        Balance the number of positive and negative samples using SMOTE.
    """
    data_raw = dataset.data_raw.copy()
    preprocess_func = dataset.preprocess_func

    pos_samples = data_raw[data_raw[:, -1] == label_pos]
    neg_samples = data_raw[data_raw[:, -1] == label_neg]

    num_pos = pos_samples.shape[0]
    num_neg = neg_samples.shape[0]

    if num_pos / num_neg >= target_ratio:
        return dataset

    target_pos_count = int(num_neg * target_ratio)
    synthetic_sample_count = target_pos_count - num_pos

    # KNN
    features = pos_samples[:, :-1]
    nn = NearestNeighbors(n_neighbors = k_neighbors).fit(features)
    neighbors = nn.kneighbors(features, return_distance = False)

    synthetic_samples = []
    for _ in range(synthetic_sample_count):
        idx = np.random.randint(0, num_pos)
        neighbor_idx = np.random.choice(neighbors[idx, 1:])

        sample1 = features[idx]
        sample2 = features[neighbor_idx]

        # interpolation
        diff = sample2 - sample1
        synthetic_sample = sample1 + np.random.rand() * diff
        synthetic_samples.append(np.hstack((synthetic_sample, label_pos)))

    synthetic_samples = np.array(synthetic_samples)

    balanced_data = np.vstack((pos_samples, synthetic_samples, neg_samples))
    np.random.shuffle(balanced_data)

    balanced_dataset = Dataset(data=balanced_data, preprocess_func=preprocess_func)

    return balanced_dataset

