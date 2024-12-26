import numpy as np

from ..data.dataset import *


class KMeans:
    def __init__(self, k, max_iteration = 1000, tolerance = 1e-6):
        self.k = k
        self.max_iteration = max_iteration
        self.tolerance = tolerance

        self.centroids = None
        self.labels = None

    def _choose_initial_centroids(self, features: np.ndarray):
        n_samples = features.shape[0]
        centroids = []

        # randomly select the first centroid
        first_centroid_idx = np.random.randint(n_samples)
        centroids.append(features[first_centroid_idx])

        # select the farthest centroid iteratively
        for _ in range(1, self.k):
            distances = np.min(
                [np.linalg.norm(features - centroid, axis = 1) for centroid in centroids], axis = 0
            )
            probabilities = distances / np.sum(distances)
            next_centroid_idx = np.random.choice(n_samples, p = probabilities)
            centroids.append(features[next_centroid_idx])

        return np.array(centroids)
    
    def train(self, dataset: Dataset):
        features = dataset.datas[0]
        labels = dataset.datas[1]

        n_samples = features.shape[0]

        # initialize centroids
        self.centroids = self._choose_initial_centroids(features)

        for _ in range(self.max_iteration):
            # assign each sample to the nearest centroid
            distances = np.array([
                np.linalg.norm(features - centroid, axis = 1) for centroid in self.centroids
            ])
            cluster_assignments = np.argmin(distances, axis = 0)

            # update centroids
            new_centroids = []
            for i in range(self.k):
                cluster_points = features[cluster_assignments == i]
                if len(cluster_points) > 0:
                    new_centroids.append(np.mean(cluster_points, axis = 0))
                else:
                    # handle empty clusters, TODO: is necessary?
                    new_centroids.append(features[np.random.randint(n_samples)])

            new_centroids = np.array(new_centroids)

            # Check for convergence
            if np.allclose(self.centroids, new_centroids, atol = self.tolerance):
                break

            self.centroids = new_centroids

        # assign labels
        self.labels = np.zeros(self.k, dtype = int)
        for i in range(self.k):
            cluster_points = labels[cluster_assignments == i].T[0].astype(int)
            if len(cluster_points) > 0:
                self.labels[i] = np.bincount(cluster_points).argmax()

    def predict(self, input_features: np.ndarray):
        distances = np.array([
            np.linalg.norm(input_features - centroid, axis = 1) for centroid in self.centroids
        ])
        cluster_assignments = np.argmin(distances, axis = 0)
        return np.array([self.labels[cluster] for cluster in cluster_assignments])


class SoftKMeans:
    def __init__(self, k, max_iteration = 1000, tolerance = 1e-6, beta = 1.0):
        self.k = k
        self.max_iteration = max_iteration
        self.tolerance = tolerance
        self.beta = beta

        self.centroids = None
        self.membership_probabilities = None
        self.labels = None

    def _choose_initial_centroids(self, features: np.ndarray):
        n_samples = features.shape[0]
        centroids = []

        # randomly select the first centroid
        first_centroid_idx = np.random.randint(n_samples)
        centroids.append(features[first_centroid_idx])

        # select the farthest centroid iteratively
        for _ in range(1, self.k):
            distances = np.min(
                [np.linalg.norm(features - centroid, axis = 1) for centroid in centroids], axis = 0
            )
            probabilities = distances / np.sum(distances)
            next_centroid_idx = np.random.choice(n_samples, p = probabilities)
            centroids.append(features[next_centroid_idx])

        return np.array(centroids)

    def train(self, dataset: Dataset):
        features = dataset.datas[0]
        labels = dataset.datas[1]

        n_samples = features.shape[0]
        
        # initialize centroids
        self.centroids = self._choose_initial_centroids(features)

        for _ in range(self.max_iteration):
            # calculate distances to centroids
            distances = np.array([
                np.linalg.norm(features - centroid, axis = 1) for centroid in self.centroids
            ])

            # compute membership probabilities using softmax-like formula
            exp_values = np.exp(-self.beta * distances)
            self.membership_probabilities = exp_values / np.sum(exp_values, axis = 0, keepdims = True)

            # update centroids as weighted mean of points
            new_centroids = np.array([
                np.sum(self.membership_probabilities[i][:, np.newaxis] * features, axis = 0) / np.sum(self.membership_probabilities[i])
                for i in range(self.k)
            ])

            # check for convergence
            if np.allclose(self.centroids, new_centroids, atol = self.tolerance):
                break

            self.centroids = new_centroids

        # assign labels based on majority vote within each cluster
        self.labels = np.zeros(self.k, dtype = int)
        cluster_assignments = np.argmax(self.membership_probabilities, axis = 0)
        for i in range(self.k):
            cluster_points = labels[cluster_assignments == i].T[0].astype(int)
            if len(cluster_points) > 0:
                self.labels[i] = np.bincount(cluster_points).argmax()

    def predict(self, input_features: np.ndarray):
        distances = np.array([
            np.linalg.norm(input_features - centroid, axis = 1) for centroid in self.centroids
        ])
        
        exp_values = np.exp(-self.beta * distances)
        membership_probabilities = exp_values / np.sum(exp_values, axis = 0, keepdims = True)

        cluster_assignments = np.argmax(membership_probabilities, axis = 0)
        return np.array([self.labels[cluster] for cluster in cluster_assignments])

