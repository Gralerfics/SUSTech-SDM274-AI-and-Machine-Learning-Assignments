import numpy as np

# from sklearn.metrics import mean_squared_error

from ..data.dataset import *
from ..data.samples import *


class PCA:
    def __init__(self, n: int):
        self.n = n
        self.center = None
        self.U_pca = None
        # self.explained_variance = None

    def train(self, dataset: Dataset):
        X = dataset.datas[0]

        # center the datas
        self.center = np.mean(X, axis=0)
        X_centered = X - self.center

        # covariance matrix
        C = np.cov(X_centered.T)

        # decomposition
        eigenvalues, eigenvectors = np.linalg.eig(C)

        # sort eigenpairs by eigenvalues
        eigenpairs = [(np.abs(eigenvalues[i]), eigenvectors[:, i]) for i in range(len(eigenvalues))]
        eigenpairs.sort(key = lambda k: k[0], reverse = True) # in descending order

        # pick the top n components
        self.U_pca = np.hstack([eigenpairs[i][1][:, np.newaxis] for i in range(self.n)])
        # self.explained_variance = np.array([eigenpairs[i][0] for i in range(self.n)])

    def project(self, dataset: Dataset):
        X = dataset.datas[0]
        X_centered = X - self.center
        X_pca = X_centered.dot(self.U_pca)
        data = np.hstack([X_pca, dataset.datas[1]])
        return Dataset(data = data, preprocess_func = label_split_for_single_output_dataset)

    def reconstruct(self, dataset: Dataset):
        X_pca = dataset.datas[0]
        X_reconstructed = X_pca.dot(self.U_pca.T) + self.center
        data = np.hstack([X_reconstructed, dataset.datas[1]])
        return Dataset(data = data, preprocess_func = label_split_for_single_output_dataset)

    # def reconstruction_error(self, X):
    #     X_pca = self.project(X)
    #     X_reconstructed = self.reconstruct(X_pca)
    #     return mean_squared_error(X, X_reconstructed)

