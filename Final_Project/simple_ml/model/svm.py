"""
    Partially referenced from given source code.
"""

import numpy as np

from ..data.dataset import Dataset
from ..data.samples import label_split_for_single_output_dataset


class SMOSolver:
    def __init__(self,
                 Q: np.ndarray,
                 p: np.ndarray,
                 y: np.ndarray,
                 C: float,
                 tol: float = 1e-5) -> None:
        problem_size = p.shape[0]
        assert problem_size == y.shape[0]
        if Q is not None:
            assert problem_size == Q.shape[0]
            assert problem_size == Q.shape[1]

        self.Q = Q
        self.p = p
        self.y = y
        self.C = C
        self.tol = tol
        self.alpha = np.zeros(problem_size)

        self.neg_y_grad = -y * p

    def working_set_select(self):
        Iup = np.argwhere(
            np.logical_or(
                np.logical_and(self.alpha < self.C, self.y > 0),
                np.logical_and(self.alpha > 0, self.y < 0),
            )).flatten()
        Ilow = np.argwhere(
            np.logical_or(
                np.logical_and(self.alpha < self.C, self.y < 0),
                np.logical_and(self.alpha > 0, self.y > 0),
            )).flatten()

        find_fail = False
        try:
            i = Iup[np.argmax(self.neg_y_grad[Iup])]
            j = Ilow[np.argmin(self.neg_y_grad[Ilow])]
        except:
            find_fail = True

        if find_fail or self.neg_y_grad[i] - self.neg_y_grad[j] < self.tol:
            return -1, -1
        return i, j

    def update(self, i: int, j: int, func=None):
        Qi, Qj = self.get_Q(i, func), self.get_Q(j, func)
        yi, yj = self.y[i], self.y[j]
        alpha_i, alpha_j = self.alpha[i], self.alpha[j]

        quad_coef = Qi[i] + Qj[j] - 2 * yi * yj * Qi[j]
        if quad_coef <= 0:
            quad_coef = 1e-12

        if yi * yj == -1:
            delta = (self.neg_y_grad[i] * yi +
                     self.neg_y_grad[j] * yj) / quad_coef
            diff = alpha_i - alpha_j
            self.alpha[i] += delta
            self.alpha[j] += delta

            if diff > 0:
                if (self.alpha[j] < 0):
                    self.alpha[j] = 0
                    self.alpha[i] = diff

            else:
                if (self.alpha[i] < 0):
                    self.alpha[i] = 0
                    self.alpha[j] = -diff

            if diff > 0:
                if (self.alpha[i] > self.C):
                    self.alpha[i] = self.C
                    self.alpha[j] = self.C - diff

            else:
                if (self.alpha[j] > self.C):
                    self.alpha[j] = self.C
                    self.alpha[i] = self.C + diff

        else:
            delta = (self.neg_y_grad[j] * yj -
                     self.neg_y_grad[i] * yi) / quad_coef
            sum = self.alpha[i] + self.alpha[j]
            self.alpha[i] -= delta
            self.alpha[j] += delta

            if sum > self.C:
                if self.alpha[i] > self.C:
                    self.alpha[i] = self.C
                    self.alpha[j] = sum - self.C

            else:
                if self.alpha[j] < 0:
                    self.alpha[j] = 0
                    self.alpha[i] = sum

            if sum > self.C:
                if self.alpha[j] > self.C:
                    self.alpha[j] = self.C
                    self.alpha[i] = sum - self.C

            else:
                if self.alpha[i] < 0:
                    self.alpha[i] = 0
                    self.alpha[j] = sum

        delta_i = self.alpha[i] - alpha_i
        delta_j = self.alpha[j] - alpha_j
        self.neg_y_grad -= self.y * (delta_i * Qi + delta_j * Qj)
        return delta_i, delta_j

    def calculate_rho(self) -> float:
        sv = np.logical_and(
            self.alpha > 0,
            self.alpha < self.C,
        )
        if sv.sum() > 0:
            rho = -np.average(self.neg_y_grad[sv])
        else:
            ub_id = np.logical_or(
                np.logical_and(self.alpha == 0, self.y < 0),
                np.logical_and(self.alpha == self.C, self.y > 0),
            )
            lb_id = np.logical_or(
                np.logical_and(self.alpha == 0, self.y > 0),
                np.logical_and(self.alpha == self.C, self.y < 0),
            )
            try:
                rho = -(self.neg_y_grad[lb_id].min() + self.neg_y_grad[ub_id].max()) / 2
            except:
                rho = 0
        return rho

    def get_Q(self, i: int, func = None):
        return self.Q[i]


class SVM:
    def __init__(self, C = 1.0, max_iteration = 10000, tolerance = 1e-3, kernel = 'linear'):
        self.C = C
        self.max_iteration = max_iteration
        self.tolerance = tolerance
        self.kernel = kernel
    
    def register_kernal(self, std: float):
        if self.kernel == 'linear':
            return lambda x, y: np.matmul(x, y.T)
        elif self.kernel == 'gaussian':
            return lambda x, y: np.exp(-np.linalg.norm(x[:, None] - y, axis = 2) ** 2 / (2 * std ** 2))
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")

    def train(self, dataset: Dataset, positive_label = 1):
        X, y = dataset.datas[0], dataset.datas[1].flatten()
        y = np.where(y == positive_label, 1, -1)
        n_samples, n_features = X.shape

        p = -np.ones(n_samples)
        kernel_func = self.register_kernal(X.std())

        Q = y.reshape(-1, 1) * y * kernel_func(X, X)
        solver = SMOSolver(Q, p, y, self.C, self.tolerance)    

        def func(i):
            return y * kernel_func(X, X[i:i + 1]).flatten() * y[i]
        
        for _ in range(self.max_iteration):
            i, j = solver.working_set_select()
            if i < 0:
                break
            solver.update(i, j, func)
        
        self._decision_function = lambda x: np.matmul(solver.alpha * y, kernel_func(X, x)) - solver.calculate_rho()

    def predict(self, dataset: Dataset):
        X = dataset.datas[0]
        return (self._decision_function(X) >= 0).astype(int)


class MultiClassSVM:
    def __init__(self, C = 1.0, max_iteration = 10000, tolerance = 1e-3, kernel = 'linear'):
        self.C = C
        self.max_iteration = max_iteration
        self.tolerance = tolerance
        self.kernel = kernel
        self.models = {}

    def train(self, dataset: Dataset):
        X, y = dataset.datas[0], dataset.datas[1].flatten()
        self.classes = np.unique(y)

        for i, class_label in enumerate(self.classes):
            binary_y = np.where(y == class_label, 1, -1)
            binary_dataset = Dataset(data = np.hstack([X, binary_y[:, None]]), preprocess_func = label_split_for_single_output_dataset)

            svm = SVM(C = self.C, max_iteration = self.max_iteration, tolerance = self.tolerance, kernel = self.kernel)
            svm.train(binary_dataset, positive_label = 1)
            
            self.models[class_label] = svm

    def predict(self, dataset: Dataset):
        X = dataset.datas[0]
        decision_values = []

        for class_label, svm in self.models.items():
            decision_values.append(svm._decision_function(X))

        decision_values = np.array(decision_values).T
        predictions = self.classes[np.argmax(decision_values, axis = 1)]

        return predictions

