import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

from ..data.dataset import Dataset
from ..data.types import Variable
from ..model import Model


cmap_nnp = LinearSegmentedColormap('nnp', {
    'red':   [(0.0, 237 / 255, 237 / 255), (0.5, 233 / 255, 233 / 255), (1.0, 39 / 255, 39 / 255)],
    'green': [(0.0, 153 / 255, 153 / 255), (0.5, 233 / 255, 233 / 255), (1.0, 122 / 255, 122 / 255)],
    'blue':  [(0.0, 65 / 255, 65 / 255), (0.5, 233 / 255, 233 / 255), (1.0, 185 / 255, 185 / 255)]
})


class TwoFeaturesModelVisualizer:
    POINT_SIZE = 26

    def __init__(self, model: Model, train_dataset: Dataset, test_dataset: Dataset, x1_range = (-6, 6, 100), x2_range = (-6, 6, 100), output_range = (-1, 1), z_func = None):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.x1_range = x1_range
        self.x2_range = x2_range
        self.z_func = z_func

        self.x1, self.x2 = np.meshgrid(np.linspace(*x1_range), np.linspace(*x2_range))

        # initialize the plot
        self.fig = plt.figure(constrained_layout = True, figsize = (20, 5))
        self.gs = GridSpec(2, 5, figure = self.fig, wspace = 0.1, hspace = 0.1)
        self.ax_data = self.fig.add_subplot(self.gs[0:2, 0:2])
        img = self.ax_data.imshow(np.zeros_like(self.x1), extent = (*x1_range[:2], *x2_range[:2]), origin = 'lower', cmap = cmap_nnp, vmin = output_range[0], vmax = output_range[1])
        plt.colorbar(img, ax = self.ax_data, label = "Model Output")
        self.ax_train_loss = self.fig.add_subplot(self.gs[0, 2:3])
        self.ax_train_accuracy = self.fig.add_subplot(self.gs[0, 3:4])
        self.ax_test_loss = self.fig.add_subplot(self.gs[1, 2:3])
        self.ax_test_accuracy = self.fig.add_subplot(self.gs[1, 3:4])

    def calculate_2d_output_mesh(self):
        """
            Model input and output dimensions should be 2 and 1 respectively.
            Or z_func (np.ndarray -> float) should be provided if output dimension is not 1.
        """
        # assert model.layers[0].n == 2 and (model.layers[-1].m == 1 or self.z_func is not None) # TODO: activation function
        mesh_features = np.c_[self.x1.ravel(), self.x2.ravel()]
        mesh_prediction = self.model(Variable(mesh_features, derivable = False))
        if self.z_func is not None:
            mesh_prediction = self.z_func(mesh_prediction)
        return mesh_prediction.value.reshape(self.x1.shape)
    
    def update(self, epoch, train_loss_history = [], train_accuracy_history = [], test_loss_history = [], test_accuracy_history = [], delay = 0.001):
        self.ax_data.clear()
        self.ax_data.scatter(self.train_dataset.data_raw[:, 0], self.train_dataset.data_raw[:, 1], c = self.train_dataset.data_raw[:, 2], s = self.POINT_SIZE, cmap = cmap_nnp, vmin = -1, vmax = 1, edgecolors = 'white', linewidths = 1)
        self.ax_data.scatter(self.test_dataset.data_raw[:, 0], self.test_dataset.data_raw[:, 1], c = self.test_dataset.data_raw[:, 2], s = self.POINT_SIZE, cmap = cmap_nnp, vmin = -1, vmax = 1, edgecolors = 'black', linewidths = 1)
        self.ax_data.imshow(self.calculate_2d_output_mesh(), extent = (*self.x1_range[:2], *self.x2_range[:2]), origin = 'lower', cmap = cmap_nnp, vmin = -1, vmax = 1)
        self.ax_data.set_aspect(1)
        self.ax_data.set_xlabel("Feature 1")
        self.ax_data.set_ylabel("Feature 2")
        self.ax_data.set_title(f"Epoch {epoch}")
        
        self.ax_train_loss.clear()
        self.ax_train_loss.plot(train_loss_history, color = 'black')
        self.ax_train_loss.set_xlabel("Epoch")
        self.ax_train_loss.set_ylabel("Train Loss")

        self.ax_train_accuracy.clear()
        self.ax_train_accuracy.plot(train_accuracy_history, color = 'black')
        self.ax_train_accuracy.set_xlabel("Epoch")
        self.ax_train_accuracy.set_ylabel("Train Accuracy")

        self.ax_test_loss.clear()
        self.ax_test_loss.plot(test_loss_history, color = 'black')
        self.ax_test_loss.set_xlabel("Epoch")
        self.ax_test_loss.set_ylabel("Test Loss")

        self.ax_test_accuracy.clear()
        self.ax_test_accuracy.plot(test_accuracy_history, color = 'black')
        self.ax_test_accuracy.set_xlabel("Epoch")
        self.ax_test_accuracy.set_ylabel("Test Accuracy")

        plt.pause(delay)

