import numpy as np

import matplotlib.pyplot as plt
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
        self.fig, (self.ax_data, self.ax_train_loss) = plt.subplots(1, 2, figsize = (15, 5))
        self.fig.subplots_adjust(wspace = 0.3)
        
        self.ax_data.scatter([], [], c = [], s = self.POINT_SIZE, cmap = cmap_nnp, vmin = output_range[0], vmax = output_range[1], edgecolors = 'white', linewidths = 1)
        self.ax_data.scatter([], [], c = [], s = self.POINT_SIZE, cmap = cmap_nnp, vmin = output_range[0], vmax = output_range[1], edgecolors = 'black', linewidths = 1)
        img = self.ax_data.imshow(np.zeros_like(self.x1), extent = (*x1_range[:2], *x2_range[:2]), origin = 'lower', cmap = cmap_nnp, vmin = output_range[0], vmax = output_range[1])
        plt.colorbar(img, ax = self.ax_data, label = "Model Output")
        self.ax_data.set_aspect(1)
        self.ax_data.set_xlabel("Feature 1")
        self.ax_data.set_ylabel("Feature 2")

        self.ax_train_loss.set_xlabel("Epoch")
        self.ax_train_loss.set_ylabel("Training Loss")

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
    
    def update(self, current_epoch, train_loss_history, delay = 0.001):
        self.ax_data.clear()
        self.ax_data.scatter(self.train_dataset.data_raw[:, 0], self.train_dataset.data_raw[:, 1], c = self.train_dataset.data_raw[:, 2], s = self.POINT_SIZE, cmap = cmap_nnp, vmin = -1, vmax = 1, edgecolors = 'white', linewidths = 1)
        self.ax_data.scatter(self.test_dataset.data_raw[:, 0], self.test_dataset.data_raw[:, 1], c = self.test_dataset.data_raw[:, 2], s = self.POINT_SIZE, cmap = cmap_nnp, vmin = -1, vmax = 1, edgecolors = 'black', linewidths = 1)
        
        img = self.ax_data.imshow(self.calculate_2d_output_mesh(), extent = (*self.x1_range[:2], *self.x2_range[:2]), origin = 'lower', cmap = cmap_nnp, vmin = -1, vmax = 1)
        self.ax_data.set_title(f"Epoch {current_epoch}")

        self.ax_train_loss.clear()
        self.ax_train_loss.plot(train_loss_history, color = 'black')
        self.ax_train_loss.set_title("Train Loss")

        plt.pause(delay)

