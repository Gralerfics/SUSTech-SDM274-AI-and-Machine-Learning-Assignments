import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from simple_ml import Variable, Dataset, DataIterator, split_train_and_test_dataset
from simple_ml.data.samples import generate_2d_classification_circle
from simple_ml.model import Model, Sequential
from simple_ml.model.layers import Linear, ReLU, Sigmoid
from simple_ml.training.loss import MSELoss, CrossEntropyLoss
from simple_ml.training.optimizer import GD, Adam


# l = Linear(3, 2)
# print(l.W)

# x = Variable(np.array([[1, 2, 3], [4, 5, 6]]), derivable = True)
# y = x
# y.value = np.array([[1, 2, 3]])
# print(x)

# x = Variable(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), derivable = True)
# x[1, 1:] = [1, 2]
# print(x[1, :])
# print(x[1, :] @ x[1, :].T)
# print(x[1, :] @ [1, 2, 3])
# print(x[1, :] - 5)

# X = Variable(np.array([
#     [1, 2, 3],
#     [4, 5, 6],
#     [7, 8, 9]
# ]), derivable = True)

# T = np.array([
#     [1, 0, 0],
#     [0, 1, 0],
#     [0, 0, 1]
# ])

# model = Sequential([
#     Linear(13, 2),
#     Sigmoid(),
#     Linear(2, 3),
#     Sigmoid()
# ])

# def label_split(data: np.ndarray):
#     return data[:, 1:], np.eye(3)[data[:, 0].astype(int) - 1] # X, T

# dataset = Dataset(file_path = "wine.data", preprocess_func = label_split)
# train_dataset, test_dataset = split_train_and_test_dataset(dataset, 0.2, seed = 42)



data_np = generate_2d_classification_circle()
# plt.scatter(data_np[:, 0], data_np[:, 1], c = data_np[:, 2], cmap = 'coolwarm', vmin = -1, vmax = 1)
# plt.colorbar(label = 'Label')
# plt.xlabel("x_1")
# plt.ylabel("x_2")
# plt.show()

def label_split(data: np.ndarray):
    return data[:, :-1], data[:, -1].reshape(-1, 1) # [x_0, x_1], t

dataset = Dataset(data = data_np, preprocess_func = label_split)
train_dataset, test_dataset = split_train_and_test_dataset(dataset, 0.2)
train_iter = DataIterator(train_dataset, batch_size = 20, shuffle = True, cyclic = False)

model = Sequential([
    Linear(2, 8),
    Sigmoid(),
    Linear(8, 8),
    Sigmoid(),
    Linear(8, 1)
])

criterion = MSELoss()
# optimizer = GD(model.params, lr = 0.01)
optimizer = Adam(model.params, lr = 0.01)

epoch_num = 1000
train_loss_history = []

# PLOT
x1, x2 = np.meshgrid(np.linspace(-6, 6, 200), np.linspace(-6, 6, 200))
mesh_features = np.c_[x1.ravel(), x2.ravel()]

cdict = {
    'red':   [(0.0, 237 / 255, 237 / 255), (0.5, 233 / 255, 233 / 255), (1.0, 39 / 255, 39 / 255)],
    'green': [(0.0, 153 / 255, 153 / 255), (0.5, 233 / 255, 233 / 255), (1.0, 122 / 255, 122 / 255)],
    'blue':  [(0.0, 65 / 255, 65 / 255), (0.5, 233 / 255, 233 / 255), (1.0, 185 / 255, 185 / 255)]
}
cmap = LinearSegmentedColormap("my_cmap", cdict)
# cmap = 'coolwarm'

fig, ax = plt.subplots()
scatter = ax.scatter([], [], c = [], s = 30, cmap = cmap, vmin = -1, vmax = 1, edgecolors = 'white', linewidths = 1)
img = ax.imshow(np.zeros_like(x1), extent = (-6, 6, -6, 6), origin = 'lower', cmap = cmap, vmin = -1, vmax = 1)
plt.colorbar(img, label = "Model Output")
plt.xlabel("x_1")
plt.ylabel("x_2")
# PLOT

for epoch in range(epoch_num):
    train_loss = 0

    for batch, [features, labels] in enumerate(train_iter):
        mesh_prediction = model(Variable(features, derivable = True))
        loss = criterion(mesh_prediction, labels)

        model.backward()
        optimizer.step()

        train_loss += loss
    
    train_loss_history.append(train_loss / len(dataset))
    # print(train_loss / len(dataset))

    # PLOT
    ax.clear()
    
    ax.scatter(data_np[:, 0], data_np[:, 1], c = data_np[:, 2], s = 30, cmap = cmap, vmin = -1, vmax = 1, edgecolors = 'white', linewidths = 1)

    mesh_prediction = model(Variable(mesh_features, derivable = False))
    z = mesh_prediction.value.reshape(x1.shape)
    
    img = ax.imshow(z, extent = (-6, 6, -6, 6), origin = 'lower', cmap = cmap, vmin = -1, vmax = 1)
    ax.set_title(f"Epoch {epoch + 1}")

    plt.pause(0.001)
    # PLOT

plt.show()


# plt.scatter(data_np[:, 0], data_np[:, 1], c = data_np[:, 2], cmap = 'coolwarm', vmin = -1, vmax = 1)
# plt.imshow(z, extent = (-6, 6, -6, 6), origin = 'lower', cmap = 'coolwarm', vmin = -1, vmax = 1)
# plt.colorbar(label = "Model Output")
# plt.xlabel("x_1")
# plt.ylabel("x_2")
# plt.show()

# plt.plot(np.arange(len(train_loss_history)), train_loss_history)
# plt.show()

# test_features, test_labels = test_dataset.datas
# prediction = model(Variable(test_features, derivable = True))
# test_loss = criterion(prediction, test_labels)
# print(prediction.value)
# print(test_loss)


# print("Loss value:\n", loss, "\n")

# print("out.value:\n", model.output.value, "\n")
# print("out.gradient:\n", model.output.gradient, "\n")

# print("Y1.value:\n", model.layers[3].input.value, "\n")
# print("Y1.gradient:\n", model.layers[3].input.gradient, "\n")

# print("W1.value:\n", model.layers[2].W.value, "\n")
# print("W1.gradient:\n", model.layers[2].W.gradient, "\n")

# print("b1.value:\n", model.layers[2].b.value, "\n")
# print("b1.gradient:\n", model.layers[2].b.gradient, "\n")

# print("X1.value:\n", model.layers[2].input.value, "\n")
# print("X1.gradient:\n", model.layers[2].input.gradient, "\n")

# print("Y0.value:\n", model.layers[1].input.value, "\n")
# print("Y0.gradient:\n", model.layers[1].input.gradient, "\n")

# print("W0.value:\n", model.layers[0].W.value, "\n")
# print("W0.gradient:\n", model.layers[0].W.gradient, "\n")

# print("b0.value:\n", model.layers[0].b.value, "\n")
# print("b0.gradient:\n", model.layers[0].b.gradient, "\n")

# print("X.value:\n", model.input.value, "\n")
# print("X.gradient:\n", model.input.gradient, "\n")


# def label_split(data: np.ndarray):
#     return data[:, 1:], data[:, 0].astype(int) # X, T

# dataset = Dataset(file_path = "wine.data", preprocess_func = label_split)
# data_iter = DataIterator(dataset, batch_size = 10, shuffle = False, cyclic = False)

# for i, [x, t] in enumerate(data_iter):
#     print(i, x[:, :2], t)
#     if i >= 17:
#         break

