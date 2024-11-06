import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from simple_ml import Variable, Dataset, DataIterator, split_train_and_test_dataset
from simple_ml.data.samples import generate_2d_classification_circle, generate_2d_classification_exclusive_or
from simple_ml.model import Model, Sequential
from simple_ml.model.layers import Linear, ReLU, Sigmoid, Tanh
from simple_ml.training.loss import MSELoss, CrossEntropyLoss
from simple_ml.training.optimizer import GD, Adam


data_np = generate_2d_classification_circle()
# data_np = generate_2d_classification_exclusive_or()

def label_split(data: np.ndarray):
    return data[:, :-1], data[:, -1].reshape(-1, 1) # [x_0, x_1], t

dataset = Dataset(data = data_np, preprocess_func = label_split)
train_dataset, test_dataset = split_train_and_test_dataset(dataset, 0.2)
train_iter = DataIterator(train_dataset, batch_size = 10, shuffle = True, cyclic = False)

# model = Sequential([
#     Linear(2, 8),
#     Tanh(),
#     Linear(8, 8),
#     Tanh(),
#     Linear(8, 1)
# ])

# model = Sequential([
#     Linear(2, 4),
#     Tanh(),
#     Linear(4, 2),
#     Tanh(),
#     Linear(2, 1)
# ])

model = Sequential([
    Linear(2, 4),
    ReLU(),
    Linear(4, 2),
    ReLU(),
    Linear(2, 1)
])

criterion = MSELoss()
optimizer = GD(model.params, lr = 0.01)
# optimizer = Adam(model.params, lr = 0.008)

epoch_num = 10000
train_loss_history = []

# PLOT
x1, x2 = np.meshgrid(np.linspace(-6, 6, 100), np.linspace(-6, 6, 100))
mesh_features = np.c_[x1.ravel(), x2.ravel()]

cdict = {
    'red':   [(0.0, 237 / 255, 237 / 255), (0.5, 233 / 255, 233 / 255), (1.0, 39 / 255, 39 / 255)],
    'green': [(0.0, 153 / 255, 153 / 255), (0.5, 233 / 255, 233 / 255), (1.0, 122 / 255, 122 / 255)],
    'blue':  [(0.0, 65 / 255, 65 / 255), (0.5, 233 / 255, 233 / 255), (1.0, 185 / 255, 185 / 255)]
}
cmap = LinearSegmentedColormap("my_cmap", cdict)

fig, (ax_data, ax_loss) = plt.subplots(1, 2, figsize = (15, 5))
fig.subplots_adjust(wspace = 0.3)
ax_data.set_aspect(1)

scatter = ax_data.scatter([], [], c = [], s = 30, cmap = cmap, vmin = -1, vmax = 1, edgecolors = 'white', linewidths = 1)
img = ax_data.imshow(np.zeros_like(x1), extent = (-6, 6, -6, 6), origin = 'lower', cmap = cmap, vmin = -1, vmax = 1)
plt.colorbar(img, ax = ax_data, label = "Model Output")
ax_data.set_xlabel("x_1")
ax_data.set_ylabel("x_2")
ax_loss.set_xlabel("Epoch")
ax_loss.set_ylabel("Training Loss")
# PLOT

for epoch in range(epoch_num):
    train_loss = 0

    for batch, [features, labels] in enumerate(train_iter):
        mesh_prediction = model(Variable(features, derivable = True)) # must be wrapped by Variable
        loss = criterion(mesh_prediction, labels) # calculate loss and gradient (!)

        model.backward()
        optimizer.step()

        train_loss += loss
    
    train_loss_history.append(train_loss / len(dataset))

    # PLOT
    ax_data.clear()

    ax_data.scatter(data_np[:, 0], data_np[:, 1], c = data_np[:, 2], s = 30, cmap = cmap, vmin = -1, vmax = 1, edgecolors = 'white', linewidths = 1)
    mesh_prediction = model(Variable(mesh_features, derivable = False))
    z = mesh_prediction.value.reshape(x1.shape)
    img = ax_data.imshow(z, extent = (-6, 6, -6, 6), origin = 'lower', cmap = cmap, vmin = -1, vmax = 1)
    ax_data.set_title(f"Epoch {epoch + 1}")

    ax_loss.clear()
    ax_loss.plot(train_loss_history, color = 'black')
    ax_loss.set_title("Training Loss")

    plt.pause(0.001)
    # PLOT

plt.show()


# 数据集 -> 训练集 + 验证集
# 超参搜索：
#     重复搜索范围次：
#         交叉验证：
#             选取超参数
#             重复 k 次：
#                 训练集 -> 训练集 + 验证集
#                 训练模型，得到该超参、该划分下模型的指标
#                     [ HERE ]
#             平均（或其他）各划分后模型的好坏（不交叉验证容易过拟合）
#         得该超参下模型的好坏
#     选取最好的超参

# [ HERE ] 训练模型：
#     初始化参数
#     训练轮次（epoch）：
#         优化器（取本轮所用数据，问模型要loss和gradient，去调模型的参数）



# Project Description: Multilayer Perceptron (MLP) Implementation and Evaluation

# 1. Develop a Multilayer Perceptron (MLP) Model Using NumPy:

#     Create a Python program that leverages NumPy to implement an MLP capable of handling any number of layers and units per layer.
#     Ensure that the program includes both the forward and backward propagation processes.
#     Implement Mini-batch and Stochastic Gradient Descent Updates:

#     Write code to update the model parameters using both mini-batch and stochastic gradient descent methods.
#     Ensure that these updates are integrated into the training process of the MLP model.

# 2. Cross-Validation Implementation:

#     Develop code for k-fold cross-validation to assess the model's performance.
#     This should allow for the evaluation of different hyperparameters and their impact on model accuracy.

# 3. Nonlinear Function Approximation:

#     Select a complex nonlinear function with a single input and a single output.
#     Generate a dataset by adding noise to the function's output.
#     Utilize the MLP model with various hyperparameters (number of layers, number of units per layer) to approximate the nonlinear function.
#     Use k-fold cross-validation to determine which set of hyperparameters provides the best approximation.
#     Demonstrate the results and discuss the findings.

# 4. Classifier Performance Evaluation:

#     Create a dataset similar to the one depicted on page 14 of the provided PPT of MLP lecture.
#     Experiment with the MLP model using different hyperparameters (number of layers, number of units per layer) to classify the dataset.
#     Validate the model's classification capabilities using k-fold cross-validation and identify the best performing configuration.
#     Demonstrate the classification results and evaluate the model's performance using accuracy, recall, precision, and F1 score metrics.

# 5. Submission Requirements:

#     Ensure that your code is well-documented and follows good programming practices.
#     Include a detailed report explaining your methodology, the results of your experiments, and your conclusions.
#     The report should also include visualizations of the dataset, the model's performance, and any other relevant findings.