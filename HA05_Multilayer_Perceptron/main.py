import numpy as np

import matplotlib.pyplot as plt

from simple_ml import Variable, Dataset, DataIterator, split_train_and_test_dataset
from simple_ml.data.samples import generate_2d_classification_circle, generate_2d_classification_exclusive_or
from simple_ml.evaluation.criterion import eval_binary_accuracy, eval_binary_recall, eval_binary_precision, eval_binary_f1_score
from simple_ml.model import Model, Sequential
from simple_ml.model.layers import Linear, ReLU, Sigmoid, Tanh
from simple_ml.training.loss import MSELoss, CrossEntropyLoss
from simple_ml.training.optimizer import GD, Adam
from simple_ml.visualization.plot import TwoFeaturesModelVisualizer


""" Dataset """
data_np = generate_2d_classification_circle()
# data_np = generate_2d_classification_exclusive_or()

def label_split(data: np.ndarray):
    return data[:, :-1], data[:, -1].reshape(-1, 1) # [x_0, x_1], t

dataset = Dataset(data = data_np, preprocess_func = label_split)
train_dataset, test_dataset = split_train_and_test_dataset(dataset, 0.2)
train_iter = DataIterator(train_dataset, batch_size = 10, shuffle = True, cyclic = False)


""" Model """
model = Sequential([
    Linear(2, 4),
    ReLU(),
    Linear(4, 2),
    ReLU(),
    Linear(2, 1)
])

loss_func = MSELoss()
# loss_func = CrossEntropyLoss() # the output of the model should be in (0, 1), i.e. Sigmoid

# optimizer = GD(model.params, lr = 0.01)
optimizer = Adam(model.params, lr = 0.003)


""" Training """
epoch_num = 10000
train_loss_history = []
train_accuracy_history = []
test_loss_history = []
test_accuracy_history = []

# initialize the plot
visualizer = TwoFeaturesModelVisualizer(model, train_dataset, test_dataset, x1_range = (-6, 6, 100), x2_range = (-6, 6, 100), output_range = (-1, 1))

for epoch in range(epoch_num):
    train_loss = 0
    train_accuracy = 0

    for batch, [features, labels] in enumerate(train_iter):
        prediction = model(Variable(features, derivable = True)) # must be wrapped by Variable
        loss = loss_func(prediction, labels) # calculate loss and gradient (!)

        model.backward()
        optimizer.step()

        train_loss += loss
        train_accuracy += eval_binary_accuracy(prediction, labels)
    
    # record training loss and accuracy
    train_loss /= len(train_dataset)
    train_accuracy /= len(train_dataset)
    train_loss_history.append(train_loss)
    train_accuracy_history.append(train_accuracy)

    # record testing loss and accuracy
    test_prediction = model(Variable(test_dataset.datas[0], derivable = True))
    test_loss = loss_func(test_prediction, test_dataset.datas[1])
    test_accuracy = eval_binary_accuracy(test_prediction, test_dataset.datas[1])
    # test_recall = eval_binary_recall(test_prediction, test_dataset.datas[1])
    # test_precision = eval_binary_precision(test_prediction, test_dataset.datas[1])
    # test_f1_score = eval_binary_f1_score(test_prediction, test_dataset.datas[1])
    # print(f"Test Accuracy: {test_accuracy * 100:.2f} %\tTest Recall: {test_recall * 100:.2f} %\tTest Precision: {test_precision * 100:.2f} %\tTest F1 Score: {test_f1_score * 100:.2f} %")
    test_loss_history.append(test_loss)
    test_accuracy_history.append(test_accuracy)

    # update the plot
    if epoch % 10 == 0:
        visualizer.update(epoch, train_loss_history, train_accuracy_history, test_loss_history, test_accuracy_history)

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