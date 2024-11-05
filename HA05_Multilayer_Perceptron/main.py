import numpy as np

from simple_ml.model import Model
from simple_ml.model.layers import Linear


pass


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