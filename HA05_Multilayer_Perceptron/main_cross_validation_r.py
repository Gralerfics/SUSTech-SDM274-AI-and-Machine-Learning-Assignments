import numpy as np

import matplotlib.pyplot as plt

from simple_ml import Variable, Dataset, DataIterator, merge_datasets, split_train_and_test_dataset, split_k_fold_cross_validation_dataset
from simple_ml.data.samples import label_split_for_single_output_dataset, generate_2d_classification_circle, generate_2d_classification_exclusive_or, generate_1d_regression_with_function
from simple_ml.evaluation.criterion import eval_binary_accuracy, eval_binary_recall, eval_binary_precision, eval_binary_f1_score, eval_regression_r2
from simple_ml.model import Model, Sequential
from simple_ml.model.layers import Linear, ReLU, Sigmoid, Tanh
from simple_ml.training.loss import MSELoss, CrossEntropyLoss
from simple_ml.training.optimizer import GD, Adam
from simple_ml.visualization.plot import TwoFeaturesClassificationModelVisualizer


""" Dataset """
# data_np = generate_2d_classification_circle(N = 400, r_0 = 3, r_1 = 3, noise = 0.8)
# data_np = generate_2d_classification_exclusive_or(N = 1000)
data_np = generate_1d_regression_with_function(N = 1000, f = lambda x: np.cos(x) + np.exp(-x ** 2) + x ** 3 / 233, x_range = (-10, 10), noise = 0.2)

dataset = Dataset(data = data_np, preprocess_func = label_split_for_single_output_dataset)
train_dataset, test_dataset = split_train_and_test_dataset(dataset, 0.2)


# 1,20,20,1,sig,gd0d01,8000ep,func1000,cv5

""" Model """
model = Sequential([
    Linear(1, 20),
    Sigmoid(),
    Linear(20, 20),
    Sigmoid(),
    Linear(20, 1)
])

loss_func = MSELoss()
# loss_func = CrossEntropyLoss() # the output of the model should be in (0, 1), i.e. Sigmoid

optimizer = GD(model.params, lr = 0.01)
# optimizer = Adam(model.params, lr = 0.003)


""" Cross-Validation & Training """
K = 5
datasets = split_k_fold_cross_validation_dataset(train_dataset, k = K)

losses = []
accuracies = []
recalls = []
precisions = []
f1_scores = []
r2s = []

epoch_num = 4000

for i in range(K + 1):
    if i < K:
        print(f"[Info] Training on cross-validation fold {i + 1} / {K}")
    else:
        print(f"[Info] Training on whole training set")

    # prepare train and validation set
    if i < K:
        train_ds = merge_datasets([datasets[j] for j in range(K) if j != i])
        valid_ds = datasets[i]
    else:
        train_ds = train_dataset
        valid_ds = test_dataset
    train_iter_kfold = DataIterator(train_ds, batch_size = 10, shuffle = True, cyclic = False)

    # train
    for epoch in range(epoch_num):
        for batch, [features, labels] in enumerate(train_iter_kfold):
            prediction = model(Variable(features, derivable = True)) # must be wrapped by Variable
            loss = loss_func(prediction, labels) # calculate loss and gradient (!)

            model.backward()
            optimizer.step()

    # evaluate on validation/testing set
    prediction = model(Variable(valid_ds.datas[0], derivable = True))

    loss = loss_func(prediction, valid_ds.datas[1])
    # accuracy = eval_binary_accuracy(prediction, valid_ds.datas[1])
    # recall = eval_binary_recall(prediction, valid_ds.datas[1])
    # precision = eval_binary_precision(prediction, valid_ds.datas[1])
    # f1_score = eval_binary_f1_score(prediction, valid_ds.datas[1])
    r2 = eval_regression_r2(prediction, valid_ds.datas[1])

    # if i < K:
    #     print(f"[Info] Performance for fold {i + 1} / {K}: Loss = {loss}, Accuracy = {accuracy}, Recall = {recall}, Precision = {precision}, F1 Score = {f1_score}")
    # else:
    #     print(f"[Info] Performance on test set: Loss = {loss}, Accuracy = {accuracy}, Recall = {recall}, Precision = {precision}, F1 Score = {f1_score}")

    if i < K:
        print(f"[Info] Performance for fold {i + 1} / {K}: Loss = {loss}, R2 = {r2}")
    else:
        print(f"[Info] Performance on test set: Loss = {loss}, R2 = {r2}")

    losses.append(loss)
    # accuracies.append(accuracy)
    # recalls.append(recall)
    # precisions.append(precision)
    # f1_scores.append(f1_score)
    r2s.append(r2)
    

# categories = ["Loss", "Accuracy", "Recall", "Precision", "F1 Score"]
# validation_results = [losses[:-1], accuracies[:-1], recalls[:-1], precisions[:-1], f1_scores[:-1]]
# test_results = [losses[-1], accuracies[-1], recalls[-1], precisions[-1], f1_scores[-1]]
categories = ["Loss", "R2"]
validation_results = [losses[:-1], r2s[:-1]]
test_results = [losses[-1], r2s[-1]]

averages = [np.mean(val) for val in validation_results]
# print(f"[Info] Average performance on validation folds: Loss = {averages[0]}, Accuracy = {averages[1]}, Recall = {averages[2]}, Precision = {averages[3]}, F1 Score = {averages[4]}")
print(f"[Info] Average performance on validation folds: Loss = {averages[0]}, R2 = {averages[1]}")

bar_colors = ['#FF8F31', '#FF8F31', '#FF8F31', '#FF8F31', '#FF8F31', '#FF6820', '#544943']
bar_colors = ['#FF8F31', '#FF8F31', '#FF8F31', '#FF8F31', '#FF8F31', '#FF6820', '#544943']
bar_width = 0.1
bar_positions = np.arange(len(categories))

plt.figure(figsize = (12, 6))
for i in range(K):
    plt.bar(bar_positions + i * bar_width, [val[i] for val in validation_results], width = bar_width, label = f'Validation Fold {i + 1}', color = bar_colors[i])
plt.bar(bar_positions + K * bar_width, averages, width = bar_width, label = 'Validation Average', color = bar_colors[K])
plt.bar(bar_positions + (K + 1) * bar_width, test_results, width = bar_width, label = 'Test', color = bar_colors[K + 1])

plt.xlabel('Metrics')
plt.ylabel('Values')
plt.xticks(bar_positions + (K + 1) * bar_width / 2, categories)
plt.legend()
plt.show()


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