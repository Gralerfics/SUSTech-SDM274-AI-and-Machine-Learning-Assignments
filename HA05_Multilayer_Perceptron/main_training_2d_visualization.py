import numpy as np

import matplotlib.pyplot as plt

from simple_ml import Variable, Dataset, DataIterator, split_train_and_test_dataset
from simple_ml.data.samples import label_split_for_2d_classification_dataset, generate_2d_classification_circle, generate_2d_classification_exclusive_or, generate_2d_classification_gaussians
from simple_ml.evaluation.criterion import eval_binary_accuracy, eval_binary_recall, eval_binary_precision, eval_binary_f1_score
from simple_ml.model import Model, Sequential
from simple_ml.model.layers import Linear, ReLU, Sigmoid, Tanh
from simple_ml.training.loss import MSELoss, CrossEntropyLoss
from simple_ml.training.optimizer import GD, Adam
from simple_ml.visualization.plot import TwoFeaturesClassificationModelVisualizer


""" Dataset """
data_np = generate_2d_classification_circle(N = 1000)
# data_np = generate_2d_classification_exclusive_or(N = 1000)
# data_np = generate_2d_classification_gaussians([
#     ([4, 4], [[4, 0], [0, 4]], 200, 1),
#     ([-4, 4], [[4, 0], [0, 4]], 200, -1),
#     ([-4, -4], [[4, 0], [0, 4]], 200, 1),
#     ([4, -4], [[4, 0], [0, 4]], 200, -1)
# ])
# data_np = np.array([
#     [-3.2, 4.5, 1],
#     [-2.6, 4.7, 1],
#     [0.1, 4.2, 1],
#     [0.3, 2.1, 1],
#     [2.2, 3.2, 1],
#     [4.6, 2.8, 1],
#     [3.8, 1.4, 1],
#     [4.9, 0.4, 1],
#     [0.2, -0.05, 1],
#     [1.8, -0.05, 1],
#     [-0.3, -3.2, 1],
#     [4, -1, 1],
#     [5.3, -0.9, 1],
#     [0.4, -2.95, 1],
#     [2.4, -3, 1],
#     [4.1, -3.1, 1],
#     [1.6, -5, 1],
#     [-0.4, 4.5, -1],
#     [-1.8, 3.1, -1],
#     [-3.2, 2, -1],
#     [-3.35, 0.45, -1],
#     [-2.1, 1.4, -1],
#     [-0.1, 1.4, -1],
#     [1.7, 2, -1],
#     [0.05, -1.8, -1],
#     [2.05, -1.6, -1],
#     [1, -4.2, -1],
#     [1.95, -3.3, -1],
#     [-1.75, -0.3, -1],
#     [-2.8, -0.38, -1],
#     [-2, -2.1, -1],
#     [-3.8, -2, -1]
# ])

dataset = Dataset(data = data_np, preprocess_func = label_split_for_2d_classification_dataset)
train_dataset, test_dataset = split_train_and_test_dataset(dataset, 0)


""" Model """
model = Sequential([
    Linear(2, 4),
    Sigmoid(),
    Linear(4, 2),
    Sigmoid(),
    Linear(2, 1)
])

# model = Sequential([
#     Linear(2, 8),
#     ReLU(),
#     Linear(8, 16),
#     ReLU(),
#     Linear(16, 16),
#     ReLU(),
#     Linear(16, 8),
#     ReLU(),
#     Linear(8, 1)
# ])

# model = Sequential([
#     Linear(2, 20),
#     Tanh(),
#     Linear(20, 1)
# ])

loss_func = MSELoss()
# loss_func = CrossEntropyLoss() # the output of the model should be in (0, 1), i.e. Sigmoid

# optimizer = GD(model.params, lr = 0.01)
optimizer = Adam(model.params, lr = 0.01)


""" Training """
train_iter = DataIterator(train_dataset, batch_size = 10, shuffle = True, cyclic = False)

epoch_num = 100000
train_loss_history = []
train_accuracy_history = []
test_loss_history = []
test_accuracy_history = []

# initialize the plot
visualizer = TwoFeaturesClassificationModelVisualizer(model, train_dataset, test_dataset, x1_range = (-6, 6, 100), x2_range = (-6, 6, 100), output_range = (-1, 1))
    # , z_func = lambda Y: (Y > 0) * 2 - 1)

for epoch in range(epoch_num):
    train_loss = 0
    train_accuracy = 0 # TODO: ！！！！！！！！数字好像不太对，有点低

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
    if epoch % 1 == 0:
        visualizer.update(epoch, train_loss_history, train_accuracy_history, test_loss_history, test_accuracy_history)

plt.show()

