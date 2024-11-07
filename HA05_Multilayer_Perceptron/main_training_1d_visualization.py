import numpy as np

import matplotlib.pyplot as plt

from simple_ml import Variable, Dataset, DataIterator, split_train_and_test_dataset
from simple_ml.data.samples import label_split_for_2d_classification_dataset, generate_1d_regression_with_function
from simple_ml.evaluation.criterion import eval_regression_r2
from simple_ml.model import Model, Sequential
from simple_ml.model.layers import Linear, ReLU, Sigmoid, Tanh
from simple_ml.training.loss import MSELoss, CrossEntropyLoss
from simple_ml.training.optimizer import GD, MomentumGD, Adam
from simple_ml.visualization.plot import OneFeatureRegressionModelVisualizer


""" Dataset """
def example_func(x):
    return np.cos(x) + np.exp(-x ** 2) + x ** 3 / 233

data_np = generate_1d_regression_with_function(N = 1000, f = example_func, x_range = (-10, 10), noise = 0.2)

dataset = Dataset(data = data_np, preprocess_func = label_split_for_2d_classification_dataset)
train_dataset, test_dataset = split_train_and_test_dataset(dataset, 0.2)


""" Model """
model = Sequential([
    Linear(1, 5),
    Sigmoid(),
    Linear(5, 7),
    Sigmoid(),
    Linear(7, 1)
])

loss_func = MSELoss()

# optimizer = GD(model.params, lr = 0.1)
# optimizer = MomentumGD(model.params, lr = 0.1, momentum = 0.8)
optimizer = Adam(model.params, lr = 0.01)


""" Training """
train_iter = DataIterator(train_dataset, batch_size = 32, shuffle = True, cyclic = False)

epoch_num = 1000000
train_loss_history = []
train_r2_history = []
test_loss_history = []
test_r2_history = []

# initialize the plot
visualizer = OneFeatureRegressionModelVisualizer(model, train_dataset, test_dataset, x_range = (-10, 10, 100))

for epoch in range(epoch_num):
    train_loss = 0
    train_r2 = 0

    for batch, [features, labels] in enumerate(train_iter):
        prediction = model(Variable(features, derivable = True)) # must be wrapped by Variable
        loss = loss_func(prediction, labels) # calculate loss and gradient (!)

        model.backward()
        optimizer.step()

        train_loss += loss
        train_r2 += eval_regression_r2(prediction, labels)
    
    # record training loss and R2
    train_loss /= len(train_dataset)
    train_r2 /= len(train_dataset)
    train_loss_history.append(train_loss)
    train_r2_history.append(train_r2)

    # record testing loss and R2
    test_prediction = model(Variable(test_dataset.datas[0], derivable = True))
    test_loss = loss_func(test_prediction, test_dataset.datas[1])
    test_r2 = eval_regression_r2(test_prediction, test_dataset.datas[1])
    test_loss_history.append(test_loss)
    test_r2_history.append(test_r2)

    # update the plot
    if epoch % 200 == 0 or epoch < 10:
        visualizer.update(epoch, train_loss_history, train_r2_history, test_loss_history, test_r2_history)

plt.show()

