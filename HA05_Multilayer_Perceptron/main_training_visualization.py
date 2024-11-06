import numpy as np

import matplotlib.pyplot as plt

from simple_ml import Variable, Dataset, DataIterator, split_train_and_test_dataset
from simple_ml.data.samples import label_split_for_2d_classification_dataset, generate_2d_classification_circle, generate_2d_classification_exclusive_or
from simple_ml.evaluation.criterion import eval_binary_accuracy, eval_binary_recall, eval_binary_precision, eval_binary_f1_score
from simple_ml.model import Model, Sequential
from simple_ml.model.layers import Linear, ReLU, Sigmoid, Tanh
from simple_ml.training.loss import MSELoss, CrossEntropyLoss
from simple_ml.training.optimizer import GD, Adam
from simple_ml.visualization.plot import TwoFeaturesModelVisualizer


""" Dataset """
data_np = generate_2d_classification_circle(N = 1000)
# data_np = generate_2d_classification_exclusive_or()

dataset = Dataset(data = data_np, preprocess_func = label_split_for_2d_classification_dataset)
train_dataset, test_dataset = split_train_and_test_dataset(dataset, 0.2)


""" Model """
model = Sequential([
    Linear(2, 4),
    Sigmoid(),
    Linear(4, 2),
    Sigmoid(),
    Linear(2, 1)
])

loss_func = MSELoss()
# loss_func = CrossEntropyLoss() # the output of the model should be in (0, 1), i.e. Sigmoid

# optimizer = GD(model.params, lr = 0.01)
optimizer = Adam(model.params, lr = 0.003)


""" Training """
train_iter_kfold = DataIterator(train_dataset, batch_size = 10, shuffle = True, cyclic = False)

epoch_num = 500
train_loss_history = []
train_accuracy_history = []
test_loss_history = []
test_accuracy_history = []

# initialize the plot
visualizer = TwoFeaturesModelVisualizer(model, train_dataset, test_dataset, x1_range = (-6, 6, 100), x2_range = (-6, 6, 100), output_range = (-1, 1))

for epoch in range(epoch_num):
    train_loss = 0
    train_accuracy = 0

    for batch, [features, labels] in enumerate(train_iter_kfold):
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

