import numpy as np

from simple_ml import Variable, Dataset, DataIterator
from simple_ml.model import Model, Sequential
from simple_ml.model.layers import Linear, ReLU, Sigmoid
from simple_ml.training.loss import MSELoss, CrossEntropyLoss
from simple_ml.training.optimizer import GD #, Adam


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
#     Linear(3, 4), # , debug_init_weights = True),
#     Sigmoid(),
#     Linear(4, 3), # , debug_init_weights = True),
#     ReLU()
# ])

# criterion = MSELoss()
# optimizer = GD(model.params, lr = 0.01)


# dataset = Dataset(file_path = "wine.data")
# data_iter = DataIterator(dataset, batch_size = 10, shuffle = True, cyclic = True)


# for epoch in range(100):
#     for X_np in data_iter:
#         out = model(X_np)
#         loss = criterion(out, )

#         # optimizer.zeroize_gradients()
#         model.backward()
#         optimizer.step()


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


def label_split(data: np.ndarray):
    return data[:, 1:], data[:, 0].astype(int) # X, T

dataset = Dataset(file_path = "wine.data", preprocess_func = label_split)
data_iter = DataIterator(dataset, batch_size = 10, shuffle = False, cyclic = True)

for i, [x, t] in enumerate(data_iter):
    print(i, x[:, :2], t)
    if i >= 17:
        break

