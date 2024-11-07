import asyncio

import numpy as np

from sanic import Sanic, response
from sanic import HTTPResponse, Request, Websocket

from simple_ml import Variable, Dataset, DataIterator, split_train_and_test_dataset
from simple_ml.data.samples import label_split_for_2d_classification_dataset, generate_1d_regression_with_function
from simple_ml.evaluation.criterion import eval_regression_r2
from simple_ml.model import Model, Sequential
from simple_ml.model.layers import Linear, ReLU, Sigmoid, Tanh
from simple_ml.training.loss import MSELoss, CrossEntropyLoss
from simple_ml.training.optimizer import GD, MomentumGD, Adam

from simple_ml.dashboard.comm import WebSocketClientsPool
from simple_ml.dashboard.logger import TrainingDataLogger


""" App """
APP_NAME = 'simple_ml_dashboard'
APP_HOST = 'localhost'
APP_PORT = 2333
SERIAL_POLL_INTERVAL = 0.2

app = Sanic(APP_NAME)
clients = WebSocketClientsPool()


""" CORS Middleware """
@app.middleware('request')
async def cors_middle_req(request: Request):
    if request.method.lower() == 'options':
        allow_headers = [
            'Authorization',
            'content-type'
        ]
        headers = {
            'Access-Control-Allow-Methods': ', '.join(['GET', 'POST', 'OPTIONS']),
            'Access-Control-Max-Age': '86400',
            'Access-Control-Allow-Headers': ', '.join(allow_headers),
        }
        return HTTPResponse('', headers=headers)

@app.middleware('response')
def cors_middle_res(request: Request, response: HTTPResponse):
    allow_origin = '*'
    response.headers.update(
        {
            'Access-Control-Allow-Origin': allow_origin,
        }
    )


""" Routes """
@app.route('/')
async def index(request: Request):
    return response.html(f'<p>{APP_NAME}</p>')

@app.websocket('/notify') # new client
async def notify(request: Request, ws: Websocket):
    clients.append(ws)
    try:
        async for msg in ws:
            print(f'Received: {msg}')
    except Exception as e:
        print(f'WebSocket error: {e}')


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

try:
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

        # record testing loss and R2
        test_prediction = model(Variable(test_dataset.datas[0], derivable = True))
        test_loss = loss_func(test_prediction, test_dataset.datas[1])
        test_r2 = eval_regression_r2(test_prediction, test_dataset.datas[1])
        
        
except KeyboardInterrupt:
    clients.close()

