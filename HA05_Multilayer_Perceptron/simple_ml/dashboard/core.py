import os
import time
import json
import threading

import numpy as np

import asyncio
from sanic import Sanic

from .. import Variable, Dataset, DataIterator, split_train_and_test_dataset
from ..data.samples import label_split_for_2d_classification_dataset, generate_2d_classification_circle, generate_1d_regression_with_function
from ..evaluation.criterion import eval_binary_accuracy
from ..model import Model, Sequential
from ..model.layers import Linear, ReLU, Sigmoid, Tanh
from ..training.loss import MSELoss, CrossEntropyLoss
from ..training.optimizer import GD, MomentumGD, Adam

from .clients import ClientsPool



class Task:
    def __init__(self, task_id, pool, components: dict):
        self.task_id = task_id
        self.pool: ClientsPool = pool

        # TODO: assert
        self.train_dataset = components.get('train_dataset', None)
        self.test_dataset = components.get('test_dataset', None)
        self.model = components.get('model', None)
        self.loss = components.get('loss', None)
        self.optimizer = components.get('optimizer', None)
    
        self.is_stopped = threading.Event() # False
        self.is_resumed = threading.Event()
        self.is_resumed.set() # True

        self.is_updated = threading.Event() # False
        self.update_msg = None
        self.update_msg_lock = threading.Lock()

        self.epoch = 0
    
    def stop(self):
        self.is_stopped.set() # True
    
    def pause(self):
        self.is_resumed.clear() # False
    
    def resume(self):
        self.is_resumed.set() # True
    
    def reset(self):
        pass # TODO: reset epoch, model weights, etc.

    def run(self): # TODO: now only for 2d classification
        train_iter = DataIterator(self.train_dataset, batch_size = 32, shuffle = True, cyclic = False)

        train_loss_buffer = []
        train_accuracy_buffer = []

        # for model output visualization (temporary) TODO
        x1_range = (-6, 6, 50)
        x2_range = (-6, 6, 50)
        x1, x2 = np.meshgrid(np.linspace(*x1_range), np.linspace(*x2_range))
        model_output_features = Variable(np.c_[x1.ravel(), x2.ravel()])

        while not self.is_stopped.is_set(): # run if is_stopped = False
            # block until is_resumed = True
            self.is_resumed.wait()

            train_loss = 0
            train_accuracy = 0

            for batch, [features, labels] in enumerate(train_iter):
                prediction = self.model(Variable(features, derivable = True)) # must be wrapped by Variable
                loss = self.loss(prediction, labels) # calculate loss and gradient (!)

                self.model.backward()
                self.optimizer.step()

                train_loss += loss * features.shape[0]
                train_accuracy += eval_binary_accuracy(prediction, labels) * features.shape[0]
            
            # record training loss and accuracy
            train_loss /= len(self.train_dataset)
            train_accuracy /= len(self.train_dataset)
            train_loss_buffer.append(train_loss)
            train_accuracy_buffer.append(train_accuracy)

            # record testing loss and accuracy
            # test_prediction = self.model(Variable(self.test_dataset.datas[0], derivable = True))
            # test_loss = self.loss(test_prediction, self.test_dataset.datas[1])
            # test_accuracy = eval_binary_accuracy(test_prediction, self.test_dataset.datas[1])
            # test_loss_buffer.append(test_loss)
            # test_accuracy_buffer.append(test_accuracy)

            # broadcast
            msg = json.dumps({
                'epoch': self.epoch,
                'train_loss_buffer': train_loss_buffer,
                'train_accuracy_buffer': train_accuracy_buffer,
                'model_output': {
                    'type': '2i1o',
                    'in': [
                        {'name': 'x_1', 'range': x1_range},
                        {'name': 'x_2', 'range': x2_range}
                    ],
                    'out': [
                        {'name': 'output', 'range': (-1, 1)}
                    ],
                    'data': self.model(model_output_features).value.reshape(x1.shape).tolist()
                },
            })
            with self.update_msg_lock:
                self.update_msg = msg
                self.is_updated.set()

            # next epoch
            self.epoch += 1
    
    async def async_forwarding_task(self):
        while not self.is_stopped.is_set():
            if self.is_updated.is_set(): # don't use .wait() here, which will block the event loop
                self.is_updated.clear()
                with self.update_msg_lock:
                    msg = self.update_msg
                await self.pool.broadcast(msg)
            await asyncio.sleep(0.02)


class DashboardCore:
    def __init__(self, app, pool):
        self.app: Sanic = app
        self.pool = pool
        
        self.task = None
    
    def get_state(self):
        if self.task is None:
            state = 'stopped'
        else:
            if self.task.is_stopped.is_set():
                state = 'stopped'
            elif self.task.is_resumed.is_set():
                state = 'running'
            else:
                state = 'paused'
        return {
            'state': state,
            'task_id': self.task.task_id if self.task is not None else None
        }
    
    def launch_task(self, conf):
        # TODO: stop previous task (currently only one task is allowed)
        if self.task is not None:
            self.stop_task()

        """
        {
            'dataset': {
                'type': 'builtin',
                'name': '<function_name>',
                'test_ratio': 0.2,
                'batch_size': ...,
                'params': {...: ...}
            },
            # 'dataset': {
            #     'type': 'direct',
            #     'test_ratio': 0.2,
            #     'batch_size': ...,
            #     'datas': [
            #         [[[...]]],
            #         [[...]],
            #         ...
            #     ]
            # },
            'model': [
                {'type': 'Linear', 'params': {...: ...}},
                {'type': 'Sigmoid'},
                ...
            ],
            'loss': {
                'type': 'MSELoss',
                'params': {...: ...}
            },
            'optimizer': {
                'type': 'GD',
                'params': {...: ...}
            }
        }
        """
        
        # dataset, TODO: parser
        # data_np = generate_1d_regression_with_function(N = 1000, f = lambda x: np.cos(x) + np.exp(-x ** 2) + x ** 3 / 233, x_range = (-10, 10), noise = 0.2)
        data_np = generate_2d_classification_circle(N = 1000)
        dataset = Dataset(data = data_np, preprocess_func = label_split_for_2d_classification_dataset)
        train_dataset, test_dataset = split_train_and_test_dataset(dataset, 0.2)

        # model, TODO: parser
        model = Sequential([
            Linear(2, 4),
            Sigmoid(),
            Linear(4, 2),
            Sigmoid(),
            Linear(2, 1)
        ])

        # loss, TODO: parser
        loss_func = MSELoss()

        # optimizer, TODO: parser
        # optimizer = GD(model.params, lr = 0.1)
        # optimizer = MomentumGD(model.params, lr = 0.1, momentum = 0.8)
        optimizer = Adam(model.params, lr = 0.01)

        # launch task
        self.task = Task(
            task_id = os.urandom(16).hex(), # TODO: collision
            pool = self.pool,
            components = {
                'train_dataset': train_dataset,
                'test_dataset': test_dataset,
                'model': model,
                'loss': loss_func,
                'optimizer': optimizer
            }
        )
        task_thread = threading.Thread(target = self.task.run)
        task_thread.start()
        self.app.add_task(self.task.async_forwarding_task())

        return True

    def stop_task(self):
        if self.task is not None:
            self.task.stop()
            self.task = None
        return True # TODO

    def pause_task(self):
        if self.task is not None:
            self.task.pause()
        return True # TODO
    
    def resume_task(self):
        if self.task is not None:
            self.task.resume()
        return True # TODO

