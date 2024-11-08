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

        # self.is_updated = threading.Event() # False
        self.msg_buffer = {}
        self.msg_buffer_lock = threading.Lock()

        self.epoch = 0
    
    def update_msg_buffer(self, msg: dict):
        with self.msg_buffer_lock:
            self.msg_buffer.update(msg)
    
    def get_msg_buffer(self):
        with self.msg_buffer_lock:
            return self.msg_buffer.copy()
    
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

            # message preparation
            self.update_msg_buffer({
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

            # next epoch
            self.epoch += 1
    
    # async def async_forwarding_task(self):
    #     while not self.is_stopped.is_set():
    #         if self.is_updated.is_set(): # don't use .wait() here, which will block the event loop
    #             self.is_updated.clear()
    #             with self.msg_buffer_lock:
    #                 msg = self.msg_buffer
    #             await self.pool.broadcast(msg)
    #         await asyncio.sleep(0.02)


class DashboardCore:
    def __init__(self, pool):
        self.pool = pool

        self.train_dataset = None
        self.test_dataset = None
        self.model = None
        self.loss = None
        self.optimizer = None

        self.epoch = 0
    
        self.is_stopped = threading.Event()
        self.is_stopped.set() # True
        self.is_resumed = threading.Event()
        self.is_resumed.set() # True

        self.msg_buffer = {}
        self.msg_buffer_lock = threading.Lock()
    
    def is_runnable(self):
        return self.train_dataset is not None and self.test_dataset is not None and self.model is not None and self.loss is not None and self.optimizer is not None
    
    def recursively_merge(self, target, source):
        for key in source:
            if key in target and isinstance(target[key], dict) and isinstance(source[key], dict):
                self.recursively_merge(target[key], source[key])
            else:
                target[key] = source[key]

    def update_msg_buffer(self, msg: dict):
        with self.msg_buffer_lock:
            self.recursively_merge(self.msg_buffer, msg)
    
    def get_msg_buffer(self):
        with self.msg_buffer_lock:
            return self.msg_buffer.copy()
    
    def get_state(self):
        return 'stopped' if self.is_stopped.is_set() else ('running' if self.is_resumed.is_set() else 'paused')
    
    def launch(self, conf):
        # stop the previous task
        self.reset()
        
        # dataset, TODO: parser
        data_np = generate_2d_classification_circle(N = 1000)
        dataset = Dataset(data = data_np, preprocess_func = label_split_for_2d_classification_dataset)
        self.train_dataset, self.test_dataset = split_train_and_test_dataset(dataset, 0.2)

        # model, TODO: parser
        self.model = Sequential([
            Linear(2, 4),
            Sigmoid(),
            Linear(4, 2),
            Sigmoid(),
            Linear(2, 1)
        ])

        # loss, TODO: parser
        self.loss = MSELoss()

        # optimizer, TODO: parser
        self.optimizer = Adam(self.model.params, lr = 0.01)

        # launch task
        if self.is_runnable():
            self.is_stopped.clear() # False
            task_thread = threading.Thread(target = self.run)
            task_thread.start()
            return True
        else:
            return False

    def stop(self):
        self.is_stopped.set() # True

    def pause(self):
        self.is_resumed.clear() # False
    
    def resume(self):
        self.is_resumed.set() # True
    
    def reset(self):
        self.stop()
        self.resume()

        # TODO: reset, perhaps broadcast an empty update to the whole pool
    
    def run(self): # TODO: now only for 2d classification
        train_iter = DataIterator(self.train_dataset, batch_size = 32, shuffle = True, cyclic = False)

        train_loss_buffer = []
        train_accuracy_buffer = []
        test_loss_buffer = []
        test_accuracy_buffer = []

        # for model output visualization (temporary) TODO
        x1_range = (-6, 6, 50)
        x2_range = (-6, 6, 50)
        x1, x2 = np.meshgrid(np.linspace(*x1_range), np.linspace(*x2_range))
        model_output_features = Variable(np.c_[x1.ravel(), x2.ravel()])

        # invariant message
        self.update_msg_buffer({
            'model_output': {
                'type': '2i1o',
                'in': [
                    {'name': 'x_1', 'range': x1_range},
                    {'name': 'x_2', 'range': x2_range}
                ],
                'out': [
                    {'name': 'output', 'range': (-1, 1)}
                ]
            },
            'train_dataset': {
                'type': '2i1o',
                'data_in': self.train_dataset.datas[0].tolist(),
                'data_out': self.train_dataset.datas[1].tolist()
            },
            'test_dataset': {
                'type': '2i1o',
                'data_in': self.test_dataset.datas[0].tolist(),
                'data_out': self.test_dataset.datas[1].tolist()
            }
        })

        while not self.is_stopped.is_set(): # continue if is_stopped = False
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
            test_prediction = self.model(Variable(self.test_dataset.datas[0], derivable = True))
            test_loss = self.loss(test_prediction, self.test_dataset.datas[1])
            test_accuracy = eval_binary_accuracy(test_prediction, self.test_dataset.datas[1])
            test_loss_buffer.append(test_loss)
            test_accuracy_buffer.append(test_accuracy)

            # message update
            self.update_msg_buffer({
                'epoch': self.epoch,
                'train_loss_buffer': train_loss_buffer,
                'train_accuracy_buffer': train_accuracy_buffer,
                'test_loss_buffer': test_loss_buffer,
                'test_accuracy_buffer': test_accuracy_buffer,
                'model_output': {
                    'data': self.model(model_output_features).value.reshape(x1.shape).tolist()
                }
            })

            # next epoch
            self.epoch += 1
            # print(self.epoch)

        # destroy the task
        self.train_dataset = None
        self.test_dataset = None
        self.model = None
        self.loss = None
        self.optimizer = None
        
        self.epoch = 0

        self.msg_buffer.clear()
    
    async def async_ws_listener(self, ws):
        try:
            async for msg in ws:
                data = json.loads(msg)
                """
                {
                    'type': 'poll',
                    'prop_name': '<...>.<...>...<...>',
                }
                """
                if data['type'] == 'poll':
                    # print(f"[Info] a client polled: {data['prop_name']}")
                    if 'prop_name' not in data.keys():
                        continue # invalid cmd, do not respond, TODO

                    prop_value = self.get_msg_buffer()
                    prop_names = data['prop_name'].split('.')[1:]
                    for prop_name in prop_names:
                        prop_value = prop_value.get(prop_name, None)
                        if prop_value is None:
                            break
                    # print(f"[Info] response: {prop_value}")
                    await ws.send(json.dumps({
                        'type': 'poll_response',
                        'prop_name': data['prop_name'],
                        'prop_value': prop_value # if not found, None
                    }))
                else:
                    continue # invalid cmd, do not respond, TODO

        except Exception as e:
            print(f'WebSocket error: {e}')
        
        print("[Info] a client cancelled")

