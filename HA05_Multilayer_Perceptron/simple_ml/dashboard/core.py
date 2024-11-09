import os
import time
import json
import threading

import numpy as np

import asyncio
from sanic import Sanic

from .. import *
from ..data.samples import *
from ..evaluation.criterion import *
from ..model import *
from ..model.layers import *
from ..training.loss import *
from ..training.optimizer import *


class DashboardCore:
    def __init__(self, pool):
        self.pool = pool

        self.train_dataset = None
        self.test_dataset = None
        self.batch_size = None
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
        # Reset previous tasks
        self.reset()
        
        # Parse dataset
        dataset_conf = conf['dataset']
        dataset_type = dataset_conf['type']
        
        if dataset_type == 'builtin':
            generate_func = globals().get(dataset_conf['name'])
            preprocess_func = globals().get(dataset_conf.get('proc', None))
            
            data_np = generate_func(**dataset_conf.get('params', {}))
            dataset = Dataset(data = data_np, preprocess_func = preprocess_func)
            test_ratio = dataset_conf.get('test_ratio', 0.2)
            self.train_dataset, self.test_dataset = split_train_and_test_dataset(dataset, test_ratio)
            self.batch_size = dataset_conf.get('batch_size', 10)

        # Parse model structure
        model_layers = []
        for layer_conf in conf.get('model', []):
            layer_type = layer_conf.get('type')
            layer_params = layer_conf.get('params', {})
            layer_class = globals().get(layer_type)
            model_layers.append(layer_class(**layer_params))
        self.model = Sequential(model_layers)

        # Parse loss function
        loss_conf = conf['loss']
        loss_type = loss_conf.get('type', 'MSELoss')
        loss_params = loss_conf.get('params', {})
        loss_class = globals().get(loss_type)
        self.loss = loss_class(**loss_params)

        # Parse optimizer
        optimizer_conf = conf['optimizer']
        optimizer_type = optimizer_conf.get('type', 'GD')
        optimizer_params = optimizer_conf.get('params', { 'lr': 0.01 })
        optimizer_class = globals().get(optimizer_type)
        self.optimizer = optimizer_class(self.model.params, **optimizer_params)

        # Launch task
        if self.is_runnable():
            self.is_stopped.clear()  # False
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
    
    def run(self): # TODO: now only for 2d classification
        train_iter = DataIterator(self.train_dataset, batch_size = self.batch_size, shuffle = True, cyclic = False)

        train_loss_buffer = []
        train_accuracy_buffer = []
        test_loss_buffer = []
        test_accuracy_buffer = []

        # for model output visualization (temporary) TODO
        x1_range = (-6, 6, 60)
        x2_range = (-6, 6, 60)
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

        # destroy the task
        self.train_dataset = None
        self.test_dataset = None
        self.model = None
        self.loss = None
        self.optimizer = None
        
        self.epoch = 0

        # clear frontend view database
        self.update_msg_buffer({ key: None for key in self.get_msg_buffer().keys() })
    
    async def async_ws_listener(self, ws):
        try:
            async for msg in ws:
                data = json.loads(msg)
                """
                {
                    'type': 'poll',
                    'prop_names': ['...', ...] (list of views) or not provided (root)
                }
                """
                if data['type'] == 'poll':
                    if 'prop_names' not in data.keys():
                        # root
                        prop_values = self.get_msg_buffer()
                    else:
                        # list or str
                        prop_names = data['prop_names']
                        if isinstance(prop_names, str):
                            prop_names = [prop_names]
                        prop_values = { prop_name: self.get_msg_buffer().get(prop_name, None) for prop_name in prop_names }
                    
                    await ws.send(json.dumps({
                        'type': 'poll_response',
                        'prop_values': prop_values
                    }))
                else:
                    continue # unknown cmd, do not respond, TODO

        except Exception as e:
            print(f'WebSocket error: {e}')
        
        print("[Info] a client cancelled")

