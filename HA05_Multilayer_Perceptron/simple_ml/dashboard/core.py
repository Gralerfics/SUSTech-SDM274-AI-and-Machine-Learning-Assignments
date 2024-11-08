import time
import json
import threading

import numpy as np

from .. import Variable, Dataset, DataIterator, split_train_and_test_dataset
from ..data.samples import label_split_for_2d_classification_dataset, generate_2d_classification_circle, generate_1d_regression_with_function
from ..evaluation.criterion import eval_binary_accuracy
from ..model import Model, Sequential
from ..model.layers import Linear, ReLU, Sigmoid, Tanh
from ..training.loss import MSELoss, CrossEntropyLoss
from ..training.optimizer import GD, MomentumGD, Adam

from .vc import ViewID, ViewClientsPool



class Task:
    def __init__(self, task_id, vc_pool, components: dict):
        self.task_id = task_id
        self.vc_pool: ViewClientsPool = vc_pool

        # TODO: assert
        self.train_dataset = components.get('train_dataset', None)
        self.test_dataset = components.get('test_dataset', None)
        self.model = components.get('model', None)
        self.loss = components.get('loss', None)
        self.optimizer = components.get('optimizer', None)
    
        self.is_stopped = threading.Event() # False
        self.is_resumed = threading.Event()
        self.is_resumed.set() # True

        self.epoch = 0
    
    def stop(self):
        self.is_stopped.set() # True
    
    def pause(self):
        self.is_resumed.clear() # False
    
    def resume(self):
        self.is_resumed.set() # True
    
    def reset(self):
        pass # TODO: reset epoch, model weights, etc.

    async def run(self): # TODO: now only for 2d classification
        train_iter = DataIterator(self.train_dataset, batch_size = 32, shuffle = True, cyclic = False)

        train_loss_buffer = []

        while not self.is_stopped.is_set(): # run if is_stopped = False
            # block until is_resumed = True
            self.is_resumed.wait()

            train_loss = 0
            train_accuracy = 0

            for batch, [features, labels] in enumerate(train_iter):
                prediction = self.model(Variable(features, derivable = True)) # must be wrapped by Variable
                loss = self.loss_func(prediction, labels) # calculate loss and gradient (!)

                self.model.backward()
                self.optimizer.step()

                train_loss += loss
                train_accuracy += eval_binary_accuracy(prediction, labels)
            
            # record training loss and accuracy
            train_loss /= len(self.train_dataset)
            train_accuracy /= len(self.train_dataset)
            train_loss_buffer.append(train_loss)
            # train_accuracy_history.append(train_accuracy)

            # record testing loss and accuracy
            test_prediction = self.model(Variable(self.test_dataset.datas[0], derivable = True))
            test_loss = self.loss_func(test_prediction, self.test_dataset.datas[1])
            test_accuracy = eval_binary_accuracy(test_prediction, self.test_dataset.datas[1])
            # test_loss_history.append(test_loss)
            # test_accuracy_history.append(test_accuracy)

            """ Inform the dashboard views """
            with self.vc_pool.lock:
                for vc in self.vc_pool.clients:
                    if vc.task_id == self.task_id:
                        if vc.view_id == ViewID.EPOCH:
                            msg = json.dumps({
                                'epoch': self.epoch
                            })
                        elif vc.view_id == ViewID.TRAIN_LOSS:
                            msg = json.dumps({
                                'train_loss': train_loss_buffer
                            })
                            train_loss_buffer.clear()
                        
                        try:
                            await client.send(msg)
                        except Exception as e:
                            # print(f"Failed to send message to a client: {e}")
                            print(f"Deprecated client removed.")
                            self.vc_pool.remove(client) # TODO
            """ Inform the dashboard views """

            # next epoch
            self.epoch += 1


class DashboardCore:
    def __init__(self, vc_pool):
        self.vc_pool = vc_pool
        
        self.task = None
    
    def get_state(self):
        return {
            'task_id': self.task.task_id if self.task is not None else None
        }
    
    def launch_task(self, conf):
        """
        {
            'model': [
                {'type': 'Linear', 'params': {...: ...}},
                {'type': 'Sigmoid'},
                ...
            ],
            'loss': {
                'type': 'MSELoss',
                'params': {...: ...}
            },
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
            task_id = 'task_id',
            vc_pool = self.vc_pool,
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

        return True

