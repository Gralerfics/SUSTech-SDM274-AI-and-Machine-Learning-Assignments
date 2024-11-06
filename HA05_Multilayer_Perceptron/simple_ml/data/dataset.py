import os
import random

import numpy as np
import pandas as pd

from ..data.types import Variable


class Dataset:
    """
    def __init__(self, **kwargs):
        self.data: np.ndarray = kwargs.get('data', None)
        self.file_path: str = kwargs.get('file_path', None)
        assert (self.data is not None and self.file_path is None) or (self.data is None and self.file_path is not None) # only one of them should be provided

        self.in_memory: bool = kwargs.get('in_memory', True)
        
        if self.file_path is not None:
            # from disk
            self.file_type = kwargs.get('file_type', 'csv')
            self.file_is_temp = False
            if self.file_type == 'csv':
                # csv
                header = kwargs.get('header', None)
                self.data = pd.read_csv(self.file_path, header = header).to_numpy() if self.in_memory else None # no header in default; read if in_memory, or read in __getitem__
            else:
                pass # TODO: other file types
        else:
            # from given data
            if not self.in_memory:
                self.file_path = os.urandom(16).hex() + '.csv'
                while os.path.exists(self.file_path):
                    self.file_path = os.urandom(16).hex() + '.csv'
                self.file_type = 'csv'
                self.file_is_temp = True
                pd.DataFrame(self.data).to_csv(self.file_path, index = False) # temporary file

                del self.data
                self.data = None
        
        if self.in_memory:
            self.length = self.data.shape[0]
        else:
            if self.file_type == 'csv':
                self.length = sum(1 for _ in open(self.file_path)) - (1 if header is not None else 0)
            else:
                pass # TODO: other file types
    """

    def __init__(self, **kwargs):
        self.data: np.ndarray = kwargs.get('data', None)
        self.file_path: str = kwargs.get('file_path', None)
        assert (self.data is not None and self.file_path is None) or (self.data is None and self.file_path is not None) # only one of them should be provided

        if self.file_path is not None:
            # from disk
            file_type = kwargs.get('file_type', 'csv')
            if file_type == 'csv':
                # csv
                header = kwargs.get('header', None) # no header in default
                self.data = pd.read_csv(self.file_path, header = header).to_numpy() # TODO: header
            else:
                pass # TODO: other file types

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]


class DataIterator:
    def __init__(self, dataset, batch_size = 1, shuffle = False, cyclic = False):
        self.dataset: Dataset = dataset
        self.batch_size: int = batch_size
        self.shuffle: bool = shuffle
        self.cyclic: bool = cyclic

        self.indices: list = list(range(len(dataset)))
        self.next_idx: int = batch_size

        if shuffle:
            random.shuffle(self.indices)

    def __iter__(self):
        return self

    def __next__(self):
        exceed = self.next_idx - len(self.dataset)
        if exceed >= 0:
            if self.cyclic:
                self.next_idx = exceed
                batch_indices = self.indices[(self.next_idx - self.batch_size):]
                if self.shuffle:
                    random.shuffle(self.indices)
                batch_indices.extend(self.indices[:exceed])
            else:
                raise StopIteration
        else:
            batch_indices = self.indices[(self.next_idx - self.batch_size):self.next_idx]
        
        self.next_idx += self.batch_size
        batch_data = self.dataset[batch_indices]
        return batch_data # return in np.ndarray, Model().forward(X) can accept that and automatically wrap it into Variable.


# def split_dataset_train_and_test(dataset, test_ratio = 0.2):


