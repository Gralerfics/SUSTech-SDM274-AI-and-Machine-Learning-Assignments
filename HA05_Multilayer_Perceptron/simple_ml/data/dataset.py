import os
import random
from typing import Union

import numpy as np
import pandas as pd

from ..data.types import Variable


class Dataset:
    def __init__(self, **kwargs):
        self.data_raw: np.ndarray = kwargs.get('data', None) # the whole table
        self.file_path: str = kwargs.get('file_path', None)
        assert (self.data_raw is not None and self.file_path is None) or (self.data_raw is None and self.file_path is not None) # only one of them should be provided

        self.preprocess_func = kwargs.get('preprocess_func', self.default_preprocess_func)

        if self.file_path is not None:
            # from disk
            file_type = kwargs.get('file_type', 'csv')
            if file_type == 'csv':
                # csv
                header = kwargs.get('header', None) # no header in default
                self.data_raw = pd.read_csv(self.file_path, header = header).to_numpy() # TODO: header
            else:
                pass # TODO: other file types
        
        self.length = self.data_raw.shape[0]

        self.datas: Union[np.ndarray, list[np.ndarray]] = self.preprocess_func(self.data_raw)
            # np.ndarray for the case of only one data source, list[np.ndarray] for the case of multiple data sources
    
    """ @Override """
    def default_preprocess_func(self, data: np.ndarray):
        return data # only one element, no list wrapping

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if isinstance(self.datas, np.ndarray):
            return self.datas[index]
        elif isinstance(self.datas, list) or isinstance(self.datas, tuple):
            return [d[index] for d in self.datas]


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


