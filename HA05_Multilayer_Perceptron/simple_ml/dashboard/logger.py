import json
import numpy as np


class TrainingDataLogger:
    def __init__(self, client):
        """
            TODO: 暂时依照 1D Regression 的例子实现
        """
        self.client = client

        self.train_loss_history = []
        self.train_r2_history = []
        self.test_loss_history = []
        self.test_r2_history = []

    def log(self, epoch, train_loss, train_r2, test_loss, test_r2):
        # record the data
        self.train_loss_history.append(train_loss)
        self.train_r2_history.append(train_r2)
        self.test_loss_history.append(test_loss)
        self.test_r2_history.append(test_r2)

        # prepare the data
        data = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_r2': train_r2,
            'test_loss': test_loss,
            'test_r2': test_r2,
            'train_loss_history': self.train_loss_history,
            'train_r2_history': self.train_r2_history,
            'test_loss_history': self.test_loss_history,
            'test_r2_history': self.test_r2_history
        }

        # send the data to the frontend
        self.client.send(json.dumps(data))

    def reset(self):
        self.train_loss_history.clear()
        self.train_r2_history.clear()
        self.test_loss_history.clear()
        self.test_r2_history.clear()

