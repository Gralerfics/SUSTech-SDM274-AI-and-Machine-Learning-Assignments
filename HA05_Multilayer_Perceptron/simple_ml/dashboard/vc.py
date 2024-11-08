import json
import threading

from enum import Enum

from sanic import Websocket


class ViewID(Enum):
    EPOCH = 0,
    TRAIN_DATASET = 1
    TEST_DATASET = 2
    MODEL_OUTPUT = 3
    TRAIN_LOSS_BUFFER = 4


class ViewClient:
    def __init__(self, ws: Websocket, task_id: str, view_id, params: dict = {}):
        self.ws = ws
        self.task_id = task_id
        self.view_id = view_id
        self.params = params


class ViewClientsPool:
    def __init__(self):
        self.clients: list[ViewClient] = []
        self.lock = threading.Lock()
    
    def add(self, ws: Websocket, task_id: str, view_id):
        with self.lock:
            self.clients.append(ViewClient(ws, task_id, view_id))
    
    def remove(self, vc: ViewClient):
        with self.lock:
            self.clients.remove(vc)
    
    async def broadcast(self, msg: str):
        with self.lock:
            for vc in self.clients:
                try:
                    await vc.ws.send(msg)
                except Exception as e:
                    # print(f"Failed to send message to a client: {e}")
                    print(f"Deprecated client removed.")
                    self.clients.remove(vc)
    
    def close_all(self):
        with self.lock:
            for vc in self.clients:
                try:
                    vc.ws.close()
                except Exception as e:
                    print(f"Failed to close a client: {e}")
            self.clients = []

