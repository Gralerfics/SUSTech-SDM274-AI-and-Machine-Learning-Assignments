import json
import threading

from sanic import Websocket


class ClientsPool:
    def __init__(self):
        self.clients: list[Websocket] = []
        self.lock = threading.Lock()
    
    def add(self, ws: Websocket):
        with self.lock:
            self.clients.append(ws)
    
    def remove(self, ws: Websocket):
        with self.lock:
            self.clients.remove(ws)
    
    async def broadcast(self, msg: str):
        with self.lock:
            for ws in self.clients:
                try:
                    await ws.send(msg)
                except Exception as e:
                    # print(f"Failed to send message to a client: {e}")
                    print(f"[Info] Deprecated client removed")
                    self.clients.remove(ws)
    
    def close_all(self):
        with self.lock:
            for ws in self.clients:
                try:
                    ws.close()
                except Exception as e:
                    print(f"Failed to close a client: {e}")
            self.clients.clear()

