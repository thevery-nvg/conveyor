import threading


class ClientManager:
    def __init__(self):
        self.active_clients = set()
        self.lock = threading.Lock()

    def add_client(self, client_id):
        with self.lock:
            self.active_clients.add(client_id)

    def remove_client(self, client_id):
        with self.lock:
            if client_id in self.active_clients:
                self.active_clients.remove(client_id)

class AppState:
    def __init__(self):
        self.producer = None
        self.consumer = None
        self.queues = None
        self.shared_dict=None


app_state = AppState()
client_manager = ClientManager()