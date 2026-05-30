from client import client

class FLSimulator():
    def __init__(self, allocation, args):
        self.num_clients = args.num_clients
        self.data_allocation = allocation
        self.clients = self.create_client_devices(self.num_clients, args, self.data_allocation)
        self.args = args

    def create_client_devices(self, num_clients, args, allocation):
        clients = {}
        for i in range(num_clients):
            clients[i] = client(i, args, Distribute_Array=allocation)
        return clients