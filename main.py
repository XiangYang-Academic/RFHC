import os
import torch
import warnings
import random
from FLSimulator import FLSimulator
from preprocess import load_label
from server import server

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class Args():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_clients = 5
        self.num_clients_round = 1

        self.num_local_epochs = 1
        self.lr = 1e-6
        self.max_no_better = 150
        self.dataset = "IP"
        self.data_dir = "DataArray/{self.dataset}_X.npy"
        self.num_label = 16
        self.label_gt = load_label(self.dataset, self.num_clients)

        self.windowsize = 27
        self.encoded_dim = 128
        self.batch_size = 128
        self.workers = 1
        self.Patch_channel = 30
        self.agg_method = "fedavg"


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    args = Args()
    loaded_allocation = list(range(args.num_clients))
    random.shuffle(loaded_allocation)
    for attr_num in loaded_allocation:
        expected_file = f'DataArray/{args.dataset}_{attr_num}.npy'
        if not os.path.exists(expected_file):
            raise FileNotFoundError(
                f"\n[Critical Error] Missing client data block file: {expected_file}\n"
            )

    fl_simulator = FLSimulator(loaded_allocation, args)
    central_server = server(fl_simulator.clients, args)
    central_server.test()
    central_server.federated_train(100, 10, 100)