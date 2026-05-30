import abc
import time
import torch
import random
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from models import Enc_AE
from preprocess import load_label
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from torch.utils.data.dataloader import DataLoader
from utils import my_clustering_gpu, read_npy_file, AEDataset

class AggregationStrategy(abc.ABC):
    @abc.abstractmethod
    def aggregate(self, global_sd, updates):
        pass

class FedAvgStrategy(AggregationStrategy):
    def aggregate(self, global_sd, updates):
        sum_n = sum(update['sampleCount'] for update in updates)
        layers = updates[0]['update'].keys()
        new_global_sd = {layer: torch.zeros_like(global_sd[layer], dtype=torch.float32) for layer in layers}
        for update in updates:
            client_sd = update['update']
            n_k = update['sampleCount']
            weight = n_k / sum_n

            for layer in layers:
                new_global_sd[layer] += client_sd[layer].float() * weight

        return new_global_sd

class FedNovaStrategy(AggregationStrategy):
    def aggregate(self, global_sd, updates):
        sum_n = sum(update['sampleCount'] for update in updates)
        layers = updates[0]['update'].keys()
        aggregated_delta = {layer: torch.zeros_like(global_sd[layer], dtype=torch.float32) for layer in layers}

        for update in updates:
            client_sd = update['update']
            n_k = update['sampleCount']
            tau_k = update['tau_k']
            delta_k = {}
            for layer in layers:
                client_val = client_sd[layer].float()
                global_val = global_sd[layer].float()
                delta_k[layer] = client_val - global_val
                normalized_delta = delta_k[layer] / tau_k
                aggregated_delta[layer] += (n_k / sum_n) * normalized_delta
        new_global_sd = {}
        for layer in global_sd:
            new_global_sd[layer] = global_sd[layer].float() + aggregated_delta[layer].float()

        return new_global_sd

class FederatedAggregation():
    def __init__(self, strategy_name: str = 'fednova', args=None):
        self.args = args
        strategy_name = strategy_name.lower()

        if strategy_name == 'fedavg':
            self.strategy = FedAvgStrategy()
        elif strategy_name == 'fednova':
            self.strategy = FedNovaStrategy()
        else:
            raise ValueError(f"Unsupported aggregation strategies: {strategy_name}.")

    def aggregate(self, global_model, client_state_dict):
        current_global_sd = global_model.state_dict()
        new_global_sd = self.strategy.aggregate(current_global_sd, client_state_dict)
        global_model.load_state_dict(new_global_sd)

def order_sam_for_diag(x, y):
    x_new = np.zeros(x.shape)
    y_new = np.zeros(y.shape)
    start = 0
    for i in np.unique(y):
        idx = np.nonzero(y == i)
        stop = start + idx[0].shape[0]
        x_new[start:stop] = x[idx]
        y_new[start:stop] = y[idx]
        start = stop
    return x_new, y_new

class server():
    def __init__(self, clients, args):
        self.clients = clients
        strategy_name = getattr(args, 'agg_method', 'fednova')
        self.aggregator = FederatedAggregation(strategy_name=strategy_name, args=args)
        self.args = args
        self.num_clients = args.num_clients
        self.device = args.device
        self.dataset_name = args.dataset
        self.encoded_dim = args.encoded_dim
        self.batch_size = args.batch_size
        self.windowsize = args.windowsize
        self.num_label = args.num_label
        self.Patch_channel = args.Patch_channel
        self.cluster_global = None
        XPath = f'DataArray/{args.dataset}_X.npy'
        self.Patch_dataset = AEDataset(XPath)
        self.globalEnc_patch = Enc_AE(channel=self.Patch_channel, output_dim=self.encoded_dim,
                                      windowSize=self.windowsize).to(self.device)

    def random_select_clients(self, clients, number):
        return random.sample(list(clients), number)

    def federated_train(self, num_comm_round, instance_coeff, cross_cor_coeff):
        test_accuracies = []
        best_oa = -1
        best_metrics = None
        start_time = time.time()
        for round in tqdm(range(num_comm_round), desc=f"Federated Training ({self.dataset_name})"):
            selected_clients = self.random_select_clients(self.clients, self.num_clients)

            round_updates, local_encodings, local_ids = [], [], []
            for client_id in selected_clients:
                client_update, client_encoding, attr_num = self.clients[client_id].localTrain(
                    self.globalEnc_patch.state_dict(),
                    cluster_global=self.cluster_global,
                    local_cluster=self.local_cluster,
                    instance_coeff=instance_coeff,
                    cross_cor_coeff=cross_cor_coeff,
                    Round=round
                )
                round_updates.append(client_update)
                local_encodings.append(client_encoding)
                local_ids.append(attr_num)
            if len(round_updates) == 0:
                raise ValueError("round_updates is empty; the average cannot be calculated")
            self.aggregator.aggregate(self.globalEnc_patch, round_updates)

            sorted_pairs = sorted(zip(local_ids, local_encodings), key=lambda x: x[0])
            local_encodings_sorted = [pair[1] for pair in sorted_pairs]
            merged_array = np.concatenate(local_encodings_sorted, axis=0)

            kmeans = KMeans(n_clusters=self.num_label, n_init=10)
            cluster_labels = kmeans.fit_predict(merged_array)
            self.local_cluster = []
            self.local_cluster.append(cluster_labels[:self.index[0]])
            for i in range(1, self.num_clients):
                self.local_cluster.append(cluster_labels[self.index[i - 1]:self.index[i]])
            merged_array = torch.tensor(merged_array, dtype=torch.float32)
            self.cluster_global = self.compute_centers(self.num_label, merged_array, cluster_labels)
            metrics = self.test()
            oa = metrics[0]
            test_accuracies.append(oa)
            if oa > best_oa:
                best_oa = oa
                best_metrics = metrics

        total_time = time.time() - start_time
        if best_metrics:
            oa, aa, kappa, nmi, ari, f1, precision, recall, purity = best_metrics
            print(f"\n--- RFHC on {self.dataset_name} ---")
            print(
                f"{'OA':>12} | {'AA':>12} | {'Kappa':>12} | {'NMI':>12} | {'ARI':>12} | {'F1':>12} | {'Precision':>12} | {'Recall':>12} | {'Purity':>12} | {'Time':>8}")
            print("-" * 135)
            print(
                f"{oa:12.8f} | {aa:12.8f} | {kappa:12.8f} | {nmi:12.8f} | {ari:12.8f} | {f1:12.8f} | {precision:12.8f} | {recall:12.8f} | {purity:12.8f} | {total_time:8.2f}")
            print("\n")

        return test_accuracies

    def compute_centers(self, N_CLASSES, x, cluster_labels):
        n_samples = x.size(0)
        if len(torch.from_numpy(cluster_labels).size()) > 1:
            weight = cluster_labels.T
        else:
            weight = torch.zeros(N_CLASSES, n_samples).to(x)
            weight[cluster_labels, torch.arange(n_samples)] = 1
        weight = F.normalize(weight, p=1, dim=1)
        centers = torch.mm(weight, x)
        centers = F.normalize(centers, dim=1)
        return centers

    def test(self):
        Data = self.Patch_dataset
        eval_loader = DataLoader(dataset=Data, batch_size=4096, shuffle=False)
        self.globalEnc_patch.eval()
        global_encodings = torch.FloatTensor([])
        with torch.set_grad_enabled(False):
            for data in eval_loader:
                data = data.float().to(self.device)
                encoding = self.globalEnc_patch(data)
                global_encodings = torch.cat([global_encodings, encoding.detach().cpu()], dim=0)

        global_encodings = global_encodings.numpy()
        scaler = StandardScaler()
        global_encodings = scaler.fit_transform(global_encodings)

        self.label_gt = load_label(self.dataset_name)
        self.N_CLASSES = np.unique(self.label_gt).shape[0]
        clustering_results = my_clustering_gpu(global_encodings, self.label_gt, self.N_CLASSES, self.dataset_name)
        if self.cluster_global is None:
            kmeans = KMeans(n_clusters=self.num_label, n_init=10)
            cluster_labels = kmeans.fit_predict(global_encodings)
            encoding_tensor = torch.tensor(global_encodings, dtype=torch.float32)
            self.index = []
            for i in range(self.num_clients):
                _, client_index, _ = read_npy_file(f'DataArray/{self.dataset_name}_{i}.npy')
                self.index.append(client_index[0])
            for i in range(1, self.num_clients):
                self.index[i] = self.index[i - 1] + self.index[i]
            self.local_cluster = []
            self.local_cluster.append(cluster_labels[:self.index[0]])
            for i in range(1, self.num_clients):
                self.local_cluster.append(cluster_labels[self.index[i - 1]:self.index[i]])
            self.cluster_global = self.compute_centers(self.num_label, encoding_tensor, cluster_labels)

        oa = clustering_results[0]
        aa = clustering_results[1]
        kappa = clustering_results[2]
        nmi = clustering_results[3]
        ari = clustering_results[4]
        f1 = clustering_results[5]
        precision = clustering_results[6]
        recall = clustering_results[7]
        purity = clustering_results[8]

        return oa, aa, kappa, nmi, ari, f1, precision, recall, purity