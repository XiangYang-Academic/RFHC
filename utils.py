import torch
from torch_clustering import PyTorchKMeans
from sklearn import cluster, metrics
from torch.utils.data.dataset import Dataset
import random
import numpy as np
from cal_metric import full_metric
import scipy.sparse as sp
import scipy.io as sio
import os
from scipy.optimize import linear_sum_assignment as linear_assignment


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class AEDataset(Dataset):
    def __init__(self, Datapath, judge=-1):
        self.Datalist = np.load(Datapath)
        excluded_datasets = ('DataArray/IP_X.npy')
        if Datapath not in excluded_datasets and judge >= 0:
            np.random.shuffle(self.Datalist)

    def __getitem__(self, index):
        Data = self.Datalist[index].astype(np.float32)
        Data_tensor = torch.from_numpy(Data).permute(2, 0, 1)
        Data_tensor = Data_tensor.unsqueeze(0)
        return Data_tensor

    def __len__(self):
        return len(self.Datalist)

def read_mat(filename):
    mat = sio.loadmat(filename)
    keys = [k for k in mat.keys() if k != '__version__' and k != '__header__' and k != '__globals__']
    arr = mat[keys[0]]
    return arr


def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)


def mkmeans(X, nclusters):
    k_means = cluster.MiniBatchKMeans(n_clusters=nclusters)
    k_means.fit(X)
    y_pred = k_means.predict(X)
    return y_pred


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    w = np.transpose(metrics.confusion_matrix(y_true, y_pred))
    row_ind, col_ind = linear_assignment(w.max() - w)
    return np.sum([w[row_ind[i], col_ind[i]] for i in range(0, len(row_ind))]) * 1.0 / y_pred.size


def my_clustering_gpu(feature, true_labels, cluster_num, judge):
    kmeans = PyTorchKMeans(metric='euclidean', init='k-means++', n_clusters=cluster_num, n_init=10, verbose=False)
    feature = torch.tensor(feature, dtype=torch.float32)
    predict_labels = kmeans.fit_predict(feature)
    center = kmeans.cluster_centers_
    dis = torch.cdist(feature, center, p=2)
    dis = dis.cpu()
    predict_labels = predict_labels.cpu().numpy()
    pixel_pred = predict_labels
    OA, AA, KAPPA, NMI, ARI, F1, PRECISION, RECALL, PURITY = full_metric(true_labels, pixel_pred, is_refined=False)
    return OA, AA, KAPPA, NMI, ARI, F1, PRECISION, RECALL, PURITY, predict_labels, dis


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def process_adj(adj, norm='sym', renorm=True):
    adj = adj - np.diag(np.diag(adj))
    adj = sp.csr_matrix(adj)
    adj.eliminate_zeros()
    adj = sp.coo_matrix(adj)
    if renorm:
        adj = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    if norm == 'sym':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj.dot(degree_mat_inv_sqrt).transpose().dot(
            degree_mat_inv_sqrt).tocoo()
    elif norm == 'left':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -1.).flatten())
        adj_normalized = degree_mat_inv_sqrt.dot(adj).tocoo()
    adj_normalized = sparse_mx_to_torch_sparse_tensor(adj_normalized)
    return adj_normalized

def random_allocation(Patch_dataset, num_clients):
    dataset_length = len(Patch_dataset)
    if num_clients > dataset_length:
        raise ValueError("num_clients cannot be greater than the length of Patch_dataset")
    indices = list(range(dataset_length))
    random.shuffle(indices)
    allocation = indices[:num_clients]
    return allocation


def load_allocation(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} Not found.")
    with open(file_path, "r") as file:
        content = file.read()
        allocation = list(map(int, content.split(",")))
    return allocation


def read_npy_file(file_path):
    try:
        data = np.load(file_path)
        data_shape = data.shape
        data_size = data.size

        return data, data_shape, data_size
    except Exception as e:
        print(f"Error occurred while reading file: {e}")