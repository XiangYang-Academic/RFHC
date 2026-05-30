import torch
from torch.utils.data import Dataset
from processor import Processor
from preprocess import loadData, applyPCA
from sklearn.preprocessing import scale
import numpy as np

class HSI_Data(Dataset):
    def __init__(self, dataset, patch_size=(27, 27), pca_dim=30, is_labeled=True, num_clients=5):
        dataset_info_map = {
            'IP': ('HSI_datasets/Indian_pines_corrected.mat', 'HSI_datasets/Indian_pines_gt.mat'),
            'SA': ('HSI_datasets/Salinas_corrected.mat', 'HSI_datasets/Salinas_gt.mat'),
            'PU': ('HSI_datasets/PaviaU.mat', 'HSI_datasets/PaviaU_gt.mat')
        }
        prefix = dataset.split('_')[0]
        data_1, data_2 = dataset_info_map[prefix]
        X = loadData(prefix)
        parts = dataset.split('_')
        cnt = int(parts[1]) if len(parts) > 1 else -1
        num_splits = num_clients
        splits = np.array_split(X, num_splits, axis=0)
        p = Processor()
        if cnt != -1:
            img = splits[cnt]
            _, gt_full = p.prepare_data(data_1, data_2)
            label_splits = np.array_split(gt_full, num_splits, axis=0)
            gt = label_splits[cnt]
        else:
            img, gt = p.prepare_data(data_1, data_2)
        img = applyPCA(img, numComponents=pca_dim)
        x_patches, y_ = p.get_HSI_patches_rw(img, gt, (patch_size[0], patch_size[1]))
        y = p.standardize_label(y_)
        self.n_classes = np.unique(y).shape[0] - (0 if is_labeled else 1)
        n_samples, n_row, n_col, _ = x_patches.shape
        self.data_size = n_samples
        x_patches = scale(x_patches.reshape((n_samples, -1))).reshape((n_samples, n_row, n_col, -1))
        x_patches = np.transpose(x_patches, axes=(0, 3, 1, 2))
        
        self.x_tensor = torch.from_numpy(x_patches).type(torch.FloatTensor)
        self.y_tensor = torch.from_numpy(y).type(torch.LongTensor)

    def __getitem__(self, idx):
        return self.x_tensor[idx], self.y_tensor[idx]

    def __len__(self):
        return self.data_size