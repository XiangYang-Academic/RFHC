import numpy as np
import scipy.io as sio
import os
from sklearn.decomposition import PCA
from operator import truediv
from utils import read_mat
from processor import Processor


def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX


def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX


def createImageCubes(X, windowSize=18):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]), dtype=np.float32)
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchIndex = patchIndex + 1
    return patchesData


def loadData(name):
    data_path = 'HSI_datasets'
    if name == 'IP':
        data = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
    elif name == 'SA':
        data = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
    elif name == 'PU':
        data = sio.loadmat(os.path.join(data_path, 'PaviaU.mat'))['paviaU']
    return data


def feature_normalize(data):
    mu = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return truediv((data - mu), std)


def Preprocess(XPath, dataset, Windowsize=25, Patch_channel=15, num_splits=5):
    dataset_map = {
        'IP': ('HSI_datasets/Indian_pines_corrected.mat', 'HSI_datasets/Indian_pines_gt.mat'),
        'SA': ('HSI_datasets/Salinas_corrected.mat', 'HSI_datasets/Salinas_gt.mat'),
        'PU': ('HSI_datasets/PaviaU.mat', 'HSI_datasets/PaviaU_gt.mat')
    }
    
    if dataset not in dataset_map:
        raise ValueError(f"Unsupported dataset: {dataset}")
        
    data_1, data_2 = dataset_map[dataset]
    p = Processor()
    _, gt = p.prepare_data(data_1, data_2)
    X_raw = loadData(dataset)
    splits = np.array_split(X_raw, num_splits, axis=0)
    label_splits = np.array_split(gt, num_splits, axis=0)
    X_global = applyPCA(X_raw, numComponents=Patch_channel)
    X_global, _ = p.get_HSI_patches_rw(X_global, gt, (Windowsize, Windowsize))
    X_global = feature_normalize(X_global)
    np.save(XPath, X_global)
    for i in range(num_splits):
        split_pca = applyPCA(splits[i], numComponents=Patch_channel)
        split_patched, _ = p.get_HSI_patches_rw(split_pca, label_splits[i], (Windowsize, Windowsize))
        split_norm = feature_normalize(split_patched)
        split_path = f'DataArray/{dataset}_{i}.npy'
        np.save(split_path, split_norm)

def load_label(dataset_name, num_splits=5):
    dataset_path_map = {
        'IP': 'HSI_datasets/Indian_pines_gt.mat',
        'SA': 'HSI_datasets/Salinas_gt.mat',
        'PU': 'HSI_datasets/PaviaU_gt.mat'
    }
    parts = dataset_name.split('_')
    prefix = parts[0]
    cnt = int(parts[1]) if len(parts) == 2 else -1
    label_path = dataset_path_map.get(prefix)
    if not label_path:
        raise ValueError(f"Unknown dataset prefix: {prefix}")
    label = read_mat(label_path).astype(np.int32) - 1
    if cnt < 0:
        label_flat = label.reshape(-1)
        return label_flat[label_flat >= 0]
    else:
        label_splits = np.array_split(label, num_splits, axis=0)
        target_split_flat = label_splits[cnt].reshape(-1)
        return target_split_flat[target_split_flat >= 0]