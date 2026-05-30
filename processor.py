from __future__ import print_function
import numpy as np
from rolling_window import rolling_window as rw
import spectral as spy
import torch
import torch.nn.functional as F

class Processor:
    def __init__(self):
        pass

    def prepare_data(self, img_path, gt_path):
        if img_path[-3:] == 'mat':
            import scipy.io as sio
            img_mat = sio.loadmat(img_path)
            gt_mat = sio.loadmat(gt_path)
            img_keys = img_mat.keys()
            gt_keys = gt_mat.keys()
            img_key = [k for k in img_keys if k != '__version__' and k != '__header__' and k != '__globals__']
            gt_key = [k for k in gt_keys if k != '__version__' and k != '__header__' and k != '__globals__']
            return img_mat.get(img_key[0]).astype('float64'), gt_mat.get(gt_key[0]).astype('int8')
        else:
            import spectral as spy
            img = spy.open_image(img_path).load()
            gt = spy.open_image(gt_path)
            a = spy.principal_components()
            a.transform()
            return img, gt.read_band(0)

    def get_correct(self, img, gt):
        gt_1D = gt.reshape(-1)
        index = gt_1D.nonzero()
        gt_correct = gt_1D[index]
        img_2D = img.reshape(img.shape[0] * img.shape[1], img.shape[2])
        img_correct = img_2D[index]
        return img_correct, gt_correct

    def get_tr_tx_index(self, y, test_size=0.9):
        from sklearn.model_selection import train_test_split
        train_index, test_index, y_train_, y_test_ = \
            train_test_split(np.arange(0, y.shape[0]), y, test_size=test_size)
        return train_index, test_index

    def divide_img_blocks(self, img, gt, block_size=(5, 5)):
        w_1, w_2 = int((block_size[0] - 1) / 2), int((block_size[1] - 1) / 2)
        img_padding = np.pad(img, ((w_1, w_2),
                                   (w_1, w_2), (0, 0)), 'reflect')
        gt_padding = np.pad(gt, ((w_1, w_2),
                                 (w_1, w_2)), 'reflect')
        img_blocks = rw(img_padding, block_size, axes=(1, 0))
        gt_blocks = rw(gt_padding, block_size, axes=(1, 0))
        i_1, i_2 = int((block_size[0] - 1) / 2), int((block_size[0] - 1) / 2)
        nonzero_index = gt_blocks[:, :, i_1, i_2].nonzero()
        img_blocks_nonzero = img_blocks[nonzero_index]
        gt_blocks_nonzero = (gt_blocks[:, :, i_1, i_2])[nonzero_index]
        return img_blocks_nonzero, gt_blocks_nonzero

    def get_HSI_patches(self, x, gt, ksize, stride=(1, 1), padding='reflect', indix=False):
        new_height = np.ceil(x.shape[0] / stride[0])
        new_width = np.ceil(x.shape[1] / stride[1])
        pad_needed_height = (new_height - 1) * stride[0] + ksize[0] - x.shape[0]
        pad_needed_width = (new_width - 1) * stride[1] + ksize[1] - x.shape[1]
        pad_top = int(pad_needed_height / 2)
        pad_down = int(pad_needed_height - pad_top)
        pad_left = int(pad_needed_width / 2)
        pad_right = int(pad_needed_width - pad_left)
        x = np.pad(x, ((pad_top, pad_down), (pad_left, pad_right), (0, 0)), padding)
        gt = np.pad(gt, ((pad_top, pad_down), (pad_left, pad_right)), padding)
        n_row, n_clm, n_band = x.shape
        x_t = torch.from_numpy(x.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        ks = (ksize[0], ksize[1])
        st = (stride[0], stride[1])
        x_unf = F.unfold(x_t, kernel_size=ks, stride=st)
        L = x_unf.shape[-1]
        C = n_band
        x_patches = x_unf[0].transpose(0, 1).contiguous().view(L, C, ks[0], ks[1]).permute(0, 2, 3, 1).numpy()
        y_t = torch.from_numpy(np.reshape(gt, (n_row, n_clm)).astype(np.float32)).unsqueeze(0).unsqueeze(0)
        y_unf = F.unfold(y_t, kernel_size=ks, stride=st)
        y_patches = y_unf[0].transpose(0, 1).contiguous().view(L, 1, ks[0], ks[1]).permute(0, 2, 3, 1).numpy()
        i_1, i_2 = int((ksize[0] - 1) // 2), int((ksize[1] - 1) // 2)
        y_center_label = y_patches[:, i_1, i_2, 0]
        nonzero_index = np.nonzero(y_center_label)
        x_patches_nonzero = x_patches[nonzero_index]
        y_patches_nonzero = y_center_label[nonzero_index]
        if indix is True:
            return x_patches_nonzero, y_patches_nonzero, nonzero_index
        return x_patches_nonzero, y_patches_nonzero

    def get_HSI_patches_rw(self, x, gt, ksize, stride=(1, 1), padding='reflect', indix=False):
        new_height = np.ceil(x.shape[0] / stride[0])
        new_width = np.ceil(x.shape[1] / stride[1])
        pad_needed_height = (new_height - 1) * stride[0] + ksize[0] - x.shape[0]
        pad_needed_width = (new_width - 1) * stride[1] + ksize[1] - x.shape[1]
        pad_top = int(pad_needed_height / 2)
        pad_down = int(pad_needed_height - pad_top)
        pad_left = int(pad_needed_width / 2)
        pad_right = int(pad_needed_width - pad_left)
        x = np.pad(x, ((pad_top, pad_down), (pad_left, pad_right), (0, 0)), padding)
        gt = np.pad(gt, ((pad_top, pad_down), (pad_left, pad_right)), padding)
        n_row, n_clm, n_band = x.shape
        x = np.reshape(x, (n_row, n_clm, n_band))
        y = np.reshape(gt, (n_row, n_clm))
        ksizes_ = (ksize[0], ksize[1])
        x_patches = rw(x, ksizes_, axes=(1, 0))
        y_patches = rw(y, ksizes_, axes=(1, 0))
        i_1, i_2 = int((ksize[0] - 1) // 2), int((ksize[0] - 1) // 2)
        nonzero_index = y_patches[:, :, i_1, i_2].nonzero()
        x_patches_nonzero = x_patches[nonzero_index]
        y_patches_nonzero = (y_patches[:, :, i_1, i_2])[nonzero_index]
        x_patches_nonzero = np.transpose(x_patches_nonzero, [0, 2, 3, 1])
        if indix is True:
            return x_patches_nonzero, y_patches_nonzero, nonzero_index
        return x_patches_nonzero, y_patches_nonzero

    def split_tr_tx(self, X, y, test_size=0.4):
        from sklearn.model_selection import train_test_split
        return train_test_split(X, y, test_size=test_size)

    def split_each_class(self, X, y, each_train_size=10):
        X_tr, y_tr, X_ts, y_ts = [], [], [], []
        for c in np.unique(y):
            y_index = np.nonzero(y == c)[0]
            np.random.shuffle(y_index)
            cho, non_cho = np.split(y_index, [each_train_size, ])
            X_tr.append(X[cho])
            y_tr.append(y[cho])
            X_ts.append(X[non_cho])
            y_ts.append(y[non_cho])
        X_tr, X_ts, y_tr, y_ts = np.asarray(X_tr), np.asarray(X_ts), np.asarray(y_tr), np.asarray(y_ts)
        return X_tr.reshape(X_tr.shape[0] * X_tr.shape[1], X.shape[1]),\
               X_ts.reshape(X_ts.shape[0] * X_ts.shape[1], X.shape[1]), \
               y_tr.flatten(), y_ts.flatten()

    def stratified_train_test_index(self, y, train_size):
        train_idx, test_idx = [], []
        for i in np.unique(y):
            idx = np.nonzero(y == i)[0]
            np.random.shuffle(idx)
            num = np.sum(y == i)
            if 0. < train_size < 1.:
                train_size_ = int(np.ceil(train_size * num))
            elif train_size > num or train_size <= 0.:
                raise Exception('Invalid training size.')
            else:
                train_size_ = np.copy(train_size)
            train_idx += idx[:train_size_].tolist()
            test_idx += idx[train_size_:].tolist()
        train_idx = np.asarray(train_idx).reshape(-1)
        test_idx = np.asarray(test_idx).reshape(-1)
        np.random.shuffle(train_idx)
        np.random.shuffle(test_idx)
        return train_idx, test_idx

    def save_experiment(self, y_pre, y_test, file_neme=None, parameters=None):
        import os
        home = os.getcwd() + '/experiments'
        if not os.path.exists(home):
            os.makedirs(home)
        if parameters == None:
            parameters = [None]
        if file_neme == None:
            file_neme = home + '/scores.npz'
        else:
            file_neme = home + '/' + file_neme + '.npz'

        ca, oa, aa, kappa = [], [], [], []
        if np.array(y_pre).shape.__len__() > 1:
            for y in y_pre:
                ca_, oa_, aa_, kappa_ = self.score(y_test, y)
                ca.append(ca_), oa.append(oa_), aa.append(aa_), kappa.append(kappa_)
        else:
            ca, oa, aa, kappa = self.score(y_test, y_pre)
        np.savez(file_neme, y_test=y_test, y_pre=y_pre, CA=np.array(ca), OA=np.array(oa), AA=aa, Kappa=kappa,
                 param=parameters)
        print('the experiments have been saved in experiments/scores.npz')

    def majority_filter(self, classes_map, selems):
        from skimage.filters.rank import modal
        classes_map__ = classes_map.astype(np.uint16)
        out = classes_map__
        for selem in selems:
            out = modal(classes_map__, selem)
            classes_map__ = out
        return out.astype(np.int8)

    def score(self, y_test, y_predicted):
        from sklearn.metrics import accuracy_score
        oa = accuracy_score(y_test, y_predicted)
        n_classes = max([np.unique(y_test).__len__(), np.unique(y_predicted).__len__()])
        ca = []
        for c in np.unique(y_test):
            y_c = y_test[np.nonzero(y_test == c)]
            y_c_p = y_predicted[np.nonzero(y_test == c)]
            acurracy = accuracy_score(y_c, y_c_p)
            ca.append(acurracy)
        ca = np.array(ca)
        aa = ca.mean()
        kappa = self.kappa(y_test, y_predicted)
        return ca, oa, aa, kappa

    def result2gt(self, y_predicted, test_indexes, gt):
        n_row, n_col = gt.shape
        gt_1D = gt.reshape((n_row * n_col))
        gt_1D[test_indexes] = y_predicted
        return gt_1D.reshape(n_row, n_col)

    def extended_morphological_profile(self, components, disk_radius):
        rows, cols, bands = components.shape
        n = disk_radius.__len__()
        import numpy as np
        emp = np.zeros((rows * cols, bands * (2 * n + 1)))
        from skimage.morphology import opening, closing, disk
        for band in range(bands):
            position = band * (n * 2 + 1) + n
            emp_ = np.zeros((rows, cols, 2 * n + 1))
            emp_[:, :, n] = components[:, :, band]
            i = 1
            for r in disk_radius:
                closed = closing(components[:, :, band], selem=disk(r))
                opened = opening(components[:, :, band], selem=disk(r))
                emp_[:, :, n - i] = closed
                emp_[:, :, n + i] = opened
                i += 1
            emp[:, position - n:position + n + 1] = emp_.reshape((rows * cols, 2 * n + 1))
        return emp.reshape(rows, cols, bands * (2 * n + 1))

    def texture_feature(self, components, theta_arr=None, frequency_arr=None):
        if theta_arr == None:
            theta_arr = np.arange(0, 8) * np.pi / 4
        if frequency_arr == None:
            frequency_arr = np.pi / (2 ** np.arange(1, 5))

        from skimage.filters import gabor
        results = []
        for img in components.transpose():
            for theta in theta_arr:
                for fre in frequency_arr:
                    filt_real, filt_imag = gabor(img, frequency=fre, theta=theta)
                    results.append(filt_real)
        return np.array(results).transpose()

    def pca_transform(self, n_components, samples):
        HSI_or_not = samples.shape.__len__() == 3
        n_row, n_column, n_bands = 0, 0, 0
        if HSI_or_not:
            n_row, n_column, n_bands = samples.shape
            samples = samples.reshape((n_row * n_column, n_bands))
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        trans_samples = pca.fit_transform(samples)
        if HSI_or_not:
            return trans_samples.reshape((n_row, n_column, n_components))
        return trans_samples

    def normlize_HSI(self, img):
        from sklearn.preprocessing import normalize
        n_row, n_column, n_bands = img.shape
        norm_img = normalize(img.reshape(n_row * n_column, n_bands))
        return norm_img.reshape(n_row, n_column, n_bands)

    def each_class_OA(self, y_test, y_predicted):
        classes = np.unique(y_test)
        results = []
        for c in classes:
            y_c = y_test[np.nonzero(y_test == c)]
            y_c_p = y_predicted[np.nonzero(y_test == c)]
            acurracy = self.score(y_c, y_c_p)
            results.append(acurracy)
        return np.array(results)

    def kappa(self, y_test, y_predicted):
        from sklearn.metrics import cohen_kappa_score
        return round(cohen_kappa_score(y_test, y_predicted), 3)

    def color_legend(self, color_map, label):
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        size = len(label)
        patchs = []
        m = 255.
        color_map_ = (color_map / m)[1:]
        for i in range(0, size):
            patchs.append(mpatches.Patch(color=color_map_[i], label=label[i]))
        return patchs

    def get_tr_ts_index_num(self, y, n_labeled=10):
        import random
        classes = np.unique(y)
        X_train_index, X_test_index = np.empty(0, dtype='int8'), np.empty(0, dtype='int8')
        for c in classes:
            index_c = np.nonzero(y == c)[0]
            random.shuffle(index_c)
            X_train_index = np.append(X_train_index, index_c[:n_labeled])
            X_test_index = np.append(X_test_index, index_c[n_labeled:])
        return X_train_index, X_test_index

    def save_res_4kfolds_cv(self, y_pres, y_tests, file_name=None, verbose=False):
        ca, oa, aa, kappa = [], [], [], []
        for y_p, y_t in zip(y_pres, y_tests):
            ca_, oa_, aa_, kappa_ = self.score(y_t, y_p)
            ca.append(np.asarray(ca_)), oa.append(np.asarray(oa_)), aa.append(np.asarray(aa_)),
            kappa.append(np.asarray(kappa_))
        ca = np.asarray(ca) * 100
        oa = np.asarray(oa) * 100
        aa = np.asarray(aa) * 100
        kappa = np.asarray(kappa)
        ca_mean, ca_std = np.round(ca.mean(axis=0), 2), np.round(ca.std(axis=0), 2)
        oa_mean, oa_std = np.round(oa.mean(), 2), np.round(oa.std(), 2)
        aa_mean, aa_std = np.round(aa.mean(), 2), np.round(aa.std(), 2)
        kappa_mean, kappa_std = np.round(kappa.mean(), 3), np.round(kappa.std(), 3)
        if file_name is not None:
            file_name = 'scores.npz'
            np.savez(file_name, y_test=y_tests, y_pre=y_pres,
                     ca_mean=ca_mean, ca_std=ca_std,
                     oa_mean=oa_mean, oa_std=oa_std,
                     aa_mean=aa_mean, aa_std=aa_std,
                     kappa_mean=kappa_mean, kappa_std=kappa_std)
            print('the experiments have been saved in ', file_name)

        if verbose is True:
            print('---------------------------------------------')
            print('ca\t\t', '\taa\t\t', '\toa\t\t', '\tkappa\t\t')
            print(ca_mean, '+-', ca_std)
            print(aa_mean, '+-', aa_std)
            print(oa_mean, '+-', oa_std)
            print(kappa_mean, '+-', kappa_std)

        return np.asarray([ca_mean, ca_std]), np.asarray([aa_mean, aa_std]), \
               np.asarray([oa_mean, oa_std]), np.asarray([kappa_mean, kappa_std])

    def view_clz_map_spyversion4single_img(self, gt, y_test_index, y_predicted, save_path=None, show_error=False,
                                           show_axis=False):
        n_row, n_column = gt.shape
        gt_1d = gt.reshape(-1).copy()
        nonzero_index = gt_1d.nonzero()
        gt_corrected = gt_1d[nonzero_index]
        if show_error:
            t = y_predicted.copy()
            correct_index = np.nonzero(y_predicted == gt_corrected[y_test_index])
            t[correct_index] = 0
            gt_corrected[:] = 0
            gt_corrected[y_test_index] = t
            gt_1d[nonzero_index] = t
        else:
            gt_corrected[y_test_index] = y_predicted
            gt_1d[nonzero_index] = gt_corrected
        gt_map = gt_1d.reshape((n_row, n_column)).astype('uint8')
        spy.imshow(classes=gt_map)
        if save_path != None:
            import matplotlib.pyplot as plt
            spy.save_rgb('temp.png', gt_map, colors=spy.spy_colors)
            if show_axis:
                plt.savefig(save_path, format='eps', bbox_inches='tight')
            else:
                plt.axis('off')
                plt.savefig(save_path, format='eps', bbox_inches='tight')
            print('the figure is saved in ', save_path)

    def classification_map(self, map, groundTruth, dpi, savePath):
        import matplotlib.pyplot as plt
        fig = plt.figure(frameon=False)
        fig.set_size_inches(groundTruth.shape[1] * 2.0 / dpi, groundTruth.shape[0] * 2.0 / dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        fig.add_axes(ax)

        ax.imshow(map, aspect='normal')
        plt.imshow()
        fig.savefig(savePath, dpi=dpi, format='eps')
        return 0

    def view_clz_map_mlpversion(self, test_index, results, sub_indexes, labels, save_name=None):
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        import copy
        n_res = results.__len__()
        gt = copy.deepcopy(results[0])
        n_row, n_column = gt.shape
        gt_1d = gt.reshape(-1).copy()
        nonzero_index = gt_1d.nonzero()
        for i in range(n_res):
            if i == 0:
                gt_map = gt
            else:
                gt_corrected = copy.deepcopy(gt_1d[nonzero_index])
                gt_corrected[test_index] = results[i]
                gt_1d_temp = copy.deepcopy(gt.reshape(-1))
                gt_1d_temp[nonzero_index] = gt_corrected
                gt_map = gt_1d_temp.reshape((n_row, n_column)).astype('uint8')
            axe = plt.subplot(sub_indexes[i])
            im = axe.imshow(gt_map, cmap='jet')
            axe.set_title(labels[i], fontdict={'fontsize': 10})
            axe.axis('off')
        values = np.unique(gt.ravel())
        colors = [im.cmap(im.norm(value)) for value in values]
        patches = [mpatches.Patch(color=colors[i], label="{l}".format(l=values[i])) for i in range(len(values))]
        axe_legend = plt.subplot(sub_indexes[-1])
        axe_legend.legend(handles=patches, loc=10, ncol=6)
        axe_legend.axis('off')
        plt.show()
        plt.savefig(save_name, format='eps', dpi=1000)
        print('the figure is saved in ', save_name)

    def show_class_map(self, y_pre, y_indx, gt, show=True, save=False):
        import copy
        import matplotlib.pyplot as plt

        gt_pre = copy.deepcopy(gt)
        gt_pre_flatten = gt_pre.reshape(-1)
        gt_pre_flatten[y_indx] = y_pre
        gt_pre_2d = np.reshape(gt_pre_flatten, gt.shape)
        fig, ax = plt.subplots()
        ax.imshow(gt_pre_2d, cmap='nipy_spectral')
        plt.axis('off')
        plt.tight_layout()
        if save is not False:
            plt.savefig(save, format='pdf', bbox_inches='tight')
        if show:
            plt.show()

    def standardize_label(self, y):
        import copy
        classes = np.unique(y)
        standardize_y = copy.deepcopy(y)
        for i in range(classes.shape[0]):
            standardize_y[np.nonzero(y == classes[i])] = i
        return standardize_y

    def one2array(self, y):
        n_classes = np.unique(y).__len__()
        y_expected = np.zeros((y.shape[0], n_classes))
        for i in range(y.shape[0]):
            y_expected[i][y[i]] = 1
        return y_expected

    def zca_whitening(self, x, epsilon=1e-6, mean=None, whitening=None):
        if not x.size:
            return x, mean, whitening
        data_shape = x.shape
        size = data_shape[0]
        white_data = x.reshape((size, -1))

        if mean is None:
            mean = white_data.mean(axis=0)
        white_data -= mean
        
        if whitening is None:
            cov = np.dot(white_data.T, white_data) / size
            U, S, V = np.linalg.svd(cov)
            whitening = np.dot(np.dot(U, np.diag(1. / np.sqrt(S + epsilon))), U.T)

        white_data = np.dot(white_data, whitening)
        return white_data.reshape(data_shape), mean, whitening
