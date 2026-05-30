import os
import torch
import warnings
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
from preprocess import load_label
from models import Enc_AE
from utils import AEDataset
from sklearn.preprocessing import StandardScaler
from torch.utils.data.dataloader import DataLoader
import dataset, networks, contrastive_loss

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class client:
    def __init__(self, Id, args, Distribute_Array=[]):
        self.Id = Id
        self.dataset_name = args.dataset
        self.AttributeNumber = Distribute_Array[self.Id]
        self.client_dataset_name = f'{self.dataset_name}_{self.AttributeNumber}'
        self.encoded_dim = args.encoded_dim
        self.batch_size = args.batch_size
        self.windowsize = args.windowsize
        self.Patch_channel = args.Patch_channel
        self.lr = args.lr
        self.device = args.device
        self.num_clients = args.num_clients
        self.num_label = args.num_label
        self.epochs = args.num_local_epochs
        self.label_gt = load_label(self.client_dataset_name, self.num_clients)
        self.Data = self.get_training_data()
        self.dataset_train = dataset.HSI_Data(
            self.client_dataset_name,
            patch_size=(args.windowsize, args.windowsize),
            pca_dim=self.Patch_channel,
            num_clients=self.num_clients
        )
        self.data_loader = torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=args.workers,
        )
        self.Enc_patch = Enc_AE(channel=self.Patch_channel, output_dim=self.encoded_dim, windowSize=self.windowsize).to(
            self.device)
        self.optim_enc = Adam(self.Enc_patch.parameters(), lr=self.lr, weight_decay=5e-4)
        self.Global_Enc_patch = Enc_AE(channel=self.Patch_channel, output_dim=self.encoded_dim,
                                       windowSize=self.windowsize).to(self.device)

    def get_training_data(self, judge=-1):
        data_path = f'DataArray/{self.dataset_name}_{self.AttributeNumber}.npy'
        return AEDataset(data_path, judge=judge)

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

    def compute_prototype_relationship_loss(self, local_prototypes, global_prototypes, pseudo_labels, num_cluster):
        local_prototypes = local_prototypes.to(self.device)
        global_prototypes = global_prototypes.to(self.device)
        local_prototypes = F.normalize(local_prototypes, p=2, dim=1)
        global_prototypes = F.normalize(global_prototypes, p=2, dim=1)
        pseudo_labels = np.concatenate(pseudo_labels)
        existing_classes = torch.unique(torch.from_numpy(pseudo_labels).to(torch.int64)).to(self.device)
        if len(existing_classes) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        pos_term_sum = 0.0
        neg_term_sum = 0.0
        for k in existing_classes:
            pos_sim = torch.sum(local_prototypes[k] * global_prototypes[k])
            pos_term_sum += pos_sim
            valid_neg_classes = [j for j in range(num_cluster) if j != k.item()]
            if valid_neg_classes:
                sampled_j = np.random.choice(valid_neg_classes)
                neg_sim = torch.sum(local_prototypes[k] * global_prototypes[sampled_j])
                neg_term_sum += neg_sim
        loss = (neg_term_sum - pos_term_sum) / len(existing_classes)
        return loss

    def localTrain(self, globalModelSD=None, globaldecGD=None, pretrain_epochs=1, cluster_global=None,
                   local_cluster=None, instance_coeff=0.01, cross_cor_coeff=0.01, Round=0):
        if globalModelSD is not None:
            self.Global_Enc_patch.load_state_dict(globalModelSD)

        N_CLASSES = np.unique(self.label_gt).shape[0]
        model = networks.Network(self.Enc_patch, self.Global_Enc_patch, self.encoded_dim, N_CLASSES, self.windowsize,
                               self.Patch_channel)
        model.to(self.device)
        for epoch in range(1, self.epochs + 1):
            self.Enc_patch.train()
            model.train()
            
            loss = loss_cross_cor = loss_instance = total_loss_instance_epoch = total_loss_cross_cor_epoch = 0.0
            self.optim_enc.zero_grad()
            Encoding = torch.FloatTensor([])

            for step, (x, _) in enumerate(self.data_loader):
                current_batch_size = x.shape[0]
                self.criterion_instance = contrastive_loss.InstanceLoss(current_batch_size, 0.5, self.device).to(
                    self.device)
                self.criterion_cross_cor = contrastive_loss.CrossCorrelationLoss(N_CLASSES, 0.005, self.device).to(
                    self.device)
                x_i = x.reshape(current_batch_size, 1, self.Patch_channel, self.windowsize, self.windowsize).to(
                    self.device)
                x_j = x.reshape(current_batch_size, 1, self.Patch_channel, self.windowsize, self.windowsize).to(
                    self.device)

                with torch.set_grad_enabled(True):
                    is_single_sample = (current_batch_size == 1)
                    if is_single_sample:
                        model.eval()
                        self.criterion_cross_cor.eval()

                    z_i, z_j, c_i, c_j = model(x_i, x_j)
                    loss_instance = instance_coeff * self.criterion_instance(c_i, c_j)
                    loss_cross_cor = cross_cor_coeff * self.criterion_cross_cor(c_i, c_j)

                    if is_single_sample:
                        model.train()
                        self.criterion_cross_cor.train()

                    total_loss_instance_epoch += loss_instance.item()
                    total_loss_cross_cor_epoch += loss_cross_cor.item()
                    Encoding = torch.cat([Encoding, z_i.detach().cpu()], dim=0)

                step_loss = loss_instance + loss_cross_cor
                step_loss.backward(retain_graph=True)
                self.optim_enc.step()

            if len(Encoding) > 0:
                cluster_local = self.compute_centers(self.num_label, Encoding, local_cluster[self.AttributeNumber])
                self.optim_enc.zero_grad()
                L_c = self.compute_prototype_relationship_loss(cluster_local, cluster_global, local_cluster,
                                                               self.num_label)
                L_c.requires_grad = True
                L_c.backward()
                loss += L_c.cpu().detach().numpy().item() + total_loss_instance_epoch + total_loss_cross_cor_epoch
                self.optim_enc.step()

        saved_state_dict = self.Enc_patch.state_dict()
        full_state_dict = self.Enc_patch.state_dict()
        saved_state_dict.pop('projector.0.weight', None)
        saved_state_dict.pop('projector.0.bias', None)
        save_dir = f'ClientData/pretrain_checkpoints/{self.Id}/'
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        torch.save(saved_state_dict, f'{save_dir}{self.client_dataset_name}_pretrain_weight.pth')
        
        eval_loader = DataLoader(dataset=self.Data, batch_size=4096, shuffle=False)
        self.Enc_patch.eval()
        feature_encoding = torch.FloatTensor([])
        with torch.set_grad_enabled(False):
            for data in eval_loader:
                data = data.float().to(self.device)
                encoding = self.Enc_patch(data)
                feature_encoding = torch.cat([feature_encoding, encoding.detach().cpu()], dim=0)

        feature_encoding = feature_encoding.numpy()
        scaler = StandardScaler()
        feature_encoding = scaler.fit_transform(feature_encoding)

        N_CLASSES = np.unique(self.label_gt).shape[0]
        update = {
            "update": full_state_dict,
            "sampleCount": len(self.Data) * self.epochs,
            "tau_k": int(self.epochs)
        }
        return update, feature_encoding, self.AttributeNumber