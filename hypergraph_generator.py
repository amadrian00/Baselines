import os
import torch
import numpy as np
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import dense_to_sparse
from thfcn.mutitask_Lasso_manual import mutitaskLasso
from thfcn.THFCN_ADNI import normalize, create_dataset4
from thfcn.construct_hyper_graph_KNN import construct_H_with_KNN, generate_S_from_H
from sklearn.linear_model import Lasso


class FCHypergraph(Dataset):
    def __init__(self, data, k=2, th = 0.05, torch_device='cpu', **kwargs):
        super(FCHypergraph, self).__init__(**kwargs)
        self.device = torch_device
        self.device = 'cpu'

        self.data = data

        self.k = k
        self.threshold = th

        self.y = torch.tensor(np.stack(data['label']), dtype=torch.long, device=self.device)
        self.x = torch.tensor(np.stack(data['corr']), dtype=torch.float32, device=self.device)

        try:
            if os.path.exists('calculated_ts.pt'):
                self.calculated_ts = torch.load('calculated_ts.pt')
                self.loaded = True
            else:
                raise FileNotFoundError
        except FileNotFoundError:
            self.calculated_ts = []
            self.loaded = False



        self.graphs = self.generate_graphs()

    def generate_graphs(self):
        result = []
        for i, matrix in enumerate(self.x):
            hyperedge_index, hyperedge_weights = self.create_hyperedges(matrix)
            ts_modelling_index, ts_modelling_weight = self.ts_modelling(self.data['time'][i][0], i)
            fc_modelling_index, fc_modelling_weight = self.fc_modelling(matrix)
            random_hyperedge_index, random_hyperedge_weight = self.create_random_hyperedges(matrix)
            edge_index, weights = dense_to_sparse((matrix >= -0.3) & (matrix <= 0.3).int())
            thfcn = self.thfcn(self.data['time'][i][0])

            graph = Data(x=matrix, y=self.y[i].view(-1, 1),

                         hyperedge_index = hyperedge_index,
                         hyperedge_attr=hyperedge_weights,
                         hyperedge_weight=hyperedge_weights,

                         ts_modelling_index = ts_modelling_index,
                         ts_modelling_attr = ts_modelling_weight,
                         ts_modelling_weight = ts_modelling_weight,

                         fc_modelling_index = fc_modelling_index,
                         fc_modelling_attr = fc_modelling_weight,
                         fc_modelling_weight = fc_modelling_weight,

                         random_hyperedge_index = random_hyperedge_index,
                         random_hyperedge_attr = random_hyperedge_weight,
                         random_hyperedge_weight = random_hyperedge_weight,

                         edge_index= edge_index,
                         weight = weights,
                         eye = torch.eye(matrix.shape[0]),

                         thfcn = thfcn,
                         )

            self.calculated_ts.append(ts_modelling_index)

            result.append(graph)
        torch.save(self.calculated_ts, 'calculated_ts.pt')
        self.get_dist()
        return result


    def ts_modelling(self, ts, i):
        if self.loaded:
            return self.calculated_ts[i], torch.tensor([0])

        alphas = []
        hyperedges = []

        ts = ts.T

        rois = ts.shape[0]

        indices = [i for i in range(rois)]

        for i in range(rois):
            roi = ts[i,:].reshape(-1, 1)
            other_rois_indices = indices[:i] + indices[i+1:]
            Ai = ts[other_rois_indices, :].T

            lasso_model = Lasso(alpha=0.1, fit_intercept=False, max_iter=1000)
            lasso_model.fit(Ai, roi.ravel())

            alpha_i = lasso_model.coef_
            alphas.append(alpha_i)

            contributing_rois_indices_in_A = np.where(alpha_i > 0)[0]

            contributing_rois_global_indices = [other_rois_indices[idx] for idx in contributing_rois_indices_in_A]

            current_hyperedge = set([i] + contributing_rois_global_indices)
            hyperedges.append(current_hyperedge)

        hyperedge_index = [[], []]
        for i, nodes in enumerate(hyperedges):
            hyperedge_index[0].extend(nodes)
            hyperedge_index[1].extend([i] * len(nodes))

        hyperedge_index = torch.tensor(hyperedge_index, device=self.device)
        return hyperedge_index, torch.tensor([0])

    def thfcn(self, ts, knn=10):
        new_ts_run1 = ts[:, 0:200]
        normalize_style = '-01'  # （-1，1）
        dataset_run1, maxV, minV = normalize(new_ts_run1, normalize_style)

        H = construct_H_with_KNN(dataset_run1.T, knn)
        S, L = generate_S_from_H(H)
        L = np.array(L)

        X, Y = create_dataset4(dataset_run1, 1)
        W, _ = mutitaskLasso(X, Y, L, 0.01, 0.01,  1e-24, 0)
        if W is None:
            W = [[0],[0]]
        return W.flatten()

    def fc_modelling(self, matrix):
        n = matrix.shape[0]

        matrix_dropped = (torch.abs(matrix) > 0.3).int().float()

        alpha = 1
        I = torch.eye(matrix_dropped.size(0), dtype=matrix_dropped.dtype, device=matrix_dropped.device)
        Z = alpha * torch.inverse(alpha * matrix_dropped.T @ matrix_dropped + I) @ matrix_dropped.T @ matrix_dropped

        row, col = torch.where(Z > self.threshold)
        H = torch.zeros((n, n))

        H[row, col] = 1

        h = torch.concat([H, matrix_dropped], dim=1)
        index, weight = dense_to_sparse(h)
        return index, torch.flatten(weight)

    def create_random_hyperedges(self, data):
        N, M = data.shape

        rand_indices = []
        for i in range(N):
            candidates = torch.cat([torch.arange(0, i), torch.arange(i + 1, M)])
            sampled = candidates[torch.randperm(M - 1)[:self.k]]
            indices = torch.cat([torch.tensor([i], device=data.device), sampled])
            rand_indices.append(indices)

        rand_indices = torch.stack(rand_indices)

        hyperedge_attr = torch.gather(data, 1, rand_indices)

        hyperedge_index = [[], []]
        for k, nodes in enumerate(rand_indices):
            hyperedge_index[0].extend(nodes)
            hyperedge_index[1].extend([k] * len(nodes))  # k is the hyperedge index

        hyperedge_index = torch.tensor(hyperedge_index, device=self.device)

        return hyperedge_index, hyperedge_attr

    def create_hyperedges(self, data):
        distances = torch.cdist(data, data, p=2)  # Euclidean distance

        _, knn = torch.topk(distances, self.k + 1, largest=False)
        hyperedge_weights = torch.gather(data, 1, knn)

        hyperedge_index = [[], []]
        for k, nodes in enumerate(knn):
            hyperedge_index[0].extend(nodes)
            hyperedge_index[1].extend([k] * len(nodes))  # k is the hyperedge index

        hyperedge_index = torch.tensor(hyperedge_index, device=self.device)

        return hyperedge_index, torch.flatten(hyperedge_weights)

    def get_dist(self):
        print(' Shape: ', self.y.shape[0])
        print(
            f"        Number of 0s: {self.data['label'].value_counts().get(0, 0) / len(self.y) * 100:.2f}%, "
            f"          Number of 1s: {self.data['label'].value_counts().get(1, 0) / len(self.y) * 100:.2f}%")
        print(f"        Site Distribution: " +
              ",        ".join([f"{site_id}: {count / len(self.data) * 100:.2f}%"
                                for site_id, count in self.data['site_id'].value_counts().items()]))

    def __getitem__(self, index):
        return self.graphs[index]

    def __len__(self):
        return len(self.y)