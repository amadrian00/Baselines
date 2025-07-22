import os
import torch
import numpy as np
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import dense_to_sparse
from thfcn.THFCN_ADNI import normalize
from thfcn.construct_hyper_graph_KNN import construct_H_with_KNN
from sklearn.linear_model import Lasso
from hypergraph.fc_hypergraph_learning2 import CorrelationToIncidenceTransformer


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

            thfcn_index, thfcn_weights = self.thfcn(self.data['time'][i][0])

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
                         weight = weights.float(),
                         eye = torch.eye(matrix.shape[0]),

                         thfcn_index = thfcn_index,
                         thfcn_weight = thfcn_weights.float(),
                         thfcn_attr=thfcn_weights.float(),
                         )

            if not self.loaded: self.calculated_ts.append(ts_modelling_index)
            result.append(graph)

        if not self.loaded: torch.save(self.calculated_ts, 'calculated_ts.pt')

        self.get_dist()
        return result

    def ts_modelling(self, ts, i):
        if self.loaded:
            return self.calculated_ts[i],  torch.ones(len(self.calculated_ts[i][0]))

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
        return hyperedge_index, torch.ones(len(self.calculated_ts[i][0]))

    def thfcn(self, ts):
        dataset_run1, maxV, minV = normalize(ts, '-01')

        H = construct_H_with_KNN(dataset_run1.T, 10)

        h, weights = dense_to_sparse(torch.tensor(H))

        return h, weights

    def fc_modelling(self, matrix):
        n = matrix.shape[0]

        matrix_dropped = (torch.abs(matrix) > 0.3).int().float()

        alpha = 1
        I = torch.eye(matrix_dropped.size(0), dtype=matrix_dropped.dtype, device=matrix_dropped.device)
        Z = alpha * torch.inverse(alpha * matrix_dropped.T @ matrix_dropped + I) @ matrix_dropped.T @ matrix_dropped

        row, col = torch.where(Z > self.threshold)
        H = torch.zeros((n, n))

        H[row, col] = Z[row,col]

        h = (H.T+matrix_dropped).T
        index, weight = dense_to_sparse(h)
        return index, torch.flatten(weight)

    def create_random_hyperedges(self, data):
        N, _ = data.shape
        device = data.device
        rand_indices = []
        for i in range(N):
            candidates = torch.cat([torch.arange(0, i), torch.arange(i + 1, N)]).to(device)
            sampled = candidates[torch.randperm(N - 1, device=device)[:self.k]]
            full_hyperedge = torch.cat([torch.tensor([i], device=device), sampled])
            rand_indices.append(full_hyperedge)

        rand_indices = torch.stack(rand_indices)
        hyperedge_weights = torch.gather(data, 1, rand_indices)

        hyperedge_index = [[], []]
        for hyperedge_id, nodes in enumerate(rand_indices):
            hyperedge_index[0].extend(nodes.tolist())
            hyperedge_index[1].extend([hyperedge_id] * len(nodes))

        hyperedge_index = torch.tensor(hyperedge_index, device=device)
        hyperedge_weights[hyperedge_weights < 0] = 0.001
        return hyperedge_index, torch.flatten(hyperedge_weights)

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



class SecondFCHypergraph(Dataset):
    def __init__(self, data, model, device, **kwargs):
        super(SecondFCHypergraph, self).__init__(**kwargs)
        self.device = device

        self.data = data

        self.model = model

        self.graphs = self.generate_graphs()

    def generate_graphs(self):
        result = []
        for batch in self.data:
            for i, fc in enumerate(batch.x.view(-1,200,200)):
                proposed_hyperedge_index, proposed_hyperedge_weight, proposed_x = self.proposed(fc.to(self.device))

                graph = Data(x=fc, y=batch.y[i].view(-1, 1),

                             proposed_hyperedge_index=proposed_hyperedge_index,
                             proposed_weight=proposed_hyperedge_weight,
                             proposed_x=proposed_x,
                             )

                result.append(graph)
        return result

    def proposed(self, fc):
        incidence, _, _, x, hyperedge_attr, _ = self.model(fc)

        proposed_hyperedge_index = incidence.detach().squeeze()
        proposed_hyperedge_index = torch.where(proposed_hyperedge_index < 0.2, torch.tensor(0.0),
                                               proposed_hyperedge_index)

        weight = torch.tensor([0])

        return proposed_hyperedge_index, weight, x.squeeze().detach()

    def __getitem__(self, index):
        return self.graphs[index]

    def __len__(self):
        return len(self.graphs)