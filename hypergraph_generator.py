import torch
import numpy as np
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import dense_to_sparse

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

        self.graphs = self.generate_graphs()

    def generate_graphs(self):
        result = []
        for i, matrix in enumerate(self.x):
            hyperedge_index, hyperedge_attr = self.create_hyperedges(matrix)
            ts_modelling_index = self.ts_modelling(matrix)
            fc_modelling_index = self.fc_modelling(matrix)
            random_hyperedge_index = self.create_random_hyperedges(matrix)

            graph = Data(x=matrix, y=self.y[i].view(-1, 1),
                         hyperedge_attr=hyperedge_attr,
                         hyperedge_weight=torch.mean(hyperedge_attr, dim=1),

                         hyperedge_index = hyperedge_index,
                         ts_modelling_index = ts_modelling_index,
                         fc_modelling_index = fc_modelling_index,
                         random_hyperedge_index = random_hyperedge_index,

                         edge_index=dense_to_sparse((matrix >= -0.3) & (matrix <= 0.3).int())[0],
                         eye = torch.eye(matrix.shape[0]))

            result.append(graph)
        self.get_dist()
        return result

    def ts_modelling(self, matrix):
        return torch.tensor([self.threshold*matrix[0][0]])

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

        return h

    def create_random_hyperedges(self, data):
        N, M = data.shape

        rand_indices = []
        for i in range(N):
            candidates = torch.cat([torch.arange(0, i), torch.arange(i + 1, M)])
            sampled = candidates[torch.randperm(M - 1)[:self.k]]
            indices = torch.cat([torch.tensor([i], device=data.device), sampled])
            rand_indices.append(indices)

        rand_indices = torch.stack(rand_indices)

        hyperedge_index = [[], []]
        for k, nodes in enumerate(rand_indices):
            hyperedge_index[0].extend(nodes)
            hyperedge_index[1].extend([k] * len(nodes))  # k is the hyperedge index

        hyperedge_index = torch.tensor(hyperedge_index, device=self.device)

        return hyperedge_index

    def create_hyperedges(self, data):
        distances = torch.cdist(data, data, p=2)  # Euclidean distance

        _, knn = torch.topk(distances, self.k + 1, largest=False)
        hyperedge_attr = torch.gather(data, 1, knn)

        hyperedge_index = [[], []]
        for k, nodes in enumerate(knn):
            hyperedge_index[0].extend(nodes)
            hyperedge_index[1].extend([k] * len(nodes))  # k is the hyperedge index

        hyperedge_index = torch.tensor(hyperedge_index, device=self.device)

        return hyperedge_index, hyperedge_attr

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