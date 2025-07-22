"""
Adrián Ayuso Muñoz 2024-12-11.

Class of the functional connectivity hypergraph sub_module.
    Receives fc_matrix, and ROI features as input and generates the hypergraph.
"""
import yaml
import torch
import warnings
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data, Dataset
from nilearn.connectome import ConnectivityMeasure
from torch_geometric.utils import dense_to_sparse

def create_hyperedges(data, device, top=3):
    """Function that constructs the hypergraph.
    :param data: (np.array) FC matrix.
    :param top: (int) top k elements to get.
    :return: (Hypergraph) The constructed hyperedge index and weights.
    """
    distances = torch.cdist(data, data, p=2)  # Euclidean distance

    _, knn = torch.topk(distances, top + 1, largest=False)

    knn = knn[:, 1:]

    hyperedge_index = [[], []]
    for k, nodes in enumerate(knn):
        hyperedge_index[0].extend(nodes)
        hyperedge_index[1].extend([k] * len(nodes))  # k is the hyperedge index

    hyperedge_index = torch.tensor(hyperedge_index, device=device)

    return hyperedge_index

def set_overlap_ratio(matrix):
    res = []
    n = len(matrix)
    for i in range(n):
        for j in range(i + 1, n):
            set1 = set(matrix[i].tolist())
            set2 = set(matrix[j].tolist())
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            ratio = intersection / union if union > 0 else 0.0
            res.append(ratio)
    return sum(res) / len(res) if res else 0.0

class FCHypergraph(Dataset):
    def __init__(self, data, k=3, th = 0.05, torch_device='cpu', **kwargs):
        super(FCHypergraph, self).__init__(**kwargs)
        self.device = torch_device
        self.device = 'cpu'

        self.data = data
        self.k = k

        self.threshold = th

        self.y = torch.tensor(np.stack(data['label']), dtype=torch.long, device=self.device)

        self.x = torch.tensor(np.stack(data['corr']), dtype=torch.float32, device=self.device)

        self.acumm = []

        self.graphs = self.generate_graphs()

    def generate_graphs(self):
        result = []
        for i, matrix in enumerate(self.x):
            hyperedge_index, hyperedge_attr = self.create_hyperedges(matrix)
            analytical_hyperedge_index = self.analytical_solution_hyper(matrix)

            graph = Data(x=matrix, hyperedge_index=hyperedge_index, y=self.y[i].view(-1, 1),
                         hyperedge_weight=torch.mean(hyperedge_attr, dim=1), hyperedge_attr=hyperedge_attr,
                         sex=torch.tensor(self.data['sex'][i]), age=torch.tensor(self.data['age'][i]),
                         handedness=torch.tensor(self.data['handedness'][i]),
                         analytical_hyperedge=analytical_hyperedge_index, eye=torch.eye(matrix.shape[0]),
                         edge_index=dense_to_sparse((matrix >= -0.5) & (matrix <= 0.5).int())[0])

            result.append(graph)
        self.get_dist()
        return result

    def analytical_solution_hyper(self, matrix):
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

    def create_hyperedges(self, data):
        """Function that constructs the hypergraph.
        :param data: (np.array) FC matrix.
        :return: (Hypergraph) The constructed hyperedge index and weights.
        """
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

class FCDynamicHypergraph(torch.utils.data.Dataset):
    def __init__(self, data, torch_device='cpu'):
        self.device = torch_device
        self.device = 'cpu'

        self.data = data

        time = [matrix[0] for matrix in self.data['time']]
        self.x = [self.sliding_window_fc_hamming(matrix.T) for matrix in time]

        self.original_lengths =  [x.shape[0] for x in self.x]
        self.x = pad_sequence(self.x, batch_first=True)

        self.y = torch.tensor(np.stack(data['label']), dtype=torch.long, device=self.device)

        self.sex = torch.tensor(np.stack(data['sex']), dtype=torch.long, device=self.device)
        self.age = torch.tensor(np.stack(data['age']), dtype=torch.long, device=self.device)
        self.handedness = torch.tensor(np.stack(data['handedness']), dtype=torch.long, device=self.device)

    def sliding_window_fc_hamming(self, signals, win_length=40, step=40):
        """
        Idea from https://www.sciencedirect.com/science/article/pii/S0925492723001592?via%3Dihub
        """
        n_rois, T = signals.shape
        window = np.hamming(win_length)
        fc_series = []

        for start in range(0, T - win_length + 1, step):
            segment = signals[:, start:start + win_length]

            windowed = segment * window

            correlation_matrix = correlation_measure.fit_transform([windowed.T])[0]
            correlation_matrix[np.isnan(correlation_matrix)] = 0.0
            correlation_matrix = torch.tensor(correlation_matrix, dtype=torch.float32, device=self.device)
            fc_series.append(correlation_matrix)

        return torch.stack(fc_series)

    def __getitem__(self, index):
        return {'x': self.x[index], 'y': self.y[index], 'sex': self.sex[index], 'age': self.age[index],
                'handedness': self.handedness[index], 'original_lengths': self.original_lengths[index]}

    def __len__(self):
        return len(self.y)