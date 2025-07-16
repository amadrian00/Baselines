import torch
import numpy as np
from torch.utils.data import Subset
from prepare_dataset import get_dataframe
from hypergraph_generator import FCHypergraph
from torch_geometric.loader import DataLoader
from sklearn.model_selection import RepeatedStratifiedKFold


def prepare_dataloader(graphs, batch_size, shuffle=False):
    return DataLoader(graphs, batch_size=batch_size, drop_last=False, shuffle=shuffle)

def get_folds(n_folds, n_repeats, batch_size, device):
    whole_dataset = FCHypergraph(get_dataframe(), torch_device=device)

    kf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=42)
    folds = []
    for train_idx, val_idx in kf.split(whole_dataset.x.cpu().numpy(), whole_dataset.y.cpu().numpy()):
        train_subset = Subset(whole_dataset, train_idx)
        val_subset = Subset(whole_dataset, val_idx)

        train_labels = torch.stack([whole_dataset[i]['y'] for i in train_idx]).float()

        train_loader = prepare_dataloader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = prepare_dataloader(val_subset, batch_size=len(val_idx))

        folds.append((train_loader, val_loader, train_labels))

    x_shape = whole_dataset[0].x.shape
    return folds, x_shape

def compute_stats(metric_array):
    mean = np.mean(metric_array)
    std = np.std(metric_array, ddof=1)
    return mean, std
