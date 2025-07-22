import json
import torch
import numpy as np
from torch.utils.data import Subset
from prepare_dataset import get_dataframe
from hypergraph_generator import FCHypergraph
from torch_geometric.loader import DataLoader
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit

def prepare_dataloader(graphs, batch_size, shuffle=False):
    return DataLoader(graphs, batch_size=batch_size, drop_last=False, shuffle=shuffle)

def get_folds(n_folds, n_repeats, batch_size, device):
    whole_dataset = FCHypergraph(get_dataframe(), torch_device=device)

    kf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=42)
    strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    
    folds = []
    ids = {'test':{}, 'train':{}, 'val':{}}
    for i, (train_idx, test_idx) in enumerate(kf.split(whole_dataset.x.cpu().numpy(), whole_dataset.y.cpu().numpy())):
        train_labels = torch.stack([whole_dataset[i]['y'] for i in train_idx]).float().squeeze()
        train_idx_sub, val_idx = next(strat_split.split(train_idx, train_labels))

        val_idx = train_idx[val_idx]
        train_idx = train_idx[train_idx_sub]

        train_subset = Subset(whole_dataset, train_idx)
        test_subset = Subset(whole_dataset, test_idx)
        val_subset = Subset(whole_dataset, val_idx)

        train_loader = prepare_dataloader(train_subset, batch_size=batch_size, shuffle=True)
        test_loader = prepare_dataloader(test_subset, batch_size=len(test_idx))
        val_loader = prepare_dataloader(val_subset, batch_size=len(val_idx))

        folds.append((train_loader, test_loader, val_loader, train_labels))
        for key, ids_list  in zip(['train', 'val', 'test'], [train_idx, val_idx, test_idx]):
            ids[key][f"fold{i}"] = ids_list.tolist()
        folds.append((train_loader, test_loader, val_loader, train_labels))

    with open("fold_dict.json", "w") as f:
        json.dump(ids, f, indent=4)

    x_shape = whole_dataset[0].x.shape
    return folds, x_shape

def compute_stats(metric_array):
    mean = np.mean(metric_array)
    std = np.std(metric_array, ddof=1)
    return mean, std