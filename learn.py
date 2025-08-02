import torch
import copy
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch import Tensor
import matplotlib.pyplot as plt
from collections import OrderedDict
from dhg.models.hypergraphs.hgnnp import HGNNP
from dhg.models.hypergraphs.hgnn import HGNN
from torch_geometric.nn.conv import HypergraphConv, GATConv, GCNConv, SAGEConv
from torch_geometric.nn.pool import global_max_pool, global_mean_pool
from torcheval.metrics import BinaryAUROC, BinaryF1Score, BinaryRecall, BinaryConfusionMatrix, BinaryAccuracy

from baseline_models.bnt import BrainNetworkTransformer
from baseline_models.brainnetcnn import BrainNetCNN

class FCHypergraphLearning(torch.nn.Module):
    def __init__(self, in_size, hidden_size, dropout, device, y: Tensor, num_hyperedges, name, model_name, fold, use_features):
        super(FCHypergraphLearning, self).__init__()

        self.name = name
        self.fold = fold
        self.use_features = use_features

        self.device = device
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

        num_positive = torch.sum(y == 1).float()
        num_negative = torch.sum(y == 0).float()
        pos_weight = num_negative / (num_positive + 1e-20)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.roi = in_size
        self.hidden_size = hidden_size
        self.num_hyperedges = num_hyperedges

        self.model_name = model_name

        if model_name == 'hgnn':
            self.model = HGNN(in_size, hidden_size, int(hidden_size/2), True).to(self.device)
        elif model_name == 'hgnnplus':
            self.model = HGNNP(in_size, hidden_size, int(hidden_size/2), True).to(self.device)
        elif name == 'gat':
            self.conv1 = GATConv(in_size, hidden_size)
            self.conv2 = GATConv(hidden_size, int(hidden_size/2))
        elif name == 'gsage':
            self.conv1 = SAGEConv(in_size, hidden_size)
            self.conv2 = SAGEConv(hidden_size, int(hidden_size / 2))
        elif name == 'gcn':
            self.conv1 = GCNConv(in_size, hidden_size)
            self.conv2 = GCNConv(hidden_size, int(hidden_size / 2))
        elif name == 'proposed':
            in_size = in_size
            self.conv1 = HypergraphConv(in_size, hidden_size) #Hyper-Attn
            self.conv2 = HypergraphConv(hidden_size, int(hidden_size/2))
        elif name == 'btf':
            self.conv1 = BrainNetworkTransformer()
        elif name == 'brainnetcnn':
            self.conv1 = BrainNetCNN()
        else:
            self.conv1 = HypergraphConv(in_size, hidden_size) #Hyper-Attn
            self.conv2 = HypergraphConv(hidden_size, int(hidden_size/2))

        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)

        self.embeddingFinal = nn.Linear(int(hidden_size), 1)

        self.best_threshold = 0.5

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, HypergraphConv):
                torch.nn.init.xavier_uniform_(m.lin.weight)
                if m.lin.bias is not None:
                    torch.nn.init.zeros_(m.lin.bias)

            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm1d):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)

            elif isinstance(m, (GCNConv, GATConv)):
                torch.nn.init.xavier_uniform_(m.lin.weight)
                if m.lin.bias is not None:
                    torch.nn.init.zeros_(m.lin.bias)

    def forward(self, data):
        batch = data.batch
        input_x = data.x
        input_x = input_x.view(-1, input_x.shape[1], input_x.shape[1])
        if not self.use_features:
            input_x = data.eye

        if self.model_name in {'hgnn', 'hgnnplus'}:
            hypergraph = getattr(data, self.name, None)
            if hypergraph is None:
                raise ValueError(f"Invalid 'name' provided: {self.name}.")
        else:
            if self.name in {'knn', 'ts-modelling', 'fc-modelling', 'k-random', 'thfcn', 'proposed'}:
                idx_attr = {
                    'knn': ('hyperedge_index', 'hyperedge_weight'),
                    'ts-modelling': ('ts_modelling_index', 'ts_modelling_weight'),
                    'fc-modelling': ('fc_modelling_index', 'fc_modelling_weight'),
                    'k-random': ('random_hyperedge_index', 'random_hyperedge_weight'),
                    'thfcn': ('thfcn_index', 'thfcn_weight'),
                    'proposed': ('proposed_hyperedge_index', 'proposed_weight')
                }
                hyperedge_index = getattr(data, idx_attr[self.name][0])
                weights = getattr(data, idx_attr[self.name][1])
            elif self.name in {'gat', 'gsage', 'gcn'}:
                hyperedge_index = data.edge_index
                weights = data.weight
            elif self.name in {'btf', 'brainnetcnn'}:
                input_x = input_x.view(-1, self.roi, self.roi)
                x = self.conv1(input_x)[0] if self.name == 'btf' else self.conv1(input_x)
                return x
            else:
                raise ValueError(
                    f"Invalid 'name' provided: {self.name}.")

        if self.model_name == 'hgnn' or self.model_name == 'hgnnplus':
            x = torch.stack([self.model(f, h) for f, h in zip(input_x, hypergraph)])
            x = x.contiguous().view(-1, int(self.hidden_size / 2))
        else:
            if self.name!= 'gsage': x = self.conv1(input_x, hyperedge_index, weights)
            else : x = self.conv1(input_x, hyperedge_index)

            x = self.bn1(x)
            x = self.activation(x)

            if self.name!= 'gsage': x = self.conv2(x, hyperedge_index, weights)
            else: x = self.conv2(x, hyperedge_index)

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        x = self.bn2(x)
        x = self.dropout(x)

        x = self.embeddingFinal(x)

        return x

    def test(self, dataloaders, find_threshold=False):
        self.eval()
        with torch.no_grad():
            metrics = {}

            for mode, dataloader in dataloaders.items():
                total_loss = 0.0
                all_probs = []
                all_labels = []
                for batch in dataloader:
                    batch = batch.to(self.device)
                    logits = self(batch)
                    y_true = batch['y']

                    total_loss += self.loss(logits, y_true).item()

                    if isinstance(logits, tuple):
                        logits = logits[0]

                    probs = torch.sigmoid(logits).view(-1)
                    all_probs.append(probs.detach().cpu())
                    all_labels.append(y_true.view(-1).detach().cpu())

                probs = torch.cat(all_probs)
                labels = torch.cat(all_labels)

                if find_threshold and mode == "val":
                    thresholds = torch.linspace(0.1, 0.9, 17)
                    best_j = 0.0
                    for t in thresholds:
                        recall_metric = BinaryRecall(threshold=t.item())
                        recall_metric.update(probs, labels.int())
                        recall = recall_metric.compute().item()

                        confmat = BinaryConfusionMatrix(threshold=t.item())
                        confmat.update(probs, labels.int())
                        tn, fp, fn, tp = confmat.compute().flatten()
                        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

                        j = recall + specificity - 1
                        if j > best_j:
                            best_j = j
                            self.best_threshold = t.item()

                f1 = BinaryF1Score(threshold=self.best_threshold)
                f1.update(probs, labels.int())

                acc = BinaryAccuracy(threshold=self.best_threshold)
                acc.update(probs, labels.int())

                recall = BinaryRecall(threshold=self.best_threshold)
                recall.update(probs, labels.int())

                auroc = BinaryAUROC()
                auroc.update(probs, labels.int())

                confmat = BinaryConfusionMatrix(threshold=self.best_threshold)
                confmat.update(probs, labels.int())
                tn, fp, fn, tp = confmat.compute().flatten()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0


                metrics[f'{mode}_Loss'] = total_loss / len(dataloader)
                metrics[f'{mode}_AUC'] = auroc.compute().item()
                metrics[f'{mode}_F1'] = f1.compute().item()
                metrics[f'{mode}_Sensitivity'] = recall.compute().item()
                metrics[f'{mode}_Specificity'] = specificity
                metrics[f'{mode}_Accuracy'] = acc.compute().item()

            return metrics

    def loss(self, pred, y: Tensor):
        return self.loss_fn(pred, y.type_as(pred).view_as(pred))

    def learn(self, dataloaders, epochs, lr, wd):
        best_it = -1
        best_model_state = copy.deepcopy(self.state_dict())
        best_auc = float('-inf')

        t_accu, v_accu, e_accu = [], [], []
        lossL, lossLV = [], []

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)

        epoch_bar = tqdm(range(epochs), desc="Training Progress", unit="epoch", leave=True)

        for epoch in epoch_bar:
            self.train()
            epoch_loss = 0

            for batch in dataloaders['train']:
                batch = batch.to(self.device)

                optimizer.zero_grad()

                pred = self(batch)

                loss = self.loss(pred, batch['y'])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            metrics = self.test(dataloaders)

            log_loss = epoch_loss / len(dataloaders['train'])

            metric = 'Accuracy'

            ord_dict = OrderedDict([
                ('Loss', f"{log_loss:.4f}"),
                ('Loss Val', f"{metrics['val_Loss']:.4f}"),
                ('Train', f"{metrics['train_' + metric]:.4f}"),
                ('Val', f"{metrics['val_' + metric]:.4f}"),
                ('Test', f"{metrics['test_' + metric]:.4f}")
            ])
            epoch_bar.set_postfix(ord_dict)

            # Accumulate results for plotting
            t_accu.append(metrics[f"train_{metric}"])
            v_accu.append(metrics[f"val_{metric}"])
            e_accu.append(metrics[f"test_{metric}"])

            lossL.append(log_loss)
            lossLV.append(metrics['val_Loss'])

            if best_auc < metrics['val_Accuracy']:
                best_auc = metrics['val_Accuracy']
                best_model_state = copy.deepcopy(self.state_dict())
                best_it = epoch


        metrics = self.finish_training(epochs, best_it, dataloaders, [t_accu, v_accu, e_accu, best_auc], [lossL, lossLV],
                             best_model_state)
        return metrics

    def finish_training(self, epochs, best_it, dataloaders, accu, loss, best_model_state):
        self.load_state_dict(best_model_state)
        self.to(self.device)
        self.eval()

        metrics = self.test(dataloaders, find_threshold=True)


        results = []
        for batch in dataloaders['test']:
            batch = batch.to(self.device)
            output = self(batch).detach().cpu().numpy()
            results.append(output)

        np.save(f"preds/fc_{self.name}_{self.model_name}_{self.fold}_{'feats' if self.use_features else 'nofeats'}.npy",
                np.array(results))


        tqdm.write(f"       Best Model at Iteration {best_it:d}: Train: {metrics['train_Accuracy']:.4f}, "
                    f"Val: {metrics['val_Accuracy']:.4f}, Test: {metrics['test_Accuracy']:.4f}")
        # Plot Accuracy Evolution
        plt.figure()
        plt.plot(accu[0], label='Train')
        plt.plot(accu[1], label='Validation')
        plt.plot(accu[2], label='Test')

        plt.title('Accuracy Evolution')
        plt.legend()
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.xticks(range(0, epochs, max(1, epochs // 10)))
        plt.savefig(f"figures/accuracy_fc_{self.name}_{self.model_name}.svg", format='svg', dpi=1200)
        plt.clf()
        plt.close()

        # **Plot Loss Evolution**
        plt.figure()
        plt.plot(loss[0], label='Training Loss', linestyle='dashed')
        plt.plot(loss[1], label='Validation Loss', linestyle='solid')
        plt.title('Loss Evolution')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.xticks(range(0, epochs, max(1, epochs // 10)))
        plt.savefig(f"figures/loss_fc_{self.name}_{self.model_name}.svg", format='svg', dpi=1200)
        plt.clf()
        plt.close()

        return metrics