import torch
import torch.nn as nn
from tqdm import tqdm
from torch import Tensor
import matplotlib.pyplot as plt
from collections import OrderedDict
from torch_geometric.nn.conv import HypergraphConv
from torcheval.metrics import BinaryAUROC, BinaryF1Score, BinaryRecall, BinaryConfusionMatrix, BinaryAccuracy


class AttentionPooling(nn.Module):
    def __init__(self, in_dim, hidden_dim=128):
        super(AttentionPooling, self).__init__()
        self.att_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        scores = self.att_mlp(x)
        weights = nn.functional.softmax(scores, dim=1)
        x_pooled = torch.sum(weights * x, dim=1)
        return x_pooled

class FCHypergraphLearning(torch.nn.Module):
    def __init__(self, in_size, hidden_size, dropout, device, y: Tensor, name):
        super(FCHypergraphLearning, self).__init__()

        self.name = name

        self.device = device
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

        num_positive = torch.sum(y == 1).float()
        num_negative = torch.sum(y == 0).float()
        pos_weight = num_negative / (num_positive + 1e-20)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.roi = in_size
        self.hidden_size = hidden_size

        self.conv1 = HypergraphConv(in_size, hidden_size)
        self.conv2 = HypergraphConv(hidden_size, int(hidden_size/2))

        self.bn1 = nn.BatchNorm1d(num_features=hidden_size)
        self.bn2 = nn.BatchNorm1d(num_features=int(hidden_size/2))

        self.att_pool = AttentionPooling(int(hidden_size/2),hidden_size)

        self.embeddingFinal = nn.Linear(int(hidden_size/2), 1)

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

    def forward(self, data):
        batch = data.batch
        input_x = data.x

        if self.name == 'knn':
            hyperedge_index = data.hyperedge_index
        elif self.name == "ts-modelling":
            hyperedge_index = data.ts_modelling_index
        elif self.name == "fc-modelling":
            hyperedge_index = data.fc_modelling_index
        elif self.name == "k-random":
            hyperedge_index = data.random_hyperedge_index
        else:
            return 0

        x = self.conv1(input_x, hyperedge_index)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x, hyperedge_index)
        x = self.bn2(x)

        x = self.att_pool(x.view(-1, self.roi, x.shape[1]))
        x = self.dropout(x)

        x = self.embeddingFinal(x)

        return x

    def test(self, dataloaders, find_threshold=False):
        self.eval()
        with torch.no_grad():
            metrics = {}
            best_threshold = 0.5

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
                            best_threshold = t.item()

                f1 = BinaryF1Score(threshold=best_threshold)
                f1.update(probs, labels.int())

                acc = BinaryAccuracy(threshold=best_threshold)
                acc.update(probs, labels.int())

                recall = BinaryRecall(threshold=best_threshold)
                recall.update(probs, labels.int())

                auroc = BinaryAUROC()
                auroc.update(probs, labels.int())  # AUROC no necesita threshold

                confmat = BinaryConfusionMatrix(threshold=best_threshold)
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
                ('Train', f"{metrics[f"train_{metric}"]:.4f}"),
                ('Val', f"{metrics[f"val_{metric}"]:.4f}"),
                ('Test', f"{metrics[f"test_{metric}"]:.4f}")
            ])
            epoch_bar.set_postfix(ord_dict)

            # Accumulate results for plotting
            t_accu.append(metrics[f"train_{metric}"])
            v_accu.append(metrics[f"val_{metric}"])
            e_accu.append(metrics[f"test_{metric}"])

            lossL.append(log_loss)
            lossLV.append(metrics['val_Loss'])


        metrics = self.finish_training(epochs, dataloaders, [t_accu, v_accu, e_accu], [lossL, lossLV])
        return metrics

    def finish_training(self, epochs, dataloaders, accu, loss):
        self.to(self.device)
        self.eval()

        metrics = self.test(dataloaders, find_threshold=True)

        tqdm.write(f"       Train: {metrics['train_AUC']:.4f}, Val: {metrics['val_AUC']:.4f}, Test: {metrics['test_AUC']:.4f}")

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
        plt.savefig(f"hypergraph/hypergraph_learning_data/accuracy_fc_{self.type}.svg", format='svg', dpi=1200)
        plt.clf()
        plt.close()

        # **Plot Loss Evolution**
        plt.figure()
        plt.plot(loss[0], label='Training Loss', linestyle='dashed')
        plt.plot(loss[1], label='Validation Loss', linestyle='solid')
        plt.title('Loss Evolution')
        plt.ylabel('Loss'
        plt.xlabel('Epoch')
        plt.legend()
        plt.xticks(range(0, epochs, max(1, epochs // 10)))
        plt.savefig(f"hypergraph/hypergraph_learning_data/loss_fc_{self.type}.svg", format='svg', dpi=1200)
        plt.clf()
        plt.close()

        return metrics