"""
Adrián Ayuso Muñoz 2024-12-11.

Class of the hypergraph_learning submodule.
    Receives the hypergraph as input and returns the inter-features.
"""
import numpy as np
import copy
import torch
from tqdm import tqdm
import torch.nn as nn
from torch import Tensor
import matplotlib.pyplot as plt
from collections import OrderedDict
from torch_geometric.nn.conv import HypergraphConv
from torcheval.metrics import BinaryAUROC, BinaryF1Score, BinaryRecall, BinaryConfusionMatrix, BinaryAccuracy
from sklearn.metrics import roc_curve

class HypergraphConvLayer(nn.Module):
    """
    PyTorch Module for Hypergraph Convolution (W=I):
    X_out = D^-1 H W B^-1 H^T X theta.

    Args:
        in_features (int): Number of input features per node.
        out_features (int): Number of output features per node.
        bias (bool, optional): If True, adds a learnable bias to the output.
                               Default: True.
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 use_attention: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_attention = use_attention

        self.theta = nn.Parameter(torch.Tensor(in_features, out_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        if use_attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(in_features, in_features),
                nn.LeakyReLU(0.2),
                nn.Linear(in_features, 1)  # output attention score per node
            )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize learnable parameters."""
        nn.init.xavier_uniform_(self.theta)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        if self.use_attention:
            for layer in self.att_mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self,
                x: torch.Tensor,
                incidence: torch.Tensor,
                w: torch.Tensor=None,
               ) -> torch.Tensor:
        """
        Performs the forward pass of the unweighted hypergraph convolution.

        Args:
            x: Input node features (batch_size, n_nodes, f_in).
            incidence: Incidence matrix H (batch_size, n_nodes, n_edges).
                       Assumed to be dense.
            w: Hyperedge weights (batch_size, n_edges)

        Returns:
            torch.Tensor: Output node features (batch_size, n_nodes, f_out).
        """
        H = incidence
        H_t = H.transpose(1, 2)

        D = torch.sum(H, dim=2)
        D_mask = D != 0
        D_inv = torch.zeros_like(D)
        D_inv[D_mask] = 1.0 / D[D_mask]

        B = torch.sum(H, dim=1)
        B_mask = B != 0
        B_inv = torch.zeros_like(B)
        B_inv[B_mask] = 1.0 / B[B_mask]

        if self.use_attention:
            # Compute hyperedge embeddings (avg of incident node embeddings)
            H_float = H.float()
            edge_node_count = H_float.sum(dim=1, keepdim=True).clamp(min=1)
            hyperedge_embs = torch.bmm(H_t, x) / edge_node_count.transpose(1, 2)

            # Apply MLP to get attention scores
            att_scores = self.att_mlp(hyperedge_embs).squeeze(-1)  # (batch_size, n_edges)
            att_weights = torch.sigmoid(att_scores)  # or softmax over edges if you prefer
            w = att_weights
        elif w is None:
            w = x.new_ones(H.shape[0], H.shape[2])

        feats_to_hyper = torch.matmul(H_t, x)                          # H^T @ X
        hyperedge_degree_norm = B_inv.unsqueeze(-1)  * feats_to_hyper  # B^-1 * (H^T @ X)
        weighted = w.unsqueeze(-1) * hyperedge_degree_norm             # W * (B^-1 * (H^T @ X))
        feats_to_nodes = torch.matmul(H, weighted)                     # H @ (W * B^-1 * (H^T @ X))
        node_degree_norm = D_inv.unsqueeze(-1) * feats_to_nodes        # D^-1 * (H @ (W * B^-1 * (H^T @ X)))
        x_out = torch.matmul(node_degree_norm, self.theta)             # (D^-1 * (H @ (W * B^-1 * (H^T @ X)))) @ theta

        # Add bias if enabled
        if self.bias is not None:
            x_out = x_out + self.bias

        return x_out

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.in_features} -> {self.out_features})'

class CorrelationToIncidenceTransformer(nn.Module):
    def __init__(self, num_hyperedges, in_size, hidden_size, num_heads, num_layers, dropout, device, y, fold):
        super(CorrelationToIncidenceTransformer, self).__init__()
        self.num_hyperedges = num_hyperedges
        self.name = 'proposed'
        self.fold = fold
        self.device = device
        
        self.roi = in_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_hyperedges = 0

        self.finished_training = False

        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

        num_positive = torch.sum(y == 1).float()
        num_negative = torch.sum(y == 0).float()
        pos_weight = num_negative / (num_positive + 1e-20)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)


        self.conv1 = HypergraphConvLayer(in_size, hidden_size, use_attention=False)
        self.conv2 = HypergraphConvLayer(hidden_size, int(hidden_size / 2), use_attention=False)


        self.bn1 = nn.BatchNorm1d(num_features=hidden_size)
        self.bn2 = nn.BatchNorm1d(num_features=int(hidden_size / 2))

        self.embeddingFinal = nn.Linear(hidden_size, 1)


        self.best_threshold = 0.5


        self.transformer_encoder = nn.TransformerEncoder(
                                        nn.TransformerEncoderLayer(d_model=in_size, nhead=num_heads, dropout=dropout,
                                                                   activation='gelu',
                                                                   dim_feedforward=in_size * 4, batch_first=True),
                                        num_layers=num_layers,
                                        norm=nn.LayerNorm(in_size))
        self.fc = nn.Linear(in_size*in_size, in_size*num_hyperedges)

        self.param = nn.Parameter(torch.Tensor([0.0]).repeat(self.num_hyperedges), requires_grad=True)


        self._initialize_weights()

    def forward(self, data):
        try:
            input_x = data.x.view(-1, self.roi, self.roi)
        except:
            input_x = data.view(-1, self.roi, self.roi)

        attn = self.transformer_encoder(input_x)
        x = self.dropout(attn)
        x = self.fc(x.view(x.shape[0], -1)).view(-1, self.roi, self.num_hyperedges)

        incidence_matrix_sig = torch.sigmoid(x)
        incidence_matrix = torch.relu(incidence_matrix_sig-torch.sigmoid(self.param))

        input_x = torch.eye(self.roi).unsqueeze(0).repeat(input_x.shape[0], 1, 1).to(self.device)

        if self.finished_training: self.print_matrices(input_x[0], attn[0], x[0], incidence_matrix[0])

        incidence_matrix_d = self.dropout(incidence_matrix)
        input_x_d = self.dropout(input_x)

        emb1 = self.conv1(input_x_d, incidence_matrix_d)
        emb1 = emb1.view(-1, emb1.shape[2])
        x = self.bn1(emb1).view(-1, self.roi, emb1.shape[1])
        x = self.activation(x)

        emb2 = self.conv2(x, incidence_matrix_d)
        emb2 = emb2.view(-1, emb2.shape[2])
        x = self.bn2(emb2).view(-1, self.roi, emb2.shape[1])


        x_mean = torch.mean(x, dim=1)
        x_max = torch.max(x, dim=1)[0]
        x = torch.cat([x_mean, x_max], dim=1)

        x = self.dropout(x)

        x = self.embeddingFinal(x)

        return incidence_matrix, x, input_x, incidence_matrix_sig

    def loss(self, pred, y: Tensor, **kwargs):
        class_loss = self.loss_fn(pred[1], y.type_as(pred[1]).view_as(pred[1]))

        total_weight = pred[0].mean()

        return class_loss + total_weight

    def plot_corr(self, x, suffix='', vmin=0):
        fig, ax = plt.subplots(figsize=(20, 20))
        cax = ax.matshow(x.detach().cpu(), cmap='viridis', vmin=vmin, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(cax)
        plt.savefig(f"correlation_matrix{suffix}_{self.num_hyperedges}.svg", format="svg", bbox_inches="tight")
        plt.close()

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

    def test(self, dataloaders, find_threshold=False):
        self.eval()
        with torch.no_grad():
            metrics = {}

            for mode, dataloader in dataloaders.items():
                total_loss = 0.0
                all_probs = []
                all_labels = []
                im = []
                for batch in dataloader:
                    batch = batch.to(self.device)
                    logits = self(batch)
                    y_true = batch['y']

                    total_loss += self.loss(logits, y_true, x=batch['x']).item()

                    if isinstance(logits, tuple):
                        im.append(logits[0])
                        logits = logits[1]

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

    def learn(self, dataloaders, wandb = None, epochs=0, lr=0, wd=0,
              save=False, get_best=True):
        best_it = -1
        best_model_state = copy.deepcopy(self.state_dict())
        best_acc = float('-inf')

        self.finished_training = False

        t_accu, v_accu, e_accu = [], [], []
        lossL, lossLV = [], []

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)

        # Single progress bar for all epochs
        epoch_bar = tqdm(range(epochs), desc="Training Progress", unit="epoch", leave=True)

        for epoch in epoch_bar:
            self.train()
            epoch_loss = 0

            for batch in dataloaders['train']:
                batch= batch.to(self.device)

                optimizer.zero_grad()

                pred = self(batch)

                loss = self.loss(pred, batch['y'], x=batch['x'])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            metrics = self.test(dataloaders)

            log_loss = epoch_loss / len(dataloaders['train'])

            metric = 'Accuracy'

            if not wandb:
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

            if get_best & (best_acc < metrics['val_Accuracy']):
                best_acc = metrics['val_Accuracy']
                best_model_state = copy.deepcopy(self.state_dict())
                best_it = epoch

            if wandb: wandb.log({"epoch": epoch, " loss": log_loss, "val_acc": metrics['val_AUC'],
                                 "test_acc": metrics['test_AUC'], "train_acc": metrics['train_AUC'] })


        metrics = self.finish_training(epochs, best_it, dataloaders, [t_accu, v_accu, e_accu, best_acc], [lossL, lossLV],
                             get_best, best_model_state, save, wandb)
        return metrics

    def finish_training(self, epochs, best_it, dataloaders, accu, loss, get_best, best_model_state, save, wandb):
        if get_best: self.load_state_dict(best_model_state)
        else: best_it=epochs-1
        self.to(self.device)
        self.eval()

        metrics = self.test(dataloaders, find_threshold=True)

        if not wandb:
            tqdm.write(f"       Best Model at Iteration {best_it:d}: Train: {metrics['train_Accuracy']:.4f}, "
                       f"Val: {metrics['val_Accuracy']:.4f}, Test: {metrics['test_Accuracy']:.4f}")

            results = []
            for batch in dataloaders['test']:
                batch = batch.to(self.device)
                output = self(batch)[1].detach().cpu().numpy()
                results.append(output)

            np.save(f"preds/fc_{self.name}_hyper_{self.num_hyperedges}_{self.fold}.npy", np.array(results))

            # Plot Accuracy Evolution
            plt.figure()
            plt.plot(accu[0], label='Train')
            plt.plot(accu[1], label='Validation')
            plt.plot(accu[2], label='Test')
            if best_it >= 0:
                plt.scatter(best_it, accu[3], color='k', marker='+', s=100, label='Best Model')
            plt.title('Accuracy Evolution')
            plt.legend()
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.xticks(range(0, epochs, max(1, epochs // 10)))
            plt.savefig(f"hypergraph/hypergraph_learning_data/accuracy_fc_{self.name}.svg", format='svg', dpi=1200)
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
            plt.savefig(f"hypergraph/hypergraph_learning_data/loss_fc_{self.name}.svg", format='svg', dpi=1200)
            plt.clf()
            plt.close()

            all_probs = []
            all_labels = []
            for batch in dataloaders['test']:
                batch = batch.to(self.device)
                logits = self(batch)[1]
                probs = torch.sigmoid(logits).view(-1)
                labels = batch['y'].view(-1)
                all_probs.append(probs.detach().cpu())
                all_labels.append(labels.detach().cpu())

            probs = torch.cat(all_probs).numpy()
            labels = torch.cat(all_labels).numpy()
            fpr, tpr, thresholds = roc_curve(labels, probs)

            plt.figure()
            plt.plot(fpr, tpr, label=f'AUROC = {metrics["test_AUC"]:.4f}')
            plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC)')
            plt.legend(loc='lower right')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"hypergraph/hypergraph_learning_data/roc_curve_fc_{self.name}.svg", format='svg', dpi=1200)
            plt.clf()
            plt.close()

        return metrics


    def print_matrices(self, input_x, attn, fc, incidence_matrix):
        incidence_matrix = incidence_matrix.detach()
        incidence_matrix = incidence_matrix / incidence_matrix.max(dim=0, keepdim=True).values
        self.plot_corr(input_x, "_origin")
        self.plot_corr(attn, '_attn', -1)
        self.plot_corr(fc, '_fc', -1)
        self.plot_corr(incidence_matrix, '_activation')
        self.plot_corr(incidence_matrix, '_final')
        hyperedge_degrees = (incidence_matrix > 0.01).float().sum(dim=1)
        plt.hist(hyperedge_degrees.cpu().numpy().flatten(), bins=20)
        plt.title("Hyperedge Degrees Distribution (Test)")
        plt.xlabel("Number of Nodes per Hyperedge")
        plt.ylabel("Frequency")
        plt.savefig("hyperedge_histogram_test.svg")
        print(f"            Hyperedge Degrees - Min: {torch.min(hyperedge_degrees).item():.2f}, "
              f"Max: {torch.max(hyperedge_degrees).item():.2f}, "
              f"Average: {torch.mean(hyperedge_degrees).item():.2f}")
        plt.close()
