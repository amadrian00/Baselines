"""
Adrián Ayuso Muñoz 2024-12-11.

Class of the hypergraph_learning submodule.
    Receives the hypergraph as input and returns the inter-features.
"""
import yaml
import copy
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch import Tensor
import matplotlib.pyplot as plt
from collections import OrderedDict
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data
from torch_geometric.nn.conv import HypergraphConv
from torch_geometric.loader import DataLoader as DataLoaderG
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from brain_encoding.fc_hypergraph import FCHypergraph, create_hyperedges
from torcheval.metrics import BinaryAUROC, BinaryF1Score, BinaryRecall, BinaryConfusionMatrix, BinaryAccuracy

config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
configH =config['hypergraph_learning_parameters']
configG = config['general']

def prepare_dataloader(graphs, batch_size=configH['batch_size'], shuffle=False):
    if type(graphs.dataset) == FCHypergraph:
        return DataLoaderG(graphs, batch_size=batch_size, drop_last=False, shuffle=shuffle)
    else:
        return DataLoader(graphs, batch_size=batch_size, drop_last=False, shuffle=shuffle)

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
                 bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.theta = nn.Parameter(torch.Tensor(in_features, out_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize learnable parameters."""
        nn.init.xavier_uniform_(self.theta)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

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

        if w is None:
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

class FCHypergraphLearning(torch.nn.Module):
    def __init__(self, in_size, hidden_size, seq_len, num_layers, dropout, device, y: Tensor, custom_layer=True):
        super(FCHypergraphLearning, self).__init__()

        self.type = 'static'
        self.finished_training = False

        self.device = device
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

        num_positive = torch.sum(y == 1).float()
        num_negative = torch.sum(y == 0).float()
        pos_weight = num_negative / (num_positive + 1e-20)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.roi = in_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len


        if custom_layer:
            self.conv1 = HypergraphConvLayer(in_size, hidden_size)
            self.conv2 = HypergraphConvLayer(hidden_size, int(hidden_size/2))
        else:
            self.conv1 = HypergraphConv(in_size, hidden_size)
            self.conv2 = HypergraphConv(hidden_size, int(hidden_size/2))

        self.bn1 = nn.BatchNorm1d(num_features=hidden_size)
        self.bn2 = nn.BatchNorm1d(num_features=int(hidden_size/2))

        if seq_len > 0:
            self.dynamic = True
            self.gru = nn.GRU( input_size=hidden_size, hidden_size=int(hidden_size/2), num_layers=num_layers, batch_first=True,
                           dropout=dropout, bidirectional=True)
            self.embeddingFinal = nn.Linear(hidden_size * 2, 1)
        else:
            self.dynamic = False
            self.embeddingFinal = nn.Linear(hidden_size, 1)

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
        if self.dynamic:
            original_lengths = data['original_lengths']
            input_x = torch.cat([out[:length] for out, length in zip(data['x'], original_lengths)], dim=0)

            graphs = [Data(x=x,hyperedge_index=create_hyperedges(x, self.device)) for x in input_x]
            data = Batch.from_data_list(graphs)

        else:
            original_lengths = []

        batch = data.batch
        batch_size = len(data.ptr) - 1

        input_x = data.x.view(-1, self.roi, self.roi)

        #hyperedge_index = to_dense_adj(data.hyperedge_index, batch, max_num_nodes=self.roi, batch_size=batch_size)
        input_x = data.eye.to(device=self.device).view(-1, self.roi, self.roi)

        hyperedge_index = data.analytical_hyperedge.view(batch_size, self.roi, -1)

        #hyperedge_index_d, _ = dropout_edge(hyperedge_index, self.dropout.p, training=self.training)
        #incidence_matrix_d = self.dropout(hyperedge_index)
        #input_x = self.dropout(input_x)

        emb1 = self.conv1(input_x, hyperedge_index)
        emb1 = emb1.view(-1, emb1.shape[2])
        x = self.bn1(emb1).view(-1, self.roi, emb1.shape[1])
        x = self.activation(x)

        emb2 = self.conv2(x, hyperedge_index)
        emb2 = emb2.view(-1, emb2.shape[2])
        x = self.bn2(emb2).view(-1, self.roi, emb2.shape[1])

        #x = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        x_mean = torch.mean(x, dim=1)
        x_max = torch.max(x, dim=1)[0]
        x = torch.cat([x_mean, x_max], dim=1)
        x = self.dropout(x)

        if self.dynamic:  # Temporal encoder
            x = torch.split(x, original_lengths.tolist())
            graph_output = torch.cat([t.mean(dim=0, keepdim=True) for t in x])

            x = pack_padded_sequence(pad_sequence(x, batch_first=True), original_lengths, batch_first=True,
                                     enforce_sorted=False)

            x, hn = self.gru(x, torch.zeros(self.gru.num_layers * 2, batch_size, self.gru.hidden_size).to(self.device))
            hn = torch.cat((hn[0], hn[1]), dim=1)
            hn = self.dropout(hn)

            x = torch.cat([graph_output, hn], dim=1)

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
                    batch['x'] = batch['x'].to(self.device)
                    batch['y'] = batch['y'].to(self.device)
                    batch['batch'] = batch['batch'].to(self.device)
                    batch['hyperedge_index'] = batch['hyperedge_index'].to(self.device)
                    batch['analytical_hyperedge'] = batch['analytical_hyperedge'].to(self.device)
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

                        print(f"recall: {recall:.4f}, specificity: {specificity:.4f}")

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

    def learn(self, dataloaders, wandb = None, epochs=configH['epochs'], lr=configH['lr'], wd=configH['weight_decay'],
              save=configG['save'][3], get_best=True):
        best_it = -1
        best_model_state = copy.deepcopy(self.state_dict())
        best_acc = float('-inf')

        self.finished_training = False

        t_accu, v_accu, e_accu = [], [], []
        lossL, lossLV = [], []

        optimizer = torch.optim.Adam(self.parameters(), weight_decay=wd, lr=lr)

        # Single progress bar for all epochs
        epoch_bar = tqdm(range(epochs), desc="Training Progress", unit="epoch", leave=True)

        for epoch in epoch_bar:
            self.train()
            epoch_loss = 0

            for batch in dataloaders['train']:
                batch['x'] = batch['x'].to(self.device)
                batch['y'] = batch['y'].to(self.device)
                batch['batch'] = batch['batch'].to(self.device)
                batch['hyperedge_index'] = batch['hyperedge_index'].to(self.device)
                batch['analytical_hyperedge'] = batch['analytical_hyperedge'].to(self.device)
                optimizer.zero_grad()

                pred = self(batch)

                loss = self.loss(pred, batch['y'])
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

            if get_best & (best_acc < metrics['val_AUC']):
                best_acc = metrics['val_Loss']
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
            tqdm.write(f"       Best Model at Iteration {best_it:d}: Train: {metrics['train_AUC']:.4f}, "
                       f"Val: {metrics['val_AUC']:.4f}, Test: {metrics['test_AUC']:.4f}")

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
            plt.savefig(f"hypergraph/hypergraph_learning_data/accuracy_fc_{self.type}.svg", format='svg', dpi=1200)
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
            plt.savefig(f"hypergraph/hypergraph_learning_data/loss_fc_{self.type}.svg", format='svg', dpi=1200)
            plt.clf()
            plt.close()

        if save:
            np.save(f"hypergraph/hypergraph_learning_data/inter_features_{self.type}.npy", self(dataloaders['train']))
        return metrics

class CorrelationToIncidenceTransformer(FCHypergraphLearning):
    def __init__(self, num_hyperedges, in_size, hidden_size, seq_len, num_heads, num_layers, dropout, device, y):
        super(CorrelationToIncidenceTransformer, self).__init__(in_size, hidden_size, seq_len, num_layers,
                                                                dropout, device, y, custom_layer=True)
        self.type = 'incidence_static'
        self.num_hyperedges = num_hyperedges

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=in_size, nhead=num_heads, dropout=dropout,
                                       activation='gelu',
                                       dim_feedforward=in_size * 4, batch_first=True),
            num_layers=num_layers,
            norm=nn.LayerNorm(in_size))

        #self.conv_embedding = nn.Parameter(
        #   torch.randn(1, num_hyperedges, in_size), requires_grad=True
        #)

        #pe = torch.zeros(self.roi, self.roi).to(self.device)
        #position = torch.arange(0,  self.roi).unsqueeze(1)  # shape (n,1)
        #div_term = torch.exp(torch.arange(0, self.roi, 2) * (-torch.log(torch.tensor(10000.0)) / self.roi))
        #pe[:, 0::2] = torch.sin(position * div_term)
        #pe[:, 1::2] = torch.cos(position * div_term)

        #self.pe = pe

        #self.transformer_decoder = nn.TransformerDecoder(
        #    nn.TransformerDecoderLayer(d_model=in_size, nhead=num_heads, dropout=dropout,
        #                               activation='gelu',
        #                               dim_feedforward=in_size * 4, batch_first=True),
        #    num_layers=num_layers,
        #    norm=nn.LayerNorm(in_size)
        #)

        self.fc = nn.Linear(in_size*in_size, in_size*num_hyperedges)

        self.param = nn.Parameter(0.1 + 0.8 * torch.rand(self.num_hyperedges), requires_grad=True)

        self._initialize_weights()

    def forward(self, data):
        if self.dynamic:
            original_lengths = data['original_lengths']
            input_x = torch.cat([out[:length] for out, length in zip(data['x'], original_lengths)], dim=0)
        else:
            original_lengths = []
            input_x = data.x.view(-1, self.roi, self.roi)

        input_x = self.dropout(input_x)

        x = self.transformer_encoder(input_x)
        x = self.fc(x.view(x.shape[0], -1)).view(-1, self.roi, self.num_hyperedges)

        #x = input_x + self.pe

        #x = self.transformer_decoder(self.conv_embedding.expand(x.shape[0], -1, -1) , x)

        #x = x.transpose(1, 2)

        incidence_matrix_sig = torch.sigmoid(x)
        incidence_matrix = torch.relu(incidence_matrix_sig-torch.sigmoid(self.param))

        input_x = torch.eye(self.roi).unsqueeze(0).repeat(input_x.shape[0], 1, 1).to(self.device)

        if self.finished_training:
            self.plot_corr(data['x'].view(-1, self.roi, self.roi)[0], "_origin")
            self.plot_corr(x[0], '_fc')
            self.plot_corr(incidence_matrix_sig[0], '_activation')

            incidence_matrix += torch.sigmoid(self.param).view(1, 1, -1).expand_as(incidence_matrix) * incidence_matrix > 0
            self.plot_corr(incidence_matrix[0], '_final')

            hyperedge_degrees = (incidence_matrix>0).float().sum(dim=1)

            min_deg = torch.min(hyperedge_degrees).item()
            max_deg = torch.max(hyperedge_degrees).item()
            mean_deg = torch.mean(hyperedge_degrees).item()
            # Histogram of degrees across the whole batch
            plt.hist(hyperedge_degrees.cpu().numpy().flatten(), bins=20)
            plt.title("Hyperedge Degrees Distribution (Test)")
            plt.xlabel("Number of Nodes per Hyperedge")
            plt.ylabel("Frequency")
            plt.savefig("hyperedge_histogram_test.svg")
            print(f"            Hyperedge Degrees - Min: {min_deg:.2f}, Max: {max_deg:.2f}, Average: {mean_deg:.2f}")
            plt.close()

            return incidence_matrix, input_x

        incidence_matrix_d = self.dropout(incidence_matrix)
        input_x = self.dropout(input_x)

        emb1 = self.conv1(input_x, incidence_matrix_d)
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

        if self.dynamic: # Temporal encoder
            x = torch.split(x, original_lengths.tolist())

            graph_output = torch.cat([t.mean(dim=0, keepdim=True) for t in x])

            x = pack_padded_sequence(pad_sequence(x, batch_first=True), original_lengths, batch_first=True,
                                     enforce_sorted=False)

            x, hn = self.gru(x, torch.zeros(self.gru.num_layers * 2, x.shape[0], self.gru.hidden_size).to(self.device))
            hn = torch.cat((hn[0], hn[1]), dim=1)
            hn = self.dropout(hn)

            x = torch.cat([graph_output, hn], dim=1)

        x = self.embeddingFinal(x)

        return x, incidence_matrix_sig, incidence_matrix

    def density_loss(self, incidence_matrix_sig, incidence_matrix):
        total_weight = incidence_matrix_sig.sum(dim=(1,2)).mean()
        total_weight = total_weight / (self.num_hyperedges*self.roi)

        hyperedge_var_loss = torch.var(incidence_matrix.sum(dim=1), dim=1, unbiased=False).mean()
        node_var_loss = torch.var(incidence_matrix.sum(dim=2), dim=1, unbiased=False).mean()

        return total_weight, hyperedge_var_loss, node_var_loss

    def loss(self, pred, y: Tensor):
        #class_loss = self.loss_fn(pred[0], y.type_as(pred[0]).view_as(pred[0]))
        num_positive = torch.sum(y == 1).float()
        num_negative = torch.sum(y == 0).float()
        pos_weight = num_negative / (num_positive + 1e-20)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(pred[0].device))
        class_loss = loss_fn(pred[0], y.type_as(pred[0]).view_as(pred[0]))


        total_weight, hyperedge_var_loss, node_var_loss = self.density_loss(pred[1], pred[2])

        prinat = False
        if prinat and self.training:
         print(f'Class: {class_loss.item():.4f} (w*L={class_loss.item():.4f}) | '
           f'Density: {total_weight.item():.4f} (w*L={total_weight.item():.4f}) | '
           f'VarH: {hyperedge_var_loss.item():.4f} (w*L={0.1 * hyperedge_var_loss.item():.4f}) | '
           f'VarN: {node_var_loss.item():.4f} (w*L={node_var_loss.item():.4f})'
         )

        return class_loss + total_weight

    @staticmethod
    def plot_corr(x, suffix=''):
        fig, ax = plt.subplots(figsize=(20, 20))
        cax = ax.matshow(x.detach().cpu(), cmap='viridis', vmin=-1, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(cax)
        plt.savefig(f"correlation_matrix{suffix}.svg", format="svg", bbox_inches="tight")
        plt.close()