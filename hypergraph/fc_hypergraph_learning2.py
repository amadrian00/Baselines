"""
Adrián Ayuso Muñoz 2024-12-11.

Class of the hypergraph_learning submodule.
    Receives the hypergraph as input and returns the inter-features.
"""
import copy
import torch
from torch_geometric.utils import to_dense_adj
from tqdm import tqdm
import torch.nn as nn
from torch import Tensor
import matplotlib.pyplot as plt
from collections import OrderedDict
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data
from torch_geometric.nn.conv import HypergraphConv
from torch_geometric.loader import DataLoader as DataLoaderG
from brain_encoding.fc_hypergraph import FCHypergraph, create_hyperedges
from torcheval.metrics import BinaryAUROC, BinaryF1Score, BinaryRecall, BinaryConfusionMatrix, BinaryAccuracy
import torch.nn.functional as F
from .bnt import BrainNetworkTransformer

def prepare_dataloader(graphs, batch_size, shuffle=False):
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

        self.tau = 1.0
        self.anneal_rate = 0.99
        self.min_tau = 0.1

        if custom_layer:
            self.conv1 = HypergraphConvLayer(in_size, hidden_size, use_attention=False)
            self.conv2 = HypergraphConvLayer(hidden_size, int(hidden_size/2), use_attention=False)
        else:
            self.conv1 = HypergraphConv(in_size, hidden_size)
            self.conv2 = HypergraphConv(hidden_size, hidden_size)

        self.bn1 = nn.BatchNorm1d(num_features=hidden_size)
        self.bn2 = nn.BatchNorm1d(num_features=int(hidden_size/2))

        self.att_pool = AttentionPooling(int(hidden_size/2),hidden_size)

        self.show = False

        self.dynamic = False
        self.embeddingFinal = nn.Linear(hidden_size, 1)
            # Careful this needs to be changed to hidden_size for knn and nslr

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
        #input_x = data.eye.to(device=self.device).view(-1, self.roi, self.roi)

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

        x_mean = torch.mean(x, dim=1)
        x_max = torch.max(x, dim=1)[0]
        x = torch.cat([x_mean, x_max], dim=1)

        #x = self.att_pool(x)
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

    def loss(self, pred, y: Tensor, **kwargs):
        return self.loss_fn(pred, y.type_as(pred).view_as(pred))

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

            self.tau = max(self.min_tau, self.tau * self.anneal_rate)

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

        torch.save(self.state_dict(), 'proposed_model.pth')
        return metrics

class BrainAttentionLayer(nn.Module):
    def __init__(self, axis_dim, dropout, num_hyperedges):
        super(BrainAttentionLayer, self).__init__()
        self.axis_dim = axis_dim
        self.attn = nn.MultiheadAttention(embed_dim=axis_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(axis_dim)
        self.learnable_query  = nn.Parameter(torch.randn(num_hyperedges, self.axis_dim))


    def forward(self, x):
        attn_output, attn_weights = self.attn(query=self.learnable_query.unsqueeze(0).expand(x.size(0), -1, -1), key=x, value=x)
        attn_output = self.norm(attn_output)
        return attn_output, attn_weights

class BrainAttentionModel(nn.Module):
    def __init__(self, num_rois=200, num_hyperedges=10, num_layers=1, dropout=0):
        super(BrainAttentionModel, self).__init__()
        self.axis_dim = num_rois
        self.num_hyperedges = num_hyperedges


        self.layers = nn.ModuleList([
            BrainAttentionLayer(self.axis_dim, dropout, num_hyperedges) for _ in range(num_layers)
        ])

    def forward(self, fc_matrix):
        x = fc_matrix
        attn_weights_all = []

        for layer in self.layers:
            x, attn_weights = layer(x)  # queries fijos en todas las capas
            attn_weights_all.append(attn_weights)

        return x, attn_weights_all

class CorrelationToIncidenceTransformer(FCHypergraphLearning):
    def __init__(self, num_hyperedges, in_size, hidden_size, seq_len, num_heads, num_layers, dropout, device, y):
        super(CorrelationToIncidenceTransformer, self).__init__(in_size, hidden_size, seq_len, num_layers,
                                                                dropout, device, y, custom_layer=True)
        self.type = 'incidence_static'
        self.num_hyperedges = num_hyperedges

        self.transformer = BrainNetworkTransformer()
        self.transformer2 = BrainAttentionModel(num_hyperedges=num_hyperedges, dropout=dropout)

        self.W_recon = nn.Parameter(torch.ones(num_hyperedges))

        self._initialize_weights()

    def gumbel_softmax_sample(self, logits):
        U = torch.rand_like(logits)
        gumbel = -torch.log(-torch.log(U + 1e-9) + 1e-9)
        return torch.softmax((logits + gumbel) / self.tau, dim=-1)

    def forward(self, data):
        try:
            input_x = data.x.view(-1, self.roi, self.roi)
        except:
            input_x = data.view(-1, self.roi, self.roi)

        # ======== GENERACIÓN DE H ========
        _, output, _ = self.transformer(input_x)
        output = self.dropout(input_x)
        output, attn_weights = self.transformer2(output)
        x = output.permute(0, 2, 1)
        incidence_matrix = self.gumbel_softmax_sample(x)

        # ======== DEBUG OPCIONAL ========
        if self.show:
            print("     \n[INCIDENCE STATS]")
            #print("         Batch mean incidence:", incidence_matrix.mean(dim=(1, 2)))
            print("         [DEBUG] H.std()", incidence_matrix.std().item())
            print("         Overall max:", incidence_matrix.max().item(), "min:", incidence_matrix.min().item())
            print("         Sparsity (ratio < 0.01):", (incidence_matrix < 0.01).float().mean().item())
            cos_sim = []
            l1_diff = []
            for i in range(incidence_matrix.shape[0] - 1):
                h1 = incidence_matrix[i].flatten()
                h2 = incidence_matrix[i + 1].flatten()
                cos_sim.append(F.cosine_similarity(h1, h2, dim=0).item())
                l1_diff.append(torch.abs(h1 - h2).mean().item())
            print("         Mean Cosine Similarity:", sum(cos_sim) / len(cos_sim))
            print("         Mean L1 Difference:", sum(l1_diff) / len(l1_diff))

        # ======== TEST TIME VISUALIZACIÓN ========
        if self.finished_training:
            incidence_matrix = incidence_matrix.detach()
            self.plot_corr(input_x[0], "_origin")
            self.plot_corr(x[0], '_fc', -1)
            self.plot_corr(incidence_matrix[0], '_activation')
            self.plot_corr(incidence_matrix[0], '_final')
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

        # ======== DROPOUT PRE-GNN ========
        input_x = self.dropout(input_x)


        emb1 = self.conv1(input_x, incidence_matrix)
        emb1 = emb1.view(-1, emb1.shape[2])
        x = self.bn1(emb1).view(-1, self.roi, emb1.shape[1])
        x = self.activation(x)

        emb2 = self.conv2(x, incidence_matrix)
        emb2 = emb2.view(-1, emb2.shape[2])
        x = self.bn2(emb2).view(-1, self.roi, emb2.shape[1])

        # ======== POOLING Y CLASIFICADOR (también bloqueado si warmup) ========
        x_mean = torch.mean(x, dim=1)
        x_max = torch.max(x, dim=1)[0]
        x_g = torch.cat([x_mean, x_max], dim=1)
        x = self.dropout(x_g)

        x = self.embeddingFinal(x)

        return incidence_matrix, x, input_x, x_g

    def contrastive_loss2(self, x, labels, margin=1.0):
        """
        x: [B, D] embeddings
        labels: [B] etiquetas (0 o 1)
        margin: margen para los negativos
        """
        B = x.size(0)

        # Expandimos para pares (i, j)
        x_i = x.unsqueeze(1)  # [B, 1, D]
        x_j = x.unsqueeze(0)  # [1, B, D]

        # Distancias euclídeas entre todos los pares
        dists = torch.norm(x_i - x_j, dim=-1)  # [B, B]

        # Creamos la máscara de pares positivos y negativos
        labels_i = labels.unsqueeze(1)  # [B, 1]
        labels_j = labels.unsqueeze(0)  # [1, B]

        same_class = (labels_i == labels_j).float()  # [B, B]
        diff_class = 1.0 - same_class  # [B, B]

        # Evitamos la diagonal (pares idénticos)
        mask_offdiag = ~torch.eye(B, dtype=torch.bool, device=x.device)
        same_class = same_class * mask_offdiag
        diff_class = diff_class * mask_offdiag
        dists = dists * mask_offdiag

        # Pérdida contrastiva:
        pos_loss = (same_class * dists.pow(2)).sum()
        neg_loss = (diff_class * torch.clamp(margin - dists, min=0).pow(2)).sum()

        # Normalizamos
        num_pos = same_class.sum().clamp(min=1)
        num_neg = diff_class.sum().clamp(min=1)

        loss = (pos_loss / num_pos) + (neg_loss / num_neg)
        return loss

    def js_divergence(self, p, q, eps=1e-8):
        """
        Jensen-Shannon divergence entre dos distribuciones p y q,
        con p,q shape = [dim] y valores >=0 que suman 1.
        """
        m = 0.5 * (p + q)
        p_log = torch.log(p + eps)
        q_log = torch.log(q + eps)
        m_log = torch.log(m + eps)
        return 0.5 * (torch.sum(p * (p_log - m_log), dim=-1) + torch.sum(q * (q_log - m_log), dim=-1))

    def contrastive_loss(self, H, labels, margin=1.0):
        # H: [B, N, E], distribuciones fila a fila (softmax)
        B, N, E = H.shape

        # Expand dimensiones para pares (broadcast)
        H_i = H.unsqueeze(1)  # [B,1,N,E]
        H_j = H.unsqueeze(0)  # [1,B,N,E]

        # Calcular JS divergencia para cada par y fila
        # Resultado shape: [B,B,N]
        js_per_row = self.js_divergence(H_i, H_j)  # vectorizado sobre último eje E

        # Promediar sobre filas (nodos) para un único valor por par
        dist = js_per_row.mean(dim=2)  # [B,B]

        # Máscaras de clase
        labels = labels.view(-1, 1)  # [B,1]
        same_class = (labels == labels.T).float()  # [B,B]
        diff_class = 1.0 - same_class  # [B,B]

        # Máscara para ignorar diagonal (pares i==j)
        mask = ~torch.eye(B, dtype=torch.bool, device=H.device)

        # Pérdida positiva (pares mismos labels)
        pos_loss = (dist * same_class)[mask].sum()

        # Pérdida negativa (pares diferentes labels)
        neg_loss = (F.relu(margin - dist) * diff_class)[mask].sum()

        total_pairs = B * (B - 1)

        loss = (pos_loss + neg_loss) / total_pairs

        return loss

    def recon_loss(self, H, FC):
        W = torch.diag(self.W_recon)
        S = H @ W @ H.transpose(1, 2)  # [B, N, N]

        alpha = 0.5
        beta = 0.5

        # Flatten para correlación
        S_flat = S.view(S.size(0), -1)
        FC_flat = FC.view(FC.size(0), -1)

        S_mean = S_flat.mean(dim=1, keepdim=True)
        FC_mean = FC_flat.mean(dim=1, keepdim=True)

        S_centered = S_flat - S_mean
        FC_centered = FC_flat - FC_mean

        numerator = (S_centered * FC_centered).sum(dim=1)
        denominator = torch.sqrt((S_centered ** 2).sum(dim=1)) * torch.sqrt((FC_centered ** 2).sum(dim=1))
        corr = numerator / (denominator + 1e-8)

        # Pérdida de primer orden: 1 - correlación
        loss_corr = 1 - corr.mean()

        # Pérdida de segundo orden: similitud entre perfiles de conectividad
        S_norm = F.normalize(S, p=2, dim=-1)
        FC_norm = F.normalize(FC, p=2, dim=-1)
        sim_S = S_norm @ S_norm.transpose(1, 2)
        sim_FC = FC_norm @ FC_norm.transpose(1, 2)
        loss_second_order = F.mse_loss(sim_S, sim_FC)

        return alpha * loss_corr + beta * loss_second_order

    def loss(self, pred, y: Tensor, **kwargs):
        class_loss = self.loss_fn(pred[1], y.type_as(pred[1]).view_as(pred[1]))

        entropy_reg = entropy_loss(pred[0])

        #recon_loss = self.recon_loss(pred[0], kwargs.get('x').view(-1, self.roi, self.roi))
        contr_loss = self.contrastive_loss(pred[0], y)
        contr_loss2 = self.contrastive_loss2(pred[3], y)
        #weight_loss = pred[0].mean()

        if self.show:
            print("\n[LOSS COMPONENTS]")
            print(f"Class 1 loss: {class_loss.item():.4f}")
            #print(f"Class 2 loss: {class_loss2.item():.4f}")
            print(f"Entropy loss: {entropy_reg.item():.4f}")
            print(f"Reconstruction loss: {recon_loss.item():.4f}")

        return 1 * class_loss + 0 * contr_loss * 0.5*contr_loss2 + 0.5*entropy_reg #+ 0.25*recon_loss

    @staticmethod
    def plot_corr(x, suffix='', vmin=0):
        fig, ax = plt.subplots(figsize=(20, 20))
        cax = ax.matshow(x.detach().cpu(), cmap='viridis', vmin=vmin, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(cax)
        plt.savefig(f"correlation_matrix{suffix}.svg", format="svg", bbox_inches="tight")
        plt.close()


def entropy_loss(H):
    H = torch.clamp(H, 1e-8, 1 - 1e-8)
    return -(H * H.log()).sum(dim=-1).mean()
