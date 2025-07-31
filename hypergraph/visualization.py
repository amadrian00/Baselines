import re
import torch
import pandas as pd
import numpy as  np
from collections import Counter
from collections import defaultdict
import matplotlib.pyplot as plt
from pycircos import Gcircle, Garc
from matplotlib.lines import Line2D
from torch_geometric.utils import to_dense_adj


device = 'cuda' if torch.cuda.is_available() else 'cpu'

colorlist = [
"#ff8a80", "#ff80ab", "#ea80fc", "#b388ff", "#8c9eff", "#82b1ff", "#84ffff", "#a7ffeb", "#b9f6ca",
"#ccff90", "#f4ff81", "#ffff8d", "#ffe57f", "#ffd180", "#ff9e80", "#bcaaa4", "#eeeeee", "#b0bec5",
"#ff5252", "#ff4081", "#e040fb", "#7c4dff", "#536dfe", "#448aff", "#18ffff", "#64ffda", "#69f0ae",
"#b2ff59", "#eeff41", "#ffff00", "#ffd740", "#ffab40", "#ff6e40", "#a1887f", "#e0e0e0", "#90a4ae"]

edgecolors = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#bcf60c",
    "#fabebe", "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1",
    "#000075", "#808080", "#ffffff", "#000000", "#ff4500", "#7f7fff", "#40e0d0", "#ff1493", "#00ff00",
    "#ff6347", "#6a5acd", "#ff69b4", "#7fff00", "#dc143c", "#00ced1", "#9400d3", "#ff8c00", "#8b0000",
    "#00bfff", "#228b22", "#d2691e", "#ffdead", "#ff00ff", "#b22222", "#5f9ea0", "#7fff00", "#1e90ff",
    "#ff7f50", "#6495ed", "#f0e68c", "#dda0dd", "#ffb6c1"]


def plot_full_heatmaps(matrices, suffix):
    matrices = [np.array(matrix) for matrix in matrices]

    fig, axes = plt.subplots(1, 3, figsize=(24, 8), constrained_layout=True)  # Aumento del tamaño general

    vmin = min(matrix.min() for matrix in matrices)
    vmax = max(matrix.max() for matrix in matrices)

    titles = ['Positive Mean', 'Negative Mean', 'Difference']

    # Aumento de la separación entre gráficos
    plt.subplots_adjust(wspace=0.1)

    # Pintar cada heatmap
    for i, ax in enumerate(axes):
        cax = ax.matshow(matrices[i], cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_title(titles[i], fontsize=22)
        ax.set_xticks([])
        ax.set_yticks([])

    # Barra de color global
    cbar = fig.colorbar(cax, ax=axes.ravel().tolist(), orientation='vertical')
    cbar.ax.tick_params(labelsize=16)

    plt.savefig(f"images/{suffix}.svg", format="svg")
    plt.close()

def plot_full_heatmap(matrix, suffix):
    matrix = np.array(matrix)
    if np.max(matrix) == np.min(matrix):
        matrix = np.zeros_like(matrix)
    else:
        matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))

    fig, ax = plt.subplots(figsize=(20, 20))
    cax = ax.matshow(matrix, cmap='viridis', aspect='auto')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(cax)

    plt.savefig(f"images/{suffix}.svg", format="svg")
    plt.close()

class HypergraphPlotter():
    def __init__(self):
        self.incidence_accumulator, self.incidence_accumulator_neg = [], []
        self.k_accumulator, self.k_accumulator_neg = [], []
        self.fc_accumulator, self.fc_accumulator_neg = [], []

        self.df= pd.read_csv('hypergraph/CC200_ROI_labels.csv')
        self.roi_names = []

        self.roi = 200

        self.connectivity_patterns =  []
        self.connectivity_patterns_control =  []

        for idx, row in self.df.iterrows():
            aal = row['AAL']
            ez = row['Eickhoff-Zilles']

            if isinstance(aal, str) and aal.strip() != '':
                name = aal.split()[0].strip(':').replace('[', '').replace('"', '')
                self.roi_names.append(name)
            elif isinstance(ez, str) and ez.strip() != '':
                name = ez.split()[0].strip(':').replace('[', '').replace('"', '')
                self.roi_names.append(name)
            else:
                self.roi_names.append('None')

    def add_results(self, hgl, data):
        for batch in data:
            for k in batch.keys():
                batch[k] = batch[k].to(device)
            im, logits, _, _ = hgl(batch)

            correct_mask = (logits > hgl.best_threshold).long() == batch.y
            correct_indices = torch.nonzero(correct_mask, as_tuple=True)[0].detach().cpu()

            im = im.detach().cpu()

            indices = (batch.y == 1).nonzero(as_tuple=True)[0].detach().cpu()
            indices_not = (batch.y != 1).nonzero(as_tuple=True)[0].detach().cpu()

            indices = indices[torch.isin(indices, correct_indices)]
            indices_not = indices_not[torch.isin(indices_not, correct_indices)]

            knn = to_dense_adj(batch.hyperedge_index, batch.batch, max_num_nodes=200).detach().cpu()
            fc = batch.x.view(-1,  self.roi,  self.roi).detach().cpu()

            plot_full_heatmap(im[indices[1]], 'proposed/example')
            plot_full_heatmap(knn[indices[1]], 'knn/example')
            plot_full_heatmap(fc[indices[1]], 'fc/example')

            if len(indices) > 0:
                im_matrices = [im[i] for i in indices]

                self.incidence_accumulator.extend(im_matrices)
                self.k_accumulator.extend([knn[i] for i in indices])
                self.fc_accumulator.extend([fc[i] for i in indices])

                self.find_patterns(im_matrices, self.connectivity_patterns)

            if len(indices_not) > 0:
                im_matrices = [im[i] for i in indices_not]

                self.incidence_accumulator_neg.extend(im_matrices)
                self.k_accumulator_neg.extend([knn[i] for i in indices_not])
                self.fc_accumulator_neg.extend([fc[i] for i in indices_not])

                self.find_patterns(im_matrices, self.connectivity_patterns_control)

    def plot(self):
        graph_type = ['proposed', 'knn', 'fc']
        positive_sources = [self.incidence_accumulator, self.k_accumulator,
                            self.fc_accumulator]
        negative_sources = [self.incidence_accumulator_neg, self.k_accumulator_neg,
                            self.fc_accumulator_neg]

        for name, pos_src, neg_src in zip(graph_type, positive_sources, negative_sources):
            positive_mean = torch.stack(pos_src, dim=0).mean(dim=0)
            negative_mean = torch.stack(neg_src, dim=0).mean(dim=0)

            plot_full_heatmap(positive_mean, f'{name}/disease')
            plot_full_heatmap(negative_mean, f'{name}/negative')

            difference = positive_mean - negative_mean
            plot_full_heatmap(difference, f'{name}/difference')

            plot_full_heatmaps([positive_mean, negative_mean, difference], f'{name}/joined')
            self.write_csv(difference, name)

            used_hyperedges = self.plot_circos(positive_mean,f'{name}/disease')
            plot_legend(used_hyperedges, f'{name}_disease_legend')

            diff_min = difference.min()
            diff_max = difference.max()
            if diff_max != diff_min:
                diff_norm = (difference - diff_min) / (diff_max - diff_min)
                diff_scaled = diff_norm * 2 - 1

                used_hyperedges = self.plot_circos(negative_mean, f'{name}/negative')
                plot_legend(used_hyperedges, f'{name}_negative_legend')
                used_hyperedges = self.plot_circos(diff_scaled, f'{name}/difference')
                plot_legend(used_hyperedges, f'{name}_difference_legend')

            plot_legend(used_hyperedges, f'{name}_legend')

        self.get_most_common_patterns(self.connectivity_patterns)
        self.get_most_common_patterns(self.connectivity_patterns_control, "control ")

    def plot_circos(self, incidence_matrix, suffix):
        assert incidence_matrix.dim() == 2, "Tensor must be 2D (n_rois, n_hyperedges)"
        n_rois, n_hyperedges = incidence_matrix.shape

        circle = Gcircle(figsize=(15,15))


        if incidence_matrix.shape[1] >= 199:
            incidence_matrix.fill_diagonal_(0)

        all_values = torch.abs(incidence_matrix)
        flattened_values = all_values.view(-1)

        k = int(0.005 * flattened_values.shape[0])
        top_values, top_indices = torch.topk(flattened_values, k)

        rows = top_indices // incidence_matrix.shape[1]
        cols = top_indices % incidence_matrix.shape[1]

        mask = torch.zeros_like(incidence_matrix, dtype=torch.bool)
        mask[rows, cols] = True

        #for i, roi in enumerate(self.roi_names):
        #    circle.add_garc(Garc(arc_id=roi, interspace=0.5, raxis_range=(950,1000), facecolor=colorlist[i%36], label_visible=True))

        selected_indices = torch.unique(torch.cat([rows, cols]))
        for i in selected_indices.tolist():
            if i < 200:
                roi = self.roi_names[i]
                if roi != 'None':
                    circle.add_garc(
                        Garc(
                            arc_id=roi,
                            interspace=0.5,
                            raxis_range=(950, 1000),
                            facecolor=colorlist[i % 36],
                            label_visible=True
                        )
        )

        circle.set_garcs()
        used_hyperedges = {}
        for hyperedge_idx in range(n_hyperedges):
            connected_indices = torch.where(mask[:, hyperedge_idx])[0]

            match = re.match(r'^([^/]+)/', suffix)
            if match.group(1) != 'proposed':
                connected_indices = torch.cat((connected_indices, torch.tensor([hyperedge_idx%incidence_matrix.shape[0]])))

            drawn = False
            color = edgecolors[hyperedge_idx % len(edgecolors)]
            for i in range(len(connected_indices)):
                for j in range(i + 1, len(connected_indices)):
                    roi_start = self.roi_names[connected_indices[i].item()]
                    roi_end = self.roi_names[connected_indices[j].item()]

                    if roi_start != 'None' and roi_end != 'None':
                        source = (roi_start, 0, 180, 950)
                        destination = (roi_end, 0, 180, 950)

                        circle.chord_plot(source,destination, edgecolor=color)
                        drawn = True
            if drawn: used_hyperedges[hyperedge_idx] = color

        circle.figure.savefig(f"images/{suffix}_circos.svg", format="svg", bbox_inches="tight")
        circle.figure.savefig(f"images/{suffix}_circos.png", format="png")
        return used_hyperedges

    def write_csv(self, difference, name):
        expected_columns = ['AAL', 'Eickhoff-Zilles', 'Talairach-Tournoux', 'Harvard-Oxford', ' center of mass']
        data = []
        difference = abs(difference)
        for i in range(difference.shape[0]):
            for j in range(difference.shape[1]):
                val = difference[i, j]
                row_info = self.df.loc[i, expected_columns]
                entry = {
                    'ROI': i,
                    'Hyperedge': j,
                    'Val': val.item() if hasattr(val, 'item') else val,
                    **row_info.to_dict()
                }
                data.append(entry)


        if len(data) > 0:
            result_df = pd.DataFrame(data)
            columns_to_clean = ['AAL', 'Eickhoff-Zilles', 'Talairach-Tournoux', 'Harvard-Oxford']

            for col in columns_to_clean:
                result_df[col] = result_df[col].apply(extract_keys)

            result_df['Val_Normalized'] = result_df.groupby('ROI')['Val'].transform(
                lambda x: x / x.sum() if x.sum() != 0 else 0
            )

            result_df.to_csv(f'images/roi_difference_{name}.csv', index=False)

    def find_patterns(self, matrices, list_data, k=5):
        patterns = defaultdict(list)

        for idx_matrix, mat in enumerate(matrices):
            mat_T = mat.T
            topk_indices = torch.topk(mat_T, k=k, dim=1).indices
            top_sets = [frozenset(row.tolist()) for row in topk_indices]
            for idx_col, s in enumerate(top_sets):
                patterns[s].append((idx_matrix, idx_col))

        list_data.extend(patterns)

    def get_most_common_patterns(self, patterns, control='', threshold=3):
        element_to_indices = defaultdict(set)

        for i, pattern in enumerate(patterns):
            for element in pattern:
                element_to_indices[element].add(i)

        n = len(patterns)
        used = set()
        groups = []

        for i in range(n):
            if i in used:
                continue
            p1 = patterns[i]
            candidate_indices = set()
            for element in p1:
                candidate_indices.update(element_to_indices[element])
            candidate_indices -= used
            candidate_indices.discard(i)

            current_group = [p1]
            used.add(i)

            for j in candidate_indices:
                if j in used:
                    continue
                p2 = patterns[j]
                if len(p1 & p2) >= threshold:
                    current_group.append(p2)
                    used.add(j)

            groups.append(current_group)

        if not groups:
            print("No pattern groups found.")
            return []

        groups.sort(key=len, reverse=True)
        most_common_groups = groups[0:3]

        for most_common_group in most_common_groups:
            frequency = len(most_common_group)
            print(f"Most common pattern {control}group (≥{threshold}/5 overlap): {frequency} occurrences")

            for pattern in most_common_group:
                print([self.roi_names[element] for element in pattern])
            print()



def extract_keys(text):
    keys = re.findall(r'\"([^\"]+)\":', text)
    return ", ".join(keys)

def plot_legend(hyperedges, suffix, cols=4):
    sorted_hyperedges = sorted(hyperedges.items())  # Lista de tuplas (idx, color)

    handles = [Line2D([0], [1], color=color, lw=3) for _, color in sorted_hyperedges]
    labels = [f'Hyperedge {idx + 1}' for idx, _ in enumerate(sorted_hyperedges)]

    fig, ax = plt.subplots()
    ax.axis('off')

    ax.legend(
        handles,
        labels,
        loc='center',
        ncol=cols,
        frameon=False,
        handlelength=2
    )

    plt.savefig(f"images/{suffix}.svg", format="svg", bbox_inches='tight')
    plt.close()


