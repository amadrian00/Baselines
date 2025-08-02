import numpy as np
from utils import compute_stats
from collections import defaultdict
import os

models = ['hyper', 'hgnn', 'hgnnplus']
names = ['proposed', 'btf', 'knn', 'k-random', 'gat', 'gcn', 'gsage', 'braingnn', 'thfcn', 'brainnetcnn', 'ts-modelling', 'fc-modelling']

orden_deseado = [
    'brainnetcnn', 'btf', 'gcn', 'graphsage', 'gat', 'braingnn',
    'k-random', 'knn', 'fc-modelling', 'ts-modelling', 'thfcn', 'proposed'
]

# Diccionario para almacenar los resultados por modelo base
results_table = defaultdict(list)

for model in models:
    for name in names:
        # Construir la ruta del archivo correspondiente
        if name == 'proposed':
            path = f"paper/5r 5f_no feats/results/fold_results_proposed_{model}.npy"
        else:
            path = f"paper/5r 5f_feats/results/fold_results_{name}_{model}.npy"

        if not os.path.exists(path):
            print(f"[AVISO] Fichero no encontrado: {path}")
            continue

        res = np.load(path)  # Dimensión esperada: (R, 6)

        try:
            acc = res[:,:, 1]
            auc = res[:,:, 2]
            f1 = res[:, :,3]
        except:
            acc = res[:, 1]
            auc = res[:, 2]
            f1 = res[:,3]

        acc_mean, acc_std = compute_stats(acc)
        auc_mean, auc_std = compute_stats(auc)
        f1_mean, f1_std = compute_stats(f1)

        results_table[model].append({
            'base': name,
            'acc_mean': acc_mean, 'acc_std': acc_std,
            'auc_mean': auc_mean, 'auc_std': auc_std,
            'f1_mean': f1_mean, 'f1_std': f1_std
        })

# Mostrar resultados en consola de forma estructurada
for model in models:
    print(f"\n=== {model.upper()} ===")
    print(f"{'Base Model':<15} {'AUROC (±)':<20} {'Accuracy (±)':<20} {'F1 Score (±)':<20}")
    print("-" * 75)

    # Convertir a dict por nombre para acceso rápido
    model_results = {entry['base']: entry for entry in results_table[model]}

    for name in orden_deseado:
        if name not in model_results:
            continue
        entry = model_results[name]
        print(f"{entry['base']:<15} "
              f"{entry['auc_mean']:.4f} ± {entry['auc_std']:.4f}   "
              f"{entry['acc_mean']:.4f} ± {entry['acc_std']:.4f}   "
              f"{entry['f1_mean']:.4f} ± {entry['f1_std']:.4f}")
