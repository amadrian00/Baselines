import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
import torch
import pandas as pd

models = ['hyper', 'hgnn', 'hgnnplus']
names = ['proposed','btf','knn', 'k-random', 'gat', 'gcn', 'gsage', 'thfcn', 'braingnn', 'brainnetcnn', 'ts-modelling', 'fc-modelling']
names_short = ['proposed','knn', 'k-random', 'thfcn', 'ts-modelling', 'fc-modelling']

results_data = []

for model in models:
    if model == 'hyper':
        names_model = names
    else:
        names_model = names_short
    for model_name in names_model:

        if model_name == 'proposed':
            continue

        b01_total = 0  # proposed acierta, otro falla
        b10_total = 0  # otro acierta, proposed falla

        for fold in range(25):
            try:
                y_proposed = np.load(f"paper/5r 5f_no feats/preds/fc_proposed_{model}_30_{fold}.npy")
                y_other = np.load(f"paper/5r 5f_feats/preds/fc_{model_name}_{model}_{fold}.npy")
            except FileNotFoundError:
                print(f"Fichero no encontrado para {model_name} vs proposed en fold {fold}: paper/5r 5f_feats/preds/fc_{model_name}_{model}_{fold}.npy")
                continue

            if y_proposed.shape != y_other.shape:
                print(f"Shape mismatch entre proposed y {model_name} en fold {fold}")
                continue

            y_proposed = (torch.sigmoid(torch.tensor(y_proposed)) > 0.5).numpy()
            y_other = (torch.sigmoid(torch.tensor(y_other)) > 0.5).numpy()
            b01 = np.sum((y_proposed == 1) & (y_other == 0))
            b10 = np.sum((y_proposed == 0) & (y_other == 1))

            b01_total += b01
            b10_total += b10

        table = [[0, b01_total],
                 [b10_total, 0]]

        print(f"\n==== Modelo '{model_name}' vs 'proposed' en familia '{model}' ====")
        print("Tabla de contingencia 2x2:")
        print(np.array(table))

        if b01_total + b10_total == 0:
            print("No hay discordancias: McNemar no se puede aplicar.")
            stat = 0
            pval = 1.0
            decision = "Test no aplicable"
        else:
            # Decide si usar test exacto o no
            exact = (b01_total + b10_total) < 25
            result = mcnemar(table, exact=exact)
            stat = result.statistic
            pval = result.pvalue
            alpha = 0.05
            if pval > alpha:
                decision = 'No Significative'
            else:
                decision = 'Significative'

            print(f"Estadístico de McNemar: {stat:.4f}")
            print(f"P-valor: {pval:.4f}")
            print(f"Decisión: {decision}")

        # Guardar resultados
        results_data.append({
            'Familia': model,
            'Comparado_con': model_name,
            'B01 (proposed=1, otro=0)': b01_total,
            'B10 (proposed=0, otro=1)': b10_total,
            'Estadístico': stat,
            'P-valor': pval,
            'Decisión': decision,
            'Exacto': exact if (b01_total + b10_total) != 0 else None
        })

# Guardar en CSV
df_results = pd.DataFrame(results_data)
df_results.to_csv("mcnemar_results.csv", index=False)
print("\n✅ Resultados guardados en 'mcnemar_results.csv'")

import numpy as np
from utils import compute_stats
from collections import defaultdict

models = ['hyper', 'hgnn', 'hgnnplus']
names = ['proposed','btf','knn', 'k-random', 'gat', 'gcn', 'gsage', 'thfcn', 'brainnetcnn', 'ts-modelling', 'fc-modelling']
folds = range(25)

# Diccionario para agrupar los resultados
results_table = defaultdict(list)

for model in models:
    for name in names:
        all_metrics = []

        for fold in folds:
            # Cargar archivo correcto según modelo y nombre base
            if name == 'proposed':
                path = f"paper/5r 5f_no feats/results/fold_results_proposed_{model}_30_{fold}.npy"
            else:
                path = f"paper/5r 5f_feats/results/fold_results_{name}_{model}_{fold}.npy"

            try:
                res = np.load(path)
                all_metrics.append(res)  # (R, 6) array
            except FileNotFoundError:
                print(f"[AVISO] Fichero no encontrado: {path}")
                continue

        if not all_metrics:
            continue

        all_metrics = np.stack(all_metrics)  # (folds, R, 6)
        all_metrics = all_metrics.reshape(-1, 6)

        acc = all_metrics[:, 1]
        auc = all_metrics[:, 2]
        f1 = all_metrics[:, 3]

        acc_mean, acc_std = compute_stats(acc)
        auc_mean, auc_std = compute_stats(auc)
        f1_mean, f1_std = compute_stats(f1)

        results_table[model].append({
            'base': name,
            'acc_mean': acc_mean, 'acc_std': acc_std,
            'auc_mean': auc_mean, 'auc_std': auc_std,
            'f1_mean': f1_mean, 'f1_std': f1_std
        })

# Imprimir resultados tabulados
for model in models:
    print(f"\n=== {model.upper()} ===")
    print(f"{'Base Model':<15} {'AUROC (±)':<20} {'Accuracy (±)':<20} {'F1 Score (±)':<20}")
    print("-" * 75)
    for entry in sorted(results_table[model], key=lambda x: x['base']):
        print(f"{entry['base']:<15} "
              f"{entry['auc_mean']:.3f} ± {entry['auc_std']:.3f}   "
              f"{entry['acc_mean']:.3f} ± {entry['acc_std']:.3f}   "
              f"{entry['f1_mean']:.3f} ± {entry['f1_std']:.3f}")

