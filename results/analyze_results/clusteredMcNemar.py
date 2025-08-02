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
