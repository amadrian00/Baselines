# ===============================================
# Comparación estadística directa: PROPOSED vs BTF en familia 'hyper'
# ===============================================
import numpy as np
from scipy.stats import shapiro, ttest_rel, wilcoxon
import pandas as pd

# Cargar los resultados desde los archivos .npy
results_btf = np.load('paper/5r 5f_no feats/results/fold_results_btf_hyper.npy', allow_pickle=True)
results_proposed = np.load('paper/5r 5f_no feats/results/fold_results_proposed_hyper_30.npy', allow_pickle=True)

# Preparar los datos (5 repeticiones x 5 folds)
btf_all_folds = results_btf.reshape(-1, 12)
proposed_all_folds = results_proposed.reshape(-1, 12)

# Extraer métricas de test (columnas 0 a 5)
btf_test = btf_all_folds[:, 0:6]
proposed_test = proposed_all_folds[:, 0:6]

# Nombres de las métricas que se compararán (omitimos test_loss)
metric_names = ["Accuracy", "AUC", "F1", "Sensitivity", "Specificity"]

results = []

# Comparación métrica por métrica
for i, name in enumerate(metric_names, start=1):  # start=1 para omitir test_loss
    btf_values = btf_test[:, i]
    proposed_values = proposed_test[:, i]
    differences = proposed_values - btf_values

    # Normalidad
    stat_normality, p_normality = shapiro(differences)
    normal = p_normality > 0.05

    # Test estadístico
    if normal:
        stat_test, p_test = ttest_rel(proposed_values, btf_values)
        test_name = "t-test (paired)"
    else:
        stat_test, p_test = wilcoxon(proposed_values, btf_values)
        test_name = "Wilcoxon"

    results.append({
        "Metric": name,
        "Test": test_name,
        "p-value": p_test,
        "Normality p": p_normality,
        "Proposed > BTF Mean Diff": differences.mean()
    })

# Mostrar resultados
df_results = pd.DataFrame(results)
print("\n📊 Comparación estadística directa entre PROPOSED y BTF en la familia 'hyper':\n")
print(df_results.to_string(index=False))
