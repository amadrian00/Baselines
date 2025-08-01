import numpy as np
import os
import glob
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import re

carpeta_archivos = "paper/braingnn/"
nombre_base = "braingnn_hyper"

def extraer_numero(filename):
    match = re.search(r'Fold_(\d+)', filename)
    return int(match.group(1)) if match else -1

archivos = glob.glob(os.path.join(carpeta_archivos, "*.csv"))
archivos = sorted(archivos, key=extraer_numero)

assert len(archivos) == 25, "Deben ser exactamente 25 archivos .csv"

# Lista para guardar resultados de cada archivo
resultados = []

for file in archivos:
    data = np.genfromtxt(file, delimiter=',', skip_header=1)
    numero = extraer_numero(file)
    np.save(f"paper/5r 5f_feats/preds/fc_{nombre_base}_{numero}.npy", data)

    preds = data[:, 1]
    labels = data[:, 2]

    acc = accuracy_score(labels, preds)
    try:
        auroc = roc_auc_score(labels, preds)
    except ValueError:
        auroc = np.nan
    f1 = f1_score(labels, preds)

    resultados.append([numero, acc, auroc, f1])

# Ordenar resultados por fold por si acaso
resultados = sorted(resultados, key=lambda x: x[0])

# Guardar array 25x4
np.save(f"paper/5r 5f_feats/results/fold_results_{nombre_base}.npy", np.array(resultados))
print(np.array(resultados).shape)
