import numpy as np
from utils import compute_stats

model = 'gat'  # ['knn', 'ts-modelling', 'fc-modelling', 'k-random', 'gat', 'thfcn', 'gcn', 'gsage', 'proposed']
all_results_np = np.load(f'results/fold_results_{model}_hyper.npy')

test_acc = all_results_np[:, :, 1].flatten()
test_auc = all_results_np[:, :, 2].flatten()
test_f1 = all_results_np[:, :, 3].flatten()
test_sensitivity = all_results_np[:, :, 4].flatten()
test_specificity = all_results_np[:, :, 5].flatten()

test_acc_mean, test_acc_std = compute_stats(test_acc)
test_auc_mean, test_auc_std = compute_stats(test_auc)
test_f1_mean, test_f1_std = compute_stats(test_f1)
test_sensitivity_mean, test_sensitivity_std = compute_stats(test_sensitivity)
test_specificity_mean, test_specificity_std = compute_stats(test_specificity)

print(f"{'Metric':<20} {'Mean':>10} {'Std':>10}")
print(f"{'-' * 48}")
print(
    f"{'Test Accuracy':<20} {test_acc_mean:>10.4f} {test_acc_std:>10.4f}")
print(
    f"{'Test AUC':<20} {test_auc_mean:>10.4f} {test_auc_std:>10.4f}")
print(
    f"{'Test F1':<20} {test_f1_mean:>10.4f} {test_f1_std:>10.4f}")
print(
    f"{'Test Sensitivity':<20} {test_sensitivity_mean:>10.4f} {test_sensitivity_std:>10.4f}")
print(
    f"{'Test Specificity':<20} {test_specificity_mean:>10.4f} {test_specificity_std:>10.4f}")
