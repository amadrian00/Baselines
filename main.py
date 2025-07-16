import numpy as np
import torch
from learn import FCHypergraphLearning
from utils import get_folds, compute_stats

def learn_framework(dataloaders, shape, train_labels, graph_name):
    hgl = FCHypergraphLearning(in_size=shape[-1], hidden_size= hidden_size, dropout=dropout,
                               device=device, y=train_labels, name=graph_name).to(device)

    return hgl.learn(dataloaders, lr=lr, wd=wd, epochs=epochs)

def k_folds():
    all_results = []

    rep_folds, shape = get_folds(n_folds, n_repeats, batch_size, device)

    fold_batches = [rep_folds[i:i + n_folds] for i in range(n_repeats)]

    for repeat, folds in enumerate(fold_batches):
        print(f"Starting repetition {repeat + 1}/{n_repeats}...")

        fold_results = []

        for fold, (train_loader, val_loader, train_labels) in enumerate(folds):
            print(f"    Repeat {repeat + 1} - Fold {fold + 1}/{len(folds)} - Training and Validation")

            dataloaders = {'train': train_loader, 'val': val_loader, 'test': val_loader}
            metrics = learn_framework(dataloaders, shape, train_labels, name)

            # Log fold results
            fold_results.append([
                metrics.get("val_Loss", None),
                metrics.get("val_Accuracy", None),
                metrics.get("val_AUC", None),
                metrics.get("val_F1", None),
                metrics.get("val_Sensitivity", None),
                metrics.get("val_Specificity", None),

                metrics.get("train_Loss", None),
                metrics.get("train_F1", None),
                metrics.get("train_Sensitivity", None),
                metrics.get("train_AUC", None),
                metrics.get("train_Specificity", None),
                metrics.get("train_Accuracy", None)
            ])

            print(f"        Repeat {repeat + 1} - Fold {fold + 1}/{len(folds)} results:\n"
                  f"            Train Accuracy:      {metrics['train_Accuracy']:.4f}   |  Val Accuracy:      {metrics['val_Accuracy']:.4f}\n"
                  f"            Train AUC:           {metrics['train_AUC']:.4f}   |  Val AUC:           {metrics['val_AUC']:.4f}\n"
                  f"            Train F1:            {metrics['train_F1']:.4f}   |  Val F1:            {metrics['val_F1']:.4f}\n"
                  f"            Train Sensitivity:   {metrics['train_Sensitivity']:.4f}   |  Val Sensitivity:   {metrics['val_Sensitivity']:.4f}\n"
                  f"            Train Specificity:   {metrics['train_Specificity']:.4f}   |  Val Specificity:   {metrics['val_Specificity']:.4f}")

        all_results.append(fold_results)

        val_acc_mean = np.mean(np.array(all_results)[:, :, 1].flatten())
        val_auc_mean = np.mean(np.array(all_results)[:, :, 2].flatten())
        val_f1_mean = np.mean(np.array(all_results)[:, :, 3].flatten())
        val_sen_mean = np.mean(np.array(all_results)[:, :, 4].flatten())
        val_spe_mean = np.mean(np.array(all_results)[:, :, 5].flatten())

        print(f"Repeat {repeat + 1}/{n_repeats} - "
              f"Val Acc Mean: {val_acc_mean:.4f}, "
              f"Val AUC Mean: {val_auc_mean:.4f}, "
              f"Val F1 Mean: {val_f1_mean:.4f}, "
              f"Val Sensitivity Mean: {val_sen_mean:.4f}, "
              f"Val Specificity Mean: {val_spe_mean:.4f}")

    #vis.plot()

    all_results_np = np.array(all_results)

    np.save(f'fold_results_{name}', all_results_np)

    val_acc = all_results_np[:, :, 1].flatten()
    val_auc = all_results_np[:, :, 2].flatten()
    val_f1 = all_results_np[:, :, 3].flatten()
    val_sensitivity = all_results_np[:, :, 4].flatten()
    val_specificity = all_results_np[:, :, 5].flatten()

    val_acc_mean, val_acc_std = compute_stats(val_acc)
    val_auc_mean, val_auc_std = compute_stats(val_auc)
    val_f1_mean, val_f1_std = compute_stats(val_f1)
    val_sensitivity_mean, val_sensitivity_std = compute_stats(val_sensitivity)
    val_specificity_mean, val_specificity_std = compute_stats(val_specificity)

    print(f"{'Metric':<20} {'Mean':>10} {'Std':>10}")
    print(f"{'-' * 48}")
    print(
        f"{'Val Accuracy':<20} {val_acc_mean:>10.4f} {val_acc_std:>10.4f}")
    print(
        f"{'Val AUC':<20} {val_auc_mean:>10.4f} {val_auc_std:>10.4f}")
    print(
        f"{'Val F1':<20} {val_f1_mean:>10.4f} {val_f1_std:>10.4f}")
    print(
        f"{'Val Sensitivity':<20} {val_sensitivity_mean:>10.4f} {val_sensitivity_std:>10.4f}")
    print(
        f"{'Val Specificity':<20} {val_specificity_mean:>10.4f} {val_specificity_std:>10.4f}")


if __name__ == '__main__':
    name = 'knn'
    n_folds = 5
    n_repeats = 1

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size = 64

    dropout = 0.3
    lr = 1e-3
    wd = 1e-4
    hidden_size = 64

    epochs = 100

    k_folds()