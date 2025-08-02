import numpy as np
import torch
from learn import FCHypergraphLearning
from hypergraph.fc_hypergraph_learning import CorrelationToIncidenceTransformer as Pretrain
from utils import get_folds, compute_stats, prepare_dataloader
from hypergraph.hypergraph_generator.hypergraph_generator import SecondFCHypergraph
from hypergraph.hypergraph_generator.hypergraph_generator_dhg import SecondFCHypergraph as SecondFCHypergraphDHG
from hypergraph.visualization import HypergraphPlotter


def learn_framework(dataloaders, shape, train_labels, graph_name, model_name, fold, vis, use_features):
    device_model = 'cuda' if model_name == 'hyper' and device  == 'cuda' else 'cpu'
    if graph_name == 'proposed':
        pretrain = Pretrain(in_size=shape[-1], hidden_size=hidden_size, dropout=dropout,
                               device=device_model, y=train_labels, num_layers=num_layers,
                            num_hyperedges=num_hyperedges, num_heads=num_heads, fold=fold).to(device_model)


        metrics = pretrain.learn(dataloaders, lr=lr, wd=wd, epochs=epochs)

        if vis is not None:
            vis.add_results(pretrain, dataloaders['test'])

        pretrain.finished_training = True
        pretrain(next(iter(dataloaders['test'])).to(device_model))

        return metrics
        new_dataloaders = {}
        for mode in dataloaders:
            if model_name == 'hyper': dataset = SecondFCHypergraph(dataloaders[mode], pretrain, device_model)
            else :dataset = SecondFCHypergraphDHG(dataloaders[mode], pretrain, device_model)
            if mode == 'train':
                new_dataloaders[mode] = prepare_dataloader(dataset, batch_size=batch_size, shuffle=True)
            else:
                new_dataloaders[mode] = prepare_dataloader(dataset, batch_size=len(dataset))

        pretrain(next(iter(new_dataloaders['test'])).to(device_model))
    else:
        new_dataloaders = dataloaders
    hgl = FCHypergraphLearning(in_size=shape[-1], hidden_size=hidden_size, dropout=dropout,
                               device=device_model, y=train_labels, name=graph_name,
                               model_name=model_name, num_hyperedges=num_hyperedges, fold=fold, use_feats=use_features).to(device_model)

    if graph_name == 'btf':
        lr1 = 1e-4
    else:
        lr1 = 5e-4

    metrics = hgl.learn(new_dataloaders, lr=lr1, wd=1e-5, epochs=100)

    return metrics

def k_folds():
    rep_folds, shape = get_folds(n_folds, n_repeats, batch_size, device)

    fold_batches = [rep_folds[i:i + n_folds] for i in range(n_repeats)]

    list_names = ['btf','knn', 'k-random', 'gat', 'gcn', 'gsage', 'thfcn', 'ts-modelling', 'fc-modelling', 'proposed']
    list_names_short = ['knn', 'k-random', 'ts-modelling', 'fc-modelling', 'thfcn', 'proposed']

    vis = HypergraphPlotter()

    for model in ['hyper', 'hgnn', 'hgnnplus']:
        if model== 'hyper':
             names = list_names
        else:
            names = list_names_short
        for name in names:
            all_results = []
            total_folds = 0
            for repeat, folds in enumerate(fold_batches):
                print(f"Starting repetition {repeat + 1}/{n_repeats}...")

                fold_results = []

                for fold, (train_loader, test_loader, val_loader, train_labels) in enumerate(folds):
                    print(f"    Repeat {repeat + 1} - Fold {fold + 1}/{len(folds)} - Training and Validation")

                    dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
                    metrics = learn_framework(dataloaders, shape, train_labels, name, model, total_folds, vis, use_features)
                    total_folds+=1

                    # Log fold results
                    fold_results.append([
                        metrics.get("test_Loss", None),
                        metrics.get("test_Accuracy", None),
                        metrics.get("test_AUC", None),
                        metrics.get("test_F1", None),
                        metrics.get("test_Sensitivity", None),
                        metrics.get("test_Specificity", None),

                        metrics.get("train_Loss", None),
                        metrics.get("train_F1", None),
                        metrics.get("train_Sensitivity", None),
                        metrics.get("train_AUC", None),
                        metrics.get("train_Specificity", None),
                        metrics.get("train_Accuracy", None)
                    ])

                    print(f"        Repeat {repeat + 1} - Fold {fold + 1}/{len(folds)} results:\n"
                          f"            Train Accuracy:      {metrics['train_Accuracy']:.4f}   |  Test Accuracy:      {metrics['test_Accuracy']:.4f}\n"
                          f"            Train AUC:           {metrics['train_AUC']:.4f}   |  Test AUC:           {metrics['test_AUC']:.4f}\n"
                          f"            Train F1:            {metrics['train_F1']:.4f}   |  Test F1:            {metrics['test_F1']:.4f}\n"
                          f"            Train Sensitivity:   {metrics['train_Sensitivity']:.4f}   |  Test Sensitivity:   {metrics['test_Sensitivity']:.4f}\n"
                          f"            Train Specificity:   {metrics['train_Specificity']:.4f}   |  Test Specificity:   {metrics['test_Specificity']:.4f}")

                all_results.append(fold_results)

                test_acc_mean = np.mean(np.array(all_results)[:, :, 1].flatten())
                test_auc_mean = np.mean(np.array(all_results)[:, :, 2].flatten())
                test_f1_mean = np.mean(np.array(all_results)[:, :, 3].flatten())
                test_sen_mean = np.mean(np.array(all_results)[:, :, 4].flatten())
                test_spe_mean = np.mean(np.array(all_results)[:, :, 5].flatten())

                print(f"Repeat {repeat + 1}/{n_repeats} - "
                      f"Test Acc Mean: {test_acc_mean:.4f}, "
                      f"Test AUC Mean: {test_auc_mean:.4f}, "
                      f"Test F1 Mean: {test_f1_mean:.4f}, "
                      f"Test Sensitivity Mean: {test_sen_mean:.4f}, "
                      f"Test Specificity Mean: {test_spe_mean:.4f}")
            if name == 'proposed': vis.plot()
            all_results_np = np.array(all_results)

            np.save(f'results_performance/fold_results_{name}_{model}_{'feats' if use_features else 'nofeats'}'
                    , all_results_np)

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


if __name__ == '__main__':
    n_folds = 5
    n_repeats = 5

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size = 64

    use_features = True  # Set to False if you want to use the hypergraph without features

    dropout = 0.3
    lr = 2.5e-6
    wd = 1e-5
    hidden_size = 64

    num_layers = 1
    num_heads = 2

    epochs = 100

    num_hyperedges = 30

    k_folds()