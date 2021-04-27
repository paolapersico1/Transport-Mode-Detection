import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from os import path
import preprocessing
from pytorch.model_runner import train_loop, test_loop
import torch
from pytorch.model import Feedforward
from pytorch.dataset import TMDDataset
from torch.utils.data import DataLoader, Subset
import itertools
import time


def run(X, y):
    hidden_sizes = [64, 50, 32, 16]
    nums_epochs = [500, 400, 250, 100]
    batch_sizes = [32, 64, 128, 256]
    gamma = [0.01, 0.03, 0.05, 0.08]
    learning_rate = 0.1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: {}'.format(device))
    fs = X.shape[1]

    prefix = path.join('pytorch', 'saved_models')
    model_file = path.join(prefix, 'NN_{}.torch'.format(fs))
    result_file = path.join(prefix, 'csvs', 'NN_{}.csv'.format(fs))

    hyperparams = itertools.product(hidden_sizes, nums_epochs, batch_sizes, gamma)

    train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=42)
    X[train_idx], X[test_idx] = preprocessing.remove_nan(X[train_idx], X[test_idx])
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, stratify=y[train_idx], random_state=42)

    scaler = MinMaxScaler()
    scaler.fit(X[train_idx])
    X[train_idx] = scaler.transform(X[train_idx])
    X[val_idx] = scaler.transform(X[val_idx])
    X[test_idx] = scaler.transform(X[test_idx])
    dataset = TMDDataset(X, y)

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    test_subset = Subset(dataset, test_idx)

    best_val_score = 0

    if not path.exists(model_file):
        for hidden_size, num_epochs, batch_size, gamma in hyperparams:
            print('---------------------------------------------------------------')
            print('Dataset size: {}, hidden_size: {}, num_epochs: {}, batch_size: {}, gamma: {}'.format(fs,
                                                                                                        hidden_size,
                                                                                                        num_epochs,
                                                                                                        batch_size,
                                                                                                        gamma))
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False)
            val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)
            test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)

            model = Feedforward(dataset.X.shape[1], hidden_size, dataset.num_classes)
            model.to(device)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
            lambda1 = lambda epoch: 1 / (1 + gamma * epoch)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

            time_before = time.time()
            train_loop(train_loader, model, criterion, optimizer, scheduler, num_epochs, device)
            time_after = time.time() - time_before

            train_score = test_loop(DataLoader(train_subset, batch_size=1, shuffle=False), model, device)
            val_score = test_loop(val_loader, model, device)
            test_score = test_loop(test_loader, model, device)

            print('Dataset size: {}, hidden_size: {}, num_epochs: {}, batch_size: {}, gamma: {}'.format(
                    fs, hidden_size, num_epochs, batch_size, gamma))
            print("Train accuracy: {}".format(train_score))
            print("Validation accuracy: {}".format(val_score))

            if val_score > best_val_score:
                best_model = {"NN_" + str(fs): {"pipeline": "mlp",
                                                "hidden_size": hidden_size,
                                                "epochs": num_epochs,
                                                "batch_size": batch_size,
                                                "decay": gamma,
                                                "mean_train_score": train_score,
                                                "mean_test_score": val_score,
                                                "mean_fit_time": time_after,
                                                "final_test_score": test_score}}
                best_val_score = val_score
                best_nn = model

        pd.DataFrame(best_model).transpose().to_csv(result_file)
        torch.save(best_nn.state_dict(), model_file)

    else:
        result = pd.read_csv(result_file, index_col=0)
        model = Feedforward(dataset.X.shape[1], result['hidden_size'][0], dataset.num_classes)
        model.load_state_dict(torch.load(model_file), strict=False)
        model.to(device)
        best_model = result.transpose().to_dict()

    return best_model
