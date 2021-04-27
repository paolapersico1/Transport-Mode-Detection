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
    hidden_sizes = [16, 32, 64]
    nums_epochs = [100, 250, 350]
    batch_sizes = [8, 16, 32]
    gamma = [0.1, 0.5]
    learning_rate = 0.01

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: {}'.format(device))
    fs = X.shape[1]
    model_name = "mlp_" + str(fs)

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

    for hidden_size, num_epochs, batch_size, gamma in hyperparams:
        print('---------------------------------------------------------------')
        print('hidden_size: {}, num_epochs: {}, batch_size: {}, gamma: {}'.format(
            hidden_size, num_epochs, batch_size, gamma))
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)
        test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)

        model = Feedforward(dataset.X.shape[1], hidden_size, dataset.num_classes)
        model.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)

        name = path.join('pytorch', 'csvs', '{}.{}.{}.{}.{}.csv'.format(fs, hidden_size, num_epochs, batch_size, gamma))
        if path.exists(name):
            print('Results for dataset size: {}, hidden_size: {}, num_epochs: {}, batch_size: {}, gamma: {}'.format(fs,
                hidden_size, num_epochs, batch_size, gamma))

            print(pd.read_csv(name))
            ## TODO laod model and test on test
            # train_loop(train_loader, model, criterion, optimizer, num_epochs, device)
            # test_loop(test_loader, model, device)
        else:
            time_before = time.time()
            train_loop(train_loader, model, criterion, optimizer, scheduler, num_epochs, device)
            time_after = time.time() - time_before
            print('Results for dataset size: {}, hidden_size: {}, num_epochs: {}, batch_size: {}, gamma: {}'.format(fs,
                hidden_size, num_epochs, batch_size, gamma))
            
        train_score = test_loop(DataLoader(train_subset, batch_size=1, shuffle=False), model, device, name)
        val_score = test_loop(val_loader, model, device, name)
        test_score = test_loop(test_loader, model, device, name)

        # model, loss_values = train_model(
        #     model, criterion, optimizer, num_epochs, train_loader)
        # # plt.plot([x.detach().numpy() for x in loss_values])
        # # plt.title('Depth: {}, Epochs: {}, Batch: {}'.format(
        # #     hidden_size, num_epochs, batch_size))
        # # plt.show()
        # test_model(model, val_loader)

        if val_score > best_val_score:
            best_model = {model_name : {"pipeline": "mlp",
                                        "hidden_size": hidden_size,
                                        "epochs": num_epochs,
                                        "batch_size": batch_size,
                                        "decay": gamma,
                                        "mean_train_score" : train_score,
                                        "mean_test_score" : val_score,
                                        "mean_fit_time" : time_after,
                                        "final_test_score" : test_score}}
            best_val_score = val_score

    return pd.DataFrame(best_model)
