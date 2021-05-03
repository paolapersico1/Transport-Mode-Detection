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


# starting point for neural network training and testing
def run(X, y, nn_models_dir, use_saved_if_available, save_models):
    force_train = False

    # hyperparameters
    hidden_sizes = [64, 50, 32, 16]
    nums_epochs = [500, 400, 250, 100]
    batch_sizes = [32, 64, 128, 256]
    gamma = [0.01, 0.03, 0.05, 0.08]
    learning_rate = 0.1

    # exploit gpu if possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print('Device: {}'.format(device))
    fs = X.shape[1]

    model_file = path.join(nn_models_dir, 'NN_{}.torch'.format(fs))
    result_file = path.join(nn_models_dir, 'csvs', 'NN_{}.csv'.format(fs))

    # Cartesian product of the sets of hyperparameters
    hyperparams = itertools.product(hidden_sizes, nums_epochs, batch_sizes, gamma)

    # split the indices (72% train, 8% val, 20% test)
    train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=42)
    # remove missing values
    X[train_idx], X[test_idx] = preprocessing.remove_nan(X[train_idx], X[test_idx])
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, stratify=y[train_idx], random_state=42)

    # Min-Max Scaling
    scaler = MinMaxScaler()
    scaler.fit(X[train_idx])
    X[train_idx] = scaler.transform(X[train_idx])
    X[val_idx] = scaler.transform(X[val_idx])
    X[test_idx] = scaler.transform(X[test_idx])

    # save the dataset as a set of tensors
    dataset = TMDDataset(X, y)

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    test_subset = Subset(dataset, test_idx)

    best_val_score = 0
    losses = None
    if not use_saved_if_available or not path.exists(model_file):
        # validation of hyperparameters
        for hidden_size, num_epochs, batch_size, gamma in hyperparams:
            print('---------------------------------------------------------------')
            print('Dataset size: {}, hidden_size: {}, num_epochs: {}, batch_size: {}, gamma: {}'.format(fs,
                                                                                                        hidden_size,
                                                                                                        num_epochs,
                                                                                                        batch_size,
                                                                                                        gamma))

            # create dataloaders of each set to iterate more easily
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False)
            val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)
            test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)

            # create the neural network
            model = Feedforward(dataset.X.shape[1], hidden_size, dataset.num_classes)
            model.to(device)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # stochastic gradient descent
            # scheduler fo learning rate decay
            lambda1 = lambda epoch: 1 / (1 + gamma * epoch)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

            # train the neural network
            time_before = time.time()
            losses = train_loop(train_loader, model, criterion, optimizer, scheduler, num_epochs, device)
            time_after_train = time.time() - time_before

            # evaluate the neural network on validation and test set
            train_score = test_loop(DataLoader(train_subset, batch_size=1, shuffle=False), model, device)
            time_before = time.time()
            val_score = test_loop(val_loader, model, device)
            time_after_val = time.time() - time_before
            test_score = test_loop(test_loader, model, device)

            print('Dataset size: {}, hidden_size: {}, num_epochs: {}, batch_size: {}, gamma: {}'.format(
                fs, hidden_size, num_epochs, batch_size, gamma))
            print("Train accuracy: {}".format(train_score))
            print("Validation accuracy: {}".format(val_score))

            # save the neural network if it has the best validation score until now
            if val_score > best_val_score:
                best_model = {"NN_" + str(fs): {"pipeline": "mlp",
                                                "hidden_size": hidden_size,
                                                "epochs": num_epochs,
                                                "batch_size": batch_size,
                                                "decay": gamma,
                                                "mean_train_score": train_score,
                                                "mean_test_score": val_score,
                                                "mean_fit_time": time_after_train,
                                                "mean_score_time": time_after_val,
                                                "final_test_score": test_score}}
                best_val_score = val_score
                best_nn = model

        # save on disk the best neural network
        if save_models:
            pd.DataFrame(best_model).transpose().to_csv(result_file)
            torch.save(best_nn.state_dict(), model_file)
    else:
        # load the best neural network
        print("Saved model found: NN_{}".format(fs))
        result = pd.read_csv(result_file, index_col=0)
        model = Feedforward(dataset.X.shape[1], result['hidden_size'][0], dataset.num_classes)
        if force_train:
            model.to(device)
            train_loader = DataLoader(train_subset, batch_size=int(result['batch_size'][0]), shuffle=False)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
            lambda1 = lambda epoch: 1 / (1 + result['decay'][0] * epoch)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

            losses = train_loop(train_loader, model, criterion, optimizer, scheduler, result['epochs'][0], device)
        else:
            model.load_state_dict(torch.load(model_file), strict=False)
            model.to(device)

        best_model = result.transpose().to_dict()
    return best_model, losses
