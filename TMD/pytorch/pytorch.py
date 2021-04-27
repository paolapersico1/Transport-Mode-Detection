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


def run(X, y):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: {}'.format(device))

    prefix = path.join('pytorch', 'saved_models')
    model_file = path.join(prefix, 'NN_{}.torch'.format(X.shape[1]))
    result_file = path.join(prefix, 'csvs', 'NN_{}.csv'.format(X.shape[1]))

    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.2, stratify=y, random_state=42)
    X[train_idx], X[test_idx] = preprocessing.remove_nan(X[train_idx], X[test_idx])
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, stratify=y[train_idx], random_state=42)

    scaler = MinMaxScaler()
    scaler.fit(X[train_idx])
    X[train_idx] = scaler.transform(X[train_idx])
    X[val_idx] = scaler.transform(X[val_idx])
    X[test_idx] = scaler.transform(X[test_idx])
    dataset = TMDDataset(X, y)

    if path.exists(model_file):
        result = pd.read_csv(result_file, index_col=0)
        model = Feedforward(dataset.X.shape[1], result['hidden_size'], dataset.num_classes)
        model.load_state_dict(torch.load(model_file), strict=False)
        model.to(device)
        # TODO test on test
        # test_loop(test_loader, model, device)
    else:
        # hidden_sizes = [16, 32, 64]
        # nums_epochs = [100, 250, 350]
        # learning_rate = [0.01, 0.001, 0.0001]
        # batch_sizes = [8, 16, 32]
        hidden_sizes = [16]
        nums_epochs = [100]
        learning_rate = [0.01, 0.001]
        batch_sizes = [8]

        hyperparams = itertools.product(hidden_sizes, nums_epochs, batch_sizes, learning_rate)

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        test_subset = Subset(dataset, test_idx)

        best_model = None
        best_result = None
        best_accuracy = 0

        for hidden_size, num_epochs, batch_size, learning_rate in hyperparams:
            print('---------------------------------------------------------------')
            print('Hidden size: {} , Epochs: {} , Batch size: {} , Learning Rate: {}'.format(hidden_size, num_epochs,
                                                                                             batch_size, learning_rate))
            train_loader = DataLoader(
                train_subset, batch_size=batch_size, shuffle=False)

            val_loader = DataLoader(
                val_subset, batch_size=1, shuffle=False)

            train_loop(train_loader, model, criterion, optimizer, num_epochs, device)
            print('Results for hidden_size: {}, num_epochs: {}, batch_size: {}, learning rate: {}'.format(
                hidden_size, num_epochs, batch_size, learning_rate))
            result = test_loop(val_loader, model, device)
            # result.to_csv(name)


            test_loader = DataLoader(
                test_subset, batch_size=1, shuffle=False)

            model = Feedforward(
                dataset.X.shape[1], hidden_size, dataset.num_classes)
            model.to(device)

            criterion = torch.nn.CrossEntropyLoss()

            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

            if result.loc['accuracy'][0] > best_accuracy:
                best_model = model
                best_result = result
                best_accuracy = result.loc['accuracy'][0]
            print(result)
            # model, loss_values = train_model(
            #     model, criterion, optimizer, num_epochs, train_loader)
            # # plt.plot([x.detach().numpy() for x in loss_values])
            # # plt.title('Depth: {}, Epochs: {}, Batch: {}'.format(
            # #     hidden_size, num_epochs, batch_size))
            # # plt.show()
            # test_model(model, val_loader)
        torch.save(best_model.state_dict(), path.join('pytorch', 'saved_models', 'NN_{}.torch'.format(X.shape[1])))
        best_result.to_csv(path.join('pytorch', 'saved_models', 'csvs', 'NN_{}.csv'.format(X.shape[1])))
        print("BEST:")
        print(best_accuracy)
        print(best_result)
