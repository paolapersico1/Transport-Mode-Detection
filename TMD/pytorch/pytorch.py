import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import preprocessing
from pytorch.model_runner import train_loop, test_loop
import torch
from pytorch.model import Feedforward
from pytorch.dataset import TMDDataset
from torch.utils.data import DataLoader, Subset
import itertools


def run(X, y):
    hidden_sizes = [64]
    nums_epochs = [100]
    learning_rate = [0.001]
    batch_sizes = [8, 16, 32]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: {}'.format(device))

    hyperparams = itertools.product(hidden_sizes, nums_epochs, batch_sizes, learning_rate)

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

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    for hidden_size, num_epochs, batch_size, learning_rate in hyperparams:
        print('---------------------------------------------------------------')
        print('hidden_size: {}, num_epochs: {}, batch_size: {}, learning rate: {}'.format(
            hidden_size, num_epochs, batch_size, learning_rate))
        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(
            val_subset, batch_size=1, shuffle=False)

        model = Feedforward(
            dataset.X.shape[1], hidden_size, dataset.num_classes)
        model.to(device)

        criterion = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        train_loop(train_loader, model, criterion, optimizer, num_epochs, device)
        test_loop(val_loader, model, device)

        # model, loss_values = train_model(
        #     model, criterion, optimizer, num_epochs, train_loader)
        # # plt.plot([x.detach().numpy() for x in loss_values])
        # # plt.title('Depth: {}, Epochs: {}, Batch: {}'.format(
        # #     hidden_size, num_epochs, batch_size))
        # # plt.show()
        # test_model(model, val_loader)
