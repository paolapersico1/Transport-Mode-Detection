import numpy as np
import torch
from sklearn.metrics import classification_report


def train_loop(dataloader, model, loss_fn, optimizer, num_epochs, device):
    model.train()
    size = len(dataloader.dataset)
    for epoch in range(num_epochs):
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print("loss: {}, [{}/{}]".format(loss, current, size))
                # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, device):
    model.eval()
    predictions = []
    actual_labels = []

    # avoid memory consumption to calculate gradient: we are not training the network
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            a = model(X)
            predictions.append(a)
            actual_labels.append(y)
    # matrix of size (size(dataloader.X),k): the higher is the value, the higher is the probability of that class
    predictions = torch.stack(predictions).squeeze()
    actual_labels = torch.stack(actual_labels).squeeze()
    # keep the class with the maximum value
    predictions = predictions.argmax(dim=1, keepdim=True)
    print(classification_report(actual_labels, predictions))
