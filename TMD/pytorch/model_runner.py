import torch


# function to train the neural network (SGD)
def train_loop(dataloader, model, loss_fn, optimizer, scheduler, num_epochs, device):
    print('----------------------------------')
    model.train()
    losses = []
    # loop through the entire dataset for num_epochs times
    for epoch in range(num_epochs):
        print("Epoch: {}/{}".format(epoch, num_epochs))
        # for each mini-batch
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)

            # backpropagation step
            loss.backward()
            optimizer.step()

        # decrease the learning rate after each epoch
        scheduler.step()
        losses.append(loss.item())
        print("loss: {}".format(loss))
    print('Done')
    print('----------------------------------')
    return losses


# function to test the neural network
def test_loop(dataloader, model, device):
    model.eval()
    predictions = []
    actual_labels = []

    # avoid memory consumption to calculate gradient: we are not training the network
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            predictions.append(model(X))
            actual_labels.append(y)

    # matrix of size (size(dataloader.X),k): the higher is the value, the higher is the probability of that class
    predictions = torch.stack(predictions).squeeze()
    actual_labels = torch.stack(actual_labels).squeeze().cpu()
    # keep the class with the maximum value
    predictions = predictions.argmax(dim=1, keepdim=True).cpu()
    # the accuracy is the the ratio of correct predictions
    score = torch.sum((predictions.squeeze() == actual_labels).float()) / actual_labels.shape[0]

    return score.numpy()
