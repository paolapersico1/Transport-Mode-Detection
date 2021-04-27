from torch import nn


class Feedforward(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        dropout = 0.2

        self.model = nn.Sequential(
            # nn.Dropout(dropout),
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            # nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            # nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            # nn.Dropout(dropout),
            nn.Linear(self.hidden_size, num_classes)
        )

    def forward(self, x):
        return self.model(x)
