import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, width=64, depth=1):
        super().__init__()

        self.recurrent = nn.LSTM(
            input_size=1,
            hidden_size=width,
            num_layers=depth,
            batch_first=True
        )

        self.output = nn.Linear(width, 1)

    def forward(self, x):
        sequence, _ = self.recurrent(x)
        return self.output(sequence[:, -1, :])