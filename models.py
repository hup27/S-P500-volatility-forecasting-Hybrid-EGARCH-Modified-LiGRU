"""Reusable neural-network model definitions."""
import torch
from torch import nn

class LiGRUCell(nn.Module):
    """Modified LiGRU cell: one update gate, ReLU candidate, no batch normalization."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.Wz = nn.Linear(input_size, hidden_size)
        self.Uz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wh = nn.Linear(input_size, hidden_size)
        self.Uh = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, h):
        z = torch.sigmoid(self.Wz(x) + self.Uz(h))
        candidate = torch.relu(self.Wh(x) + self.Uh(h))
        return z * h + (1.0 - z) * candidate

class MultiLayerLiGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=8, num_layers=2, dropout=0.2):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.cells = nn.ModuleList([
            LiGRUCell(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, h=None):
        batch = x.size(0)
        if h is None:
            h = x.new_zeros(self.num_layers, batch, self.hidden_dim)
        states = [h[i] for i in range(self.num_layers)]
        outputs = []
        for t in range(x.size(1)):
            z = x[:, t, :]
            for i, cell in enumerate(self.cells):
                states[i] = cell(z, states[i])
                z = self.dropout(states[i]) if i < self.num_layers - 1 else states[i]
            outputs.append(self.fc(states[-1]))
        return torch.stack(outputs, dim=1), torch.stack(states, dim=0)

class MultiLayerGRU(nn.Module):
    """Standard PyTorch GRU baseline with a scalar output at each timestep."""
    def __init__(self, input_dim, hidden_dim=6, num_layers=2, dropout=0.4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, h=None):
        y, h_n = self.gru(x, h)
        return self.fc(y), h_n
