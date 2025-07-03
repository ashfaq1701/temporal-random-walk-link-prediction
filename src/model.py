from torch import nn
import torch.nn.functional as F


class LinkPredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=(256, 128, 64), dropout=0.2):
        super().__init__()
        h1, h2, h3 = hidden_dims

        self.fc1 = nn.Linear(input_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc_out = nn.Linear(h3, 1)

        self.proj1 = nn.Linear(input_dim, h2) if input_dim != h2 else nn.Identity()
        self.proj2 = nn.Linear(h1, h3) if h1 != h3 else nn.Identity()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x1 = F.relu(self.fc1(x))
        x1 = self.dropout(x1)

        x2 = F.relu(self.fc2(x1))
        x2 = self.dropout(x2)

        # Residual connection from input → x2
        x2_res = x2 + self.proj1(x)  # (input_dim → h2)

        x3 = F.relu(self.fc3(x2_res))
        x3 = self.dropout(x3)

        # Residual connection from x1 → x3
        x3_res = x3 + self.proj2(x1)  # (h1 → h3)

        out = self.fc_out(x3_res)
        return out
