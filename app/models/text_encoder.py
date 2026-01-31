import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # x: [B, 128]
        return self.fc(x)
