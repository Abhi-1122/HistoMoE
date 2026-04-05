import torch
import torch.nn as nn


class GatingNetwork(nn.Module):
    def __init__(self, input_dim: int, num_experts: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_experts),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
