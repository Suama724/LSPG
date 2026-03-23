import torch
from torch import nn as nn

from pathlib import Path


class MLPSurrogate(nn.Module):
    def __init__(self, n_eval = 230):
        super().__init__()
        self.n_eval = n_eval

        self.net = nn.Sequential(
            nn.Linear(n_eval, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            
            nn.Linear(128, 2)  
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected [batch, n_eval], got shape={tuple(x.shape)}")
        return self.net(x.float())


