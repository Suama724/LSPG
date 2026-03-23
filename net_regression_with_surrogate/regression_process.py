import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from surrogate_mlp.MLP import MLPSurrogate
from config import Config


class SimpleTargetNet(nn.Module):
    def __init__(self, dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1024),
            nn.SiLU(),
            nn.Linear(1024, 256),
            #nn.SiLU(),
            #nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


def _load_surrogate(surrogate_path, surrogate_method='mlp') -> nn.Module:
    model = MLPSurrogate()
    model.load_state_dict(torch.load(surrogate_path, weights_only=True))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def _get_single_regression_target(sample_method='random') -> np.array:
    return np.random.uniform(-5, 5, size=2)


def _normalize_batch_y(y):
    mean = y.mean(dim=1, keepdim=True)
    std = y.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-8)
    return (y - mean) / std


def single_func_regression(surrogate_path="", surrogate_method="", target_sample_method="") -> nn.Module:
    cfg = Config()
    surrogate_path = cfg.surrogate_path
    surrogate_method = cfg.surrogate_method
    target_sample_method = 'random'
    epoch = cfg.epoch
    dim = cfg.dim

    surrogate = _load_surrogate(surrogate_path, surrogate_method)
    target = torch.tensor(_get_single_regression_target(target_sample_method), dtype=torch.float32).unsqueeze(0)

    X = torch.rand(230, dim)
    net = SimpleTargetNet(dim=dim)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    for epoch in range(epoch):
        y = net(X).reshape(1, -1)
        y_norm = _normalize_batch_y(y)
        pred = surrogate(y_norm)
        loss = nn.functional.mse_loss(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"epoch {epoch:>3d}  loss={loss.item():.6f}  pred={pred.detach().numpy().flatten()}")
        
        if loss.item() < 0.01:
            break

    return net, surrogate, target, X


def view_output(net, surrogate, target, X):
    with torch.no_grad():
        y = net(X).reshape(1, -1)
        y_norm = _normalize_batch_y(y)
        pred = surrogate(y_norm).numpy().flatten()
    tgt = target.numpy().flatten()

    plt.figure(figsize=(6, 6))
    plt.scatter(tgt[0], tgt[1], marker='x', s=20, c='red', label='target')
    plt.scatter(pred[0], pred[1], marker='o', s=20, c='blue', label='predicted')
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.legend()
    plt.grid(True)
    plt.title("Regression Result")
    plt.show()

if __name__ == '__main__':
    view_output(*single_func_regression())