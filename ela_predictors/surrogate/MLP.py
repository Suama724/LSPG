import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm

from dataset_generation.dataset import LatentCoordDataset
from .train_utils import normalize_batch_y, stratified_train_val_split

EPOCH = 30
BATCH = 64
DATASET_DIR = "./dataset_generation/dataset/ela_predictor_100d"
SAVE_DIR = "./surrogate/output/mlp_surrogate_best.pth"
TB_LOG_DIR = "./surrogate/output/tb_mlp"


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
            
            nn.Linear(128, 2)  # 直接输出 2D 坐标 (无激活函数)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected [batch, n_eval], got shape={tuple(x.shape)}")
        return self.net(x.float())


def train_mlp(
    run_dir=DATASET_DIR,
    epochs=EPOCH,
    batch_size=BATCH,
    lr=1e-3,
    val_ratio=0.2,
    split_seed=42,
    save_path=SAVE_DIR,
    tb_log_dir=TB_LOG_DIR,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = LatentCoordDataset(run_dir=run_dir, return_meta=False)
    train_set, val_set = stratified_train_val_split(dataset, val_ratio=val_ratio, seed=split_seed)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)

    model = MLPSurrogate(n_eval=dataset[0][0].shape[0]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_dir)
    print(f"TensorBoard logdir: {tb_log_dir}")

    best_val = float("inf")
    train_step = 0
    val_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch:03d} Train", leave=False)
        for y, z2d in train_bar:
            y, z2d = y.to(device), z2d.to(device)
            y = normalize_batch_y(y)
            pred = model(y)
            loss = criterion(pred, z2d)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * y.size(0)
            train_bar.set_postfix(loss=f"{loss.item():.6f}")
            writer.add_scalar("loss/train_batch", loss.item(), train_step)
            train_step += 1
        train_loss /= len(train_set)
        writer.add_scalar("loss/train_epoch", train_loss, epoch)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch:03d} Val", leave=False)
            for y, z2d in val_bar:
                y, z2d = y.to(device), z2d.to(device)
                y = normalize_batch_y(y)
                batch_val_loss = criterion(model(y), z2d).item()
                val_loss += batch_val_loss * y.size(0)
                val_bar.set_postfix(loss=f"{batch_val_loss:.6f}")
                writer.add_scalar("loss/val_batch", batch_val_loss, val_step)
                val_step += 1
        val_loss /= len(val_set)
        writer.add_scalar("loss/val_epoch", val_loss, epoch)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_path)
        writer.add_scalar("loss/best_val", best_val, epoch)

        print(f"Epoch {epoch:03d} | train={train_loss:.6f} | val={val_loss:.6f} | best={best_val:.6f}")

    writer.close()
    print(f"Best model saved to: {save_path}")


if __name__ == "__main__":
    train_mlp()