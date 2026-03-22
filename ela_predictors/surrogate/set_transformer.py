import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm

from dataset_generation.dataset import LatentCoordDataset
from .train_utils import normalize_batch_y, stratified_train_val_split

EPOCH = 60
BATCH = 128
DATASET_DIR = "./dataset_generation/dataset/ela_predictor_100d"
SAVE_DIR = "./surrogate/output/set_transformer_surrogate_best.pth"
TB_LOG_DIR = "./surrogate/output/tb_set_transformer"


class STMultiHeadAttention(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v, num_heads, is_layernorm=False):
        super().__init__()
        self.fc_q = nn.Linear(dim_q, dim_v)
        self.fc_k = nn.Linear(dim_k, dim_v)
        self.fc_v = nn.Linear(dim_k, dim_v)

        self.ln1 = nn.LayerNorm(dim_v) if is_layernorm else nn.Identity()
        self.ln2 = nn.LayerNorm(dim_v) if is_layernorm else nn.Identity()

        self.mha = nn.MultiheadAttention(
            embed_dim=dim_v,
            num_heads=num_heads,
            batch_first=True,
        )
        self.fc_o = nn.Sequential(
            nn.Linear(dim_v, 2 * dim_v),
            nn.ReLU(),
            nn.Linear(2 * dim_v, dim_v),
        )

    def forward(self, q_in: torch.Tensor, k_in: torch.Tensor) -> torch.Tensor:
        q = self.fc_q(q_in)
        k = self.fc_k(k_in)
        v = self.fc_v(k_in)

        attn_out, _ = self.mha(q, k, v)
        out = self.ln1(q + attn_out)
        out = self.ln2(out + self.fc_o(out))
        return out


class STSelfAttention(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super().__init__()
        self.mab = STMultiHeadAttention(dim_in, dim_in, dim_out, num_heads, ln)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mab(x, x)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, is_layer_norm=False):
        # Soft pooling ( Learnable Seed Vector )
        super().__init__()
        self.seed = nn.Parameter(torch.empty(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.seed)
        self.mab = STMultiHeadAttention(dim, dim, dim, num_heads, is_layer_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seed = self.seed.repeat(x.size(0), 1, 1)
        return self.mab(seed, x)


class SetTransformerSurrogate(nn.Module):
    def __init__(self, feat_dim=1, hidden_dim=256, num_heads=4):
        # [Batch_Size, Set_Size, Feature_Dim]
        super().__init__()
        self.enc = nn.Sequential(
            STSelfAttention(feat_dim, hidden_dim, num_heads, ln=True),
            STSelfAttention(hidden_dim, hidden_dim, num_heads, ln=True),
            STSelfAttention(hidden_dim, hidden_dim, num_heads, ln=True),
        )
        self.pma = PMA(hidden_dim, num_heads, num_seeds=1, is_layer_norm=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected [batch, n_eval], got shape={tuple(x.shape)}")
        x = x.unsqueeze(-1).float()
        z = self.enc(x)
        pooled = self.pma(z).squeeze(1)
        return self.head(pooled)


def train_set_transformer(
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

    model = SetTransformerSurrogate(feat_dim=1, hidden_dim=256, num_heads=4).to(device)
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

        print(
            f"Epoch {epoch:03d} | train={train_loss:.6f} | "
            f"val={val_loss:.6f} | best={best_val:.6f}"
        )

    writer.close()
    print(f"Best model saved to: {save_path}")


if __name__ == "__main__":
    train_set_transformer()
    
