import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init


class AutoEncoder(nn.Module):
    def __init__(self, ela_feats_num):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(ela_feats_num, 128),
            nn.PReLU(),
            nn.Linear(128, 64),
            nn.PReLU(),
            nn.Linear(64, 32),
            nn.PReLU(),
            nn.Linear(32, 16),
            nn.PReLU(),
            nn.Linear(16, 8),
            nn.PReLU(),
            nn.Linear(8, 2),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 8),
            nn.PReLU(),
            nn.Linear(8, 16),
            nn.PReLU(),
            nn.Linear(16, 32),
            nn.PReLU(),
            nn.Linear(32, 64),
            nn.PReLU(),
            nn.Linear(64, 128),
            nn.PReLU(),
            nn.Linear(128, ela_feats_num),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def load_model(model_path, ela_feats_num, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoEncoder(ela_feats_num)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    return model


def encode_ela_feats(model: AutoEncoder, input_features, device=None):
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    if not isinstance(input_features, torch.Tensor):
        input_features = torch.tensor(input_features, dtype=torch.float32)
    input_tensor = input_features.clone().detach().to(device)
    with torch.no_grad():
        encoded_features = model.encoder(input_tensor).cpu().numpy() * 5
    return np.asarray(encoded_features, dtype=np.float32)