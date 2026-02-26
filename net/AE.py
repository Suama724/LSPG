import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from tqdm import tqdm
import os
import datetime
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import MinMaxScaler
#from config import config_AE

class AutoEncoder(nn.Module):
    def __init__(self, ela_feats_num):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(ela_feats_num, 128),
            nn.PReLU(),
            nn.Linear(128,64),
            nn.PReLU(),
            nn.Linear(64, 32),
            nn.PReLU(),
            nn.Linear(32, 16),
            nn.PReLU(),
            nn.Linear(16, 8),
            nn.PReLU(),
            nn.Linear(8, 2),
            nn.Tanh()
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
            nn.Linear(64,128),
            nn.PReLU(),
            nn.Linear(128, ela_feats_num)
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
        # 乘5操作在后面getencoder处进行
        x = self.decoder(x)
        return x        

def train_autoencoder(model, train_loader, val_loader, 
                      model_dir, log_dir, device,
                      num_epochs=100, save_interval=50, lr=1e-4, ):
    time_now=datetime.datetime.now().strftime('%Y_%m_%d_%H%M%S')
    #model_dir = os.path.join(model_dir, time_now)
    #log_dir = os.path.join(log_dir, time_now)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer,0.9862327,last_epoch=-1)
    best_val_loss = float('inf')

    writer = SummaryWriter(log_dir = log_dir)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = data[0].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                inputs = data[0].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        scheduler.step()
        # Save current best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(model_dir, "autoencoder_best.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"  [Create Best] Val loss improved. Saved to {best_model_path}")
        
        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            model_path = os.path.join(model_dir,f"autoencoder_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), model_path)
            print(f'[Checkpoint] Model saved to {model_path}')

        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Learning_rate', scheduler.get_last_lr()[0], epoch)
    
    # Save the final model
    final_path = os.path.join(model_dir, "autoencoder_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Training Complete. Final model saved to {final_path}")
    writer.close()

def load_data(data, batch_size=32, val_split=0.2):
    data = data.reshape(-1, data.shape[-1])  # Flatten to shape (24*1280, ela_feats_num)
    dataset = TensorDataset(torch.tensor(data, dtype=torch.float32))
    
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def split_data(data,val_split=0.2,random_state = 42):
    data = data.reshape(-1, data.shape[-1])
    train_data, val_data = train_test_split(data, test_size=val_split, random_state=random_state,shuffle=True)
    return train_data,val_data

def make_dataset(train_data,val_data, batch_size=16):
    train_data,val_data = train_data.reshape(-1, train_data.shape[-1]) ,val_data.reshape(-1,val_data.shape[-1]) # Flatten to shape (24*1280, n_fea)
    train_data,val_data = TensorDataset(torch.tensor(train_data, dtype=torch.float32)),TensorDataset(torch.tensor(val_data, dtype=torch.float32))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

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
    return encoded_features

def normalize_data(feats):
    scaler = MinMaxScaler()
    feats_normalized = scaler.fit_transform(feats.reshape(-1, feats.shape[-1]))
    return feats_normalized, scaler

def _test_AE():
    n_fea = 21
    batch_size = 32
    num_epochs = 300
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = np.random.rand(32, 270, n_fea)

    train_loader, val_loader = load_data(data, batch_size=batch_size)
    model = AutoEncoder(n_fea)
    model_dir = os.path.join('artifacts', 'models')
    log_dir = os.path.join('artifacts', 'logs')
    train_autoencoder(model, train_loader, val_loader, device=device,
                      model_dir=model_dir, log_dir=log_dir, 
                      num_epochs=num_epochs)


if __name__ == '__main__':
    _test_AE()
