import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Select feature mode: only 'fft' is supported
mode = 'fft'
feature_cols = [
    'VALR_fft1f','VALR_fft2f','VBLR_fft1f','VBLR_fft2f','VCLR_fft1f','VCLR_fft2f',
    'VAUD_fft1f','VAUD_fft2f','VBUD_fft1f','VBUD_fft2f','VCUD_fft1f','VCUD_fft2f',
    'PXAL_fft1f','PXAL_fft2f','PXAM_fft1f','PXAM_fft2f','PXAR_fft1f','PXAR_fft2f',
    'PXBL_fft1f','PXBL_fft2f','PXBM_fft1f','PXBM_fft2f','PXBR_fft1f','PXBR_fft2f',
    'PYAL_fft1f','PYAL_fft2f','PYAM_fft1f','PYAM_fft2f','PYAR_fft1f','PYAR_fft2f',
    'PYBL_fft1f','PYBL_fft2f','PYBM_fft1f','PYBM_fft2f','PYBR_fft1f','PYBR_fft2f'
]
data_dir = os.path.join(os.path.dirname(__file__), 'signal_features2')
train_file = os.path.join(data_dir, 'train_fft.csv')
test_file = os.path.join(data_dir, 'test_fft.csv')
alpha = 3.0  # Weight for localization regression term, adjust as needed
smooth_weight = 0.3  # Weight for smoothness regularization term, adjust as needed
lr = 0.005
batch_size = 64
seq_len = 5  # History length for GRU input
epochs = 80

# Create save directory
report_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
base_save_dir = os.path.join(data_dir, f'bi_GRU_results_{report_timestamp}')
os.makedirs(base_save_dir, exist_ok=True)

# 1. Load data
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

# Standardize
scaler = StandardScaler()
train_feat = scaler.fit_transform(train_df[feature_cols].values.astype(np.float32))
test_feat = scaler.transform(test_df[feature_cols].values.astype(np.float32))

def build_sequences(feat, flag, dx, dy, seq_len):
    X, y_flag, y_dx, y_dy = [], [], [], []
    for i in range(seq_len-1, len(feat)):
        X.append(feat[i-seq_len+1:i+1])
        y_flag.append(flag[i])
        y_dx.append(dx[i])
        y_dy.append(dy[i])
    return np.stack(X), np.array(y_flag), np.array(y_dx), np.array(y_dy)

X_train, y_train_flag, y_train_dx, y_train_dy = build_sequences(
    train_feat, train_df['flag'].values, train_df['dx'].values, train_df['dy'].values, seq_len)
X_test, y_test_flag, y_test_dx, y_test_dy = build_sequences(
    test_feat, test_df['flag'].values, test_df['dx'].values, test_df['dy'].values, seq_len)

class SeqDataset(Dataset):
    def __init__(self, X, y_flag, y_dx, y_dy):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_flag = torch.tensor(y_flag, dtype=torch.float32)
        self.y_dx = torch.tensor(y_dx, dtype=torch.float32)
        self.y_dy = torch.tensor(y_dy, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y_flag[idx], self.y_dx[idx], self.y_dy[idx]

train_dataset = SeqDataset(X_train, y_train_flag, y_train_dx, y_train_dy)
test_dataset = SeqDataset(X_test, y_test_flag, y_test_dx, y_test_dy)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 2. Define GRU model
class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.GRU = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc_flag = nn.Linear(hidden_dim * 2, 1)
        self.fc_reg = nn.Linear(hidden_dim * 2, 2)
    def forward(self, x):
        out, _ = self.GRU(x)
        last = out[:,-1,:]
        flag = torch.sigmoid(self.fc_flag(last)).squeeze(-1)
        dxdy = self.fc_reg(last)
        return flag, dxdy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GRUNet(X_train.shape[2]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
bce_loss = nn.BCELoss()
mse_loss = nn.MSELoss()

# 3. Training
test_loss_log = []
run_dir = os.path.join(base_save_dir, 'run')
os.makedirs(run_dir, exist_ok=True)

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for xb, yb_flag, yb_dx, yb_dy in train_loader:
        xb, yb_flag, yb_dx, yb_dy = xb.to(device), yb_flag.to(device), yb_dx.to(device), yb_dy.to(device)
        pred_flag, pred_dxdy = model(xb)
        loss_cls = bce_loss(pred_flag, yb_flag)
        mask = (yb_flag > 0.5)
        if mask.sum() > 0:
            gt_dxdy = torch.stack([yb_dx, yb_dy], dim=1)
            loss_reg = mse_loss(pred_dxdy[mask], gt_dxdy[mask])
            pred_dxdy_masked = pred_dxdy[mask]
            if pred_dxdy_masked.shape[0] > 2:
                d2 = pred_dxdy_masked[2:] - 2 * pred_dxdy_masked[1:-1] + pred_dxdy_masked[:-2]
                loss_smooth = d2.pow(2).mean()
            else:
                loss_smooth = 0 * loss_cls
        else:
            loss_reg = 0 * loss_cls
            loss_smooth = 0 * loss_cls
        loss = loss_cls + alpha * loss_reg + smooth_weight * loss_smooth
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch+1}/{epochs} train loss: {avg_loss:.6f}')

model_path = os.path.join(run_dir, 'bi_GRU_model.pth')
torch.save(model.state_dict(), model_path)
print(f'Model saved: {model_path}')
