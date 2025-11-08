# file: timeseries_regression.py
# Utilisation: python timeseries_regression.py --csv data/btc_daily.csv --target_col close --window 30

import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# Dataset utils
# -------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, series: np.ndarray, window: int):
        """
        series: 1-D numpy array of floats (normalized)
        window: input sequence length
        returns sequences X (window,) and target y (scalar)
        """
        self.window = window
        self.series = series
        self.n = len(series) - window

    def __len__(self):
        return max(0, self.n)

    def __getitem__(self, idx):
        x = self.series[idx: idx + self.window]         # shape (window,)
        y = self.series[idx + self.window]              # scalar
        # convert to tensors: (window, 1) for model expecting features
        return torch.tensor(x, dtype=torch.float32).unsqueeze(-1), torch.tensor(y, dtype=torch.float32)


def load_series_from_csv(csv_path: str, target_col: str) -> pd.Series:
    df = pd.read_csv(csv_path, parse_dates=True, infer_datetime_format=True)
    if target_col not in df.columns:
        raise ValueError(f"{target_col} not in CSV columns: {df.columns.tolist()}")
    # sort by index if time index exists, else assume it's already chronological
    # if there's a 'date' column, sort by it
    if 'date' in df.columns:
        df = df.sort_values('date').reset_index(drop=True)
    return df[target_col].astype(float)


def train_val_split(series: np.ndarray, train_frac=0.8) -> Tuple[np.ndarray, np.ndarray]:
    n = len(series)
    split = int(n * train_frac)
    return series[:split], series[split:]


# -------------------------
# Models
# -------------------------
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, (hn, cn) = self.lstm(x)   # out: (batch, seq_len, hidden_dim)
        last = out[:, -1, :]           # (batch, hidden_dim)
        return self.fc(last).squeeze(-1)  # (batch,)


class TransformerRegressor(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_layers=3, dim_feedforward=128, dropout=0.1):
        """
        A simple TransformerEncoder-based regressor.
        We project input_dim -> d_model, add positional enc, pass through TransformerEncoder,
        take last token representation and regress to scalar.
        """
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_emb = PositionalEncoding(d_model=d_model, dropout=dropout)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)       # -> (batch, seq_len, d_model)
        x = self.pos_emb(x)
        out = self.transformer(x)    # (batch, seq_len, d_model)
        last = out[:, -1, :]         # (batch, d_model)
        return self.fc(last).squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # odd case
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


# -------------------------
# Training / Evaluation
# -------------------------
def fit_one_epoch(model, optimizer, criterion, dataloader, device):
    model.train()
    total_loss = 0.0
    for xb, yb in dataloader:
        xb = xb.to(device)  # (batch, seq_len, 1)
        yb = yb.to(device)
        preds = model(xb)
        loss = criterion(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(dataloader.dataset)


def eval_model(model, criterion, dataloader, device):
    model.eval()
    total_loss = 0.0
    preds_all = []
    trues_all = []
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            total_loss += loss.item() * xb.size(0)
            preds_all.append(preds.cpu().numpy())
            trues_all.append(yb.cpu().numpy())
    preds_all = np.concatenate(preds_all, axis=0)
    trues_all = np.concatenate(trues_all, axis=0)
    mse = mean_squared_error(trues_all, preds_all)
    return total_loss / len(dataloader.dataset), mse, preds_all, trues_all


# -------------------------
# Main routine
# -------------------------
def main(args):
    # 1) Load series
    series_pd = load_series_from_csv(args.csv, target_col=args.target_col)
    values = series_pd.values.reshape(-1, 1).astype(float)
    scaler = MinMaxScaler(feature_range=(0, 1))
    values_norm = scaler.fit_transform(values).squeeze(-1)  # 1D array normalized

    # 2) train/val split at sequence level
    train_series, val_series = train_val_split(values_norm, train_frac=args.train_frac)

    train_dataset = TimeSeriesDataset(train_series, window=args.window)
    val_dataset = TimeSeriesDataset(val_series, window=args.window)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 3) instantiate models
    lstm = LSTMRegressor(input_dim=1, hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout).to(DEVICE)
    transformer = TransformerRegressor(input_dim=1, d_model=args.d_model, nhead=args.nhead,
                                       num_layers=args.trans_layers, dim_feedforward=args.dim_feedforward,
                                       dropout=args.dropout).to(DEVICE)

    # criterion + optimizers
    criterion = nn.MSELoss()
    opt_lstm = torch.optim.Adam(lstm.parameters(), lr=args.lr)
    opt_trans = torch.optim.Adam(transformer.parameters(), lr=args.lr)

    # 4) training loop
    best_val_mse_lstm = float('inf')
    best_val_mse_trans = float('inf')
    for epoch in range(1, args.epochs + 1):
        loss_tr_lstm = fit_one_epoch(lstm, opt_lstm, criterion, train_loader, DEVICE)
        val_loss_lstm, val_mse_lstm, _, _ = eval_model(lstm, criterion, val_loader, DEVICE)

        loss_tr_trans = fit_one_epoch(transformer, opt_trans, criterion, train_loader, DEVICE)
        val_loss_trans, val_mse_trans, _, _ = eval_model(transformer, criterion, val_loader, DEVICE)

        print(f"Epoch {epoch:03d} | LSTM train loss {loss_tr_lstm:.6f} val mse {val_mse_lstm:.6f} | "
              f"Trans train loss {loss_tr_trans:.6f} val mse {val_mse_trans:.6f}")

        # save best
        os.makedirs(args.out_dir, exist_ok=True)
        if val_mse_lstm < best_val_mse_lstm:
            best_val_mse_lstm = val_mse_lstm
            torch.save(lstm.state_dict(), os.path.join(args.out_dir, "best_lstm.pth"))
        if val_mse_trans < best_val_mse_trans:
            best_val_mse_trans = val_mse_trans
            torch.save(transformer.state_dict(), os.path.join(args.out_dir, "best_transformer.pth"))

    # final evaluation on val set (best models)
    print("Training complete. Best val MSE - LSTM:", best_val_mse_lstm, "Transformer:", best_val_mse_trans)

    # Save scaler for inverse transform later
    import joblib
    joblib.dump(scaler, os.path.join(args.out_dir, "scaler.save"))
    print("Saved scaler and models in", args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="path to csv with price series")
    parser.add_argument("--target_col", type=str, default="close")
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--trans_layers", type=int, default=3)
    parser.add_argument("--dim_feedforward", type=int, default=256)
    parser.add_argument("--out_dir", type=str, default="models")
    args = parser.parse_args()
    main(args)
