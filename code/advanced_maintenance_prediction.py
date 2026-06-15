# -*- coding: utf-8 -*-
"""
Advanced Maintenance Prediction using Deep Learning & Time Series Models

This enhanced version includes:
1. LSTM/GRU neural networks for sequential learning
2. Temporal Convolutional Networks (TCN)
3. Advanced feature engineering (lag features, rolling windows)
4. Attention mechanisms for important time periods
5. Transformer-based and N-BEATS models for advanced temporal learning
6. Prophet for trend decomposition
"""

import os
import json
import argparse
import math
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Traditional ML
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (classification_report, roc_auc_score, roc_curve,
                             precision_recall_curve, average_precision_score)

# Deep Learning
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False
    print("PyTorch not available. Install with: pip install torch")

# Advanced Time Series
try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False
    print("Prophet not available. Install with: pip install prophet")

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
dump_dir = os.path.join(base_dir, "..", "data")
monthly_dir = os.path.join(dump_dir, "monthly")
result_dir = os.path.join(base_dir, "..", "result")

INACTIVE_THRESHOLD_DAYS = 90
RANDOM_STATE = 42
SEQUENCE_LENGTH = 12  # Use 12 months of history (legacy / other models)
PREDICTION_HORIZON = 3  # Predict 3 months ahead

# Mamba temporal-split workflow
TRAIN_MONTHS = 20   # months used as input features for training
LABEL_MONTHS = 4    # future months used to derive the binary activity label
TOTAL_MONTHS = TRAIN_MONTHS + LABEL_MONTHS  # 24 months loaded total

# Set random seeds
np.random.seed(RANDOM_STATE)
if HAS_PYTORCH:
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)


# ----------------------------------------------------------------------
# Data Loading (reuse from previous script)
# ----------------------------------------------------------------------
def load_crates(path: str) -> pd.DataFrame:
    """Load crates.csv with essential columns."""
    print(f"Loading crates from {path}...")
    df = pd.read_csv(path, usecols=["id", "name", "repository", "updated_at", "created_at"])
    df["id"] = df["id"].astype(int)
    df["name"] = df["name"].astype(str)
    df["repository"] = df["repository"].fillna("").astype(str)
    
    df["updated_at"] = pd.to_datetime(df["updated_at"], errors="coerce", utc=True)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    
    df = df[df["repository"].str.contains("github.com", case=False, na=False)].copy()
    
    print(f"Loaded {len(df)} open-sourced crates")
    return df


def normalize_repo_url(repo_url: str) -> str:
    """Normalize GitHub repository URL."""
    if not repo_url or not isinstance(repo_url, str):
        return ""
    
    repo_url = repo_url.lower().strip()
    if "github.com/" in repo_url:
        parts = repo_url.split("github.com/")[-1]
        parts = parts.rstrip("/").split("/")[:2]
        if len(parts) == 2:
            return f"github:{parts[0]}/{parts[1]}"
    return ""


def load_monthly_activity(monthly_dir: str, n_months: int = None):
    """Load monthly activity data and organize by repository and time.
    
    Args:
        monthly_dir: Path to directory containing monthly delta JSON files.
        n_months: Number of most recent months to load. If None, load all.
    """
    print(f"Loading monthly activity from {monthly_dir}...")
    
    all_files = sorted([f for f in os.listdir(monthly_dir) 
                        if f.startswith("delta_") and f.endswith(".json")])
    
    if n_months is not None:
        files = all_files[-n_months:]
        print(f"Using last {n_months} months (out of {len(all_files)} available)")
    else:
        files = all_files
    
    print(f"Processing {len(files)} monthly files...")
    
    # repo -> {month: activity_dict}
    repo_timeseries = defaultdict(dict)
    all_months = []
    
    for file in files:
        month = file.replace("delta_", "").replace(".json", "")
        all_months.append(month)
        file_path = os.path.join(monthly_dir, file)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for repo, activity in data.items():
            if repo.startswith("github:"):
                repo_timeseries[repo][month] = activity
    
    print(f"Loaded activity for {len(repo_timeseries)} repositories across {len(all_months)} months")
    return repo_timeseries, sorted(all_months)


# ----------------------------------------------------------------------
# Advanced Feature Engineering
# ----------------------------------------------------------------------
def create_sequence_matrix(repo_timeseries, all_months, metrics_list):
    """
    Create sequence matrices for deep learning models.
    
    Returns:
        sequences: numpy array of shape (n_repos, seq_length, n_metrics)
        repo_names: list of repo identifiers
        valid_repos: repos with sufficient history
    """
    sequences = []
    repo_names = []
    
    for repo, monthly_data in repo_timeseries.items():
        # Create time series for this repo
        time_series = []

        for month in all_months:
            if month in monthly_data:
                activity = monthly_data[month]

                # Extract metrics
                month_values = []
                for metric in metrics_list:
                    if metric == 'active_contributors_count':
                        val = len(activity.get('active_contributors', []))
                    elif metric == 'days_since_last_release':
                        # Compute days since last release if possible
                        last_release_at = activity.get('last_release_at')
                        if last_release_at:
                            try:
                                last_release_dt = pd.to_datetime(last_release_at, utc=True)
                                # Month string is like '2024_04', get last day of month
                                month_dt = pd.to_datetime(month + '_01', format='%Y_%m_%d', utc=True) + pd.offsets.MonthEnd(0)
                                days_since = (month_dt - last_release_dt).days
                                val = float(days_since) if days_since >= 0 else 0.0
                            except Exception:
                                val = 0.0
                        else:
                            val = 0.0
                    else:
                        val = activity.get(metric, 0) or 0
                    month_values.append(float(val))

                time_series.append(month_values)
            else:
                # Missing data - use zeros or forward fill
                if len(time_series) > 0:
                    time_series.append(time_series[-1])  # Forward fill
                else:
                    time_series.append([0.0] * len(metrics_list))

        # Pad with zeros if sequence is shorter than SEQUENCE_LENGTH
        if len(time_series) < SEQUENCE_LENGTH:
            padding = [[0.0] * len(metrics_list)] * (SEQUENCE_LENGTH - len(time_series))
            time_series = padding + time_series

        sequences.append(time_series)
        repo_names.append(repo)

    return np.array(sequences), repo_names


def create_lag_features(df, value_col, lags=[1, 2, 3, 6, 12]):
    """Create lag features for a time series column."""
    lag_df = df.copy()
    
    for lag in lags:
        lag_df[f'{value_col}_lag{lag}'] = df[value_col].shift(lag)
    
    return lag_df


def create_rolling_features(df, value_col, windows=[3, 6, 12]):
    """Create rolling window features."""
    roll_df = df.copy()
    
    for window in windows:
        roll_df[f'{value_col}_rolling_mean{window}'] = df[value_col].rolling(window).mean()
        roll_df[f'{value_col}_rolling_std{window}'] = df[value_col].rolling(window).std()
        roll_df[f'{value_col}_rolling_max{window}'] = df[value_col].rolling(window).max()
        roll_df[f'{value_col}_rolling_min{window}'] = df[value_col].rolling(window).min()
    
    return roll_df


def extract_advanced_features(repo_timeseries, all_months, repo):
    """Extract advanced time series features including lags and rolling windows."""
    if repo not in repo_timeseries:
        return None
    
    monthly_data = repo_timeseries[repo]
    
    # Create DataFrame of time series
    rows = []
    for month in all_months:
        if month in monthly_data:
            activity = monthly_data[month]
            row = {
                'month': month,
                'push_events': activity.get('push_events', 0) or 0,
                'issues_opened': activity.get('issues_opened', 0) or 0,
                'issues_closed': activity.get('issues_closed', 0) or 0,
                'prs_opened': activity.get('prs_opened', 0) or 0,
                'prs_merged': activity.get('prs_merged', 0) or 0,
                'active_contributors': len(activity.get('active_contributors', [])),
                'releases': activity.get('releases_published', 0) or 0,
            }
            rows.append(row)
    
    if len(rows) < SEQUENCE_LENGTH:
        return None
    
    ts_df = pd.DataFrame(rows)
    
    # Add lag features
    for col in ['push_events', 'issues_opened', 'prs_merged', 'active_contributors']:
        ts_df = create_lag_features(ts_df, col, lags=[1, 2, 3, 6])
        ts_df = create_rolling_features(ts_df, col, windows=[3, 6])
    
    # Get most recent row features
    latest = ts_df.iloc[-1]
    
    # Drop NaN and return as dict
    features = latest.drop('month').to_dict()
    features = {k: (v if not pd.isna(v) else 0) for k, v in features.items()}
    
    return features


# ----------------------------------------------------------------------
# Deep Learning Models (PyTorch)
# ----------------------------------------------------------------------
class LSTMModel(nn.Module):
    """LSTM model for sequence classification."""
    def __init__(self, n_features, units=64, dropout=0.3):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, units, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(units, units // 2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(units // 2, 32)
        self.dropout3 = nn.Dropout(dropout / 2)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x[:, -1, :])  # last time step
        x = torch.relu(self.fc1(x))
        x = self.dropout3(x)
        x = torch.sigmoid(self.fc2(x))
        return x.squeeze(-1)


class GRUModel(nn.Module):
    """GRU model for sequence classification."""
    def __init__(self, n_features, units=64, dropout=0.3):
        super().__init__()
        self.gru1 = nn.GRU(n_features, units, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.gru2 = nn.GRU(units, units // 2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(units // 2, 32)
        self.dropout3 = nn.Dropout(dropout / 2)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x, _ = self.gru1(x)
        x = self.dropout1(x)
        x, _ = self.gru2(x)
        x = self.dropout2(x[:, -1, :])
        x = torch.relu(self.fc1(x))
        x = self.dropout3(x)
        x = torch.sigmoid(self.fc2(x))
        return x.squeeze(-1)


class AttentionLSTMModel(nn.Module):
    """LSTM with attention mechanism."""
    def __init__(self, n_features, units=64):
        super().__init__()
        self.lstm = nn.LSTM(n_features, units, batch_first=True)
        self.attention_fc = nn.Linear(units, 1)
        self.fc1 = nn.Linear(units, 32)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # (batch, seq, units)
        # Attention
        attn_scores = torch.tanh(self.attention_fc(lstm_out))  # (batch, seq, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)       # (batch, seq, 1)
        context = (lstm_out * attn_weights).sum(dim=1)          # (batch, units)
        # Output
        x = torch.relu(self.fc1(context))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x.squeeze(-1)


class TCNModel(nn.Module):
    """Temporal Convolutional Network."""
    def __init__(self, n_features, filters=64, kernel_size=3):
        super().__init__()
        # Causal padding: pad left side only
        self.pad1 = nn.ConstantPad1d((kernel_size - 1, 0), 0)
        self.conv1 = nn.Conv1d(n_features, filters, kernel_size)
        self.dropout1 = nn.Dropout(0.3)
        self.pad2 = nn.ConstantPad1d((kernel_size - 1, 0), 0)
        self.conv2 = nn.Conv1d(filters, filters, kernel_size)
        self.dropout2 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(filters, 32)
        self.dropout3 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # x: (batch, seq, features) -> (batch, features, seq) for Conv1d
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv1(self.pad1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.conv2(self.pad2(x)))
        x = self.dropout2(x)
        # Global average pooling over time
        x = x.mean(dim=2)
        x = torch.relu(self.fc1(x))
        x = self.dropout3(x)
        x = torch.sigmoid(self.fc2(x))
        return x.squeeze(-1)


# ---- New Advanced Models ----

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer models."""
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerModel(nn.Module):
    """
    Transformer Encoder for sequence classification.
    Uses multi-head self-attention to capture global dependencies.
    Reference: Vaswani et al., "Attention Is All You Need", NeurIPS 2017.
    """
    def __init__(self, n_features, d_model=64, nhead=4, num_layers=2, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc1 = nn.Linear(d_model, 32)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.input_proj(x)          # (batch, seq, d_model)
        x = self.pos_enc(x)
        x = self.transformer_encoder(x) # (batch, seq, d_model)
        x = x.mean(dim=1)               # global average pooling
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x.squeeze(-1)


class InformerModel(nn.Module):
    """
    Informer-style model with ProbSparse self-attention approximation.
    Uses top-k query selection for efficient attention on long sequences.
    Reference: Zhou et al., "Informer", AAAI 2021.
    """
    def __init__(self, n_features, d_model=64, nhead=4, num_layers=2, dropout=0.3, top_k_ratio=0.5):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.top_k_ratio = top_k_ratio
        self.num_layers = num_layers

        # ProbSparse attention layers
        self.q_projs = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers)])
        self.k_projs = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers)])
        self.v_projs = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers)])
        self.out_projs = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers)])
        self.norms1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.norms2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(d_model * 4, d_model), nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_model, 32)
        self.fc2 = nn.Linear(32, 1)

    def _prob_sparse_attention(self, Q, K, V):
        """ProbSparse self-attention: select top-k queries by sparsity measure."""
        B, H, L, D = Q.shape
        top_k = max(1, int(L * self.top_k_ratio))

        # Compute query sparsity measurement: KL divergence proxy
        # M(q_i) = max(q_i * k_j^T / sqrt(d)) - mean(q_i * k_j^T / sqrt(d))
        scores_sample = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)
        M = scores_sample.max(dim=-1).values - scores_sample.mean(dim=-1)  # (B, H, L)

        # Select top-k queries
        _, top_idx = M.topk(top_k, dim=-1)  # (B, H, top_k)

        # Gather selected queries
        top_idx_expanded = top_idx.unsqueeze(-1).expand(-1, -1, -1, D)
        Q_selected = torch.gather(Q, 2, top_idx_expanded)  # (B, H, top_k, D)

        # Standard attention on selected queries
        attn_scores = torch.matmul(Q_selected, K.transpose(-2, -1)) / math.sqrt(D)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_weights, V)  # (B, H, top_k, D)

        # Scatter back; fill non-selected with mean of V
        context = V.mean(dim=2, keepdim=True).expand_as(V).clone()  # (B, H, L, D)
        top_idx_out = top_idx.unsqueeze(-1).expand(-1, -1, -1, D)
        context.scatter_(2, top_idx_out, attn_out)

        return context

    def forward(self, x):
        B, L, _ = x.shape
        x = self.input_proj(x)
        x = self.pos_enc(x)

        for i in range(self.num_layers):
            Q = self.q_projs[i](x).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
            K = self.k_projs[i](x).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
            V = self.v_projs[i](x).view(B, L, self.nhead, self.head_dim).transpose(1, 2)

            attn_out = self._prob_sparse_attention(Q, K, V)  # (B, H, L, D)
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, self.d_model)
            attn_out = self.out_projs[i](attn_out)

            x = self.norms1[i](x + self.dropout(attn_out))
            x = self.norms2[i](x + self.ffns[i](x))

        x = x.mean(dim=1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x.squeeze(-1)


class NBEATSBlock(nn.Module):
    """Single N-BEATS block with fully connected stack."""
    def __init__(self, input_size, theta_size, hidden_size=128, n_layers=4):
        super().__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        self.fc = nn.Sequential(*layers)
        self.theta_b = nn.Linear(hidden_size, theta_size)  # backcast
        self.theta_f = nn.Linear(hidden_size, theta_size)  # forecast

    def forward(self, x):
        h = self.fc(x)
        return self.theta_b(h), self.theta_f(h)


class NBEATSModel(nn.Module):
    """
    N-BEATS for sequence classification (adapted from forecasting).
    Stacks residual blocks that decompose the input into backcast/forecast.
    Reference: Oreshkin et al., "N-BEATS", ICLR 2020.
    """
    def __init__(self, n_features, seq_len=12, n_stacks=3, n_blocks=3, hidden_size=128):
        super().__init__()
        input_size = n_features * seq_len
        self.seq_len = seq_len
        self.n_features = n_features

        self.blocks = nn.ModuleList()
        for _ in range(n_stacks * n_blocks):
            self.blocks.append(NBEATSBlock(input_size, input_size, hidden_size))

        self.fc1 = nn.Linear(input_size, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x: (batch, seq, features) -> flatten
        B = x.size(0)
        residual = x.view(B, -1)  # (batch, seq*features)
        forecast_sum = torch.zeros_like(residual)

        for block in self.blocks:
            backcast, forecast = block(residual)
            residual = residual - backcast
            forecast_sum = forecast_sum + forecast

        out = torch.relu(self.fc1(forecast_sum))
        out = self.dropout(out)
        out = torch.sigmoid(self.fc2(out))
        return out.squeeze(-1)


class MambaBlock(nn.Module):
    """Simplified S4/Mamba-style selective state space block."""
    def __init__(self, d_model, d_state=16, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Input projection (expand)
        self.in_proj = nn.Linear(d_model, d_model * 2)
        # Conv for local context
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        # SSM parameters (input-dependent)
        self.dt_proj = nn.Linear(d_model, d_model)
        self.A = nn.Parameter(torch.randn(d_model, d_state))
        self.B_proj = nn.Linear(d_model, d_state)
        self.C_proj = nn.Linear(d_model, d_state)
        self.D = nn.Parameter(torch.ones(d_model))
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: (batch, seq, d_model)"""
        residual = x
        B, L, D = x.shape

        # Input projection -> gate and input
        xz = self.in_proj(x)  # (B, L, 2*D)
        x_in, z = xz.chunk(2, dim=-1)  # each (B, L, D)

        # Conv1d for local context
        x_conv = self.conv1d(x_in.transpose(1, 2)).transpose(1, 2)  # (B, L, D)
        x_conv = nn.functional.silu(x_conv)

        # Selective SSM scan
        dt = nn.functional.softplus(self.dt_proj(x_conv))  # (B, L, D)
        B_input = self.B_proj(x_conv)  # (B, L, d_state)
        C_input = self.C_proj(x_conv)  # (B, L, d_state)

        # Discretize and scan (simplified sequential scan)
        A_neg = -torch.exp(self.A)  # (D, d_state)
        h = torch.zeros(B, D, self.d_state, device=x.device)  # hidden state
        outputs = []
        for t in range(L):
            dt_t = dt[:, t, :]  # (B, D)
            dA = torch.exp(dt_t.unsqueeze(-1) * A_neg.unsqueeze(0))  # (B, D, d_state)
            dB = dt_t.unsqueeze(-1) * B_input[:, t, :].unsqueeze(1)  # (B, D, d_state)
            h = h * dA + dB * x_conv[:, t, :].unsqueeze(-1)  # (B, D, d_state)
            y_t = (h * C_input[:, t, :].unsqueeze(1)).sum(dim=-1)  # (B, D)
            y_t = y_t + self.D * x_conv[:, t, :]  # skip connection
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (B, L, D)
        y = y * nn.functional.silu(z)  # gating
        y = self.out_proj(y)
        y = self.norm(y + residual)
        return self.dropout(y)


class MambaModel(nn.Module):
    """
    Mamba-style Selective State Space Model for sequence classification.
    Uses input-dependent selection mechanism for efficient long-range modeling.
    Reference: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2023.
    """
    def __init__(self, n_features, d_model=64, n_layers=2, d_state=16, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, dropout) for _ in range(n_layers)
        ])
        self.fc1 = nn.Linear(d_model, 32)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.input_proj(x)  # (batch, seq, d_model)
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=1)  # global average pooling
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x.squeeze(-1)


# All available models
ALL_MODELS = ['LSTM', 'GRU', 'Attention_LSTM', 'TCN',
              'Transformer', 'Informer', 'NBEATS', 'Mamba']


def build_model(model_name, n_features, seq_len=12):
    """Factory function to create a PyTorch model by name."""
    if model_name == 'LSTM':
        return LSTMModel(n_features)
    elif model_name == 'GRU':
        return GRUModel(n_features)
    elif model_name == 'Attention_LSTM':
        return AttentionLSTMModel(n_features)
    elif model_name == 'TCN':
        return TCNModel(n_features)
    elif model_name == 'Transformer':
        return TransformerModel(n_features)
    elif model_name == 'Informer':
        return InformerModel(n_features)
    elif model_name == 'NBEATS':
        return NBEATSModel(n_features, seq_len=seq_len)
    elif model_name == 'Mamba':
        return MambaModel(n_features)
    else:
        raise ValueError(f"Unknown model: {model_name}. Available: {ALL_MODELS}")


# ----------------------------------------------------------------------
# Main Training Pipeline
# ----------------------------------------------------------------------
def _train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(dataloader.dataset)


def _evaluate(model, dataloader, criterion, device):
    """Evaluate model and return loss + predictions."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            loss = criterion(output, y_batch)
            total_loss += loss.item() * X_batch.size(0)
            all_preds.append(output.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())
    avg_loss = total_loss / len(dataloader.dataset)
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    auc = roc_auc_score(labels, preds) if len(np.unique(labels)) > 1 else 0.0
    return avg_loss, auc, preds


def train_deep_learning_models(X_seq, y, output_dir, model_names=None):
    """Train deep learning models using PyTorch.
    
    Args:
        X_seq: Sequence data array (n_samples, seq_len, n_features)
        y: Labels array
        output_dir: Directory to save results
        model_names: List of model names to train. If None, trains all.
    """
    if model_names is None:
        model_names = ALL_MODELS

    if not HAS_PYTORCH:
        print("PyTorch not available. Skipping deep learning models.")
        return {}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    print("\n" + "="*80)
    print("Training Deep Learning Models (PyTorch)")
    print("="*80)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Normalize sequences
    scaler = MinMaxScaler()
    n_samples, seq_len, n_features = X_train.shape

    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, n_features))
    X_train_scaled = X_train_scaled.reshape(n_samples, seq_len, n_features)

    n_test = X_test.shape[0]
    X_test_scaled = scaler.transform(X_test.reshape(-1, n_features))
    X_test_scaled = X_test_scaled.reshape(n_test, seq_len, n_features)

    # Split training into train/val (80/20)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
    )

    # Convert to PyTorch tensors
    train_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                             torch.tensor(y_tr, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                           torch.tensor(y_val, dtype=torch.float32))
    test_ds = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32),
                            torch.tensor(y_test, dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    test_loader = DataLoader(test_ds, batch_size=32)

    results = {}
    EPOCHS = 50
    PATIENCE = 10
    LR_PATIENCE = 5

    for model_name in model_names:
        print(f"\n{'='*80}")
        print(f"Training {model_name}")
        print(f"{'='*80}")

        model = build_model(model_name, n_features, seq_len=seq_len).to(device)
        print(model)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=LR_PATIENCE, min_lr=1e-6
        )

        # Training loop with early stopping
        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0
        history = {'loss': [], 'val_loss': [], 'auc': [], 'val_auc': []}

        for epoch in range(EPOCHS):
            train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_auc, _ = _evaluate(model, val_loader, criterion, device)
            _, train_auc, _ = _evaluate(model, train_loader, criterion, device)

            history['loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['auc'].append(train_auc)
            history['val_auc'].append(val_auc)

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{EPOCHS}  "
                      f"loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                      f"val_auc={val_auc:.4f}")

            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        # Restore best weights
        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(device)

        # Evaluate on test set
        _, _, y_pred_proba = _evaluate(model, test_loader, criterion, device)
        y_pred = (y_pred_proba >= 0.5).astype(int)

        roc_auc = roc_auc_score(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)


        print(f"\n{model_name} Results:")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"PR-AUC: {pr_auc:.4f}")
        print("\nClassification Report:")
        report_dict = classification_report(y_test, y_pred, target_names=['Inactive', 'Active'], output_dict=True)
        print(classification_report(y_test, y_pred, target_names=['Inactive', 'Active']))

        # Save classification report to CSV
        report_df = pd.DataFrame(report_dict).transpose()
        report_csv_path = os.path.join(output_dir, f'{model_name}_classification_report.csv')
        report_df.to_csv(report_csv_path, index=True)

        # Save model
        torch.save(model.state_dict(), os.path.join(output_dir, f'{model_name}_model.pt'))

        # Plot training history
        plot_training_history(history, model_name, output_dir)

        results[model_name] = {
            'model': model,
            'history': history,
            'y_pred_proba': y_pred_proba,
            'y_pred': y_pred,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'classification_report': report_dict,
            'classification_report_csv': report_csv_path
        }

    return results, X_test, y_test, scaler


def plot_training_history(history, model_name, output_dir):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history['loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_name} - Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # AUC
    axes[1].plot(history['auc'], label='Train')
    axes[1].plot(history['val_auc'], label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUC')
    axes[1].set_title(f'{model_name} - AUC')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_training_history.png'), dpi=150)
    plt.close()


def create_future_activity_labels(repo_timeseries: dict, future_months: list) -> dict:
    """
    Derive binary activity label from the future window.

    label = 1  if the repo shows any of the following in at least one future month:
               push_events, issues_opened, prs_opened, prs_merged,
               or active_contributors > 0
    label = 0  otherwise (repo inactive / missing in all future months)

    Args:
        repo_timeseries: dict  repo_key -> {month_str: activity_dict}
        future_months:   list  month strings to treat as the future label window

    Returns:
        dict  repo_key -> int (0 or 1)
    """
    labels = {}
    for repo, monthly_data in repo_timeseries.items():
        active = False
        for month in future_months:
            if month not in monthly_data:
                continue
            act = monthly_data[month]
            if (
                (act.get('push_events') or 0) > 0
                or (act.get('issues_opened') or 0) > 0
                or (act.get('prs_opened') or 0) > 0
                or (act.get('prs_merged') or 0) > 0
                or len(act.get('active_contributors', [])) > 0
            ):
                active = True
                break
        labels[repo] = 1 if active else 0
    return labels


def parse_args():
    parser = argparse.ArgumentParser(
        description='Advanced Maintenance Prediction with Deep Learning Models',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--models', '-m',
        nargs='+',
        choices=ALL_MODELS,
        default=None,
        help=(
            'Models to train. Choose one or more from:\n'
            '  LSTM            - Long Short-Term Memory (recurrent)\n'
            '  GRU             - Gated Recurrent Unit (recurrent)\n'
            '  Attention_LSTM  - LSTM with attention mechanism\n'
            '  TCN             - Temporal Convolutional Network\n'
            '  Transformer     - Transformer Encoder (self-attention)\n'
            '  Informer        - Informer with ProbSparse attention (AAAI 2021)\n'
            '  NBEATS          - N-BEATS residual stacks (ICLR 2020)\n'
            '  Mamba           - Selective State Space Model (2023)\n'
            'Default: all models'
        )
    )
    parser.add_argument(
        '--list-models', action='store_true',
        help='List all available models and exit'
    )
    parser.add_argument(
        '--epochs', type=int, default=50,
        help='Maximum training epochs (default: 50)'
    )
    parser.add_argument(
        '--seq-length', type=int, default=None,
        help=f'Sequence length / number of months to use (default: {SEQUENCE_LENGTH})'
    )
    parser.add_argument(
        '--batch-size', type=int, default=32,
        help='Training batch size (default: 32)'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Handle --list-models
    if args.list_models:
        print("Available models:")
        descriptions = {
            'LSTM': 'Long Short-Term Memory — captures long-term sequential dependencies',
            'GRU': 'Gated Recurrent Unit — lighter alternative to LSTM',
            'Attention_LSTM': 'LSTM + learned attention over time steps',
            'TCN': 'Temporal Convolutional Network — parallel causal convolutions',
            'Transformer': 'Transformer Encoder — multi-head self-attention (Vaswani et al. 2017)',
            'Informer': 'ProbSparse attention for efficient long sequences (Zhou et al. AAAI 2021)',
            'NBEATS': 'Neural Basis Expansion — residual FC stacks (Oreshkin et al. ICLR 2020)',
            'Mamba': 'Selective State Space Model — linear-time sequence modeling (Gu & Dao 2023)',
        }
        for name in ALL_MODELS:
            print(f"  {name:20s} {descriptions[name]}")
        return

    print("=" * 80)
    print("Mamba Activity Prediction — Temporal Split Workflow")
    print("=" * 80)
    print(f"  Training window : first {TRAIN_MONTHS} months (input features)")
    print(f"  Label window    : next  {LABEL_MONTHS} months  (activity label)")
    print(f"  Total loaded    : {TOTAL_MONTHS} months")
    print(f"  Max epochs      : {args.epochs}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(result_dir, f"advanced_prediction_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"  Output directory: {output_dir}\n")

    # ------------------------------------------------------------------ #
    # 1. Load data
    # ------------------------------------------------------------------ #
    crates_df = load_crates(os.path.join(dump_dir, "crates.csv"))

    # Load the most recent TOTAL_MONTHS (24) monthly delta files
    repo_timeseries, all_months = load_monthly_activity(monthly_dir, n_months=TOTAL_MONTHS)

    if len(all_months) < TOTAL_MONTHS:
        print(f"[WARN] Only {len(all_months)} months available; expected {TOTAL_MONTHS}. "
              "Proceeding with available data.")

    # Temporal split
    train_months  = all_months[:TRAIN_MONTHS]
    future_months = all_months[TRAIN_MONTHS:]

    print(f"Training window : {train_months[0]}  →  {train_months[-1]}  ({len(train_months)} months)")
    if future_months:
        print(f"Label window    : {future_months[0]}  →  {future_months[-1]}  ({len(future_months)} months)")
    else:
        print("[WARN] No future months available for labelling — all data used as input only.")

    # ------------------------------------------------------------------ #
    # 2. Build sequence matrices
    # ------------------------------------------------------------------ #
    metrics_list = [
        'push_events', 'issues_opened', 'issues_closed',
        'prs_opened', 'prs_closed', 'prs_merged',
        'issue_comments', 'pr_review_comments',
        'releases_published', 'watch_started',
        'active_contributors_count', 'days_since_last_release',
    ]

    print("\n" + "=" * 60)
    print("Building sequence matrices …")

    # Full 24-month matrix — used for the final prediction step
    X_full, repo_names = create_sequence_matrix(repo_timeseries, all_months, metrics_list)
    print(f"Full sequence matrix  : {X_full.shape}  (samples × months × features)")

    # Training input: first TRAIN_MONTHS columns only
    X_train_input = X_full[:, :TRAIN_MONTHS, :]
    print(f"Training input slice  : {X_train_input.shape}  (samples × {TRAIN_MONTHS} × features)")

    # ------------------------------------------------------------------ #
    # 3. Build activity labels from future LABEL_MONTHS window
    # ------------------------------------------------------------------ #
    future_labels_map = create_future_activity_labels(repo_timeseries, future_months)
    y = np.array([future_labels_map.get(repo, 0) for repo in repo_names])
    print(f"\nLabel distribution  →  Active={int(np.sum(y == 1))},  "
          f"Inactive={int(np.sum(y == 0))}")

    # ------------------------------------------------------------------ #
    # 4. Build repo → crate_name lookup
    # ------------------------------------------------------------------ #
    repo_to_crate_name: dict = {}
    for _, row in crates_df.iterrows():
        repo_key = normalize_repo_url(row['repository'])
        if repo_key:
            repo_to_crate_name[repo_key] = str(row['name'])

    # ------------------------------------------------------------------ #
    # 5. Train Mamba model on 20-month sequences
    # ------------------------------------------------------------------ #
    if not HAS_PYTORCH:
        print("PyTorch not available — cannot train Mamba model. Exiting.")
        return

    print("\n" + "=" * 60)
    print("Training Mamba model on first 20 months …")
    dl_results, X_test, y_test, scaler = train_deep_learning_models(
        X_train_input, y, output_dir, model_names=['Mamba']
    )

    if 'Mamba' not in dl_results:
        print("[ERROR] Mamba model training failed. Exiting.")
        return

    mamba_result = dl_results['Mamba']
    mamba_model  = mamba_result['model']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mamba_model = mamba_model.to(device)
    mamba_model.eval()

    scaler_path = os.path.join(output_dir, 'Mamba_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    metadata_path = os.path.join(output_dir, 'Mamba_prediction_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'model_name': 'Mamba',
                'metrics_list': metrics_list,
                'train_months': TRAIN_MONTHS,
                'label_months': LABEL_MONTHS,
                'total_months': TOTAL_MONTHS,
                'model_file': 'Mamba_model.pt',
                'scaler_file': 'Mamba_scaler.pkl',
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved Mamba scaler: {scaler_path}")
    print(f"Saved Mamba prediction metadata: {metadata_path}")

    print(f"\nMamba test-set  →  ROC-AUC={mamba_result['roc_auc']:.4f}, "
          f"PR-AUC={mamba_result['pr_auc']:.4f}")

    # ------------------------------------------------------------------ #
    # 6. Predict on all 24 months for every repo
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print(f"Predicting activity probabilities using all {TOTAL_MONTHS} months …")

    n_samples, seq_len_full, n_features = X_full.shape

    # Apply the same per-feature MinMaxScaler (fit on the 20-month training data)
    X_pred_scaled = scaler.transform(X_full.reshape(-1, n_features))
    X_pred_scaled = X_pred_scaled.reshape(n_samples, seq_len_full, n_features)

    pred_dataset = TensorDataset(torch.tensor(X_pred_scaled, dtype=torch.float32))
    pred_loader  = DataLoader(pred_dataset, batch_size=64, shuffle=False)

    all_probs: list = []
    with torch.no_grad():
        for (X_batch,) in pred_loader:
            X_batch = X_batch.to(device)
            probs = mamba_model(X_batch)
            all_probs.append(probs.cpu().numpy())

    probabilities = np.concatenate(all_probs)  # shape (n_samples,)

    # ------------------------------------------------------------------ #
    # 7. Save results as CSV: crate_name, activity_probability
    # ------------------------------------------------------------------ #
    rows = []
    for i, repo in enumerate(repo_names):
        crate_name = repo_to_crate_name.get(repo, "")
        if crate_name:
            rows.append({
                'crate_name': crate_name,
                'activity_probability': float(probabilities[i]),
            })

    result_df = (
        pd.DataFrame(rows)
        .sort_values('activity_probability', ascending=False)
        .reset_index(drop=True)
    )

    result_csv = os.path.join(output_dir, 'mamba_activity_prediction.csv')
    result_df.to_csv(result_csv, index=False)
    print(f"\nSaved activity prediction CSV ({len(result_df)} crates): {result_csv}")
    print(result_df.head(20).to_string(index=False))

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
