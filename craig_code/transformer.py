import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from scipy.ndimage import label
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT_PATH = os.path.join(PROJECT_DIR, "best_transformer_model.pth")

class TimeSeriesDataset(Dataset):
    """Dataset for time series with sliding window"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerAnomalyDetector(nn.Module):
    """Transformer model for anomaly detection"""
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super(TransformerAnomalyDetector, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # Use the last time step for prediction
        x = x[:, -1, :]
        x = self.fc(x)
        return x.squeeze(-1)


def create_sequences(data, labels, window_size=50, stride=1):
    """Create sliding window sequences from time series data"""
    X, y = [], []
    for i in range(0, len(data) - window_size + 1, stride):
        X.append(data[i:i + window_size])
        # Label is 1 if ANY point in the window is anomalous
        y.append(1 if labels[i:i + window_size].sum() > 0 else 0)
    return np.array(X), np.array(y)


def apply_smote_to_sequences(X, y, random_state=42):
    """Apply SMOTE to sequence data by flattening and reshaping"""
    original_shape = X.shape
    n_samples = original_shape[0]
    
    # Flatten sequences for SMOTE
    X_flat = X.reshape(n_samples, -1)
    
    # Apply SMOTE
    smote = SMOTE(random_state=random_state, k_neighbors=min(5, sum(y) - 1) if sum(y) > 1 else 1)
    X_resampled, y_resampled = smote.fit_resample(X_flat, y)
    
    # Reshape back to sequences
    X_resampled = X_resampled.reshape(-1, original_shape[1], original_shape[2])
    
    return X_resampled, y_resampled


def predict_sequences(model, data_loader, device='cpu'):
    """Run inference and return binary window predictions."""
    model.eval()
    all_preds = []
    with torch.no_grad():
        for X_batch, _ in data_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            all_preds.extend((outputs > 0.5).cpu().numpy())
    return np.array(all_preds)


def windows_to_point_predictions(window_preds, series_len, window_size, stride):
    """
    Map window-level predictions back to point-level predictions.
    A point is marked anomalous if any covering window is predicted anomalous.
    """
    point_preds = np.zeros(series_len, dtype=int)
    idx = 0
    for start in range(0, series_len - window_size + 1, stride):
        if idx >= len(window_preds):
            break
        end = start + window_size
        if window_preds[idx] == 1:
            point_preds[start:end] = 1
        idx += 1
    
    # If we still have leftover window predictions (shouldn't happen), apply them to the tail
    while idx < len(window_preds):
        if window_preds[idx] == 1:
            point_preds[-window_size:] = 1
        idx += 1
    return point_preds


def identify_events(labels):
    """Identify continuous anomalous segments (events)"""
    labeled_array, num_events = label(labels)
    events = []
    for event_id in range(1, num_events + 1):
        event_indices = np.where(labeled_array == event_id)[0]
        events.append((event_indices[0], event_indices[-1]))
    return events


def calculate_event_wise_metrics(y_true, y_pred, beta=0.5):
    """
    Calculate event-wise precision, recall, and F-beta score.
    An event is detected if at least one point in the event is predicted as anomalous.
    """
    true_events = identify_events(y_true)
    pred_events = identify_events(y_pred)
    
    if len(true_events) == 0:
        return 0, 0, 0
    
    # Check which true events were detected
    detected_events = 0
    for true_start, true_end in true_events:
        # Check if any prediction overlaps with this event
        if y_pred[true_start:true_end + 1].sum() > 0:
            detected_events += 1
    
    # Event-wise recall: what fraction of true events were detected
    recall = detected_events / len(true_events) if len(true_events) > 0 else 0
    
    # Event-wise precision: what fraction of predicted events correspond to true events
    if len(pred_events) == 0:
        precision = 0
    else:
        valid_pred_events = 0
        for pred_start, pred_end in pred_events:
            # Check if this prediction overlaps with any true event
            for true_start, true_end in true_events:
                if not (pred_end < true_start or pred_start > true_end):
                    valid_pred_events += 1
                    break
        precision = valid_pred_events / len(pred_events)
    
    # F-beta score (F0.5 emphasizes precision)
    if precision + recall == 0:
        f_beta = 0
    else:
        f_beta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
    
    return precision, recall, f_beta


class FocalLoss(nn.Module):
    """Focal Loss to handle class imbalance and prevent collapse to all zeros"""
    def __init__(self, alpha=0.75, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = nn.BCELoss(reduction='none')(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device='cpu', checkpoint_path=DEFAULT_CHECKPOINT_PATH):
    """Train the model with early stopping"""
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        for X_batch, y_batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend((outputs > 0.5).cpu().numpy())
            train_labels.extend(y_batch.cpu().numpy())
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                val_preds.extend((outputs > 0.5).cpu().numpy())
                val_labels.extend(y_batch.cpu().numpy())
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Check for prediction collapse
        pred_rate = np.mean(train_preds)
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Pred Rate: {pred_rate:.4f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # Load best model
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model


def evaluate_model(model, test_loader, y_true_pointwise, series_len, window_size, stride, device='cpu'):
    """Evaluate model with event-wise metrics, returning both window and point predictions."""
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            all_preds.extend((outputs > 0.5).cpu().numpy())
    
    window_preds = np.array(all_preds)
    point_preds = windows_to_point_predictions(window_preds, series_len, window_size, stride)
    
    precision, recall, f_beta = calculate_event_wise_metrics(y_true_pointwise, point_preds, beta=0.5)
    
    return precision, recall, f_beta, window_preds, point_preds


def main():
    """Main training and evaluation pipeline"""
    
    # Example: Generate synthetic data (replace with your actual data loading)
    print("Loading and preparing data...")
    # Simulated data - replace this with your actual dataframe
    # n_samples = 10000
    # n_channels = 10
    
    # Create synthetic time series data
    # np.random.seed(42)
    # data = np.random.randn(n_samples, n_channels)
    
    # # Create synthetic anomalies (5% of data in bursts)
    # is_anomaly = np.zeros(n_samples)
    # n_events = 20
    # for _ in range(n_events):
    #     start = np.random.randint(0, n_samples - 100)
    #     length = np.random.randint(5, 50)
    #     is_anomaly[start:start + length] = 1
    #     # Add signal to anomalous regions
    #     data[start:start + length] += np.random.randn(length, n_channels) * 3
    
    # print(f"Total anomalous points: {is_anomaly.sum()} ({100*is_anomaly.mean():.2f}%)")
    # print(f"Number of anomalous events: {len(identify_events(is_anomaly))}")

    parquet_train = "/home/cbelshe/CMPE-258/final_project/train.parquet"
    parquet_test = "/home/cbelshe/CMPE-258/final_project/test.parquet"
    train_df = pd.read_parquet(parquet_train)

    channels_to_use_file = "/home/cbelshe/CMPE-258/final_project/target_channels.csv"
    channels = pd.read_csv(channels_to_use_file)
    channels = list(channels["target_channels"])
    n_channels = len(channels)

    data = np.array(train_df[channels].values)
    is_anomaly = np.array(train_df["is_anomaly"].values)
    
    test_df = pd.read_parquet(parquet_test)
    full_test_data = np.array(test_df[channels].values)
    print("Full test data shape: ", full_test_data.shape)
    full_test_fake_labels = np.zeros(full_test_data.shape[0], dtype=int)
    
    # Split data
    train_idx, test_idx = train_test_split(range(len(data)), test_size=0.2, random_state=42)
    train_data, test_data = data[train_idx], data[test_idx]
    train_labels, test_labels = is_anomaly[train_idx], is_anomaly[test_idx]
    
    # Normalize data
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)
    full_test_data = scaler.transform(full_test_data)
    
    # Create sequences
    print("\nCreating sequences...")
    window_size = 50
    train_stride = 10
    eval_stride = 1
    X_train, y_train = create_sequences(train_data, train_labels, window_size=window_size, stride=train_stride)
    X_test, y_test = create_sequences(test_data, test_labels, window_size=window_size, stride=eval_stride)
    
    print(f"Training sequences: {len(X_train)}, Anomalous: {y_train.sum()} ({100*y_train.mean():.2f}%)")
    print(f"Test sequences: {len(X_test)}, Anomalous: {y_test.sum()} ({100*y_test.mean():.2f}%)")
    
    # Apply SMOTE
    print("\nApplying SMOTE...")
    X_train_balanced, y_train_balanced = apply_smote_to_sequences(X_train, y_train)
    print(f"After SMOTE - Training sequences: {len(X_train_balanced)}, "
          f"Anomalous: {y_train_balanced.sum()} ({100*y_train_balanced.mean():.2f}%)")
    
    # Create validation split
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_balanced, y_train_balanced, test_size=0.15, random_state=42
    )
    
    # Create datasets and loaders
    train_dataset = TimeSeriesDataset(X_train_final, y_train_final)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    model = TransformerAnomalyDetector(
        input_dim=n_channels,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.2
    )#.to(device)
    
    # print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # # Use Focal Loss to prevent collapse
    # criterion = FocalLoss(alpha=0.75, gamma=2.0)
    # optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # # Train model
    # print("\nTraining model...")
    # model = train_model(
    #     model,
    #     train_loader,
    #     val_loader,
    #     criterion,
    #     optimizer,
    #     num_epochs=100,
    #     device=device,
    #     checkpoint_path=DEFAULT_CHECKPOINT_PATH,
    # )

    state_dict = torch.load("best_transformer_model.pth")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Evaluate on test set
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    precision, recall, f_beta, window_preds, point_preds = evaluate_model(
        model, test_loader, test_labels, len(test_labels), window_size, eval_stride, device
    )
    
    print(f"\nEvent-wise Metrics (F0.5 Score):")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F0.5 Score: {f_beta:.4f}")
    
    print(f"\nPrediction Statistics:")
    print(f"  Predicted anomalies: {point_preds.sum()} ({100*point_preds.mean():.2f}%)")
    print(f"  True anomalies: {test_labels.sum()} ({100*test_labels.mean():.2f}%)")
    
    # Check for collapse
    if point_preds.sum() == 0:
        print("\n⚠️  WARNING: Model collapsed to predicting all zeros!")
    elif point_preds.sum() == len(point_preds):
        print("\n⚠️  WARNING: Model collapsed to predicting all ones!")
    else:
        print("\n✓ Model is making varied predictions")


    # test_data_sequences, test_data_labels = create_sequences(
    #     full_test_data, full_test_fake_labels, window_size=window_size, stride=eval_stride
    # )
    # full_test_dataset = TimeSeriesDataset(test_data_sequences, test_data_labels)
    # full_test_loader = DataLoader(full_test_dataset, batch_size=64, shuffle=False)
    # full_window_preds = predict_sequences(model, full_test_loader, device)
    # full_point_preds = windows_to_point_predictions(
    #     full_window_preds, test_data_sequences.shape[0] + window_size-1, window_size, eval_stride
    # )
    
    # print(f"\nExternal test prediction statistics:")
    # print(f"  Predicted anomalies: {full_point_preds.sum()} ({100*full_point_preds.mean():.2f}%)")
    # print(
    #     f"  Predicted normal: {len(full_point_preds) - full_point_preds.sum()} "
    #     f"({100*(1-full_point_preds.mean()):.2f}%)"
    # )

    # test_df["is_anomaly"] = full_point_preds
    # test_df = test_df[["id", "is_anomaly"]]
    # test_df.to_csv("transformer_predictions.csv", index=False)
    
    return model, scaler


if __name__ == "__main__":
    model, scaler = main()
    print("\nTraining complete! Model saved to 'best_model.pth'")

"""

Creating sequences...
Training sequences: 1178261, Anomalous: 1173596 (99.60%)
Test sequences: 2945616, Anomalous: 2934283 (99.62%)

Applying SMOTE...
After SMOTE - Training sequences: 2347192, Anomalous: 1173596 (50.00%)

Using device: cuda

==================================================
EVALUATION RESULTS
==================================================

Event-wise Metrics (F0.5 Score):
  Precision: 1.0000
  Recall: 1.0000
  F0.5 Score: 1.0000

Prediction Statistics:
  Predicted anomalies: 2945610 (100.00%)
  True anomalies: 309194 (10.50%)

✓ Model is making varied predictions

"""