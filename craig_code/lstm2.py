import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import Tuple, List
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class TimeSeriesDataset(Dataset):
    """Dataset for multivariate time series with sliding windows"""
    def __init__(self, data: np.ndarray, labels: np.ndarray, window_size: int = 50, stride: int = 1):
        self.window_size = window_size
        self.stride = stride
        self.data = data
        self.labels = labels
        
        # Create windows
        self.windows = []
        self.window_labels = []
        
        for i in range(0, len(data) - window_size + 1, stride):
            window = data[i:i + window_size]
            # Label is 1 if any point in window is anomalous
            label = np.max(labels[i:i + window_size])
            self.windows.append(window)
            self.window_labels.append(label)
        
        self.windows = np.array(self.windows, dtype=np.float32)
        self.window_labels = np.array(self.window_labels, dtype=np.float32)
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.windows[idx]), torch.FloatTensor([self.window_labels[idx]])


class LSTMAutoencoder(nn.Module):
    """LSTM-based autoencoder for anomaly detection"""
    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(LSTMAutoencoder, self).__init__()
        
        self.n_features = n_features
        self.hidden_size = hidden_size
        
        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Decoder
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.output_layer = nn.Linear(hidden_size, n_features)
        
    def forward(self, x):
        # Encode
        encoded, (hidden, cell) = self.encoder_lstm(x)
        
        # Use the encoded representation for all timesteps
        batch_size, seq_len, _ = x.shape
        decoder_input = encoded[:, -1, :].unsqueeze(1).repeat(1, seq_len, 1)
        
        # Decode
        decoded, _ = self.decoder_lstm(decoder_input, (hidden, cell))
        output = self.output_layer(decoded)
        
        return output


class LSTMVariationalAutoencoder(nn.Module):
    """LSTM VAE to prevent collapse via latent regularization"""
    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        
        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Latent space projections
        self.fc_mu = nn.Linear(hidden_size, hidden_size)
        self.fc_logvar = nn.Linear(hidden_size, hidden_size)
        
        # Decoder
        self.decoder_lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.output_layer = nn.Linear(hidden_size, n_features)
        self.num_layers = num_layers
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        batch_size, seq_len, n_features = x.shape
        
        # Encode
        _, (hidden, cell) = self.encoder_lstm(x)
        
        # Get mu and logvar from last layer's hidden state
        h = hidden[-1]  # (batch, hidden_size)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Reshape for decoder
        hidden_vae = z.unsqueeze(0).repeat(self.num_layers, 1, 1)
        
        # Decode
        decoder_input = torch.zeros(batch_size, seq_len, n_features, device=x.device)
        decoded, _ = self.decoder_lstm(decoder_input, (hidden_vae, cell))
        output = self.output_layer(decoded)
        
        return output, mu, logvar

# Modified loss function
def vae_loss(recon_x, x, mu, logvar, beta=0.01):
    # Reconstruction loss
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl_loss

class LSTMAutoencoderWithTeacherForcing(nn.Module):
    """LSTM autoencoder with teacher forcing to prevent collapse"""
    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 2, 
                 dropout: float = 0.2, teacher_forcing_ratio: float = 0.5):
        super().__init__()
        
        self.teacher_forcing_ratio = teacher_forcing_ratio
        
        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Decoder
        self.decoder_lstm = nn.LSTM(
            input_size=n_features,  # Takes previous output or input
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.output_layer = nn.Linear(hidden_size, n_features)
        
    def forward(self, x, use_teacher_forcing=None):
        batch_size, seq_len, n_features = x.shape
        
        # Encode
        _, (hidden, cell) = self.encoder_lstm(x)
        
        # Decide whether to use teacher forcing
        if use_teacher_forcing is None:
            use_teacher_forcing = self.training and (torch.rand(1).item() < self.teacher_forcing_ratio)
        
        if use_teacher_forcing:
            # Teacher forcing: use actual input
            decoder_input = x
        else:
            # No teacher forcing: start from zeros
            decoder_input = torch.zeros(batch_size, seq_len, n_features, device=x.device)
        
        # Decode
        decoded, _ = self.decoder_lstm(decoder_input, (hidden, cell))
        output = self.output_layer(decoded)
        
        return output

class TransformerAnomalyDetector(nn.Module):
    """Transformer-based model for anomaly detection"""
    def __init__(self, n_features: int, d_model: int = 64, nhead: int = 4, 
                 num_layers: int = 2, dropout: float = 0.2):
        super(TransformerAnomalyDetector, self).__init__()
        
        self.n_features = n_features
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(n_features, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Output projection for reconstruction
        self.output_projection = nn.Linear(d_model, n_features)
        
    def forward(self, x):
        # Project input
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transform
        x = self.transformer_encoder(x)
        
        # Project to output
        output = self.output_projection(x)
        
        return output


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)


def compute_event_wise_f_beta(y_true: np.ndarray, y_pred: np.ndarray, beta: float = 0.5) -> Tuple[float, dict]:
    """
    Compute event-wise F-beta score.
    
    Args:
        y_true: Binary array of ground truth labels
        y_pred: Binary array of predictions
        beta: Beta parameter for F-beta score (0.5 for F0.5)
    
    Returns:
        f_beta: F-beta score
        metrics: Dictionary with precision, recall, TP, FP, FN
    """
    # Find anomaly events in ground truth
    true_events = find_events(y_true)
    pred_events = find_events(y_pred)
    
    # Count true positives: true events with at least one detected point
    tp = 0
    for start, end in true_events:
        if np.any(y_pred[start:end+1] == 1):
            tp += 1
    
    # False negatives: true events with no detected points
    fn = len(true_events) - tp
    
    # False positives: predicted events that don't overlap with any true event
    fp = 0
    for pred_start, pred_end in pred_events:
        overlaps = False
        for true_start, true_end in true_events:
            # Check if there's any overlap
            if not (pred_end < true_start or pred_start > true_end):
                overlaps = True
                break
        if not overlaps:
            fp += 1
    
    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # F-beta score
    if precision + recall > 0:
        f_beta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
    else:
        f_beta = 0
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'n_true_events': len(true_events),
        'n_pred_events': len(pred_events)
    }
    
    return f_beta, metrics


def find_events(labels: np.ndarray) -> List[Tuple[int, int]]:
    """Find continuous segments of anomalies (events)"""
    events = []
    in_event = False
    start = 0
    
    for i in range(len(labels)):
        if labels[i] == 1 and not in_event:
            start = i
            in_event = True
        elif labels[i] == 0 and in_event:
            events.append((start, i - 1))
            in_event = False
    
    # Handle case where sequence ends in an event
    if in_event:
        events.append((start, len(labels) - 1))
    
    return events


def find_optimal_threshold(reconstruction_errors: np.ndarray, y_true: np.ndarray, 
                          beta: float = 0.5, percentiles: np.ndarray = None) -> Tuple[float, float]:
    """
    Find optimal threshold for anomaly detection based on event-wise F-beta score.
    
    Args:
        reconstruction_errors: Array of reconstruction errors
        y_true: Ground truth labels
        beta: Beta parameter for F-beta score
        percentiles: Array of percentile values to try as thresholds
    
    Returns:
        best_threshold: Optimal threshold value
        best_score: Best F-beta score achieved
    """
    # if percentiles is None:
        # percentiles = np.linspace(50, 99.9, 100)
    
    percentiles = np.linspace(0.5, 0.99, 100)
    
    best_threshold = 0
    best_score = 0
    
    for percentile in percentiles:
        # threshold = np.percentile(reconstruction_errors, percentile)
        threshold = percentile
        y_pred = (reconstruction_errors > threshold).astype(int)
        
        f_beta, _ = compute_event_wise_f_beta(y_true, y_pred, beta=beta)
        
        if f_beta > best_score:
            best_score = f_beta
            best_threshold = threshold
        if f_beta == best_score:
            print(f"scores are the same: Threshold: {threshold}, F-beta: {f_beta}")
    
    return best_threshold, best_score


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                epochs: int = 50, learning_rate: float = 0.001, device: str = 'cuda',
                patience: int = 10):
    """Train the anomaly detection model"""
    model = model.to(device)
    criterion = nn.MSELoss()
    # criterion = nn.KLDivLoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_data, _ in tqdm(train_loader, desc=f"Training epoch {epoch+1}", leave=False):
            batch_data = batch_data.to(device)
            
            optimizer.zero_grad()
            output = model(batch_data)
            loss = criterion(output, batch_data)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_data, _ in val_loader:
                batch_data = batch_data.to(device)
                output = model(batch_data)
                loss = criterion(output, batch_data)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses


def compute_reconstruction_errors(model: nn.Module, data: np.ndarray, 
                                  window_size: int, stride: int, device: str = 'cuda') -> np.ndarray:
    """Compute reconstruction errors for each point in the data"""
    model.eval()
    
    # Create dataset
    dummy_labels = np.zeros(len(data))
    dataset = TimeSeriesDataset(data, dummy_labels, window_size=window_size, stride=stride)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    all_errors = []
    
    with torch.no_grad():
        for batch_data, _ in dataloader:
            batch_data = batch_data.to(device)
            output = model(batch_data)
            
            # Compute MSE for each window
            errors = torch.mean((batch_data - output) ** 2, dim=(1, 2))
            all_errors.extend(errors.cpu().numpy())
    
    all_errors = np.array(all_errors)
    
    # Map window errors back to point-wise errors
    point_errors = np.zeros(len(data))
    point_counts = np.zeros(len(data))
    
    for i, error in enumerate(all_errors):
        start_idx = i * stride
        end_idx = start_idx + window_size
        point_errors[start_idx:end_idx] += error
        point_counts[start_idx:end_idx] += 1
    
    # Average errors for points covered by multiple windows
    point_counts[point_counts == 0] = 1
    point_errors = point_errors / point_counts
    
    return point_errors


# def plot_results(data: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, 
#                 reconstruction_errors: np.ndarray, threshold: float):
#     """Plot the results of anomaly detection"""
#     fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
#     # Plot first few channels
#     n_channels_to_plot = min(3, data.shape[1])
#     for i in range(n_channels_to_plot):
#         axes[0].plot(data.iloc[:, i], label=f'Channel {i+1}', alpha=0.7)
#     axes[0].set_title('Time Series Data (First Few Channels)')
#     axes[0].set_ylabel('Value')
#     axes[0].legend()
#     axes[0].grid(True, alpha=0.3)
    
#     # Plot reconstruction errors
#     axes[1].plot(reconstruction_errors, color='blue', alpha=0.7)
#     axes[1].axhline(y=threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.4f}')
#     axes[1].set_title('Reconstruction Errors')
#     axes[1].set_ylabel('Error')
#     axes[1].legend()
#     axes[1].grid(True, alpha=0.3)
    
#     # Plot ground truth
#     axes[2].fill_between(range(len(y_true)), 0, y_true, alpha=0.5, color='red', label='True Anomalies')
#     axes[2].set_title('Ground Truth Anomalies')
#     axes[2].set_ylabel('Anomaly')
#     axes[2].set_ylim(-0.1, 1.1)
#     axes[2].legend()
#     axes[2].grid(True, alpha=0.3)
    
#     # Plot predictions
#     axes[3].fill_between(range(len(y_pred)), 0, y_pred, alpha=0.5, color='blue', label='Predicted Anomalies')
#     axes[3].set_title('Predicted Anomalies')
#     axes[3].set_xlabel('Time')
#     axes[3].set_ylabel('Anomaly')
#     axes[3].set_ylim(-0.1, 1.1)
#     axes[3].legend()
#     axes[3].grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig('/mnt/user-data/outputs/anomaly_detection_results.png', dpi=150, bbox_inches='tight')
#     plt.close()


def main(test_size: float = 0.2, window_size: int = 50, 
         model_type: str = 'lstm', epochs: int = 50):
    """
    Main function to train and evaluate anomaly detection model.
    
    Args:
        df: DataFrame with time series channels and 'is_anomaly' column
        test_size: Fraction of data to use for testing
        window_size: Size of sliding window for sequences
        model_type: Type of model ('lstm' or 'transformer')
        epochs: Number of training epochs
    
    Returns:
        model: Trained model
        results: Dictionary with evaluation metrics
    """
    print("=" * 80)
    print("ANOMALY DETECTION WITH EVENT-WISE F0.5 OPTIMIZATION")
    print("=" * 80)
    
    # Separate features and labels
    # feature_cols = [col for col in df.columns if col != 'is_anomaly']
    # X = df[feature_cols].values
    # y = df['is_anomaly'].values
    parquet_train = "/home/cbelshe/CMPE-258/final_project/train.parquet"
    parquet_test = "/home/cbelshe/CMPE-258/final_project/test.parquet"
    train_df = pd.read_parquet(parquet_train)

    channels_to_use_file = "/home/cbelshe/CMPE-258/final_project/target_channels.csv"
    channels = pd.read_csv(channels_to_use_file)
    channels = list(channels["target_channels"])

    X = train_df[channels].values
    y = train_df["is_anomaly"].values
    
    test_df = pd.read_parquet(parquet_test)
    X_test = test_df[channels].values
    
    print(f"\nData shape: {X.shape}")
    print(f"Number of channels: {len(channels)}")
    print(f"Anomaly rate: {y.mean():.4f}")
    
    # Find anomaly events
    events = find_events(y)
    print(f"Number of anomaly events: {len(events)}")
    
    # Split data (temporal split - no shuffling for time series)
    split_idx = int(len(X) * (1 - test_size))
    X_train_final, X_val = X[:split_idx], X[split_idx:]
    y_train_final, y_val = y[:split_idx], y[split_idx:]
    
    # Further split training into train/val
    # train_split_idx = int(len(X_train) * 0.8)
    # X_train_final, X_val = X_train[:train_split_idx], X_train[train_split_idx:]
    # y_train_final, y_val = y_train[:train_split_idx], y_train[train_split_idx:]
    
    print(f"\nTrain size: {len(X_train_final)}")
    print(f"Validation size: {len(X_val)}")
    print(f"Test size: {len(X_test)}")
    
    # Normalize data
    scaler = StandardScaler()
    X_train_final = scaler.fit_transform(X_train_final)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Create datasets
    train_dataset = TimeSeriesDataset(X_train_final, y_train_final, window_size=window_size, stride=10)
    val_dataset = TimeSeriesDataset(X_val, y_val, window_size=window_size, stride=10)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else print("No GPU available")
    print(f"\nUsing device: {device}")
    
    n_features = X.shape[1]
    
    if model_type == 'lstm':
        # model = LSTMAutoencoder(n_features=n_features, hidden_size=64, num_layers=2, dropout=0.2)
        # model = LSTMVariationalAutoencoder(n_features=n_features, hidden_size=64, num_layers=2, dropout=0.2)
        model = LSTMAutoencoderWithTeacherForcing(n_features=n_features, hidden_size=64, num_layers=2, dropout=0.2)
        print("\nModel: LSTM Autoencoder")
    else:
        model = TransformerAnomalyDetector(n_features=n_features, d_model=64, nhead=4, 
                                          num_layers=2, dropout=0.2)
        print("\nModel: Transformer")
    
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\n" + "-" * 80)
    print("TRAINING")
    print("-" * 80)
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, 
        epochs=epochs, learning_rate=0.001, 
        device=device, patience=15
    )
    
    # Compute reconstruction errors on validation set
    print("\n" + "-" * 80)
    print("FINDING OPTIMAL THRESHOLD")
    print("-" * 80)
    val_errors = compute_reconstruction_errors(model, X_val, window_size=window_size, 
                                               stride=1, device=device)
    
    # Find optimal threshold on validation set
    threshold, best_val_f05 = find_optimal_threshold(val_errors, y_val, beta=0.5)
    print(f"Optimal threshold: {threshold:.6f}")
    print(f"Validation F0.5 score: {best_val_f05:.4f}")
    
    # Evaluate on test set
    print("\n" + "-" * 80)
    print("TEST SET EVALUATION")
    print("-" * 80)
    test_errors = compute_reconstruction_errors(model, X_test, window_size=window_size, 
                                               stride=1, device=device)
    
    # Apply threshold
    y_pred = (test_errors > threshold).astype(int)

    test_df["is_anomaly"] = y_pred
    test_df = test_df[["id", "is_anomaly"]]
    test_df.to_csv("lstm2_predictions.csv", index=False)
    
    # Compute event-wise F0.5
    # f05_score, metrics = compute_event_wise_f_beta(y_test, y_pred, beta=0.5)
    
    # print(f"\nTest Set Results:")
    # print(f"Event-wise F0.5 Score: {f05_score:.4f}")
    # print(f"Precision: {metrics['precision']:.4f}")
    # print(f"Recall: {metrics['recall']:.4f}")
    # print(f"True Positives: {metrics['tp']} / {metrics['n_true_events']}")
    # print(f"False Positives: {metrics['fp']}")
    # print(f"False Negatives: {metrics['fn']}")
    
    # Plot results
    # print("\nGenerating visualization...")
    # Combine data for plotting
    # test_df = pd.DataFrame(scaler.inverse_transform(X_test), columns=feature_cols)
    # plot_results(test_df, y_test, y_pred, test_errors, threshold)
    # print("Visualization saved to: /mnt/user-data/outputs/anomaly_detection_results.png")
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'threshold': threshold,
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'window_size': window_size,
        'model_type': model_type,
        'n_features': n_features
    }, 'anomaly_detection_model.pth')
    print("Model saved to: anomaly_detection_model3.pth")
    
    results = {
        # 'f05_score': f05_score,
        # 'metrics': metrics,
        'threshold': threshold,
        'test_errors': test_errors,
        'y_pred': y_pred
    }
    
    return model, results


# Example usage
if __name__ == "__main__":
    print("This script provides a complete anomaly detection solution.")
    print("\nTo use it, load your DataFrame and call:")
    print("  model, results = main(df, test_size=0.2, window_size=50, model_type='lstm', epochs=50)")
    print("\nThe DataFrame should have:")
    print("  - Multiple columns representing different channels")
    print("  - One column named 'is_anomaly' with binary labels (0 or 1)")
    # model, results =main(test_size=0.2, window_size=50, model_type='lstm', epochs=5)
    model, results =main(test_size=0.2, window_size=50, model_type='transformer', epochs=5)
    print(results)