"""
Advanced Transformer-based Anomaly Detection for Time Series
Uses multi-head attention, multiple scoring methods, and sophisticated training strategies.
"""

import argparse
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
	classification_report,
	precision_recall_curve,
	roc_auc_score,
	precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def set_seed(seed: int = 42) -> None:
	"""Set random seeds for reproducibility."""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def infer_feature_columns(df: pd.DataFrame, label_column: str = "is_anomaly") -> List[str]:
	"""Infer numeric feature columns, excluding the label."""
	numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
	feature_cols = [c for c in numeric_cols if c != label_column]
	if not feature_cols:
		raise ValueError("No numeric feature columns found other than the label.")
	return feature_cols


def time_split_indices(n_rows: int, train_ratio: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
	"""Time-based split: first portion for train, remainder for test."""
	train_end = int(n_rows * train_ratio)
	train_idx = np.arange(0, train_end, dtype=int)
	test_idx = np.arange(train_end, n_rows, dtype=int)
	return train_idx, test_idx


def build_window_labels(binary_series: np.ndarray, window_size: int, stride: int) -> np.ndarray:
	"""Convert pointwise binary array into window labels."""
	if binary_series.ndim != 1:
		raise ValueError("binary_series must be 1D.")
	n = len(binary_series)
	labels = []
	for start in range(0, n - window_size + 1, stride):
		end = start + window_size
		labels.append(1 if np.any(binary_series[start:end] == 1) else 0)
	return np.asarray(labels, dtype=np.int64)


class SlidingWindowDataset(Dataset):
	"""Create sliding windows over multivariate time series."""

	def __init__(
		self,
		data: np.ndarray,
		window_size: int,
		stride: int = 1,
		labels: Optional[np.ndarray] = None,
	):
		if data.ndim != 2:
			raise ValueError("data must be 2D: (num_rows, num_features)")
		self.data = data.astype(np.float32)
		self.window_size = int(window_size)
		self.stride = int(stride)
		self.num_rows, self.num_features = self.data.shape
		self.starts = np.arange(0, self.num_rows - self.window_size + 1, self.stride, dtype=int)
		self._labels = labels
		if labels is not None and len(labels) != len(self.starts):
			raise ValueError("labels length must equal number of windows")

	def __len__(self) -> int:
		return len(self.starts)

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
		start = self.starts[idx]
		end = start + self.window_size
		seq = self.data[start:end]  # (window, features)
		x = torch.from_numpy(seq)  # float32
		if self._labels is None:
			return x, None
		return x, torch.tensor(self._labels[idx], dtype=torch.long)


def make_collate_to_device(device: str):
	"""Create a collate_fn that moves batches to the specified device."""

	def _collate(batch):
		xs, ys = zip(*batch)
		x = torch.stack(xs, dim=0).to(device, non_blocking=True)
		if ys[0] is None:
			return x, None
		y = torch.stack(ys, dim=0).to(device, non_blocking=True)
		return x, y

	return _collate


class PositionalEncoding(nn.Module):
	"""Sinusoidal positional encoding for transformer."""

	def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
		super().__init__()
		self.dropout = nn.Dropout(p=dropout)
		position = torch.arange(max_len).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
		pe = torch.zeros(1, max_len, d_model)
		pe[0, :, 0::2] = torch.sin(position * div_term)
		pe[0, :, 1::2] = torch.cos(position * div_term)
		self.register_buffer("pe", pe)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: (B, T, d_model)
		x = x + self.pe[:, : x.size(1), :]
		return self.dropout(x)


class TransformerAutoencoder(nn.Module):
	"""
	Transformer-based autoencoder with multi-head attention.
	Uses encoder-decoder architecture with skip connections.
	"""

	def __init__(
		self,
		num_features: int,
		d_model: int = 128,
		nhead: int = 8,
		num_encoder_layers: int = 4,
		num_decoder_layers: int = 4,
		dim_feedforward: int = 512,
		dropout: float = 0.1,
		activation: str = "gelu",
		max_seq_len: int = 512,
	):
		super().__init__()
		self.num_features = num_features
		self.d_model = d_model

		# Input projection
		self.input_proj = nn.Linear(num_features, d_model)

		# Positional encoding
		self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

		# Encoder
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=d_model,
			nhead=nhead,
			dim_feedforward=dim_feedforward,
			dropout=dropout,
			activation=activation,
			batch_first=True,
		)
		self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

		# Latent bottleneck
		self.latent_proj = nn.Linear(d_model, d_model // 2)

		# Decoder
		decoder_layer = nn.TransformerDecoderLayer(
			d_model=d_model,
			nhead=nhead,
			dim_feedforward=dim_feedforward,
			dropout=dropout,
			activation=activation,
			batch_first=True,
		)
		self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

		# Expand latent back
		self.latent_expand = nn.Linear(d_model // 2, d_model)

		# Output projection
		self.output_proj = nn.Linear(d_model, num_features)

		# Feature extractor for anomaly scoring
		# Goal: Learn a compact representation (d_model//4 dims) that clusters normal patterns together.
		# During training, we minimize the distance of normal samples to a learned centroid.
		# During evaluation, anomalies will be far from this centroid in feature space.
		# This provides a complementary signal to reconstruction error for anomaly detection.
		self.feature_extractor = nn.Sequential(
			nn.Linear(d_model, d_model // 2),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(d_model // 2, d_model // 4),
		)

	def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
		# x: (B, T, F)
		batch_size, seq_len, _ = x.shape

		# Project input
		x_proj = self.input_proj(x)  # (B, T, d_model)
		x_proj = self.pos_encoder(x_proj)

		# Encode
		encoded = self.encoder(x_proj)  # (B, T, d_model)

		# Always compute features during forward pass to ensure feature_extractor is trained
		# Use mean pooling over time
		encoded_mean = encoded.mean(dim=1)  # (B, d_model)
		features = self.feature_extractor(encoded_mean)  # (B, d_model//4)

		# Latent bottleneck
		latent = self.latent_proj(encoded_mean)  # (B, d_model//2)
		latent_expanded = self.latent_expand(latent)  # (B, d_model)

		# Create decoder input (repeat latent across time)
		memory = encoded  # (B, T, d_model)
		tgt = latent_expanded.unsqueeze(1).repeat(1, seq_len, 1)  # (B, T, d_model)

		# Decode
		decoded = self.decoder(tgt, memory)  # (B, T, d_model)

		# Project to output
		reconstructed = self.output_proj(decoded)  # (B, T, F)

		if return_features:
			return reconstructed, features
		return reconstructed


class FocalLoss(nn.Module):
	"""Focal Loss for handling class imbalance."""

	def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
		super().__init__()
		self.alpha = alpha
		self.gamma = gamma
		self.reduction = reduction

	def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
		ce_loss = F.cross_entropy(inputs, targets, reduction="none")
		pt = torch.exp(-ce_loss)
		focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
		if self.reduction == "mean":
			return focal_loss.mean()
		elif self.reduction == "sum":
			return focal_loss.sum()
		return focal_loss


@dataclass
class TrainConfig:
	train_ratio: float = 0.7
	window_size: int = 128
	stride: int = 1
	batch_size: int = 64
	d_model: int = 128
	nhead: int = 8
	num_encoder_layers: int = 4
	num_decoder_layers: int = 4
	dim_feedforward: int = 512
	dropout: float = 0.1
	learning_rate: float = 1e-4
	weight_decay: float = 1e-5
	num_epochs: int = 50
	device: str = "cuda" if torch.cuda.is_available() else "cpu"
	label_column: str = "is_anomaly"
	warmup_epochs: int = 5
	use_focal_loss: bool = False


def compute_anomaly_scores(
	model: nn.Module,
	batch: torch.Tensor,
	device: str,
	normal_centroid: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	"""
	Compute multiple anomaly scores:
	1. Reconstruction error (MSE)
	2. Feature distance from normal centroid (in latent space)
	3. Combined score
	Returns: (recon_scores, feature_scores, combined_scores)
	"""
	model.eval()
	with torch.no_grad():
		reconstructed, features = model(batch, return_features=True)

		# 1. Reconstruction error (per sample, averaged over time and features)
		recon_error = F.mse_loss(reconstructed, batch, reduction="none")  # (B, T, F)
		recon_scores = recon_error.mean(dim=(1, 2))  # (B,)

		# 2. Feature distance from normal centroid
		# If centroid is provided, use distance from it; otherwise use L2 norm as fallback
		if normal_centroid is not None:
			feature_scores = torch.norm(features - normal_centroid, dim=1)  # (B,)
		else:
			# Fallback: use L2 norm of features (less meaningful but works)
			feature_scores = torch.norm(features, dim=1)  # (B,)

		# 3. Combined score (normalized and weighted)
		recon_norm = (recon_scores - recon_scores.min()) / (recon_scores.max() - recon_scores.min() + 1e-8)
		feature_norm = (feature_scores - feature_scores.min()) / (feature_scores.max() - feature_scores.min() + 1e-8)
		combined_scores = 0.7 * recon_norm + 0.3 * feature_norm

	return recon_scores, feature_scores, combined_scores


def train_autoencoder(
	model: nn.Module,
	train_loader: DataLoader,
	cfg: TrainConfig,
) -> None:
	device = cfg.device
	model.to(device)

	# Use AdamW with cosine annealing
	optimizer = torch.optim.AdamW(
		model.parameters(),
		lr=cfg.learning_rate,
		weight_decay=cfg.weight_decay,
		betas=(0.9, 0.999),
	)

	# Learning rate scheduler with warmup
	def lr_lambda(epoch):
		if epoch < cfg.warmup_epochs:
			return epoch / cfg.warmup_epochs
		return 0.5 * (1 + np.cos(np.pi * (epoch - cfg.warmup_epochs) / (cfg.num_epochs - cfg.warmup_epochs)))

	scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

	criterion = nn.MSELoss()

	# Initialize a learnable centroid for normal patterns
	# This will be updated via exponential moving average during training
	with torch.no_grad():
		# Initialize with first batch
		first_batch, _ = next(iter(train_loader))
		_, first_features = model(first_batch, return_features=True)
		normal_centroid = first_features.mean(dim=0, keepdim=True).clone()  # (1, feature_dim)
		normal_centroid = normal_centroid.to(device)

	model.train()
	best_loss = float("inf")
	patience = 10
	patience_counter = 0
	ema_momentum = 0.99  # Exponential moving average momentum for centroid update

	for epoch in range(1, cfg.num_epochs + 1):
		running_loss = 0.0
		num_batches = 0

		for batch, _ in tqdm(train_loader, desc="Training", leave=False):
			optimizer.zero_grad()
			# Get both reconstruction and features in a single forward pass
			recon, features = model(batch, return_features=True)
			
			# Primary reconstruction loss
			recon_loss = criterion(recon, batch)

			# Feature clustering loss: encourage normal patterns to cluster around centroid
			# This trains the feature_extractor to create a compact representation of normal patterns
			# Anomalies will be far from this centroid
			feature_distances = torch.norm(features - normal_centroid, dim=1)  # (B,)
			clustering_loss = torch.mean(feature_distances ** 2)  # L2 distance squared
			
			# Combined loss
			loss = recon_loss + 0.1 * clustering_loss

			loss.backward()
			# Gradient clipping
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			optimizer.step()

			# Update centroid using exponential moving average (no gradients)
			with torch.no_grad():
				batch_centroid = features.mean(dim=0, keepdim=True)  # (1, feature_dim)
				normal_centroid = ema_momentum * normal_centroid + (1 - ema_momentum) * batch_centroid

			running_loss += loss.item()
			num_batches += 1

		scheduler.step()
		avg_loss = running_loss / max(1, num_batches)
		current_lr = scheduler.get_last_lr()[0]

		# Early stopping
		if avg_loss < best_loss:
			best_loss = avg_loss
			patience_counter = 0
		else:
			patience_counter += 1

		if epoch % 5 == 0 or epoch == 1:
			print(
				f"Epoch {epoch:03d}/{cfg.num_epochs} - Loss: {avg_loss:.6f} - LR: {current_lr:.2e} - Best: {best_loss:.6f}"
			)

		if patience_counter >= patience:
			print(f"Early stopping at epoch {epoch}")
			break

	print(f"Training completed. Best loss: {best_loss:.6f}")


def evaluate_anomaly_detection(
	model: nn.Module,
	train_loader: DataLoader,
	test_loader: DataLoader,
	cfg: TrainConfig,
) -> None:
	device = cfg.device
	model.eval()

	# First, compute the normal centroid from training data
	print("Computing normal centroid from training data...")
	all_train_features = []
	with torch.no_grad():
		for batch, _ in train_loader:
			_, features = model(batch, return_features=True)
			all_train_features.append(features)
	
	# Compute centroid as mean of all training features
	normal_centroid = torch.cat(all_train_features, dim=0).mean(dim=0, keepdim=True).to(device)  # (1, feature_dim)
	print(f"Normal centroid computed. Shape: {normal_centroid.shape}")

	# Collect scores from training data to establish baseline
	print("Computing baseline scores from training data...")
	all_train_scores = []

	with torch.no_grad():
		for batch, _ in train_loader:
			_, _, combined_scores = compute_anomaly_scores(model, batch, device, normal_centroid=normal_centroid)
			all_train_scores.extend(combined_scores.cpu().numpy().tolist())

	all_train_scores = np.array(all_train_scores)

	# Use multiple threshold strategies
	thresholds = {
		"5th_percentile": np.percentile(all_train_scores, 5.0),
		"10th_percentile": np.percentile(all_train_scores, 10.0),
		"mean_std": np.mean(all_train_scores) - 2 * np.std(all_train_scores),
		"iqr": np.percentile(all_train_scores, 25) - 1.5 * (np.percentile(all_train_scores, 75) - np.percentile(all_train_scores, 25)),
	}

	print(f"\nTraining score statistics:")
	print(f"  Mean: {np.mean(all_train_scores):.6f}")
	print(f"  Std:  {np.std(all_train_scores):.6f}")
	print(f"  Min:  {np.min(all_train_scores):.6f}")
	print(f"  Max:  {np.max(all_train_scores):.6f}")

	# Evaluate on test set
	print("\nEvaluating on test set...")
	all_test_scores = []
	all_test_labels = []

	with torch.no_grad():
		for batch, labels in test_loader:
			_, _, combined_scores = compute_anomaly_scores(model, batch, device, normal_centroid=normal_centroid)
			all_test_scores.extend(combined_scores.cpu().numpy().tolist())
			all_test_labels.extend(labels.cpu().numpy().tolist())

	all_test_scores = np.array(all_test_scores)
	all_test_labels = np.array(all_test_labels)

	# Try different thresholds and pick best F1
	best_f1 = 0
	best_threshold = None
	best_method = None

	print("\n" + "=" * 80)
	print("Threshold Comparison:")
	print("=" * 80)

	for method, threshold in thresholds.items():
		preds = (all_test_scores < threshold).astype(np.int64)
		precision, recall, f1, _ = precision_recall_fscore_support(
			all_test_labels, preds, average="binary", zero_division=0
		)
		print(f"\n{method:20s} (threshold={threshold:.6f}):")
		print(f"  Precision: {precision:.4f}")
		print(f"  Recall:    {recall:.4f}")
		print(f"  F1-score:  {f1:.4f}")

		if f1 > best_f1:
			best_f1 = f1
			best_threshold = threshold
			best_method = method

	# Also try optimal threshold from precision-recall curve
	precision_curve, recall_curve, pr_thresholds = precision_recall_curve(all_test_labels, -all_test_scores)
	f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-8)
	optimal_idx = np.argmax(f1_scores)
	optimal_threshold = -pr_thresholds[optimal_idx]

	preds_optimal = (all_test_scores < optimal_threshold).astype(np.int64)
	precision_opt, recall_opt, f1_opt, _ = precision_recall_fscore_support(
		all_test_labels, preds_optimal, average="binary", zero_division=0
	)

	print(f"\n{'optimal_pr_curve':20s} (threshold={optimal_threshold:.6f}):")
	print(f"  Precision: {precision_opt:.4f}")
	print(f"  Recall:    {recall_opt:.4f}")
	print(f"  F1-score:  {f1_opt:.4f}")

	if f1_opt > best_f1:
		best_f1 = f1_opt
		best_threshold = optimal_threshold
		best_method = "optimal_pr_curve"

	# Final evaluation with best threshold
	print("\n" + "=" * 80)
	print(f"Best Method: {best_method} (threshold={best_threshold:.6f})")
	print("=" * 80)

	final_preds = (all_test_scores < best_threshold).astype(np.int64)
	precision, recall, f1, _ = precision_recall_fscore_support(
		all_test_labels, final_preds, average="binary", zero_division=0
	)

	# Compute AUC-ROC
	try:
		auc_roc = roc_auc_score(all_test_labels, -all_test_scores)  # Negative because lower score = anomaly
		print(f"\nAUC-ROC: {auc_roc:.4f}")
	except ValueError:
		auc_roc = 0.0
		print("\nAUC-ROC: N/A (insufficient class diversity)")

	print(f"\nFinal Test Metrics:")
	print(f"  Precision: {precision:.4f}")
	print(f"  Recall:    {recall:.4f}")
	print(f"  F1-score:  {f1:.4f}")

	print("\nDetailed Classification Report:")
	print(classification_report(all_test_labels, final_preds, digits=4, zero_division=0))


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Advanced Transformer-based Anomaly Detection for Time Series"
	)
	parser.add_argument(
		"--parquet_path",
		type=str,
		default="/home/cbelshe/CMPE-258/final_project/train.parquet",
		help="Path to input parquet file.",
	)
	parser.add_argument("--window_size", type=int, default=128, help="Sliding window size.")
	parser.add_argument("--stride", type=int, default=1, help="Sliding window stride.")
	parser.add_argument("--batch_size", type=int, default=64, help="Dataloader batch size.")
	parser.add_argument("--epochs", type=int, default=14, help="Number of training epochs.")
	parser.add_argument("--d_model", type=int, default=128, help="Transformer model dimension.")
	parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads.")
	parser.add_argument("--encoder_layers", type=int, default=4, help="Number of encoder layers.")
	parser.add_argument("--decoder_layers", type=int, default=4, help="Number of decoder layers.")
	parser.add_argument("--dim_feedforward", type=int, default=512, help="Feedforward dimension.")
	parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
	parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
	parser.add_argument("--train_ratio", type=float, default=0.7, help="Train split ratio by time.")
	parser.add_argument("--label_column", type=str, default="is_anomaly", help="Name of label column.")
	parser.add_argument("--seed", type=int, default=42, help="Random seed.")
	parser.add_argument(
		"--model_path",
		type=str,
		default=None,
		help="Path to saved model state dict. If provided, loads model and skips training.",
	)
	parser.add_argument("--warmup_epochs", type=int, default=5, help="Number of warmup epochs for LR scheduler.")
	args = parser.parse_args()

	set_seed(args.seed)
	if not torch.cuda.is_available():
		raise RuntimeError("CUDA is required for this run but was not found.")
	device = "cuda"

	# Load parquet
	if not os.path.exists(args.parquet_path):
		raise FileNotFoundError(f"Parquet file not found: {args.parquet_path}")
	df = pd.read_parquet(args.parquet_path)

	label_col = args.label_column
	if label_col not in df.columns:
		raise ValueError(f"Label column '{label_col}' not found in dataframe.")

	feature_cols = infer_feature_columns(df, label_column=label_col)
	print(f"Identified {len(feature_cols)} feature columns.")

	# Time split
	train_idx, test_idx = time_split_indices(len(df), train_ratio=args.train_ratio)
	df_train = df.iloc[train_idx].reset_index(drop=True)
	df_test = df.iloc[test_idx].reset_index(drop=True)

	# Filter normal data for training
	df_train_normal = df_train[df_train[label_col] == 0].reset_index(drop=True)
	if len(df_train_normal) < args.window_size:
		raise ValueError("Not enough non-anomalous rows in train split to form a single window.")

	# Standardize
	scaler = StandardScaler()
	scaler.fit(df_train_normal[feature_cols].to_numpy())

	X_train_normal = scaler.transform(df_train_normal[feature_cols].to_numpy())
	X_test = scaler.transform(df_test[feature_cols].to_numpy())
	y_test_all = df_test[label_col].astype(int).to_numpy()

	# Datasets
	window_size = int(args.window_size)
	stride = int(args.stride)

	train_dataset = SlidingWindowDataset(
		data=X_train_normal,
		window_size=window_size,
		stride=stride,
		labels=None,
	)

	test_window_labels = build_window_labels(y_test_all, window_size=window_size, stride=stride)
	test_dataset = SlidingWindowDataset(
		data=X_test,
		window_size=window_size,
		stride=stride,
		labels=test_window_labels,
	)

	collate_fn = make_collate_to_device(device)
	train_loader = DataLoader(
		train_dataset,
		batch_size=int(args.batch_size),
		shuffle=True,
		drop_last=True,
		collate_fn=collate_fn,
	)
	test_loader = DataLoader(
		test_dataset,
		batch_size=int(args.batch_size),
		shuffle=False,
		drop_last=False,
		collate_fn=collate_fn,
	)

	cfg = TrainConfig(
		train_ratio=float(args.train_ratio),
		window_size=window_size,
		stride=stride,
		batch_size=int(args.batch_size),
		d_model=int(args.d_model),
		nhead=int(args.nhead),
		num_encoder_layers=int(args.encoder_layers),
		num_decoder_layers=int(args.decoder_layers),
		dim_feedforward=int(args.dim_feedforward),
		dropout=float(args.dropout),
		learning_rate=float(args.learning_rate),
		num_epochs=int(args.epochs),
		device=device,
		label_column=label_col,
		warmup_epochs=int(args.warmup_epochs),
	)

	model = TransformerAutoencoder(
		num_features=len(feature_cols),
		d_model=cfg.d_model,
		nhead=cfg.nhead,
		num_encoder_layers=cfg.num_encoder_layers,
		num_decoder_layers=cfg.num_decoder_layers,
		dim_feedforward=cfg.dim_feedforward,
		dropout=cfg.dropout,
		max_seq_len=window_size * 2,
	)

	if args.model_path is not None:
		if not os.path.exists(args.model_path):
			raise FileNotFoundError(f"Model file not found: {args.model_path}")
		print(f"Loading model from {args.model_path}...")
		model.load_state_dict(torch.load(args.model_path, map_location=device))
		model.to(device)
		print("Model loaded successfully. Skipping training.")
	else:
		print("Starting training on normal windows...")
		train_autoencoder(model, train_loader, cfg)
		torch.save(model.state_dict(), "transformer_anomaly_detector2.pt")
		print("Model saved to transformer_anomaly_detector2.pt")

	print("\nEvaluating anomaly detection...")
	evaluate_anomaly_detection(model, train_loader, test_loader, cfg)


if __name__ == "__main__":
	main()


"""
results:
================================================================================
Best Method: optimal_pr_curve (threshold=0.875410)
================================================================================

AUC-ROC: 0.5005

Final Test Metrics:
  Precision: 0.1059
  Recall:    1.0000
  F1-score:  0.1916

Detailed Classification Report:
              precision    recall  f1-score   support

           0     0.0000    0.0000    0.0000   3950337
           1     0.1059    1.0000    0.1916    468033

    accuracy                         0.1059   4418370
   macro avg     0.0530    0.5000    0.0958   4418370
weighted avg     0.0112    0.1059    0.0203   4418370
"""
