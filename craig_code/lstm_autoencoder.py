import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


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
	"""
	Convert a pointwise binary array into window labels where a window is labeled 1
	if any timepoint in the window is anomalous.
	"""
	if binary_series.ndim != 1:
		raise ValueError("binary_series must be 1D.")
	n = len(binary_series)
	labels = []
	for start in range(0, n - window_size + 1, stride):
		end = start + window_size
		labels.append(1 if np.any(binary_series[start:end] == 1) else 0)
	return np.asarray(labels, dtype=np.int64)


class SlidingWindowDataset(Dataset):
	"""
	Create sliding windows over multivariate time series.
	- data: ndarray (num_rows, num_features)
	- labels: optional sequence labels (num_windows,) with 0/1 for normal/anomaly
	"""

	def __init__(self, data: np.ndarray, window_size: int, stride: int = 1, labels: Optional[np.ndarray] = None):
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
		# batch: List[Tuple[Tensor(T,F), Optional[Tensor(())]]]
		xs, ys = zip(*batch)
		x = torch.stack(xs, dim=0).to(device, non_blocking=True)
		if ys[0] is None:
			return x, None
		# ys are 0-dim tensors on CPU; stack then move
		y = torch.stack(ys, dim=0).to(device, non_blocking=True)
		return x, y
	return _collate


class LSTMAutoencoder(nn.Module):
	"""
	Small LSTM autoencoder:
	- Encoder: LSTM -> take last hidden state -> linear to latent
	- Decoder: repeat latent across time -> LSTM -> project to features
	"""

	def __init__(
		self,
		num_features: int,
		hidden_size: int = 64,
		latent_size: int = 16,
		num_layers: int = 1,
	):
		super().__init__()
		self.num_features = num_features
		self.hidden_size = hidden_size
		self.latent_size = latent_size
		self.num_layers = num_layers

		self.encoder = nn.LSTM(
			input_size=num_features,
			hidden_size=hidden_size,
			num_layers=num_layers,
			batch_first=True,
			bidirectional=False,
		)
		self.to_latent = nn.Linear(hidden_size, latent_size)

		self.decoder_input = nn.Linear(latent_size, hidden_size)
		self.decoder = nn.LSTM(
			input_size=hidden_size,
			hidden_size=hidden_size,
			num_layers=num_layers,
			batch_first=True,
			bidirectional=False,
		)
		self.to_features = nn.Linear(hidden_size, num_features)
		self.activation = nn.Tanh()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: (B, T, F)
		enc_out, (h_n, c_n) = self.encoder(x)
		h_last = h_n[-1]  # (B, hidden)
		z = self.activation(self.to_latent(h_last))  # (B, latent)

		# Repeat latent across time dimension as decoder input
		batch_size, seq_len, _ = x.shape
		dec_in = self.activation(self.decoder_input(z))  # (B, hidden)
		dec_in = dec_in.unsqueeze(1).repeat(1, seq_len, 1)  # (B, T, hidden)

		dec_out, _ = self.decoder(dec_in)
		recon = self.to_features(dec_out)  # (B, T, F)
		return recon


@dataclass
class TrainConfig:
	train_ratio: float = 0.7
	window_size: int = 64
	stride: int = 1
	batch_size: int = 128
	hidden_size: int = 64
	latent_size: int = 16
	num_layers: int = 1
	learning_rate: float = 1e-3
	weight_decay: float = 0.0
	num_epochs: int = 20
	device: str = "cuda" if torch.cuda.is_available() else "cpu"
	label_column: str = "is_anomaly"


def compute_cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
	"""
	Cosine similarity between flattened sequences.
	a, b: (B, T, F)
	return: (B,) similarity in [-1, 1]
	"""
	a_flat = a.flatten(start_dim=1)
	b_flat = b.flatten(start_dim=1)
	return nn.functional.cosine_similarity(a_flat, b_flat, dim=1)


def train_autoencoder(
	model: nn.Module,
	train_loader: DataLoader,
	cfg: TrainConfig,
) -> None:
	device = cfg.device
	model.to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
	criterion = nn.MSELoss()

	model.train()
	for epoch in range(1, cfg.num_epochs + 1):
		running_loss = 0.0
		num_batches = 0
		for batch, _ in train_loader:
			optimizer.zero_grad()
			recon = model(batch)
			loss = criterion(recon, batch)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
			num_batches += 1
		avg_loss = running_loss / max(1, num_batches)
		print(f"Epoch {epoch:02d}/{cfg.num_epochs} - train MSE: {avg_loss:.6f}")


def evaluate_with_cosine_similarity(
	model: nn.Module,
	train_loader: DataLoader,
	test_loader: DataLoader,
	cfg: TrainConfig,
) -> None:
	device = cfg.device
	model.eval()
	all_train_sims: List[float] = []

	with torch.no_grad():
		for batch, _ in train_loader:
			recon = model(batch)
			sims = compute_cosine_similarity(batch, recon)
			all_train_sims.extend(sims.detach().cpu().numpy().tolist())

	# Define threshold from train distribution (lower similarity => anomaly)
	threshold = np.percentile(all_train_sims, 5.0)  # 5th percentile
	print(f"Cosine similarity threshold set at 5th percentile of train: {threshold:.6f}")

	# Evaluate on test windows
	all_scores: List[float] = []
	all_preds: List[int] = []
	all_labels: List[int] = []

	with torch.no_grad():
		for batch, labels in test_loader:
			recon = model(batch)
			sims = compute_cosine_similarity(batch, recon).detach().cpu().numpy()
			preds = (sims < threshold).astype(np.int64)  # anomaly if similarity is below threshold
			all_scores.extend(sims.tolist())
			all_preds.extend(preds.tolist())
			# labels already on device; bring to CPU numpy
			all_labels.extend(labels.detach().cpu().numpy().tolist())

	precision, recall, f1, _ = precision_recall_fscore_support(
		all_labels, all_preds, average="binary", zero_division=0
	)
	print("Test classification metrics (window-level):")
	print(f"- Precision: {precision:.4f}")
	print(f"- Recall:    {recall:.4f}")
	print(f"- F1-score:  {f1:.4f}")
	print()
	print("Detailed classification report:")
	print(classification_report(all_labels, all_preds, digits=4, zero_division=0))


def main() -> None:
	parser = argparse.ArgumentParser(description="Train LSTM Autoencoder on non-anomalous data and evaluate via cosine similarity.")
	parser.add_argument(
		"--parquet_path",
		type=str,
		default="/home/cbelshe/CMPE-258/final_project/train.parquet",
		help="Path to input parquet file.",
	)
	parser.add_argument("--window_size", type=int, default=64, help="Sliding window size.")
	parser.add_argument("--stride", type=int, default=1, help="Sliding window stride.")
	parser.add_argument("--batch_size", type=int, default=128, help="Dataloader batch size.")
	parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
	parser.add_argument("--hidden_size", type=int, default=64, help="LSTM hidden size.")
	parser.add_argument("--latent_size", type=int, default=16, help="Latent size.")
	parser.add_argument("--layers", type=int, default=1, help="Number of LSTM layers.")
	parser.add_argument("--train_ratio", type=float, default=0.7, help="Train split ratio by time.")
	parser.add_argument("--label_column", type=str, default="is_anomaly", help="Name of label column.")
	parser.add_argument("--seed", type=int, default=42, help="Random seed.")
	parser.add_argument(
		"--model_path",
		type=str,
		default=None,
		help="Path to saved model state dict. If provided, loads model and skips training, only runs evaluation.",
	)
	args = parser.parse_args()

	set_seed(args.seed)
	if not torch.cuda.is_available():
		raise RuntimeError("CUDA is required for this run but was not found. Please ensure a GPU runtime is available.")
	device = "cuda"

	# Load parquet with pandas
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

	# Filter normal data for training scaler and training windows
	df_train_normal = df_train[df_train[label_col] == 0].reset_index(drop=True)
	if len(df_train_normal) < args.window_size:
		raise ValueError("Not enough non-anomalous rows in train split to form a single window.")

	# Standardize using only normal train rows
	scaler = StandardScaler()
	scaler.fit(df_train_normal[feature_cols].to_numpy())

	X_train_all = scaler.transform(df_train[feature_cols].to_numpy())
	X_train_normal = scaler.transform(df_train_normal[feature_cols].to_numpy())
	X_test = scaler.transform(df_test[feature_cols].to_numpy())

	y_train_all = df_train[label_col].astype(int).to_numpy()
	y_test_all = df_test[label_col].astype(int).to_numpy()

	# Datasets
	window_size = int(args.window_size)
	stride = int(args.stride)

	# Train on normal windows only
	train_dataset = SlidingWindowDataset(
		data=X_train_normal,
		window_size=window_size,
		stride=stride,
		labels=None,
	)

	# Test set windows and labels
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
		hidden_size=int(args.hidden_size),
		latent_size=int(args.latent_size),
		num_layers=int(args.layers),
		num_epochs=int(args.epochs),
		device=device,
		label_column=label_col,
	)

	model = LSTMAutoencoder(
		num_features=len(feature_cols),
		hidden_size=cfg.hidden_size,
		latent_size=cfg.latent_size,
		num_layers=cfg.num_layers,
	)

	if args.model_path is not None:
		# Load saved model and skip training
		if not os.path.exists(args.model_path):
			raise FileNotFoundError(f"Model file not found: {args.model_path}")
		print(f"Loading model from {args.model_path}...")
		model.load_state_dict(torch.load(args.model_path, map_location=device))
		model.to(device)
		print("Model loaded successfully. Skipping training.")
	else:
		# Train the model
		print("Starting training on normal windows...")
		train_autoencoder(model, train_loader, cfg)
		# save the model
		torch.save(model.state_dict(), "lstm_autoencoder_model.pt")
		print("Model saved to lstm_autoencoder_model.pt")

	print("Evaluating with cosine similarity on test windows...")
	evaluate_with_cosine_similarity(model, train_loader, test_loader, cfg)


if __name__ == "__main__":
	main()

"""
Evaluating with cosine similarity on test windows...
Cosine similarity threshold set at 5th percentile of train: 0.289993
Test classification metrics (window-level):
- Precision: 0.0907
- Recall:    0.1149
- F1-score:  0.1013

Detailed classification report:
              precision    recall  f1-score   support

           0     0.8926    0.8646    0.8784   3953722
           1     0.0907    0.1149    0.1013    464712

    accuracy                         0.7857   4418434
   macro avg     0.4916    0.4897    0.4899   4418434
weighted avg     0.8082    0.7857    0.7966   4418434
"""