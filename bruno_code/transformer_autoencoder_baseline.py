from __future__ import annotations

import os
import json
import math
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

import yaml

from src.data_from_dirs import ingest_folder_layout as _ingest


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    Expects input of shape (batch, seq_len, d_model) when batch_first=True.
    """

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class TransformerAutoencoder(nn.Module):
    """
    Sequence autoencoder based on a Transformer encoder.

    - Projects input features to d_model
    - Adds positional encoding
    - TransformerEncoder over the sequence
    - Mean-pools over time to get a global latent
    - Repeats latent across time and decodes back to feature space
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        self.input_dim = input_dim
        self.d_model = d_model

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.decoder = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode sequence batch into latent vectors.
        x: (batch, seq_len, input_dim)
        returns: (batch, d_model)
        """
        z = self.input_proj(x)
        z = self.pos_encoder(z)
        z_enc = self.encoder(z)
        z_global = z_enc.mean(dim=1)
        return z_global

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Autoencoder reconstruction.
        x: (batch, seq_len, input_dim)
        returns: (batch, seq_len, input_dim)
        """
        batch_size, seq_len, _ = x.shape

        z = self.input_proj(x)
        z = self.pos_encoder(z)
        z_enc = self.encoder(z)
        z_global = z_enc.mean(dim=1)

        z_rep = z_global.unsqueeze(1).repeat(1, seq_len, 1)

        recon = self.decoder(z_rep)
        return recon


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def plot_confusion_matrix(cm, labels, title, path):
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.viridis)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_histogram(errors, labels, path, thr=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(
        errors[labels == 0], bins=50, alpha=0.7, label="Normal", density=True
    )
    ax.hist(
        errors[labels == 1], bins=50, alpha=0.7, label="Anomaly", density=True
    )
    if thr is not None:
        ax.axvline(thr, color="red", linestyle="--", label=f"Threshold={thr:.3f}")
    ax.set_xlabel("Reconstruction error (MSE)")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def make_windows(
    telem: pd.DataFrame,
    window: int = 128,
    stride: int = 4,
    max_windows: int | None = None,
    seed: int = 42,
):
    """
    telem must have:
      - 'time' column
      - 'label' column (0/1)
      - all other columns numeric features

    Returns:
      X: (num_windows, window, num_features)
      y_win: (num_windows,) window labels (1 if any anomaly in window)
      feat_cols: list of feature column names
    """
    assert "time" in telem.columns and "label" in telem.columns

    feat_cols = [c for c in telem.columns if c not in ["time", "label"]]
    X_full = telem[feat_cols].values.astype(np.float32)
    y_full = telem["label"].values.astype(np.int64)

    T, D = X_full.shape
    all_starts = np.arange(0, T - window + 1, stride, dtype=np.int64)
    total_windows = len(all_starts)

    if max_windows is not None and max_windows < total_windows:
        rng = np.random.default_rng(seed)
        sel = rng.choice(total_windows, size=max_windows, replace=False)
        starts = np.sort(all_starts[sel])
        print(
            f"[INFO] Subsampled windows: {len(starts)}/{total_windows} "
            f"(window={window}, stride={stride})"
        )
    else:
        starts = all_starts
        print(
            f"[INFO] Using all windows: {len(starts)} "
            f"(window={window}, stride={stride})"
        )

    num_windows = len(starts)
    X = np.zeros((num_windows, window, D), dtype=np.float32)
    y_win = np.zeros(num_windows, dtype=np.int64)

    for i, s in enumerate(starts):
        e = s + window
        X[i] = X_full[s:e]
        y_win[i] = 1 if y_full[s:e].max() > 0 else 0

    return X, y_win, feat_cols


def train_val_test_split(
    num_windows: int,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    idx = np.arange(num_windows)
    rng.shuffle(idx)

    n_train = int(train_frac * num_windows)
    n_val = int(val_frac * num_windows)

    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    print(
        f"[INFO] Split: train={len(train_idx)}, "
        f"val={len(val_idx)}, test={len(test_idx)}"
    )
    return train_idx, val_idx, test_idx


def train_ae(
    X_train: np.ndarray,
    device: torch.device,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int | None = None,
    dropout: float = 0.1,
    batch_size: int = 256,
    epochs: int = 10,
    lr: float = 1e-3,
    run_dir: str | None = None,
):
    num_windows, seq_len, feat_dim = X_train.shape
    print(f"[INFO] Training Transformer AE on {num_windows} windows, "
          f"seq_len={seq_len}, feats={feat_dim}")

    model = TransformerAutoencoder(
        input_dim=feat_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    ).to(device)

    criterion = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = TensorDataset(torch.from_numpy(X_train.astype(np.float32)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    history = {"epoch": [], "loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", unit="batch")
        for batch_idx, (batch_x,) in enumerate(pbar):
            batch_x = batch_x.to(device)

            if torch.isnan(batch_x).any() or torch.isinf(batch_x).any():
                print(f"[ERROR] NaN/inf in INPUT at batch {batch_idx}")
                print("  input min/max:", batch_x.min().item(), batch_x.max().item())
                raise SystemExit

            optimizer.zero_grad()
            recon = model(batch_x)

            if torch.isnan(recon).any() or torch.isinf(recon).any():
                print(f"[ERROR] NaN/inf in MODEL OUTPUT at batch {batch_idx}")
                print("  recon min/max:", recon.min().item(), recon.max().item())
                print("  input min/max:", batch_x.min().item(), batch_x.max().item())
                raise SystemExit

            loss = criterion(recon, batch_x)

            if not torch.isfinite(loss):
                print(f"[ERROR] NaN/inf LOSS at batch {batch_idx}")
                print("  input min/max:", batch_x.min().item(), batch_x.max().item())
                print("  recon min/max:", recon.min().item(), recon.max().item())
                raise SystemExit

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss
            n_batches += 1

            pbar.set_postfix(loss=batch_loss)

        epoch_loss /= max(n_batches, 1)
        history["epoch"].append(epoch)
        history["loss"].append(epoch_loss)
        print(f"[INFO] Epoch {epoch}/{epochs} mean_loss={epoch_loss:.6f}")

        if run_dir is not None:
            save_json(history, os.path.join(run_dir, "train_history.json"))

    return model, history


@torch.no_grad()
def compute_recon_errors(
    model,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
):
    """
    Compute per-window reconstruction error (MSE).
    """
    model.eval()
    dataset = TensorDataset(torch.from_numpy(X.astype(np.float32)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    all_errs = []
    for batch_idx, (batch_x,) in enumerate(
        tqdm(loader, desc="Computing reconstruction errors", unit="batch")
    ):
        batch_x = batch_x.to(device)

        if torch.isnan(batch_x).any() or torch.isinf(batch_x).any():
            print(f"[ERROR] NaN/inf in INPUT during eval at batch {batch_idx}")
            print("  input min/max:", batch_x.min().item(), batch_x.max().item())
            raise SystemExit

        recon = model(batch_x)

        if torch.isnan(recon).any() or torch.isinf(recon).any():
            print(f"[ERROR] NaN/inf in MODEL OUTPUT during eval at batch {batch_idx}")
            print("  recon min/max:", recon.min().item(), recon.max().item())
            print("  input min/max:", batch_x.min().item(), batch_x.max().item())
            raise SystemExit

        err = (recon - batch_x) ** 2
        err = err.mean(dim=(1, 2)).cpu().numpy()

        if not np.isfinite(err).all():
            print(f"[ERROR] NaN/inf in reconstruction ERROR at batch {batch_idx}")
            print("  err stats: min=", np.nanmin(err), "max=", np.nanmax(err))
            raise SystemExit

        all_errs.append(err)

    return np.concatenate(all_errs, axis=0)


@torch.no_grad()
def encode_dataset(
    model,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
):
    """
    Encode a 3D numpy array of windows into latent vectors.
    X: (num_windows, seq_len, feat_dim)
    returns: (num_windows, latent_dim)
    """
    model.eval()
    dataset = TensorDataset(torch.from_numpy(X.astype(np.float32)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    latents = []
    for (batch_x,) in tqdm(loader, desc="Encoding dataset", unit="batch"):
        batch_x = batch_x.to(device)
        z = model.encode(batch_x)
        latents.append(z.cpu().numpy())
    return np.concatenate(latents, axis=0)


def train_eval_transformer_ae(
    cfg: dict,
    telem: pd.DataFrame,
    run_id: str,
    no_plot: bool,
    save: bool,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    data_cfg = cfg["DATA"]
    window = int(data_cfg.get("WINDOW", 128))
    stride = int(data_cfg.get("STRIDE", 4))
    train_frac = float(data_cfg.get("TRAIN_SPLIT", 0.6))
    val_frac = float(data_cfg.get("VAL_SPLIT", 0.2))
    test_frac = float(data_cfg.get("TEST_SPLIT", 0.2))
    max_windows = data_cfg.get("MAX_WINDOWS", None)
    if max_windows is not None:
        max_windows = int(max_windows)

    eval_cfg = cfg.get("EVAL", {})
    thr_percentile = float(eval_cfg.get("THRESHOLD_PERCENTILE", 99.5))

    X, y_win, feat_cols = make_windows(
        telem,
        window=window,
        stride=stride,
        max_windows=max_windows,
    )

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    N = X.shape[0]
    print(f"[INFO] Windows: {N}, seq_len={X.shape[1]}, features={X.shape[2]}")
    print(f"[INFO] Positive windows (any anomaly in window): {y_win.sum()}")

    train_idx, val_idx, test_idx = train_val_test_split(
        N, train_frac=train_frac, val_frac=val_frac
    )

    train_flat = X[train_idx].reshape(-1, X.shape[-1])
    mean = train_flat.mean(axis=0, keepdims=True)
    std = train_flat.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    X = (X - mean) / std

    train_mask = (y_win[train_idx] == 0)
    X_train = X[train_idx][train_mask]
    print(f"[INFO] Train windows (normal only): {X_train.shape[0]}")

    X_val = X[val_idx]
    y_val = y_win[val_idx]
    X_test = X[test_idx]
    y_test = y_win[test_idx]

    runs_root = ensure_dir("runs/transformer_ae")
    run_dir = ensure_dir(os.path.join(runs_root, run_id))
    print(f"[INFO] Run directory: {run_dir}")

    model_cfg = cfg.get("MODEL", {})
    d_model = int(model_cfg.get("D_MODEL", 128))
    nhead = int(model_cfg.get("NHEAD", 4))
    num_layers = int(model_cfg.get("NUM_LAYERS", 2))
    dim_feedforward = model_cfg.get("DIM_FEEDFORWARD", None)
    if dim_feedforward is not None:
        dim_feedforward = int(dim_feedforward)
    dropout = float(model_cfg.get("DROPOUT", 0.1))

    train_cfg = cfg.get("TRAIN", {})
    epochs = int(train_cfg.get("EPOCHS", 5))
    batch_size = int(train_cfg.get("BATCH_SIZE", 256))
    lr = float(train_cfg.get("LR", 1e-3))

    model, history = train_ae(
        X_train=X_train,
        device=device,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        run_dir=run_dir,
    )

    if save:
        model_path = os.path.join(run_dir, "transformer_autoencoder.pt")
        torch.save(model.state_dict(), model_path)
        print(f"[INFO] Saved model to {model_path}")

    print("[INFO] Computing reconstruction errors on val/test sets...")
    val_errs = compute_recon_errors(model, X_val, device=device)
    test_errs = compute_recon_errors(model, X_test, device=device)

    val_norm_errs = val_errs[y_val == 0]
    thr = np.percentile(val_norm_errs, thr_percentile)
    print(f"[INFO] Threshold at {thr_percentile}th percentile of val normals: {thr:.6f}")

    y_score = test_errs
    y_pred = (y_score >= thr).astype(int)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    report = classification_report(y_test, y_pred, output_dict=True, digits=4)
    auc = roc_auc_score(y_test, y_score)

    print("[INFO] Transformer AE metrics on TEST (reconstruction):")
    print("Confusion matrix (rows=true [0,1], cols=pred [0,1]):")
    print(cm)
    print("[INFO] ROC-AUC (reconstruction error as score):", auc)

    if save:
        metrics = {
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "roc_auc": float(auc),
            "threshold": float(thr),
        }
        save_json(metrics, os.path.join(run_dir, "metrics_ae.json"))

        if not no_plot:
            cm_path = os.path.join(run_dir, "confusion_matrix_ae.png")
            plot_confusion_matrix(
                cm, labels=["Normal", "Anomaly"],
                title="Transformer AE Confusion Matrix",
                path=cm_path,
            )

            hist_path = os.path.join(run_dir, "recon_error_hist.png")
            plot_histogram(test_errs, y_test, hist_path, thr=thr)

    bal_cfg = cfg.get("BALANCE", {})
    if bal_cfg.get("ENABLE", False):
        method = str(bal_cfg.get("METHOD", "smote")).lower()
        target = str(bal_cfg.get("TARGET", "latent")).lower()

        if method != "smote" or target != "latent":
            print(
                f"[WARN] BALANCE.METHOD={method} TARGET={target} not supported; "
                f"only METHOD='smote', TARGET='latent' is implemented. Skipping."
            )
        else:
            print("[INFO] Starting SMOTE on latent space + RandomForest classifier...")

            X_train_full = X[train_idx]
            y_train_full = y_win[train_idx]
            X_val_full = X[val_idx]
            y_val_full = y_win[val_idx]
            X_test_full = X[test_idx]
            y_test_full = y_win[test_idx]

            Z_train = encode_dataset(model, X_train_full, device=device)
            Z_val = encode_dataset(model, X_val_full, device=device)
            Z_test = encode_dataset(model, X_test_full, device=device)

            print("[INFO] Applying SMOTE to latent space...")
            sm = SMOTE()
            Z_train_bal, y_train_bal = sm.fit_resample(Z_train, y_train_full)

            print("[INFO] Training RandomForest classifier on balanced latents...")
            clf = RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
            )
            clf.fit(Z_train_bal, y_train_bal)

            from sklearn.metrics import (
                classification_report as cr,
                confusion_matrix as cmx,
            )

            y_val_pred = clf.predict(Z_val)
            y_val_score = clf.predict_proba(Z_val)[:, 1]
            y_test_pred = clf.predict(Z_test)
            y_test_score = clf.predict_proba(Z_test)[:, 1]

            cm_val = cmx(y_val_full, y_val_pred, labels=[0, 1])
            cm_test = cmx(y_test_full, y_test_pred, labels=[0, 1])
            report_val = cr(y_val_full, y_val_pred, output_dict=True, digits=4)
            report_test = cr(y_test_full, y_test_pred, output_dict=True, digits=4)
            auc_val = roc_auc_score(y_val_full, y_val_score)
            auc_test = roc_auc_score(y_test_full, y_test_score)

            print("[INFO] Latent+SMOTE classifier metrics on TEST:")
            print("Confusion matrix:\n", cm_test)
            print("ROC-AUC:", auc_test)

            if save:
                smote_metrics = {
                    "val": {
                        "confusion_matrix": cm_val.tolist(),
                        "classification_report": report_val,
                        "roc_auc": float(auc_val),
                    },
                    "test": {
                        "confusion_matrix": cm_test.tolist(),
                        "classification_report": report_test,
                        "roc_auc": float(auc_test),
                    },
                }
                save_json(
                    smote_metrics,
                    os.path.join(run_dir, "metrics_latent_smote.json"),
                )

    return {
        "run_dir": run_dir,
        "metrics": {
            "ae": {"cm": cm, "report": report, "auc": auc, "thr": thr},
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transformer Autoencoder baseline (ESA spacecraft anomaly)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/folders.yaml",
        help="YAML config file for folder layout (TELEMETRY_DIR, LABELS_CSV, etc.)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run ID (directory name under runs/transformer_ae).",
    )
    parser.add_argument(
        "--no-telecommands",
        action="store_true",
        help="Skip telecommand channels in ingestion (sets DATA.USE_TELECOMMANDS=False).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable saving plots.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save model/metrics/plots.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))

    if args.no_telecommands:
        cfg.setdefault("DATA", {})
        cfg["DATA"]["USE_TELECOMMANDS"] = False

    if args.run_id is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_id = f"transformer_ae_{ts}"

    telem = _ingest(cfg)

    result = train_eval_transformer_ae(
        cfg=cfg,
        telem=telem,
        run_id=args.run_id,
        no_plot=args.no_plot,
        save=not args.no_save,
    )

    print("[INFO] Done. Run dir:", result["run_dir"])


if __name__ == "__main__":
    main()
