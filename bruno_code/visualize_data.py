from __future__ import annotations

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import yaml

from sklearn.decomposition import PCA

from src.data_from_dirs import ingest_folder_layout as _ingest
from src.baselines.transformer_autoencoder_baseline import (
    TransformerAutoencoder,
    make_windows,
    train_val_test_split,
    encode_dataset,
)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def plot_reconstruction(model, X, idx, device, save_path: str | None = None):
    model.eval()
    x = torch.from_numpy(X[idx : idx + 1]).float().to(device)
    with torch.no_grad():
        recon = model(x).cpu().numpy()[0]

    orig = X[idx]

    plt.figure(figsize=(12, 5))
    plt.plot(orig.mean(axis=1), label="Original (mean over features)", alpha=0.8)
    plt.plot(recon.mean(axis=1), label="Reconstruction (mean over features)", alpha=0.8)
    plt.title(f"Original vs Reconstruction (window {idx})")
    plt.xlabel("Time step")
    plt.ylabel("Normalized telemetry")
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_error_heatmap(
    model,
    X,
    idx: int,
    device,
    feat_cols: list[str],
    save_path: str | None = None,
):
    model.eval()
    x = torch.from_numpy(X[idx : idx + 1]).float().to(device)
    with torch.no_grad():
        recon = model(x).cpu().numpy()[0]

    err = (recon - X[idx]) ** 2

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        err.T,
        cmap="inferno",
        yticklabels=feat_cols,
        cbar=True,
    )
    plt.title(f"Per-feature reconstruction error (window {idx})")
    plt.xlabel("Time step")
    plt.ylabel("Feature")

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_latent_pca(Z: np.ndarray, y: np.ndarray, save_path: str | None = None):
    mask = ~np.isnan(Z).any(axis=1)
    Z = Z[mask]
    y = y[mask]
    if Z.shape[0] == 0:
        print("[WARN] All latent vectors contained NaNs; skipping PCA plot.")
        return

    max_points = 20000
    if Z.shape[0] > max_points:
        idx = np.random.choice(Z.shape[0], size=max_points, replace=False)
        Z = Z[idx]
        y = y[idx]

    pca = PCA(n_components=2)
    Z2 = pca.fit_transform(Z)

    plt.figure(figsize=(7, 6))
    sc = plt.scatter(Z2[:, 0], Z2[:, 1], c=y, cmap="coolwarm", alpha=0.4, s=5)
    plt.colorbar(sc, label="Label (0=normal, 1=anomaly)")
    plt.title("Latent space (PCA of Transformer encoder output)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_recon_bar(
    model,
    X,
    idx: int,
    device,
    feat_cols: list[str],
    save_path: str | None = None,
):
    model.eval()
    x = torch.from_numpy(X[idx : idx + 1]).float().to(device)
    with torch.no_grad():
        recon = model(x).cpu().numpy()[0]

    orig = X[idx]
    err = ((recon - orig) ** 2).mean(axis=0)

    plt.figure(figsize=(12, 4))
    plt.bar(range(len(feat_cols)), err)
    plt.title("Mean reconstruction error per feature (window {})".format(idx))
    plt.xticks(range(len(feat_cols)), feat_cols, rotation=90)
    plt.ylabel("MSE")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_attention(
    model,
    X,
    idx: int,
    device,
    save_path: str | None = None,
):
    if not hasattr(model, "last_attn"):
        print("[INFO] Model has no attribute `last_attn`; skipping attention heatmap.")
        return

    model.eval()
    x = torch.from_numpy(X[idx : idx + 1]).float().to(device)
    with torch.no_grad():
        _ = model(x)

    attn = model.last_attn
    if attn.ndim == 4:
        attn = attn[0]
    attn_mean = attn.mean(0)

    plt.figure(figsize=(8, 6))
    sns.heatmap(attn_mean, cmap="viridis")
    plt.title("Mean self-attention weights (window {})".format(idx))
    plt.xlabel("Key position")
    plt.ylabel("Query position")

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualization suite for Transformer AE spacecraft anomaly model"
    )
    parser.add_argument("--config", type=str, required=True, help="YAML config file")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained transformer_autoencoder.pt",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=10,
        help="Which global window index to visualize (will be mapped into test set).",
    )
    parser.add_argument(
        "--no-telecommands",
        action="store_true",
        help="Disable telecommand features in ingestion.",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))

    if args.no_telecommands:
        cfg.setdefault("DATA", {})
        cfg["DATA"]["USE_TELECOMMANDS"] = False

    data_cfg = cfg["DATA"]

    print("[INFO] Ingesting telemetry...")
    telem = _ingest(cfg)

    window = int(data_cfg.get("WINDOW", 128))
    stride = int(data_cfg.get("STRIDE", 4))
    max_windows = data_cfg.get("MAX_WINDOWS", None)
    if max_windows is not None:
        max_windows = int(max_windows)

    X, y_win, feat_cols = make_windows(
        telem,
        window=window,
        stride=stride,
        max_windows=max_windows,
    )

    N = X.shape[0]
    print(f"[INFO] Windows: {N}, seq_len={X.shape[1]}, features={X.shape[2]}")

    train_idx, val_idx, test_idx = train_val_test_split(
        N,
        train_frac=float(data_cfg.get("TRAIN_SPLIT", 0.6)),
        val_frac=float(data_cfg.get("VAL_SPLIT", 0.2)),
    )

    train_flat = X[train_idx].reshape(-1, X.shape[-1])
    mean = train_flat.mean(axis=0, keepdims=True)
    std = train_flat.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    X = (X - mean) / std

    if len(test_idx) == 0:
        raise RuntimeError("Test set is empty; check your split fractions.")
    idx_local = args.index % len(test_idx)
    idx_global = test_idx[idx_local]
    print(f"[INFO] Visualizing global window index {idx_global} (test[{idx_local}]).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    model_cfg = cfg.get("MODEL", {})
    d_model = int(model_cfg.get("D_MODEL", 128))
    nhead = int(model_cfg.get("NHEAD", 4))
    num_layers = int(model_cfg.get("NUM_LAYERS", 2))
    dropout = float(model_cfg.get("DROPOUT", 0.1))

    model = TransformerAutoencoder(
        input_dim=X.shape[2],
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.eval()

    model_dir = os.path.dirname(os.path.abspath(args.model))
    out_dir = ensure_dir(os.path.join(model_dir, "viz"))
    print("[INFO] Saving visualizations to:", out_dir)

    plot_reconstruction(
        model,
        X,
        idx_global,
        device,
        save_path=os.path.join(out_dir, "reconstruction_overlay.png"),
    )

    plot_error_heatmap(
        model,
        X,
        idx_global,
        device,
        feat_cols,
        save_path=os.path.join(out_dir, "reconstruction_heatmap.png"),
    )

    print("[INFO] Encoding dataset into latent space for PCA...")
    Z = encode_dataset(model, X, device=device, batch_size=256)
    plot_latent_pca(
        Z,
        y_win,
        save_path=os.path.join(out_dir, "latent_pca.png"),
    )

    plot_feature_recon_bar(
        model,
        X,
        idx_global,
        device,
        feat_cols,
        save_path=os.path.join(out_dir, "feature_recon_error.png"),
    )

    plot_attention(
        model,
        X,
        idx_global,
        device,
        save_path=os.path.join(out_dir, "attention_heatmap.png"),
    )

    print("[INFO] Visualization complete.")


if __name__ == "__main__":
    main()
