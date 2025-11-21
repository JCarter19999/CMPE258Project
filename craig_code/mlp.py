"""
MLP-based Anomaly Detection with Overfitting Reduction and Class Imbalance Handling

Techniques used to reduce overfitting:
1. L2 regularization (alpha parameter)
2. Early stopping with validation set
3. Adaptive learning rate
4. Class weights / sample weights for imbalance
5. Threshold tuning for optimal F1 score

Optional: To use SMOTE for oversampling, uncomment the SMOTE section below
and install imbalanced-learn: pip install imbalanced-learn
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, roc_curve, auc, 
                             precision_recall_curve, average_precision_score,
                             f1_score, precision_score, recall_score)
import numpy as np

# Optional: Uncomment to use SMOTE for oversampling
from imblearn.over_sampling import SMOTE

parquet_train = "/home/cbelshe/CMPE-258/final_project/train.parquet"
parquet_test = "/home/cbelshe/CMPE-258/final_project/test.parquet"
train_df = pd.read_parquet(parquet_train)

channels_to_use_file = "/home/cbelshe/CMPE-258/final_project/target_channels.csv"
channels = pd.read_csv(channels_to_use_file)
channels = list(channels["target_channels"])

X = train_df[channels].values
y = train_df["is_anomaly"].values

# Check class distribution
print("Class distribution:")
print(f"  Normal (0): {np.sum(y == 0):,} ({100 * np.sum(y == 0) / len(y):.2f}%)")
print(f"  Anomaly (1): {np.sum(y == 1):,} ({100 * np.sum(y == 1) / len(y):.2f}%)")
print(f"  Imbalance ratio: {np.sum(y == 0) / np.sum(y == 1):.2f}:1\n")

# Split into train/val/test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Train set: {len(X_train):,} samples")
print(f"Val set: {len(X_val):,} samples")
print(f"Test set: {len(X_test):,} samples\n")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Optional: Apply SMOTE for oversampling (uncomment to use)
print("Applying SMOTE for oversampling...")
smote = SMOTE(random_state=42, k_neighbors=5)
X_train, y_train = smote.fit_resample(X_train, y_train)
print(f"After SMOTE - Train set: {len(X_train):,} samples")
print(f"  Normal (0): {np.sum(y_train == 0):,}")
print(f"  Anomaly (1): {np.sum(y_train == 1):,}\n")

# Compute class weights to handle imbalance
# from sklearn.utils.class_weight import compute_class_weight
# class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
# class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
# print(f"Class weights: {class_weight_dict}\n")

# MLP with regularization and class imbalance handling
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),  # Two hidden layers: 64 and 32 neurons
    activation='relu',             # ReLU activation function
    solver='adam',                # Adam optimizer
    alpha=0.01,                   # L2 regularization (reduces overfitting)
    learning_rate='adaptive',      # Adaptive learning rate
    learning_rate_init=0.001,     # Initial learning rate
    max_iter=500,                 # Maximum iterations
    batch_size=256,               # Batch size for stochastic gradient descent
    random_state=42,
    early_stopping=True,           # Stop if validation score doesn't improve
    validation_fraction=0.1,      # Use 10% of training data for validation
    n_iter_no_change=20,          # Patience for early stopping
    tol=1e-4,                     # Tolerance for optimization
    verbose=True,
    # Note: sklearn MLPClassifier doesn't support class_weight directly,
    # so we'll use sample_weight during fit
)

# Fit with sample weights to handle class imbalance
# sample_weights = np.array([class_weight_dict[y] for y in y_train])
# mlp.fit(X_train, y_train, sample_weight=sample_weights)
mlp.fit(X_train, y_train)

# Get probabilities on validation set for threshold tuning
y_val_proba = mlp.predict_proba(X_val)[:, 1]

# Tune threshold on validation set to maximize F1 score
print("Tuning decision threshold on validation set...")
thresholds = np.arange(0.1, 0.9, 0.01)
best_threshold = 0.5
best_f1 = 0
for threshold in thresholds:
    y_val_pred_thresh = (y_val_proba >= threshold).astype(int)
    f1 = f1_score(y_val, y_val_pred_thresh)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Best threshold: {best_threshold:.3f} (F1: {best_f1:.4f})\n")

# Evaluate on validation set with default threshold
y_val_pred = mlp.predict(X_val)

print("\n" + "="*80)
print("VALIDATION SET RESULTS (Default Threshold=0.5):")
print("="*80)
print(classification_report(y_val, y_val_pred))
print(f"\nConfusion Matrix:\n{confusion_matrix(y_val, y_val_pred)}")
print(f"\nAccuracy: {accuracy_score(y_val, y_val_pred):.4f}")
print(f"Precision: {precision_score(y_val, y_val_pred):.4f}")
print(f"Recall: {recall_score(y_val, y_val_pred):.4f}")
print(f"F1-Score: {f1_score(y_val, y_val_pred):.4f}")
print(f"AUC-ROC: {auc(*roc_curve(y_val, y_val_proba)[:2]):.4f}")
print(f"Average Precision: {average_precision_score(y_val, y_val_proba):.4f}")

# Evaluate with tuned threshold
y_val_pred_tuned = (y_val_proba >= best_threshold).astype(int)
print(f"\n" + "="*80)
print(f"VALIDATION SET RESULTS (Tuned Threshold={best_threshold:.3f}):")
print("="*80)
print(classification_report(y_val, y_val_pred_tuned))
print(f"\nConfusion Matrix:\n{confusion_matrix(y_val, y_val_pred_tuned)}")
print(f"\nAccuracy: {accuracy_score(y_val, y_val_pred_tuned):.4f}")
print(f"Precision: {precision_score(y_val, y_val_pred_tuned):.4f}")
print(f"Recall: {recall_score(y_val, y_val_pred_tuned):.4f}")
print(f"F1-Score: {f1_score(y_val, y_val_pred_tuned):.4f}")

# Evaluate on test set
y_proba = mlp.predict_proba(X_test)[:, 1]
y_pred = mlp.predict(X_test)
y_pred_tuned = (y_proba >= best_threshold).astype(int)

print("\n" + "="*80)
print("TEST SET RESULTS (Default Threshold=0.5):")
print("="*80)
print(classification_report(y_test, y_pred))
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
print(f"AUC-ROC: {auc(*roc_curve(y_test, y_proba)[:2]):.4f}")
print(f"Average Precision: {average_precision_score(y_test, y_proba):.4f}")

print(f"\n" + "="*80)
print(f"TEST SET RESULTS (Tuned Threshold={best_threshold:.3f}):")
print("="*80)
print(classification_report(y_test, y_pred_tuned))
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred_tuned)}")
print(f"\nAccuracy: {accuracy_score(y_test, y_pred_tuned):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_tuned):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_tuned):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_tuned):.4f}")

# Run on test set and save as csv file
print("\n" + "="*80)
print("Generating predictions for test set...")
print("="*80)
test_df = pd.read_parquet(parquet_test)
test_X = test_df[channels].values
test_X = scaler.transform(test_X)

# Use tuned threshold for predictions
test_proba = mlp.predict_proba(test_X)[:, 1]
test_y_pred = (test_proba >= best_threshold).astype(int)

test_df["is_anomaly"] = test_y_pred
# Drop other columns except for is_anomaly and id
test_df = test_df[["id", "is_anomaly"]]
test_df.to_csv("mlp_predictions2.csv", index=False)

print(f"Predictions saved to mlp_predictions.csv")
print(f"Using threshold: {best_threshold:.3f}")
print(f"Predicted anomalies: {np.sum(test_y_pred == 1):,} ({100 * np.sum(test_y_pred == 1) / len(test_y_pred):.2f}%)")
print(f"Predicted normal: {np.sum(test_y_pred == 0):,} ({100 * np.sum(test_y_pred == 0) / len(test_y_pred):.2f}%)")
print("\nDone!")