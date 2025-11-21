"""
Isolation Forest for Anomaly Detection

Isolation Forest is an unsupervised learning algorithm that works well for anomaly detection,
especially with imbalanced data. It isolates anomalies by randomly selecting features and
splitting values, making anomalies easier to isolate (fewer splits needed).

Key advantages:
- Unsupervised (doesn't need labels during training)
- Handles high-dimensional data well
- Naturally handles class imbalance
- Fast training and prediction
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, roc_curve, auc, 
                             precision_recall_curve, average_precision_score,
                             f1_score, precision_score, recall_score)
import numpy as np


def get_events(y_binary):
    """
    Convert point-wise binary labels to events (contiguous segments).
    Returns list of (start, end) tuples for each event.
    """
    y_binary = np.asarray(y_binary).astype(int)
    edges = np.diff(np.r_[0, y_binary, 0])
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0] - 1
    return list(zip(starts, ends))


def evaluate_eventwise(y_true, y_pred):
    """
    Evaluate predictions event-wise:
    - TP: predicted event overlaps with at least one true anomaly event
    - FN: true anomaly event has no overlapping predicted event
    - FP: predicted event doesn't overlap with any true anomaly event
    
    Returns: (tp, fp, fn, precision, recall, f0_5)
    """
    true_events = get_events(y_true)
    pred_events = get_events(y_pred)
    
    # Count true positives: true events that have at least one overlapping prediction
    tp = 0
    for true_start, true_end in true_events:
        # Check if any predicted event overlaps with this true event
        if any(not (pred_end < true_start or pred_start > true_end) 
               for pred_start, pred_end in pred_events):
            tp += 1
    
    # False negatives: true events that weren't detected
    fn = len(true_events) - tp
    
    # False positives: predicted events that don't overlap with any true event
    fp = 0
    for pred_start, pred_end in pred_events:
        # Check if this predicted event overlaps with any true event
        if not any(not (true_end < pred_start or true_start > pred_end) 
                   for true_start, true_end in true_events):
            fp += 1
    
    # Calculate precision, recall, and F0.5
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # F0.5 = (1 + 0.5²) * precision * recall / (0.5² * precision + recall)
    #      = 1.25 * precision * recall / (0.25 * precision + recall)
    if precision == 0 and recall == 0:
        f0_5 = 0.0
    else:
        f0_5 = 1.25 * precision * recall / (0.25 * precision + recall)
    
    return tp, fp, fn, precision, recall, f0_5

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
# Note: Isolation Forest is unsupervised, but we still need labels for evaluation
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Train set: {len(X_train):,} samples")
print(f"Val set: {len(X_val):,} samples")
print(f"Test set: {len(X_test):,} samples\n")

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Train Isolation Forest
# Key parameters:
# - contamination: expected proportion of anomalies (can be 'auto' or a float)
# - n_estimators: number of trees (more = better but slower)
# - max_samples: samples per tree (auto = min(256, n_samples))
# - random_state: for reproducibility
print("Training Isolation Forest...")

# Estimate contamination from training data
contamination_rate = np.sum(y_train == 1) / len(y_train)
print(f"Estimated contamination rate: {contamination_rate:.4f} ({contamination_rate*100:.2f}%)\n")

# Try different contamination values and select best on validation set
contamination_values = [contamination_rate, 'auto', 0.1, 0.05, 0.15]
best_contamination = contamination_rate
best_f1 = 0
best_model = None

print("Tuning contamination parameter on validation set...")
for contam in contamination_values:
    if contam == 'auto':
        contam_val = 'auto'
        contam_str = 'auto'
    else:
        contam_val = float(contam)
        contam_str = f"{contam_val:.4f}"
    
    iso_forest = IsolationForest(
        n_estimators=250,              # Number of trees
        max_samples='auto',             # Samples per tree (auto = min(256, n_samples))
        contamination=contam_val,       # Expected proportion of anomalies
        max_features=1.0,               # Use all features
        bootstrap=False,                # Don't bootstrap (use all samples)
        random_state=42,
        n_jobs=-1,                      # Use all CPU cores
        verbose=0
    )
    
    # Fit on training data (unsupervised - no labels needed)
    iso_forest.fit(X_train)
    
    # Predict on validation set
    # Returns: -1 for anomalies, 1 for normal
    y_val_pred_iso = iso_forest.predict(X_val)
    # Convert to 0/1 format (1 = anomaly, 0 = normal)
    y_val_pred = (y_val_pred_iso == -1).astype(int)
    
    # Calculate F1 score
    f1 = f1_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred, zero_division=0)
    recall = recall_score(y_val, y_val_pred, zero_division=0)
    
    print(f"  Contamination={contam_str:8s}: F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_contamination = contam_val
        best_model = iso_forest

print(f"\nBest contamination: {best_contamination} (F1: {best_f1:.4f})\n")

# Train final model with best contamination on full training set
print("Training final model with best parameters...")
if best_contamination != best_model.contamination:
    # Retrain if needed
    iso_forest = IsolationForest(
        n_estimators=200,
        max_samples='auto',
        contamination=best_contamination,
        max_features=1.0,
        bootstrap=False,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    iso_forest.fit(X_train)
else:
    iso_forest = best_model

# Get anomaly scores (lower = more anomalous)
# We can also use decision_function which returns the anomaly score
y_val_scores = iso_forest.decision_function(X_val)
y_test_scores = iso_forest.decision_function(X_test)

# Predictions using the model's threshold
y_val_pred_iso = iso_forest.predict(X_val)
y_val_pred = (y_val_pred_iso == -1).astype(int)

y_test_pred_iso = iso_forest.predict(X_test)
y_test_pred = (y_test_pred_iso == -1).astype(int)

# Optimize threshold for event-wise F0.5 score
print("Optimizing threshold for event-wise F0.5 score on validation set...")
# Normalize scores to [0, 1] range for threshold tuning
# Note: Lower scores = more anomalous for Isolation Forest
y_val_scores_norm = (y_val_scores - y_val_scores.min()) / (y_val_scores.max() - y_val_scores.min() + 1e-8)
thresholds = np.arange(0.05, 0.95, 0.01)
best_threshold = 0.5
best_f0_5 = 0
best_event_metrics = None

for threshold in thresholds:
    # Lower scores = more anomalous, so lower threshold = more predictions
    y_val_pred_thresh = (y_val_scores_norm <= threshold).astype(int)
    
    # Evaluate event-wise
    tp, fp, fn, precision, recall, f0_5 = evaluate_eventwise(y_val, y_val_pred_thresh)
    
    if f0_5 > best_f0_5:
        best_f0_5 = f0_5
        best_threshold = threshold
        best_event_metrics = (tp, fp, fn, precision, recall, f0_5)

print(f"Best threshold: {best_threshold:.3f}")
if best_event_metrics:
    tp, fp, fn, precision, recall, f0_5 = best_event_metrics
    print(f"Event-wise metrics at best threshold:")
    print(f"  True Positives (TP): {tp}")
    print(f"  False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F0.5 Score: {f0_5:.4f}\n")

# Evaluate on validation set
print("="*80)
print("VALIDATION SET RESULTS (Model's Default Threshold):")
print("="*80)
print(classification_report(y_val, y_val_pred))
print(f"\nConfusion Matrix:\n{confusion_matrix(y_val, y_val_pred)}")
print(f"\nAccuracy: {accuracy_score(y_val, y_val_pred):.4f}")
print(f"Precision: {precision_score(y_val, y_val_pred, zero_division=0):.4f}")
print(f"Recall: {recall_score(y_val, y_val_pred, zero_division=0):.4f}")
print(f"F1-Score: {f1_score(y_val, y_val_pred):.4f}")

# Use negative scores for ROC (lower score = anomaly)
y_val_scores_roc = -y_val_scores  # Invert so higher = anomaly
try:
    print(f"AUC-ROC: {auc(*roc_curve(y_val, y_val_scores_roc)[:2]):.4f}")
    print(f"Average Precision: {average_precision_score(y_val, y_val_scores_roc):.4f}")
except ValueError:
    print("AUC-ROC: N/A (insufficient class diversity)")

# Evaluate with tuned threshold
y_val_scores_norm = (y_val_scores - y_val_scores.min()) / (y_val_scores.max() - y_val_scores.min() + 1e-8)
y_val_pred_tuned = (y_val_scores_norm <= best_threshold).astype(int)
print(f"\n" + "="*80)
print(f"VALIDATION SET RESULTS (Tuned Threshold={best_threshold:.3f}):")
print("="*80)
print("Point-wise metrics:")
print(classification_report(y_val, y_val_pred_tuned))
print(f"\nConfusion Matrix:\n{confusion_matrix(y_val, y_val_pred_tuned)}")
print(f"\nAccuracy: {accuracy_score(y_val, y_val_pred_tuned):.4f}")
print(f"Precision: {precision_score(y_val, y_val_pred_tuned, zero_division=0):.4f}")
print(f"Recall: {recall_score(y_val, y_val_pred_tuned, zero_division=0):.4f}")
print(f"F1-Score: {f1_score(y_val, y_val_pred_tuned):.4f}")

# Event-wise evaluation
print(f"\nEvent-wise metrics:")
tp, fp, fn, precision_evt, recall_evt, f0_5_evt = evaluate_eventwise(y_val, y_val_pred_tuned)
print(f"  True Positives (TP): {tp}")
print(f"  False Positives (FP): {fp}")
print(f"  False Negatives (FN): {fn}")
print(f"  Precision: {precision_evt:.4f}")
print(f"  Recall: {recall_evt:.4f}")
print(f"  F0.5 Score: {f0_5_evt:.4f}")

# Evaluate on test set
print("\n" + "="*80)
print("TEST SET RESULTS (Model's Default Threshold):")
print("="*80)
print(classification_report(y_test, y_test_pred))
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_test_pred)}")
print(f"\nAccuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_test_pred, zero_division=0):.4f}")
print(f"Recall: {recall_score(y_test, y_test_pred, zero_division=0):.4f}")
print(f"F1-Score: {f1_score(y_test, y_test_pred):.4f}")

y_test_scores_roc = -y_test_scores
try:
    print(f"AUC-ROC: {auc(*roc_curve(y_test, y_test_scores_roc)[:2]):.4f}")
    print(f"Average Precision: {average_precision_score(y_test, y_test_scores_roc):.4f}")
except ValueError:
    print("AUC-ROC: N/A (insufficient class diversity)")

# Evaluate with tuned threshold
y_test_scores_norm = (y_test_scores - y_test_scores.min()) / (y_test_scores.max() - y_test_scores.min() + 1e-8)
y_test_pred_tuned = (y_test_scores_norm <= best_threshold).astype(int)
print(f"\n" + "="*80)
print(f"TEST SET RESULTS (Tuned Threshold={best_threshold:.3f}):")
print("="*80)
print("Point-wise metrics:")
print(classification_report(y_test, y_test_pred_tuned))
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_test_pred_tuned)}")
print(f"\nAccuracy: {accuracy_score(y_test, y_test_pred_tuned):.4f}")
print(f"Precision: {precision_score(y_test, y_test_pred_tuned, zero_division=0):.4f}")
print(f"Recall: {recall_score(y_test, y_test_pred_tuned, zero_division=0):.4f}")
print(f"F1-Score: {f1_score(y_test, y_test_pred_tuned):.4f}")

# Event-wise evaluation
print(f"\nEvent-wise metrics:")
tp_test, fp_test, fn_test, precision_evt_test, recall_evt_test, f0_5_evt_test = evaluate_eventwise(y_test, y_test_pred_tuned)
print(f"  True Positives (TP): {tp_test}")
print(f"  False Positives (FP): {fp_test}")
print(f"  False Negatives (FN): {fn_test}")
print(f"  Precision: {precision_evt_test:.4f}")
print(f"  Recall: {recall_evt_test:.4f}")
print(f"  F0.5 Score: {f0_5_evt_test:.4f}")

# Run on test set and save as csv file
print("\n" + "="*80)
print("Generating predictions for test set...")
print("="*80)
test_df = pd.read_parquet(parquet_test)
test_X = test_df[channels].values
test_X = scaler.transform(test_X)

# Get anomaly scores and apply tuned threshold
test_scores = iso_forest.decision_function(test_X)
test_scores_norm = (test_scores - test_scores.min()) / (test_scores.max() - test_scores.min() + 1e-8)
test_y_pred = (test_scores_norm <= best_threshold).astype(int)

test_df["is_anomaly"] = test_y_pred
# Drop other columns except for is_anomaly and id
test_df = test_df[["id", "is_anomaly"]]
test_df.to_csv("isolation_forest_predictions2.csv", index=False)

print(f"Predictions saved to isolation_forest_predictions2.csv")
print(f"Using threshold: {best_threshold:.3f} (optimized for event-wise F0.5)")
print(f"Predicted anomalies: {np.sum(test_y_pred == 1):,} ({100 * np.sum(test_y_pred == 1) / len(test_y_pred):.2f}%)")
print(f"Predicted normal: {np.sum(test_y_pred == 0):,} ({100 * np.sum(test_y_pred == 0) / len(test_y_pred):.2f}%)")
print("\nDone!")

"""
initial results:
Class distribution:
  Normal (0): 13,184,217 (89.52%)
  Anomaly (1): 1,544,104 (10.48%)
  Imbalance ratio: 8.54:1

Train set: 11,782,656 samples
Val set: 1,472,832 samples
Test set: 1,472,833 samples

Training Isolation Forest...
Estimated contamination rate: 0.1048 (10.48%)

Tuning contamination parameter on validation set...
  Contamination=0.1048  : F1=0.2911, Precision=0.2911, Recall=0.2910
  Contamination=auto    : F1=0.2990, Precision=0.4555, Recall=0.2226
  Contamination=0.1000  : F1=0.2922, Precision=0.2993, Recall=0.2854
  Contamination=0.0500  : F1=0.2956, Precision=0.4578, Recall=0.2183
  Contamination=0.1500  : F1=0.2798, Precision=0.2377, Recall=0.3398

Best contamination: auto (F1: 0.2990)

Training final model with best parameters...
Tuning threshold on validation set...
Best threshold: 0.750 (F1: 0.3044)

================================================================================
VALIDATION SET RESULTS (Model's Default Threshold):
================================================================================
              precision    recall  f1-score   support

           0       0.91      0.97      0.94   1318422
           1       0.46      0.22      0.30    154410

    accuracy                           0.89   1472832
   macro avg       0.68      0.60      0.62   1472832
weighted avg       0.87      0.89      0.87   1472832


Confusion Matrix:
[[1277348   41074]
 [ 120043   34367]]

Accuracy: 0.8906
Precision: 0.4555
Recall: 0.2226
F1-Score: 0.2990
AUC-ROC: 0.6624
Average Precision: 0.2996

================================================================================
VALIDATION SET RESULTS (Tuned Threshold=0.750):
================================================================================
              precision    recall  f1-score   support

           0       0.92      0.96      0.94   1318422
           1       0.41      0.24      0.30    154410

    accuracy                           0.88   1472832
   macro avg       0.66      0.60      0.62   1472832
weighted avg       0.86      0.88      0.87   1472832


Confusion Matrix:
[[1265335   53087]
 [ 117162   37248]]

Accuracy: 0.8844
Precision: 0.4123
Recall: 0.2412
F1-Score: 0.3044

================================================================================
TEST SET RESULTS (Model's Default Threshold):
================================================================================
              precision    recall  f1-score   support

           0       0.91      0.97      0.94   1318422
           1       0.45      0.22      0.30    154411

    accuracy                           0.89   1472833
   macro avg       0.68      0.59      0.62   1472833
weighted avg       0.87      0.89      0.87   1472833


Confusion Matrix:
[[1276758   41664]
 [ 120268   34143]]

Accuracy: 0.8901
Precision: 0.4504
Recall: 0.2211
F1-Score: 0.2966
AUC-ROC: 0.6620
Average Precision: 0.2994

================================================================================
TEST SET RESULTS (Tuned Threshold=0.750):
================================================================================
              precision    recall  f1-score   support

           0       0.92      0.96      0.94   1318422
           1       0.41      0.24      0.30    154411

    accuracy                           0.88   1472833
   macro avg       0.66      0.60      0.62   1472833
weighted avg       0.86      0.88      0.87   1472833


Confusion Matrix:
[[1264656   53766]
 [ 117163   37248]]

Accuracy: 0.8839
Precision: 0.4093
Recall: 0.2412
F1-Score: 0.3035

================================================================================
Generating predictions for test set...
================================================================================
Predictions saved to isolation_forest_predictions.csv
Using threshold: 0.750
Predicted anomalies: 439,336 (84.28%)
Predicted normal: 81,944 (15.72%)
"""