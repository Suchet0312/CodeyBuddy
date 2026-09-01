"""
EVALUATION METRICS
===================
Interview must-knows:

CLASSIFICATION
- Confusion matrix: rows=actual, cols=predicted -> TP, FP, FN, TN.
- Accuracy = (TP+TN)/total -- MISLEADING on imbalanced data (predicting all-
  majority-class gives high accuracy but is useless).
- Precision = TP/(TP+FP) -- "of predicted positives, how many were right?"
  Optimize when FALSE POSITIVES are costly (e.g. spam filter marking real mail
  as spam).
- Recall (Sensitivity/TPR) = TP/(TP+FN) -- "of actual positives, how many did we
  catch?" Optimize when FALSE NEGATIVES are costly (e.g. cancer screening,
  fraud detection).
- F1 = harmonic mean of precision & recall = 2PR/(P+R) -- use when you need one
  number and both errors matter; harmonic mean punishes a big
  imbalance between P and R more than a plain average would.
- ROC-AUC: plots TPR vs FPR across ALL thresholds; AUC = probability a random
  positive is ranked above a random negative. Threshold-independent, but can be
  overly optimistic on very imbalanced data.
- PR-AUC (Precision-Recall curve): better than ROC-AUC when positives are RARE
  (imbalanced data) because it doesn't reward true negatives, which are
  plentiful and easy in that setting.
- Log loss (cross-entropy): penalizes confident-and-wrong predictions heavily;
  measures probability CALIBRATION, not just the final class decision.

REGRESSION
- MAE: average |error|, robust to outliers, same units as target, harder to
  optimize (not differentiable at 0).
- MSE/RMSE: penalizes large errors quadratically -> sensitive to outliers;
  RMSE is in the same units as the target (easier to interpret than MSE).
- R^2: fraction of variance explained; 1 = perfect, 0 = same as predicting the
  mean, can go negative (model worse than the mean).
- MAPE: percentage error, scale-independent, but blows up / undefined near
  y=0 -- avoid when targets can be zero or near-zero.
"""

import numpy as np
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    log_loss, mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error,
)
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------
# 1. CLASSIFICATION METRICS on an IMBALANCED dataset (90/10) to show why
#    accuracy alone is misleading
# -----------------------------------------------------------------
X, y = make_classification(n_samples=2000, weights=[0.9, 0.1], random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                      stratify=y, random_state=1)
clf = LogisticRegression().fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

# The "lazy" baseline: always predict majority class
baseline_pred = np.zeros_like(y_test)
print("Baseline (always predict 0):")
print(f"  accuracy={accuracy_score(y_test, baseline_pred):.3f}  <- looks great, "
      f"but recall on class 1 = {recall_score(y_test, baseline_pred):.3f} (catches NOTHING)")

print("\nActual model:")
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(f"  confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}")
print(f"  accuracy ={accuracy_score(y_test, y_pred):.3f}")
print(f"  precision={precision_score(y_test, y_pred):.3f}  (of predicted positives, % correct)")
print(f"  recall   ={recall_score(y_test, y_pred):.3f}  (of actual positives, % caught)")
print(f"  f1       ={f1_score(y_test, y_pred):.3f}")
print(f"  roc_auc  ={roc_auc_score(y_test, y_proba):.3f}")
print(f"  pr_auc   ={average_precision_score(y_test, y_proba):.3f}  "
      f"(more informative than ROC-AUC here since class 1 is rare)")
print(f"  log_loss ={log_loss(y_test, y_proba):.3f}")

# -----------------------------------------------------------------
# 2. Precision/Recall trade-off across thresholds
# -----------------------------------------------------------------
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
print("\nPrecision/Recall at a few thresholds:")
for t in [0.2, 0.5, 0.8]:
    idx = np.argmin(np.abs(thresholds - t))
    print(f"  threshold~{t}: precision={precisions[idx]:.3f} recall={recalls[idx]:.3f}")

fpr, tpr, roc_thresh = roc_curve(y_test, y_proba)
print(f"\nROC curve has {len(roc_thresh)} threshold points; "
      f"AUC = area under (fpr, tpr) = {roc_auc_score(y_test, y_proba):.3f}")

# -----------------------------------------------------------------
# 3. REGRESSION METRICS -- show sensitivity to outliers (MAE vs RMSE)
# -----------------------------------------------------------------
rng = np.random.default_rng(0)
y_true = rng.normal(50, 10, 100)
y_pred_reg = y_true + rng.normal(0, 2, 100)
y_pred_reg_outlier = y_pred_reg.copy()
y_pred_reg_outlier[0] += 80          # inject one big miss

print("\nRegression metrics, normal predictions vs one big outlier miss:")
for name, preds in [("normal", y_pred_reg), ("with 1 outlier", y_pred_reg_outlier)]:
    mae = mean_absolute_error(y_true, preds)
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    r2 = r2_score(y_true, preds)
    mape = mean_absolute_percentage_error(y_true, preds)
    print(f"  {name:16s} MAE={mae:6.2f} RMSE={rmse:6.2f} R2={r2:.3f} MAPE={mape:.3f}  "
          f"{'<- RMSE jumps much more than MAE (quadratic penalty)' if 'outlier' in name else ''}")

print("\nKey talking points: accuracy is misleading on imbalanced data, "
      "precision vs recall trade-off and which to prioritize per use case, "
      "PR-AUC vs ROC-AUC for rare positives, MAE robust vs RMSE outlier-"
      "sensitive, R2 vs MAPE limitations.")
