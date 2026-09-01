"""
TRAIN/TEST SPLIT, CROSS-VALIDATION, BIAS-VARIANCE, OVER/UNDERFITTING
=======================================================================
Interview must-knows:
- Train/test split: hold out data the model NEVER sees during fitting to get an
  honest estimate of generalization. Typical splits 70/30, 80/20. For time
  series, split chronologically (never randomly shuffle -- that leaks the future
  into training).
- Validation set / k-fold CV: used to tune hyperparameters WITHOUT touching the
  final test set (otherwise you leak test info into your model choices ->
  optimistic bias). k-fold CV: split into k folds, train on k-1, validate on the
  held-out fold, rotate k times, average the score -> more robust / lower
  variance estimate than a single train/val split, at the cost of k times the
  compute.
- Stratified k-fold: preserves class proportions in each fold -- important for
  imbalanced classification.
- Bias: error from wrong/oversimplified assumptions (model too simple to
  capture the true pattern) -> UNDERFITTING, poor performance on train AND test.
- Variance: error from sensitivity to the specific training sample (model too
  complex, memorizes noise) -> OVERFITTING, great train performance, poor test
  performance.
- Bias-Variance decomposition: Expected Test Error = Bias^2 + Variance +
  Irreducible Error. You can't drive both bias and variance to zero
  simultaneously with a fixed amount of data -- that's the "trade-off."
- Fixes for overfitting: more data, simpler model, regularization (L1/L2),
  dropout (NN), early stopping, pruning (trees), ensembling, feature selection,
  more aggressive cross-validation during tuning.
- Fixes for underfitting: more complex model, more/better features, less
  regularization, train longer.
"""

import numpy as np
from sklearn.model_selection import (
    train_test_split, KFold, StratifiedKFold, cross_val_score, learning_curve
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC

# -----------------------------------------------------------------
# 1. Basic train/test split
# -----------------------------------------------------------------
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)   # stratify keeps class balance
print(f"Train size={len(X_train)}, Test size={len(X_test)}")
print("Class balance train:", np.bincount(y_train) / len(y_train))
print("Class balance test: ", np.bincount(y_test) / len(y_test))

# -----------------------------------------------------------------
# 2. K-Fold vs Stratified K-Fold
# -----------------------------------------------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=0)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

print("\nPlain KFold fold class balances (can drift on imbalanced data):")
for i, (_, val_idx) in enumerate(kf.split(X, y)):
    print(f"  fold {i}: {np.bincount(y[val_idx]) / len(val_idx)}")

print("StratifiedKFold fold class balances (matches overall ratio each time):")
for i, (_, val_idx) in enumerate(skf.split(X, y)):
    print(f"  fold {i}: {np.bincount(y[val_idx]) / len(val_idx)}")

scores = cross_val_score(SVC(gamma="scale"), X, y, cv=skf)
print(f"\n5-fold CV accuracy: mean={scores.mean():.3f} std={scores.std():.3f} "
      f"(std tells you how sensitive performance is to the split)")

# -----------------------------------------------------------------
# 3. UNDERFITTING vs OVERFITTING using polynomial regression complexity
# -----------------------------------------------------------------
rng = np.random.default_rng(0)
X_poly = np.sort(rng.uniform(-3, 3, 40)).reshape(-1, 1)
y_poly = X_poly.ravel() ** 3 - 3 * X_poly.ravel() + rng.normal(scale=3, size=40)
Xp_train, Xp_test, yp_train, yp_test = train_test_split(X_poly, y_poly, test_size=0.3, random_state=1)

def poly_features(X, degree):
    return np.hstack([X ** d for d in range(1, degree + 1)])

print("\nModel complexity (polynomial degree) vs train/test error:")
for degree in [1, 3, 12]:
    Xtr_p = poly_features(Xp_train, degree)
    Xte_p = poly_features(Xp_test, degree)
    model = LinearRegression().fit(Xtr_p, yp_train)
    train_mse = np.mean((model.predict(Xtr_p) - yp_train) ** 2)
    test_mse = np.mean((model.predict(Xte_p) - yp_test) ** 2)
    verdict = "underfit" if degree == 1 else ("overfit" if degree == 12 else "good fit")
    print(f"  degree={degree:2d} train_mse={train_mse:8.2f} test_mse={test_mse:8.2f}  <- {verdict}")

# -----------------------------------------------------------------
# 4. Learning curves -- diagnose bias vs variance from train/val curves directly
# -----------------------------------------------------------------
print("\nLearning curve for a DEEP tree (expect high variance signature: "
      "big gap between train and val score):")
train_sizes, train_scores, val_scores = learning_curve(
    DecisionTreeRegressor(max_depth=None), X_poly, y_poly,
    train_sizes=[0.3, 0.6, 1.0], cv=5, scoring="neg_mean_squared_error")
for size, tr, val in zip(train_sizes, train_scores.mean(axis=1), val_scores.mean(axis=1)):
    print(f"  n_train={size:3d}  train_mse={-tr:8.2f}  val_mse={-val:8.2f}  "
          f"gap={(-val)-(-tr):8.2f}")

print("\nKey talking points: why we need a held-out set, k-fold reduces variance "
      "of the performance estimate, stratification for imbalanced classes, "
      "bias^2 + variance + irreducible error decomposition, reading learning "
      "curves to diagnose which problem you have.")
