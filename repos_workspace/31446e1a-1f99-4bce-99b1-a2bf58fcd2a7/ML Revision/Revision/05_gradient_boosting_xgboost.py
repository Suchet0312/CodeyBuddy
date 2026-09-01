"""
GRADIENT BOOSTING / XGBOOST
============================
Interview must-knows:
- Boosting is SEQUENTIAL (unlike bagging's parallel trees): each new tree is
  trained to correct the errors of the current ensemble.
- Gradient Boosting view: fit tree_m to the NEGATIVE GRADIENT of the loss
  w.r.t. the current predictions. For squared-error loss, the negative gradient
  is exactly the RESIDUAL (y - y_hat) -- that's why the classic explanation is
  "each tree fits the residuals of the previous ensemble."
    F_m(x) = F_{m-1}(x) + lr * h_m(x),  where h_m fits -dLoss/dF_{m-1}
- learning_rate (shrinkage) trades off with n_estimators: smaller lr needs more
  trees but generalizes better (classic bias/variance + regularization knob).
- Trees are usually SHALLOW (max_depth 3-6) -- weak learners on purpose; boosting
  reduces bias, so you don't want each learner to already overfit.
- Low bias, but CAN overfit if n_estimators too high / lr too high / trees too
  deep with no regularization / early stopping -> unlike Random Forest, more
  trees is NOT always safer.
- XGBoost adds on top of plain GBM:
    - Regularized objective: loss + gamma*(#leaves) + 0.5*lambda*sum(leaf_weight^2)
      (both L1 "alpha" and L2 "lambda" available) -> explicit control of tree
      complexity, not just depth/leaf-count heuristics.
    - Uses 2nd-order (Newton) gradient info: both gradient g and hessian h per
      sample to choose optimal leaf weights: w* = -sum(g) / (sum(h) + lambda)
    - Built-in handling of missing values (learns a default split direction).
    - Column & row subsampling (like RF) for extra regularization/speed.
    - Efficient parallel split-finding (parallelizes *within* one tree's split
      search, not across trees -- boosting itself is still sequential).
- LightGBM/CatBoost: same family, different growth strategy (leaf-wise vs
  level-wise) and categorical handling -- good to be able to name-drop.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import xgboost as xgb

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42)

# -----------------------------------------------------------------
# 1. GRADIENT BOOSTING FROM SCRATCH for regression (fit residuals directly)
#    -- the cleanest way to internalize the "fit the residual" idea.
# -----------------------------------------------------------------
def gradient_boosting_regressor(X, y, n_estimators=50, lr=0.1, max_depth=2):
    F = np.full(len(y), y.mean())          # initial prediction = mean (best constant)
    trees = []
    for _ in range(n_estimators):
        residual = y - F                    # negative gradient of 0.5*(y-F)^2 loss
        tree = DecisionTreeRegressor(max_depth=max_depth).fit(X, residual)
        F = F + lr * tree.predict(X)
        trees.append(tree)
    return trees, y.mean()

def predict_gb(X, trees, init_pred, lr=0.1):
    F = np.full(X.shape[0], init_pred)
    for tree in trees:
        F += lr * tree.predict(X)
    return F

rng = np.random.default_rng(0)
X_toy = rng.normal(size=(150, 2))
y_toy = X_toy[:, 0] ** 2 - 2 * X_toy[:, 1] + rng.normal(scale=0.1, size=150)
trees, init = gradient_boosting_regressor(X_toy, y_toy, n_estimators=100, lr=0.1)
mse = np.mean((y_toy - predict_gb(X_toy, trees, init)) ** 2)
print(f"Scratch GBM (regression) train MSE after boosting: {mse:.4f}")

# -----------------------------------------------------------------
# 2. sklearn GradientBoostingClassifier -- effect of learning_rate / n_estimators
# -----------------------------------------------------------------
print("\nlearning_rate vs n_estimators trade-off (classification):")
for lr in [0.01, 0.1, 0.5]:
    gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=lr,
                                      max_depth=3, random_state=42).fit(X_train, y_train)
    acc = accuracy_score(y_test, gbc.predict(X_test))
    print(f"  lr={lr:<5} n_estimators=100 -> test_acc={acc:.3f}")

# -----------------------------------------------------------------
# 3. XGBoost -- regularization knobs (lambda, gamma) and early stopping
# -----------------------------------------------------------------
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    "objective": "binary:logistic",
    "max_depth": 3,
    "eta": 0.1,                # learning rate
    "lambda": 1.0,             # L2 reg on leaf weights
    "gamma": 0.0,              # min loss reduction to make a further split (complexity penalty)
    "subsample": 0.8,          # row subsampling, like RF
    "colsample_bytree": 0.8,   # feature subsampling, like RF
    "eval_metric": "logloss",
}
evals_result = {}
booster = xgb.train(
    params, dtrain, num_boost_round=200,
    evals=[(dtrain, "train"), (dtest, "test")],
    early_stopping_rounds=15,          # stop once test logloss stops improving
    evals_result=evals_result, verbose_eval=False,
)
print(f"\nXGBoost best_iteration={booster.best_iteration} "
      f"(early stopping prevented training all 200 rounds -> anti-overfitting)")
xgb_preds = (booster.predict(dtest, iteration_range=(0, booster.best_iteration + 1)) >= 0.5).astype(int)
print(f"XGBoost test_acc={accuracy_score(y_test, xgb_preds):.3f}")

importances = booster.get_score(importance_type="gain")
top3 = sorted(importances.items(), key=lambda kv: -kv[1])[:3]
print("Top-3 features by average gain:", top3)

print("\nKey talking points: sequential residual-fitting vs bagging's parallel "
      "trees, lr/n_estimators trade-off, why shallow trees, XGBoost's 2nd-order "
      "(Newton) leaf weights + explicit L1/L2/gamma regularization, early stopping, "
      "row/column subsampling as regularization.")
