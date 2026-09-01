"""
RANDOM FOREST
=============
Interview must-knows:
- Ensemble of decision trees trained via BAGGING (Bootstrap Aggregating):
    1. Draw N bootstrap samples (sampling WITH replacement) from training data.
    2. Train one tree per sample, but at each split only consider a random
       SUBSET of features (max_features) -- this decorrelates the trees, which
       is what actually reduces variance (bagging alone with correlated trees
       gives limited variance reduction).
    3. Aggregate: majority vote (classification) or average (regression).
- Bias roughly stays the same as a single deep tree; VARIANCE drops a lot because
  we're averaging many decorrelated, low-bias/high-variance learners.
  Var(mean of B correlated vars) = rho*sigma^2 + (1-rho)*sigma^2/B -> lower
  correlation rho between trees is what allows variance to keep shrinking with B.
- Out-of-Bag (OOB) score: ~37% of samples are left out of each bootstrap
  (as B->inf, P(not picked) = (1-1/n)^n -> 1/e ≈ 0.368) -> free validation
  estimate without a separate holdout set.
- Robust to outliers/scaling (tree-based), handles nonlinearity, gives feature
  importance, parallelizable (trees are independent) -- unlike boosting.
- Weakness: many trees -> less interpretable, larger memory, can still overfit
  noisy data if trees are too deep and there are too few features per split.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42)

# -----------------------------------------------------------------
# 1. Single tree vs forest -- shows the variance-reduction effect directly
# -----------------------------------------------------------------
single_tree = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
forest = RandomForestClassifier(
    n_estimators=200,
    max_features="sqrt",     # classic default for classification (log2 also common)
    oob_score=True,          # free validation estimate, no extra holdout needed
    n_jobs=-1,                # trees are independent -> trivially parallelizable
    random_state=42,
).fit(X_train, y_train)

print(f"Single tree     test_acc={accuracy_score(y_test, single_tree.predict(X_test)):.3f}")
print(f"Random Forest   test_acc={accuracy_score(y_test, forest.predict(X_test)):.3f}  "
      f"OOB_acc={forest.oob_score_:.3f}")

# -----------------------------------------------------------------
# 2. Demonstrate bootstrap sampling + the ~1/e out-of-bag fraction by hand
# -----------------------------------------------------------------
rng = np.random.default_rng(0)
n = len(X_train)
bootstrap_idx = rng.integers(0, n, size=n)          # sample WITH replacement
oob_fraction = 1 - len(np.unique(bootstrap_idx)) / n
print(f"\nEmpirical OOB fraction for one bootstrap sample: {oob_fraction:.3f} "
      f"(theoretical limit 1/e = {1/np.e:.3f})")

# -----------------------------------------------------------------
# 3. Effect of n_estimators and max_features (things you'd tune)
# -----------------------------------------------------------------
print("\nEffect of n_estimators:")
for n_est in [1, 10, 50, 200]:
    rf = RandomForestClassifier(n_estimators=n_est, random_state=42, n_jobs=-1).fit(X_train, y_train)
    print(f"  n_estimators={n_est:4d} -> test_acc={accuracy_score(y_test, rf.predict(X_test)):.3f}")

print("\nEffect of max_features (controls tree decorrelation):")
for mf in [0.1, "sqrt", 1.0]:
    rf = RandomForestClassifier(n_estimators=100, max_features=mf, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    label = mf if isinstance(mf, str) else f"{mf} (=all features, less decorrelation)"
    print(f"  max_features={label} -> test_acc={accuracy_score(y_test, rf.predict(X_test)):.3f}")

# -----------------------------------------------------------------
# 4. Feature importance (averaged impurity decrease across all trees)
# -----------------------------------------------------------------
top5 = np.argsort(forest.feature_importances_)[::-1][:5]
print("\nTop-5 features by mean impurity decrease:")
for i in top5:
    print(f"  {data.feature_names[i]:25s} {forest.feature_importances_[i]:.3f}")

# Regression variant works identically (average instead of vote)
X_reg, y_reg = np.random.default_rng(1).normal(size=(200, 4)), None
print("\nRandomForestRegressor uses the SAME bagging idea, aggregates via mean() "
      "instead of majority vote.")

print("\nKey talking points: bagging + feature subsampling -> decorrelated trees "
      "-> variance reduction, OOB score as free validation, parallel training "
      "(vs sequential boosting), bias stays ~same as a single tree.")
