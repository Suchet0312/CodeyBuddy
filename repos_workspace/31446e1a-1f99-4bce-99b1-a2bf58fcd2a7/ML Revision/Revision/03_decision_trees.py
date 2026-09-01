"""
DECISION TREES
==============
Interview must-knows:
- Greedy, recursive partitioning: at each node pick the (feature, threshold) split
  that maximizes "impurity decrease".
- Classification impurity measures:
    Gini = 1 - sum(p_i^2)                (0 = pure, computationally cheaper)
    Entropy = -sum(p_i * log2(p_i))      (0 = pure, "Information Gain" = entropy
                                           decrease from a split)
  Gini and Entropy almost always agree in practice; Gini is faster (no log).
- Regression trees split to minimize variance / MSE within each leaf.
- No scaling needed (splits are threshold-based, monotonic transforms don't matter).
- Prone to OVERFITTING (can memorize training data with unlimited depth) ->
  control with max_depth, min_samples_split, min_samples_leaf, ccp_alpha (cost-
  complexity pruning).
- High variance, low bias model -> this is exactly why Random Forest / Boosting
  (ensembles of trees) work so well.
- Feature importance = total impurity decrease attributable to a feature, summed
  over all splits that use it (biased toward high-cardinality features -- know this
  caveat).
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------------------------------------------------
# 1. GINI / ENTROPY FROM SCRATCH -- a common "implement the impurity function" ask
# -----------------------------------------------------------------
def gini(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return 1 - np.sum(p ** 2)

def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return -np.sum(p * np.log2(p + 1e-12))

def information_gain(y, y_left, y_right, impurity_fn=entropy):
    n, n_l, n_r = len(y), len(y_left), len(y_right)
    return impurity_fn(y) - (n_l / n) * impurity_fn(y_left) - (n_r / n) * impurity_fn(y_right)

def best_split(X_col, y, impurity_fn=gini):
    """Try every unique threshold; return the one with max impurity decrease."""
    best_gain, best_thresh = -1, None
    for thresh in np.unique(X_col):
        left_mask = X_col <= thresh
        if left_mask.all() or (~left_mask).all():
            continue
        gain = information_gain(y, y[left_mask], y[~left_mask], impurity_fn)
        if gain > best_gain:
            best_gain, best_thresh = gain, thresh
    return best_thresh, best_gain

y_demo = np.array([0, 0, 0, 1, 1, 1, 1])
x_demo = np.array([1, 2, 3, 4, 5, 6, 7])
thresh, gain = best_split(x_demo, y_demo)
print(f"Gini={gini(y_demo):.3f} Entropy={entropy(y_demo):.3f}")
print(f"Best split on toy data: threshold<= {thresh}, info gain={gain:.3f}")

# -----------------------------------------------------------------
# 2. SKLEARN classifier + inspecting the learned tree
# -----------------------------------------------------------------
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42)

# Unrestricted tree -> near-perfect train acc, worse test acc (overfitting demo)
deep_tree = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
pruned_tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10,
                                      random_state=42).fit(X_train, y_train)

for name, tree in [("Unrestricted (overfits)", deep_tree), ("Pruned (max_depth=3)", pruned_tree)]:
    train_acc = accuracy_score(y_train, tree.predict(X_train))
    test_acc = accuracy_score(y_test, tree.predict(X_test))
    print(f"{name:28s} train_acc={train_acc:.3f} test_acc={test_acc:.3f} "
          f"depth={tree.get_depth()} leaves={tree.get_n_leaves()}")

top_features = np.argsort(pruned_tree.feature_importances_)[::-1][:3]
print("\nTop-3 important features (pruned tree):",
      [data.feature_names[i] for i in top_features])

print("\nText representation of the pruned tree (first few levels):")
print(export_text(pruned_tree, feature_names=list(data.feature_names), max_depth=2))

# Regression tree (splits minimize variance, not Gini/entropy)
X_reg = np.linspace(0, 10, 100).reshape(-1, 1)
y_reg = np.sin(X_reg).ravel() + np.random.default_rng(0).normal(scale=0.1, size=100)
reg_tree = DecisionTreeRegressor(max_depth=3).fit(X_reg, y_reg)
print("\nRegression tree leaf prediction is the MEAN target in that leaf, e.g. "
      f"predict(x=5)={reg_tree.predict([[5]])[0]:.3f}")

print("\nKey talking points: Gini vs Entropy vs Variance-reduction (regression), "
      "greedy top-down splitting, overfitting control knobs, feature_importance bias "
      "toward high-cardinality/continuous features, trees are the base learner for "
      "Random Forest and Gradient Boosting.")
