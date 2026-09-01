"""
K-NEAREST NEIGHBORS (KNN)
==========================
Interview must-knows:
- LAZY / instance-based learner: no explicit training phase (just stores the
  data) -- all the work happens at PREDICT time -> slow inference O(n*d) per
  query with brute force (use KD-Tree/Ball-Tree for speedup in low-medium
  dimensions, though they degrade to brute force in high dimensions).
- Classification: majority vote among k nearest neighbors (optionally
  distance-weighted). Regression: average of k nearest neighbors' targets.
- MUST scale features first -- distance metrics are dominated by large-scale
  features otherwise.
- Choice of k controls bias/variance:
    small k (e.g. 1) -> low bias, high variance, jagged/noisy decision boundary
    large k -> high bias, low variance, smoother boundary, eventually predicts
               the global majority class as k -> n
  Use odd k for binary classification to avoid ties.
- Curse of dimensionality: in high dimensions, distances between all points
  become nearly equal (points concentrate near the "surface"), so "nearest"
  stops being meaningful -> KNN degrades badly; use dimensionality reduction
  first (PCA) or a different model.
- Distance metrics: Euclidean (L2, default), Manhattan (L1, more robust to
  outliers), Minkowski (generalizes both), Cosine (good for text/embeddings,
  direction matters more than magnitude), Hamming (categorical/binary data).
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# -----------------------------------------------------------------
# 1. FROM SCRATCH -- distance computation + majority vote
# -----------------------------------------------------------------
def euclidean_distances(X_query, X_train):
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b  (vectorized, avoids explicit loops)
    return np.sqrt(
        ((X_query[:, None, :] - X_train[None, :, :]) ** 2).sum(axis=2)
    )

def knn_predict(X_train, y_train, X_query, k=5):
    dists = euclidean_distances(X_query, X_train)          # (n_query, n_train)
    knn_idx = np.argsort(dists, axis=1)[:, :k]              # k closest indices per query
    preds = []
    for idx_row in knn_idx:
        neighbor_labels = y_train[idx_row]
        values, counts = np.unique(neighbor_labels, return_counts=True)
        preds.append(values[np.argmax(counts)])             # majority vote
    return np.array(preds)

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42)
scaler = StandardScaler()                       # scaling is NOT optional for KNN
X_train_s, X_test_s = scaler.fit_transform(X_train), scaler.transform(X_test)

scratch_preds = knn_predict(X_train_s, y_train, X_test_s, k=5)
print("Scratch KNN (k=5) accuracy:", accuracy_score(y_test, scratch_preds))

# -----------------------------------------------------------------
# 2. sklearn + effect of k (bias/variance) + effect of scaling
# -----------------------------------------------------------------
print("\nEffect of k on train vs test accuracy (bias/variance trade-off):")
for k in [1, 3, 5, 15, 51]:
    knn = KNeighborsClassifier(n_neighbors=k).fit(X_train_s, y_train)
    train_acc = accuracy_score(y_train, knn.predict(X_train_s))
    test_acc = accuracy_score(y_test, knn.predict(X_test_s))
    print(f"  k={k:3d}  train_acc={train_acc:.3f}  test_acc={test_acc:.3f}")

print("\nEffect of skipping feature scaling (should hurt accuracy/consistency):")
knn_unscaled = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
knn_scaled = KNeighborsClassifier(n_neighbors=5).fit(X_train_s, y_train)
print(f"  unscaled test_acc={accuracy_score(y_test, knn_unscaled.predict(X_test)):.3f}")
print(f"  scaled   test_acc={accuracy_score(y_test, knn_scaled.predict(X_test_s)):.3f}")

# -----------------------------------------------------------------
# 3. Choosing k via cross-validation (proper way, not eyeballing test set)
# -----------------------------------------------------------------
best_k, best_score = None, -1
for k in range(1, 30, 2):
    score = cross_val_score(KNeighborsClassifier(n_neighbors=k), X_train_s, y_train, cv=5).mean()
    if score > best_score:
        best_k, best_score = k, score
print(f"\nBest k via 5-fold CV: k={best_k} (cv_acc={best_score:.3f})")

# -----------------------------------------------------------------
# 4. Curse of dimensionality demo: ratio of max/min distance -> 1 as dims grow
# -----------------------------------------------------------------
print("\nCurse of dimensionality (max_dist/min_dist ratio shrinks toward 1):")
rng = np.random.default_rng(0)
for dims in [2, 10, 100, 1000]:
    pts = rng.normal(size=(200, dims))
    d = euclidean_distances(pts[:1], pts[1:])
    print(f"  dims={dims:5d}  max/min distance ratio = {d.max()/d.min():.3f}")

print("\nKey talking points: lazy learner, must scale, k controls bias/variance, "
      "curse of dimensionality, distance metric choice, O(n*d) inference cost.")
