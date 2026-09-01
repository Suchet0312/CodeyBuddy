"""
K-MEANS CLUSTERING
====================
Interview must-knows:
- Unsupervised, partitions data into K clusters by minimizing WITHIN-CLUSTER
  SUM OF SQUARES (WCSS / inertia): sum over clusters of sum over points in
  cluster of ||x - centroid||^2.
- Algorithm (Lloyd's algorithm), iterate to convergence:
    1. Initialize K centroids (random points or k-means++ for smarter spread-out
       initialization -- reduces bad-local-minima risk).
    2. ASSIGN each point to its nearest centroid (by Euclidean distance).
    3. UPDATE each centroid to the mean of points assigned to it.
    4. Repeat 2-3 until assignments stop changing (converged) or max_iter hit.
- Guaranteed to converge (WCSS decreases monotonically each iteration) but only
  to a LOCAL minimum -> sensitive to initialization -> run multiple times with
  different seeds (n_init) and keep the best (lowest inertia) result.
- Must choose K in advance. Two common approaches:
    Elbow method: plot WCSS vs K, look for the "elbow" where the marginal
    improvement drops off.
    Silhouette score: for each point, (b-a)/max(a,b) where a=avg distance to
    own cluster, b=avg distance to nearest other cluster; ranges [-1,1], higher
    is better-separated clusters; average across all points to compare K values.
- Assumes clusters are roughly SPHERICAL, similarly SIZED, and similar DENSITY
  (uses Euclidean distance) -> fails on elongated/non-convex clusters
  (DBSCAN or spectral clustering handle those better).
- Must scale features first (Euclidean distance dominated by large-scale
  features otherwise) -- same reasoning as KNN/SVM.
- Sensitive to outliers (they pull centroids toward them, since it's a mean).
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

# -----------------------------------------------------------------
# 1. FROM SCRATCH -- Lloyd's algorithm
# -----------------------------------------------------------------
def kmeans_scratch(X, k, max_iter=100, seed=0):
    rng = np.random.default_rng(seed)
    centroids = X[rng.choice(len(X), k, replace=False)]     # naive random init
    for it in range(max_iter):
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)  # (n,k)
        assignments = dists.argmin(axis=1)                   # ASSIGN step
        new_centroids = np.array([
            X[assignments == c].mean(axis=0) if (assignments == c).any() else centroids[c]
            for c in range(k)
        ])                                                    # UPDATE step
        if np.allclose(new_centroids, centroids):
            print(f"  converged after {it+1} iterations")
            break
        centroids = new_centroids
    inertia = sum(((X[assignments == c] - centroids[c]) ** 2).sum() for c in range(k))
    return assignments, centroids, inertia

print("Scratch K-Means:")
assignments, centroids, inertia = kmeans_scratch(X_scaled, k=4)
print(f"  final inertia (WCSS) = {inertia:.2f}")

# -----------------------------------------------------------------
# 2. sklearn KMeans with k-means++ init and n_init for robustness
# -----------------------------------------------------------------
km = KMeans(n_clusters=4, init="k-means++", n_init=10, random_state=42).fit(X_scaled)
print(f"\nsklearn KMeans inertia={km.inertia_:.2f}, "
      f"cluster sizes={np.bincount(km.labels_)}")

# -----------------------------------------------------------------
# 3. ELBOW METHOD -- choosing K
# -----------------------------------------------------------------
print("\nElbow method (look for the bend where WCSS improvement flattens):")
inertias = []
for k in range(1, 8):
    km_k = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
    inertias.append(km_k.inertia_)
    print(f"  k={k}  inertia={km_k.inertia_:8.2f}")

# -----------------------------------------------------------------
# 4. SILHOUETTE SCORE -- a more principled way to pick K
# -----------------------------------------------------------------
print("\nSilhouette score by K (higher = better-separated clusters):")
for k in range(2, 7):
    km_k = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
    score = silhouette_score(X_scaled, km_k.labels_)
    print(f"  k={k}  silhouette={score:.3f}")

# -----------------------------------------------------------------
# 5. Sensitivity to initialization -- why n_init / kmeans++ matters
# -----------------------------------------------------------------
print("\nSame K, different random seeds, single init each (shows local-minima risk):")
for seed in [0, 1, 2]:
    km_seed = KMeans(n_clusters=4, init="random", n_init=1, random_state=seed).fit(X_scaled)
    print(f"  seed={seed}  inertia={km_seed.inertia_:.2f}")

print("\nKey talking points: Lloyd's iterative assign/update, guaranteed to "
      "converge but only to a local minimum, k-means++ / n_init for robustness, "
      "elbow method vs silhouette score for choosing K, assumes spherical "
      "equally-sized clusters, must scale features, sensitive to outliers.")
