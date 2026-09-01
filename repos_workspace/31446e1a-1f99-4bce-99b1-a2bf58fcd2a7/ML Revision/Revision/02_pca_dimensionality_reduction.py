"""
PCA & DIMENSIONALITY REDUCTION
================================
Interview must-knows:
- Goal: find a lower-dimensional linear subspace that preserves as much
  VARIANCE as possible. The new axes (principal components) are ORTHOGONAL to
  each other and ordered by how much variance they explain.
- Algorithm:
    1. Center the data (subtract the mean; typically also scale to unit
       variance -- PCA is scale-sensitive since it operates on variance/
       covariance, a feature with a large numeric range would dominate).
    2. Compute the covariance matrix C = (1/n) X^T X (X already centered).
    3. Eigendecompose C: eigenvectors = principal component directions,
       eigenvalues = variance explained along each direction.
    4. Project data onto the top-k eigenvectors (sorted by eigenvalue desc).
  (Equivalently done via SVD of X directly, which is what sklearn actually
  uses -- more numerically stable than eigendecomposing the covariance matrix.)
- Explained variance ratio: eigenvalue_i / sum(all eigenvalues) -- tells you
  how much information each component captures; cumulative sum helps you pick
  how many components to keep (e.g. keep enough for 95% cumulative variance).
- Uses: visualization (reduce to 2D/3D), noise reduction, speeding up
  downstream models, mitigating the curse of dimensionality, decorrelating
  features (helps some linear models).
- Limitations: only captures LINEAR relationships/structure; components are
  linear combinations of ALL original features -> loses direct
  interpretability; sensitive to outliers and to feature scale (always scale
  first); assumes high variance == high importance, which isn't always true
  for a specific downstream task (e.g. supervised label-relevant direction
  might be a low-variance direction).
- PCA vs t-SNE/UMAP: PCA is linear, deterministic, fast, preserves GLOBAL
  structure/variance, good preprocessing step. t-SNE/UMAP are non-linear,
  stochastic, better at preserving LOCAL neighborhood structure for
  visualization, but distances between distant clusters in the output aren't
  meaningful, they're slower, and typically not used as a preprocessing step for
  a downstream model (no simple inverse/transform of new points for t-SNE).
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X_scaled = StandardScaler().fit_transform(data.data)      # ALWAYS scale before PCA

# -----------------------------------------------------------------
# 1. PCA FROM SCRATCH via eigendecomposition of the covariance matrix
# -----------------------------------------------------------------
def pca_scratch(X, n_components):
    X_centered = X - X.mean(axis=0)                          # step 1: center
    cov = np.cov(X_centered, rowvar=False)                    # step 2: covariance matrix
    eigvals, eigvecs = np.linalg.eigh(cov)                     # step 3: eigendecompose
    order = np.argsort(eigvals)[::-1]                          # sort descending
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    components = eigvecs[:, :n_components]
    X_reduced = X_centered @ components                        # step 4: project
    explained_var_ratio = eigvals[:n_components] / eigvals.sum()
    return X_reduced, explained_var_ratio

X_reduced_scratch, evr_scratch = pca_scratch(X_scaled, n_components=2)
print("Scratch PCA explained variance ratio (top 2 components):", evr_scratch.round(3))

# -----------------------------------------------------------------
# 2. sklearn PCA (uses SVD internally) -- should match scratch version
# -----------------------------------------------------------------
pca = PCA(n_components=2).fit(X_scaled)
print("sklearn PCA explained variance ratio:", pca.explained_variance_ratio_.round(3))

# -----------------------------------------------------------------
# 3. Choosing number of components via cumulative explained variance
# -----------------------------------------------------------------
pca_full = PCA().fit(X_scaled)
cumulative = np.cumsum(pca_full.explained_variance_ratio_)
n_for_95 = np.argmax(cumulative >= 0.95) + 1
print(f"\nComponents needed for >=95% cumulative variance: {n_for_95} "
      f"(out of {X_scaled.shape[1]} original features)")
print("Cumulative variance at a few component counts:",
      {k: round(cumulative[k-1], 3) for k in [1, 5, 10, n_for_95]})

# -----------------------------------------------------------------
# 4. PCA as a preprocessing step for a downstream classifier
#    (shows the speed/accuracy trade-off of dimensionality reduction)
# -----------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, data.target,
                                                      test_size=0.2, random_state=42)
for n_comp in [2, 10, X_scaled.shape[1]]:
    if n_comp == X_scaled.shape[1]:
        Xtr, Xte, label = X_train, X_test, "no PCA (all features)"
    else:
        pca_step = PCA(n_components=n_comp).fit(X_train)     # fit PCA on TRAIN only
        Xtr, Xte = pca_step.transform(X_train), pca_step.transform(X_test)
        label = f"PCA n_components={n_comp}"
    clf = LogisticRegression(max_iter=5000).fit(Xtr, y_train)
    acc = accuracy_score(y_test, clf.predict(Xte))
    print(f"  {label:28s} test_acc={acc:.3f}")

# -----------------------------------------------------------------
# 5. Reconstruction error -- PCA is lossy; more components = less loss
# -----------------------------------------------------------------
pca_recon = PCA(n_components=10).fit(X_scaled)
X_proj = pca_recon.transform(X_scaled)
X_reconstructed = pca_recon.inverse_transform(X_proj)
recon_error = np.mean((X_scaled - X_reconstructed) ** 2)
print(f"\nReconstruction MSE with 10 components: {recon_error:.4f} "
      "(0 would mean lossless, i.e. using all original components)")

print("\nKey talking points: variance-maximizing orthogonal projection, "
      "eigendecomposition of covariance == SVD of centered data, must scale "
      "first, explained variance ratio to pick n_components, PCA is linear/"
      "lossy/less interpretable, t-SNE/UMAP for non-linear visualization "
      "(not for feeding into a downstream model the same way).")
