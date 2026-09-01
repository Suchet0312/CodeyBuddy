"""
SUPPORT VECTOR MACHINES (SVM)
==============================
Interview must-knows:
- Goal: find the hyperplane w.x + b = 0 that MAXIMIZES THE MARGIN between
  classes. Margin = 2/||w||, so maximizing margin == minimizing ||w||^2.
- Only points closest to the boundary ("support vectors") determine the
  hyperplane -- this is why SVMs can be memory-efficient at inference and
  robust to points far from the boundary.
- Soft margin (real-world, non-separable data): minimize
      0.5*||w||^2 + C * sum(hinge_loss_i),   hinge_loss = max(0, 1 - y_i*(w.x_i+b))
  C controls the trade-off:
      small C -> wider margin, tolerate more violations -> more regularization
                 (higher bias, lower variance)
      large C -> narrower margin, fewer violations tolerated -> can overfit
                 (lower bias, higher variance)
- KERNEL TRICK: instead of explicitly mapping x -> phi(x) into a higher-dim
  space where classes become linearly separable, a kernel function K(x,x') =
  phi(x).phi(x') computes the dot product directly, WITHOUT ever computing
  phi(x) -- huge computational saving. Common kernels:
    linear:       K = x.x'
    polynomial:   K = (gamma*x.x' + r)^d
    RBF/Gaussian: K = exp(-gamma * ||x-x'||^2)   (most common default; infinite-
                  dimensional feature space)
- gamma (RBF) controls how far a single training example's influence reaches:
      small gamma -> far reach -> smoother boundary (more bias)
      large gamma -> short reach -> wiggly boundary, can overfit (more variance)
- Must scale features (distance/dot-product based, like KNN).
- Naturally binary; multiclass via One-vs-One or One-vs-Rest.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s, X_test_s = scaler.fit_transform(X_train), scaler.transform(X_test)

# -----------------------------------------------------------------
# 1. Linear vs RBF kernel, and why linear fails on non-linearly-separable data
# -----------------------------------------------------------------
X_moons, y_moons = make_moons(n_samples=300, noise=0.2, random_state=0)
Xm_train, Xm_test, ym_train, ym_test = train_test_split(X_moons, y_moons, test_size=0.3, random_state=0)

linear_svm = SVC(kernel="linear").fit(Xm_train, ym_train)
rbf_svm = SVC(kernel="rbf", gamma="scale").fit(Xm_train, ym_train)
print("On non-linearly-separable 'moons' data:")
print(f"  linear kernel test_acc = {accuracy_score(ym_test, linear_svm.predict(Xm_test)):.3f}"
      "  (struggles: no straight line separates the moons)")
print(f"  RBF kernel    test_acc = {accuracy_score(ym_test, rbf_svm.predict(Xm_test)):.3f}"
      "  (kernel trick maps to a space where they ARE separable)")

# -----------------------------------------------------------------
# 2. Effect of C (margin width vs violations) on the real dataset
# -----------------------------------------------------------------
print("\nEffect of C (regularization strength, inverse-ish):")
for C in [0.01, 1, 100]:
    svm = SVC(kernel="rbf", C=C, gamma="scale").fit(X_train_s, y_train)
    train_acc = accuracy_score(y_train, svm.predict(X_train_s))
    test_acc = accuracy_score(y_test, svm.predict(X_test_s))
    print(f"  C={C:<6} n_support_vectors={svm.n_support_.sum():3d} "
          f"train_acc={train_acc:.3f} test_acc={test_acc:.3f}")

# -----------------------------------------------------------------
# 3. Effect of gamma (RBF reach)
# -----------------------------------------------------------------
print("\nEffect of gamma (RBF kernel width):")
for gamma in [0.001, 0.1, 10]:
    svm = SVC(kernel="rbf", C=1.0, gamma=gamma).fit(X_train_s, y_train)
    train_acc = accuracy_score(y_train, svm.predict(X_train_s))
    test_acc = accuracy_score(y_test, svm.predict(X_test_s))
    print(f"  gamma={gamma:<6} train_acc={train_acc:.3f} test_acc={test_acc:.3f}")

# -----------------------------------------------------------------
# 4. Grid search over C and gamma together (how you'd actually tune it)
# -----------------------------------------------------------------
grid = GridSearchCV(
    SVC(kernel="rbf"),
    param_grid={"C": [0.1, 1, 10], "gamma": [0.001, 0.01, 0.1]},
    cv=5, n_jobs=-1,
)
grid.fit(X_train_s, y_train)
print(f"\nBest params via GridSearchCV: {grid.best_params_} (cv_acc={grid.best_score_:.3f})")
print(f"Test accuracy with best params: {accuracy_score(y_test, grid.predict(X_test_s)):.3f}")

print("\nKey talking points: margin maximization = minimize ||w||, hinge loss + "
      "C trade-off, kernel trick avoids explicit feature-space mapping, "
      "gamma controls RBF smoothness, support vectors are the only points that "
      "matter for the boundary, must scale features.")
