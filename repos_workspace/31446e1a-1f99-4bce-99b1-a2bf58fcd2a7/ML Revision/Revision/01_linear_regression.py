"""
LINEAR REGRESSION
=================
Interview must-knows:
- Model: y_hat = X @ w + b. Loss: Mean Squared Error (convex -> unique global min).
- Closed form: w = (X^T X)^-1 X^T y (Normal Equation) -- O(n_features^3), bad for
  wide/collinear data. Gradient Descent scales better and is what's used in practice.
- Assumptions: linearity, independence of errors, homoscedasticity (constant error
  variance), no multicollinearity, errors ~ Normal (needed for inference, not for
  point prediction).
- Regularization: Ridge (L2, shrinks coefs, keeps all features, handles
  multicollinearity), Lasso (L1, can zero-out coefs -> feature selection),
  ElasticNet (mix of both).
- R^2 can only go up (or stay same) as you add features -> use Adjusted R^2 to compare
  models with different numbers of features.
"""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)
n, d = 200, 3
X = rng.normal(size=(n, d))
true_w, true_b = np.array([3.0, -2.0, 0.5]), 4.0
y = X @ true_w + true_b + rng.normal(scale=1.0, size=n)   # add noise

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# -----------------------------------------------------------------
# 1. FROM SCRATCH: batch gradient descent (be ready to derive the gradient)
# -----------------------------------------------------------------
def linreg_gradient_descent(X, y, lr=0.05, epochs=500):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    losses = []
    for _ in range(epochs):
        y_pred = X @ w + b
        error = y_pred - y                       # (n,)
        # dLoss/dw = (2/n) * X^T @ error ; dLoss/db = (2/n) * sum(error)
        grad_w = (2 / n) * X.T @ error
        grad_b = (2 / n) * error.sum()
        w -= lr * grad_w
        b -= lr * grad_b
        losses.append(np.mean(error ** 2))
    return w, b, losses

w_scratch, b_scratch, losses = linreg_gradient_descent(X_train, y_train)
print("Scratch GD weights:", w_scratch.round(3), "bias:", round(b_scratch, 3))
print("Loss decreasing:", losses[0] > losses[-1], f"(start={losses[0]:.3f}, end={losses[-1]:.3f})")

# -----------------------------------------------------------------
# 2. SKLEARN: OLS, Ridge, Lasso
# -----------------------------------------------------------------
scaler = StandardScaler()                 # scale before regularized regression!
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

ols = LinearRegression().fit(X_train_s, y_train)
ridge = Ridge(alpha=1.0).fit(X_train_s, y_train)      # alpha = lambda, higher -> more shrinkage
lasso = Lasso(alpha=0.1).fit(X_train_s, y_train)

for name, model in [("OLS", ols), ("Ridge", ridge), ("Lasso", lasso)]:
    pred = model.predict(X_test_s)
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    print(f"{name:6s} coef={np.round(model.coef_, 3)} intercept={model.intercept_:.3f} "
          f"MSE={mse:.3f} R2={r2:.3f}")

# Adjusted R^2 -- penalizes adding useless features
def adjusted_r2(r2, n_samples, n_features):
    return 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)

r2 = r2_score(y_test, ols.predict(X_test_s))
print("Adjusted R2:", round(adjusted_r2(r2, len(y_test), X.shape[1]), 3))

print("\nKey talking points: MSE is convex, gradient formula, when to pick "
      "Ridge vs Lasso vs ElasticNet, why scaling matters before regularization, "
      "R2 vs Adjusted R2, checking residual plots for homoscedasticity/linearity.")
