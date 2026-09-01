"""
LOGISTIC REGRESSION
====================
Interview must-knows:
- It's a CLASSIFICATION model despite the name "regression": models P(y=1|x)
  via sigmoid(w.x + b) = 1 / (1 + e^-z).
- Loss = Binary Cross-Entropy (log loss), NOT MSE (MSE + sigmoid is non-convex).
  BCE is convex in w -> gradient descent finds the global optimum.
- Decision boundary is LINEAR in the feature space (z = w.x + b = 0).
- Gradient of BCE w.r.t. w has the beautifully simple form: X^T (sigmoid(Xw) - y) / n
  -- identical form to linear regression's gradient! Know this, it's a favorite
  "derive it on the whiteboard" question.
- Multiclass: One-vs-Rest or Softmax (multinomial) regression.
- Regularization (L2 by default in sklearn) prevents unbounded weight growth on
  separable data.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, classification_report

X, y = make_classification(n_samples=400, n_features=5, n_informative=3,
                            n_redundant=0, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
scaler = StandardScaler()
X_train_s, X_test_s = scaler.fit_transform(X_train), scaler.transform(X_test)


# -----------------------------------------------------------------
# 1. FROM SCRATCH
# -----------------------------------------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_logreg(X, y, lr=0.1, epochs=1000, l2=0.01):
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    for _ in range(epochs):
        z = X @ w + b
        p = sigmoid(z)
        error = p - y                                  # same shape as linear reg!
        grad_w = X.T @ error / n + l2 * w              # + L2 penalty term
        grad_b = error.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b

w, b = train_logreg(X_train_s, y_train)
probs = sigmoid(X_test_s @ w + b)
preds = (probs >= 0.5).astype(int)                     # 0.5 threshold is a CHOICE, not a law
print("Scratch logistic regression accuracy:", accuracy_score(y_test, preds))
print("Scratch BCE loss:", round(log_loss(y_test, probs), 4))

# -----------------------------------------------------------------
# 2. SKLEARN
# -----------------------------------------------------------------
clf = LogisticRegression().fit(X_train_s, y_train)
sk_preds = clf.predict(X_test_s)
sk_probs = clf.predict_proba(X_test_s)[:, 1]
print("\nsklearn accuracy:", accuracy_score(y_test, sk_preds))
print("Confusion matrix:\n", confusion_matrix(y_test, sk_preds))
print(classification_report(y_test, sk_preds, digits=3))

# -----------------------------------------------------------------
# 3. THRESHOLD TUNING -- interviewers love asking "what if accuracy isn't the goal?"
# -----------------------------------------------------------------
for thresh in [0.3, 0.5, 0.7]:
    preds_t = (sk_probs >= thresh).astype(int)
    tp = ((preds_t == 1) & (y_test == 1)).sum()
    fp = ((preds_t == 1) & (y_test == 0)).sum()
    fn = ((preds_t == 0) & (y_test == 1)).sum()
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    print(f"threshold={thresh}: precision={precision:.2f} recall={recall:.2f} "
          f"(lower threshold -> catch more positives, more false alarms)")

print("\nKey talking points: sigmoid + BCE convexity, gradient looks like linear "
      "regression's, threshold is tunable (precision/recall trade-off), "
      "odds & log-odds interpretation of coefficients, multiclass via softmax/OvR.")
