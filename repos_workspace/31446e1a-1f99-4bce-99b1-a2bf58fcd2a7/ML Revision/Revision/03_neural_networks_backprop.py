"""
NEURAL NETWORKS & BACKPROPAGATION (FROM SCRATCH)
====================================================
Interview must-knows:
- A neural net is a stack of affine transforms + non-linear activations:
      z1 = X @ W1 + b1;  a1 = activation(z1)
      z2 = a1 @ W2 + b2; a2 = activation(z2)   ... etc
  Without the non-linear activation, stacking layers collapses to a single
  linear transform (composition of linear functions is linear) -> activations
  are what give NNs the ability to model non-linear functions (universal
  approximation).
- Common activations:
    Sigmoid: squashes to (0,1), suffers VANISHING GRADIENTS (derivative max is
             0.25, shrinks fast through many layers) -- mostly only used at an
             output layer for binary probabilities now.
    Tanh:    squashes to (-1,1), zero-centered (helps optimization vs sigmoid),
             still vanishes for large |z|.
    ReLU:    max(0, z), doesn't saturate for z>0 -> much less vanishing
             gradient, cheap to compute, default choice for hidden layers.
             Downside: "dying ReLU" (a unit stuck outputting 0 forever if it
             gets a big negative gradient push). Leaky ReLU / ELU address this.
    Softmax: for the OUTPUT layer of multiclass classification, turns logits
             into a probability distribution that sums to 1.
- BACKPROPAGATION = the chain rule applied systematically, layer by layer,
  from the loss backward to each weight, reusing intermediate gradients
  (dynamic-programming style) instead of recomputing them -- this reuse is
  what makes it efficient (linear in network size, not exponential).
    dLoss/dW2 = dLoss/da2 * da2/dz2 * dz2/dW2
    dLoss/dW1 = (dLoss/da2 * da2/dz2 * dz2/da1) * da1/dz1 * dz1/dW1
  Each layer only needs the gradient signal ("error") flowing back from the
  layer above it, plus its own local derivatives.
- Vanishing/exploding gradients: in deep nets, repeatedly multiplying many
  small (<1) or large (>1) derivatives through the chain rule shrinks/blows up
  the gradient exponentially with depth -> mitigations: ReLU family, careful
  weight init (Xavier/He), batch normalization, residual/skip connections,
  gradient clipping (for exploding).
- Optimizers: plain SGD, SGD+Momentum (smooths updates using a running
  average of past gradients), Adam (per-parameter adaptive learning rates
  using running estimates of both the gradient mean and variance) -- Adam is
  the most common default in practice.
- Regularization for NNs: L2 weight decay, Dropout (randomly zero a fraction
  of activations each training step -> prevents co-adaptation, acts like
  training an ensemble of sub-networks), early stopping, batch normalization
  (also has a regularizing side effect).
"""

import numpy as np

# -----------------------------------------------------------------
# Toy problem: XOR -- the classic "why do we need a hidden layer /
# non-linearity" example. Not linearly separable, so plain logistic
# regression (no hidden layer) can't solve it.
# -----------------------------------------------------------------
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)     # XOR

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(a):
    return a * (1 - a)                               # derivative in terms of the OUTPUT a

def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)


class TwoLayerNet:
    """A minimal 2-layer (1 hidden layer) fully-connected net, trained by
    hand-derived forward + backward passes -- exactly what interviewers want
    to see you can reason through on a whiteboard."""

    def __init__(self, n_in, n_hidden, n_out, seed=0):
        rng = np.random.default_rng(seed)
        # He-ish init: scale by sqrt(1/n_in) to keep activation variance stable
        self.W1 = rng.normal(scale=np.sqrt(1 / n_in), size=(n_in, n_hidden))
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = rng.normal(scale=np.sqrt(1 / n_hidden), size=(n_hidden, n_out))
        self.b2 = np.zeros((1, n_out))

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)                       # hidden layer: ReLU
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)                     # output layer: sigmoid (binary)
        return self.a2

    def backward(self, X, y, lr=0.1):
        n = X.shape[0]
        # ---- Output layer gradient ----
        # For sigmoid output + binary cross-entropy loss, dLoss/dz2 simplifies
        # beautifully to (a2 - y) -- same clean form as logistic regression.
        dz2 = self.a2 - y                                        # (n, n_out)
        dW2 = self.a1.T @ dz2 / n
        db2 = dz2.mean(axis=0, keepdims=True)

        # ---- Backprop into the hidden layer (chain rule) ----
        da1 = dz2 @ self.W2.T                                     # error flowing back
        dz1 = da1 * relu_deriv(self.z1)                            # apply local activation derivative
        dW1 = X.T @ dz1 / n
        db1 = dz1.mean(axis=0, keepdims=True)

        # ---- Gradient descent update ----
        self.W2 -= lr * dW2; self.b2 -= lr * db2
        self.W1 -= lr * dW1; self.b1 -= lr * db1

    def loss(self, y_pred, y_true):
        eps = 1e-9
        return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))


net = TwoLayerNet(n_in=2, n_hidden=4, n_out=1, seed=1)
losses = []
for epoch in range(5000):
    y_pred = net.forward(X)
    losses.append(net.loss(y_pred, y))
    net.backward(X, y, lr=0.5)

print("XOR predictions after training (hidden layer solves what logistic regression can't):")
final_pred = net.forward(X)
for xi, yi, pi in zip(X, y.ravel(), final_pred.ravel()):
    print(f"  input={xi} true={int(yi)} predicted_prob={pi:.3f} -> {'correct' if round(pi)==yi else 'WRONG'}")
print(f"Loss went from {losses[0]:.3f} to {losses[-1]:.4f} over training.")

# -----------------------------------------------------------------
# Numerical gradient check -- the standard way to VERIFY a backprop
# implementation is correct (compare analytic grad to finite differences).
# -----------------------------------------------------------------
def numerical_gradient_check(net, X, y, param_name, eps=1e-4):
    param = getattr(net, param_name)
    analytic = None
    # run forward+backward once to populate analytic grads via a fresh pass
    y_pred = net.forward(X)
    dz2 = net.a2 - y
    if param_name == "W2":
        analytic = net.a1.T @ dz2 / X.shape[0]
    grad_num = np.zeros_like(param)
    it = np.nditer(param, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = param[idx]
        param[idx] = orig + eps
        loss_plus = net.loss(net.forward(X), y)
        param[idx] = orig - eps
        loss_minus = net.loss(net.forward(X), y)
        param[idx] = orig
        grad_num[idx] = (loss_plus - loss_minus) / (2 * eps)
        it.iternext()
    return analytic, grad_num

analytic_grad, numeric_grad = numerical_gradient_check(net, X, y, "W2")
max_diff = np.max(np.abs(analytic_grad - numeric_grad))
print(f"\nGradient check on W2: max|analytic - numeric| = {max_diff:.2e} "
      "(should be tiny, e.g. <1e-5, if backprop is implemented correctly)")

print("\nKey talking points: why non-linear activations are required, ReLU vs "
      "sigmoid/tanh trade-offs, backprop = chain rule + reusing intermediate "
      "gradients, sigmoid+BCE output gradient simplifies to (pred-true), "
      "vanishing/exploding gradients and mitigations, gradient checking, "
      "Adam vs SGD, dropout/L2/early-stopping for regularization.")
