"""
NAIVE BAYES
============
Interview must-knows:
- Based on Bayes' Theorem: P(y|x) = P(x|y) * P(y) / P(x)
  For classification we only need the numerator (P(x) is constant across classes):
      y_hat = argmax_y  P(y) * P(x_1,...,x_d | y)
- The "naive" assumption: features are CONDITIONALLY INDEPENDENT given the
  class -> P(x_1,...,x_d|y) = product_i P(x_i|y). This is almost never exactly
  true in practice, but the model works surprisingly well anyway (it only needs
  to rank classes correctly, not get calibrated probabilities right).
- Work in LOG-SPACE to avoid numerical underflow from multiplying many small
  probabilities: log P(y|x) proportional-to log P(y) + sum_i log P(x_i|y)
- Variants (pick based on feature type):
    GaussianNB     -> continuous features, assumes P(x_i|y) ~ Normal
    MultinomialNB  -> discrete counts (classic for text/bag-of-words, word counts)
    BernoulliNB    -> binary features (word present/absent)
- Laplace/additive smoothing (alpha) prevents zero probability for a
  word/feature value unseen in training for a given class (which would zero out
  the entire product).
- Pros: extremely fast to train (closed-form counting, no iterative
  optimization), works well with high-dimensional sparse data (text), needs
  little training data, naturally multiclass, good baseline.
- Cons: independence assumption hurts when features are correlated, and
  predicted probabilities are often poorly calibrated (though class RANKING /
  the decision itself is often still fine).
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

# -----------------------------------------------------------------
# 1. GAUSSIAN NAIVE BAYES FROM SCRATCH
# -----------------------------------------------------------------
class GaussianNBScratch:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean, self.var, self.priors = {}, {}, {}
        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = X_c.mean(axis=0)
            self.var[c] = X_c.var(axis=0) + 1e-9        # epsilon avoids div-by-zero
            self.priors[c] = X_c.shape[0] / X.shape[0]

    def _log_gaussian_pdf(self, x, mean, var):
        # log of N(x; mean, var), summed over independent features (naive assumption)
        return -0.5 * np.log(2 * np.pi * var) - ((x - mean) ** 2) / (2 * var)

    def predict(self, X):
        log_probs = []
        for c in self.classes:
            log_prior = np.log(self.priors[c])
            log_likelihood = self._log_gaussian_pdf(X, self.mean[c], self.var[c]).sum(axis=1)
            log_probs.append(log_prior + log_likelihood)     # log P(y) + sum log P(x_i|y)
        log_probs = np.array(log_probs).T                    # (n_samples, n_classes)
        return self.classes[np.argmax(log_probs, axis=1)]

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42)

scratch_nb = GaussianNBScratch()
scratch_nb.fit(X_train, y_train)
scratch_preds = scratch_nb.predict(X_test)
print("Scratch GaussianNB accuracy:", accuracy_score(y_test, scratch_preds))

sklearn_nb = GaussianNB().fit(X_train, y_train)
print("sklearn GaussianNB accuracy:", accuracy_score(y_test, sklearn_nb.predict(X_test)))

# -----------------------------------------------------------------
# 2. MULTINOMIAL NAIVE BAYES for text (the classic spam-filter use case)
# -----------------------------------------------------------------
docs = [
    "win money now free offer", "free cash prize click now",
    "meeting scheduled for tomorrow", "please review the attached report",
    "urgent free lottery winner claim now", "let's catch up over coffee this week",
    "team sync notes attached please review", "you have won a free prize claim now",
]
labels = np.array([1, 1, 0, 0, 1, 0, 0, 1])   # 1 = spam, 0 = not spam

vectorizer = CountVectorizer()
X_text = vectorizer.fit_transform(docs)          # bag-of-words counts (sparse)

mnb = MultinomialNB(alpha=1.0)                   # alpha = Laplace smoothing strength
mnb.fit(X_text, labels)

test_docs = ["free money winner claim now", "let's schedule the review meeting"]
test_X = vectorizer.transform(test_docs)
print("\nSpam classifier predictions:")
for doc, pred, proba in zip(test_docs, mnb.predict(test_X), mnb.predict_proba(test_X)):
    print(f"  '{doc}' -> {'SPAM' if pred == 1 else 'NOT SPAM'} (P(spam)={proba[1]:.3f})")

# -----------------------------------------------------------------
# 3. Effect of smoothing (alpha) -- prevents zero-probability collapse
# -----------------------------------------------------------------
print("\nEffect of Laplace smoothing alpha:")
for alpha in [0.001, 1.0, 10.0]:
    mnb_a = MultinomialNB(alpha=alpha).fit(X_text, labels)
    print(f"  alpha={alpha:<6} P(spam | 'free money winner claim now') = "
          f"{mnb_a.predict_proba(vectorizer.transform(['free money winner claim now']))[0,1]:.4f}")

print("\nKey talking points: Bayes' theorem, conditional independence 'naive' "
      "assumption, log-space to avoid underflow, Laplace smoothing for unseen "
      "words, Gaussian vs Multinomial vs Bernoulli variants, why it's a strong "
      "baseline for text classification.")
