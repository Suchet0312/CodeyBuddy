# ML Engineer Interview — Revision Kit

Every file is **standalone and runnable**: `python3 <file>.py`. Each one mixes
a from-scratch implementation (for the "derive/implement this" questions) with
the sklearn/xgboost equivalent (for the "use it correctly" questions), plus
inline comments covering the theory an interviewer will probe on.

Suggested order: work top to bottom. Budget ~15-20 min per file to read the
code, run it, and be able to explain every comment out loud without notes.

## 01_numpy_pandas/
- `numpy_pandas_revision.py` — vectorization, broadcasting, axis semantics,
  views vs copies, groupby/merge/pivot, missing-value handling.

## 02_supervised_models/
- `01_linear_regression.py` — normal equation, gradient descent derivation,
  Ridge/Lasso/ElasticNet, R² vs Adjusted R².
- `02_logistic_regression.py` — sigmoid + BCE, gradient derivation, threshold
  tuning, multiclass.
- `03_decision_trees.py` — Gini/Entropy/Information Gain from scratch,
  overfitting control, feature importance caveats.
- `04_random_forest.py` — bagging, feature-subsampling decorrelation, OOB
  score, bias/variance intuition.
- `05_gradient_boosting_xgboost.py` — residual-fitting from scratch,
  learning-rate/n_estimators trade-off, XGBoost's 2nd-order regularized
  objective, early stopping.
- `06_knn.py` — distance metrics, k as bias/variance knob, curse of
  dimensionality demo.
- `07_svm.py` — margin maximization, hinge loss, kernel trick, C & gamma.
- `08_naive_bayes.py` — Bayes' theorem, conditional independence, log-space,
  Laplace smoothing, spam-filter example.

## 03_ml_workflow/
- `01_validation_bias_variance.py` — train/test split, k-fold vs stratified
  k-fold, bias-variance decomposition, under/overfitting demo, learning curves.
- `02_metrics.py` — confusion matrix, precision/recall/F1, ROC-AUC vs PR-AUC,
  log loss, MAE/RMSE/R²/MAPE, all on an imbalanced dataset to show why
  accuracy lies.
- `03_feature_engineering_missing_encoding_scaling.py` — imputation
  strategies, one-hot/ordinal/target encoding, Standard/MinMax/Robust scaling,
  train-only fitting to avoid leakage.
- `04_hyperparameter_tuning_pipelines.py` — Grid vs Randomized search,
  Pipeline + ColumnTransformer, why pipelines prevent CV leakage.

## 04_unsupervised_dl/
- `01_kmeans.py` — Lloyd's algorithm from scratch, k-means++, elbow method,
  silhouette score.
- `02_pca_dimensionality_reduction.py` — eigendecomposition from scratch,
  explained variance, PCA vs t-SNE/UMAP.
- `03_neural_networks_backprop.py` — 2-layer net solving XOR, backprop
  derived and implemented by hand, gradient checking, vanishing gradients.
- `04_cnn_basics.py` — 2D convolution and max-pooling from scratch, output-size
  formula, parameter-sharing savings vs a dense layer.
- `05_embeddings_basics.py` — one-hot vs dense embeddings, toy skip-gram
  trained from scratch, cosine similarity, nearest-neighbor retrieval.

## How to use this under interview time pressure
1. Can you say the **one-paragraph explanation** of the topic without looking?
2. Can you **derive the key formula/gradient** on a whiteboard?
3. Can you name the **2-3 hyperparameters** that matter and what happens at
   each extreme?
4. Can you state **one real failure mode** (when this model/technique breaks)?

If you can do all four for a topic, you're ready for it.
