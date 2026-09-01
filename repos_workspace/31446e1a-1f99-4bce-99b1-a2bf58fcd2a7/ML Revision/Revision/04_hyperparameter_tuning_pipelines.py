"""
HYPERPARAMETER TUNING + PIPELINES
====================================
Interview must-knows:

HYPERPARAMETER TUNING
- Parameters are LEARNED from data (e.g. weights). Hyperparameters are SET
  before training (e.g. k in KNN, C in SVM, max_depth in trees, learning_rate)
  and are tuned using a validation set / cross-validation, never the test set.
- Grid Search: exhaustively try every combination in a specified grid.
  Guaranteed to check everything in the grid, but cost grows exponentially with
  the number of hyperparameters ("curse of dimensionality" for search).
- Random Search: sample random combinations for a fixed budget. Often finds
  equally good or better solutions than grid search with far fewer
  evaluations, especially when only a few hyperparameters actually matter
  (Bergstra & Bengio result -- good one to cite).
- Bayesian Optimization (e.g. Optuna, Hyperopt): builds a probabilistic model
  of the objective and picks the next point to try based on
  expected-improvement -- more sample-efficient than random search for
  expensive-to-train models.
- Nested cross-validation: outer loop estimates generalization performance,
  inner loop tunes hyperparameters -- avoids the optimistic bias of using the
  same CV folds both to tune AND to report final performance.

PIPELINES
- sklearn Pipeline chains preprocessing + model into ONE object with .fit() /
  .predict(). Why it matters:
    1. Prevents data leakage: when used inside cross_val_score/GridSearchCV,
       each fold's preprocessing (scaler, imputer, encoder) is fit ONLY on that
       fold's training data, not the whole dataset.
    2. Cleaner deployment: one object to serialize (pickle/joblib) and serve.
    3. Hyperparameters of every step (including preprocessing) can be tuned
       together via GridSearchCV with the 'stepname__param' naming convention.
- ColumnTransformer: applies different preprocessing to different columns
  (e.g. scale numeric columns, one-hot encode categorical columns) within a
  single pipeline step.
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
)
from sklearn.datasets import load_breast_cancer
from scipy.stats import loguniform, randint

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42)

# -----------------------------------------------------------------
# 1. WHY PIPELINES MATTER: leakage demo (wrong way vs right way)
# -----------------------------------------------------------------
# WRONG: scaling on the full dataset before splitting/CV leaks test statistics
scaler_leaky = StandardScaler().fit(np.vstack([X_train, X_test]))   # DON'T DO THIS
# RIGHT: build a pipeline, then cross-validate the WHOLE pipeline
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf")),
])
cv_scores = cross_val_score(pipe, X_train, y_train, cv=5)
print(f"Pipeline CV accuracy (scaler refit per fold, no leakage): "
      f"{cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

# -----------------------------------------------------------------
# 2. GRID SEARCH over pipeline hyperparameters
#    Note the 'stepname__param' naming to reach into pipeline steps.
# -----------------------------------------------------------------
param_grid = {
    "svm__C": [0.1, 1, 10],
    "svm__gamma": [0.001, 0.01, 0.1],
}
grid = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1, scoring="accuracy")
grid.fit(X_train, y_train)
print(f"\nGridSearchCV best_params={grid.best_params_} best_cv_score={grid.best_score_:.3f}")
print(f"Test accuracy with best pipeline: {grid.score(X_test, y_test):.3f}")
print(f"(GridSearchCV tried {len(grid.cv_results_['params'])} combos x 5 folds = "
      f"{len(grid.cv_results_['params']) * 5} fits)")

# -----------------------------------------------------------------
# 3. RANDOMIZED SEARCH -- same budget-conscious idea, sampled distributions
# -----------------------------------------------------------------
rf_pipe = Pipeline([("rf", RandomForestClassifier(random_state=42))])
param_dist = {
    "rf__n_estimators": randint(50, 300),
    "rf__max_depth": randint(2, 20),
    "rf__max_features": loguniform(0.1, 1.0),
}
rand_search = RandomizedSearchCV(
    rf_pipe, param_dist, n_iter=15, cv=5, random_state=0, n_jobs=-1, scoring="accuracy"
)
rand_search.fit(X_train, y_train)
print(f"\nRandomizedSearchCV (only 15 samples) best_params={rand_search.best_params_}")
print(f"best_cv_score={rand_search.best_score_:.3f}  "
      f"test_acc={rand_search.score(X_test, y_test):.3f}")

# -----------------------------------------------------------------
# 4. ColumnTransformer -- different preprocessing per column type
# -----------------------------------------------------------------
import pandas as pd
mixed_df = pd.DataFrame({
    "age": [25, np.nan, 35, 40, 29],
    "income": [50000, 60000, np.nan, 80000, 55000],
    "city": ["NY", "LA", "NY", "SF", "LA"],
})
y_mixed = np.array([0, 1, 0, 1, 1])

numeric_features = ["age", "income"]
categorical_features = ["city"]

preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])

full_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(n_estimators=50, random_state=0)),
])
full_pipeline.fit(mixed_df, y_mixed)
print("\nFull mixed-type pipeline (impute+scale numeric, one-hot categorical, "
      "then model) fit successfully:", full_pipeline.predict(mixed_df))

print("\nKey talking points: parameters vs hyperparameters, grid vs random vs "
      "Bayesian search trade-offs, why Pipeline prevents leakage in CV, "
      "stepname__param syntax, ColumnTransformer for mixed dtypes, nested CV "
      "for unbiased final performance estimates.")
