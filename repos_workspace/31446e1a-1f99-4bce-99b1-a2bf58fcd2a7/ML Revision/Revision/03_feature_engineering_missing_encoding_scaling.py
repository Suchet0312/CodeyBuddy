"""
FEATURE ENGINEERING, MISSING VALUES, ENCODING, SCALING
=========================================================
Interview must-knows:

MISSING VALUES
- Types: MCAR (missing completely at random), MAR (missing depends on OTHER
  observed features), MNAR (missing depends on the missing value itself, e.g.
  high earners refuse to disclose income) -- MNAR is the hardest to handle
  correctly and can bias any method.
- Simple strategies: drop rows/columns (only if missingness is small/random),
  mean/median (median is robust to skew/outliers) imputation for numeric,
  mode for categorical, constant/"Missing" category (lets the model use
  "missingness itself" as a signal).
- Smarter: KNN imputation, model-based (regress the missing feature on others),
  add a boolean "was_missing" indicator column alongside imputed values so the
  model can still detect the original missingness pattern.
- CRITICAL: fit the imputer (compute mean/median/mode) on TRAIN ONLY, then
  transform train and test with those fitted statistics -- fitting on the
  full dataset leaks test information into training.

ENCODING CATEGORICAL FEATURES
- One-Hot Encoding: creates a binary column per category. Good for NOMINAL
  (unordered) categories with low-medium cardinality. Causes the "dummy
  variable trap" (multicollinearity) if you don't drop one column for linear
  models -- drop_first=True.
- Ordinal Encoding: maps categories to integers preserving order (e.g.
  low/medium/high -> 0/1/2). Only valid when there IS a genuine order;
  applying it to nominal data creates a fake numeric relationship.
- Label Encoding: like ordinal but for the TARGET variable, or a quick numeric
  stand-in tree models can handle (trees don't assume linear numeric
  relationships, so arbitrary integer codes are less harmful there than for
  linear/distance-based models).
- Target/Mean Encoding: replace category with the mean target value for that
  category -- powerful for high-cardinality features (e.g. zip code) but leaks
  target info -> must use out-of-fold / smoothing to avoid overfitting.
- Frequency/Count Encoding: replace category with its frequency -- simple,
  no leakage risk, loses some information.

SCALING
- Needed for: distance-based (KNN, SVM, K-Means), gradient-descent-based
  (linear/logistic regression, neural nets), PCA. NOT needed for tree-based
  models (splits are threshold/rank-based, invariant to monotonic scaling).
- StandardScaler: (x - mean) / std -> mean 0, std 1. Assumes roughly Gaussian-ish
  data; sensitive to outliers (they skew mean/std).
- MinMaxScaler: (x - min) / (max - min) -> bounded to [0,1]. Also outlier-sensitive
  (a single extreme value compresses everything else into a tiny range).
- RobustScaler: uses median and IQR instead of mean/std -> robust to outliers.
- Again: fit scaler on TRAIN ONLY, transform both train and test with it.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (
    OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler, RobustScaler
)
from sklearn.model_selection import train_test_split

df = pd.DataFrame({
    "age": [25, np.nan, 35, 40, np.nan, 29, 150],           # 150 is an outlier
    "income": [50000, 60000, np.nan, 80000, 55000, 62000, 58000],
    "city": ["NY", "LA", "NY", "SF", "LA", "SF", "NY"],       # nominal
    "education": ["HS", "Bachelors", "Masters", "PhD", "Bachelors", "HS", "Masters"],  # ordinal
})
print("Original data:\n", df)

train_df, test_df = train_test_split(df, test_size=0.3, random_state=0)

# -----------------------------------------------------------------
# 1. MISSING VALUE IMPUTATION -- fit on train, transform both
# -----------------------------------------------------------------
num_cols = ["age", "income"]
median_imputer = SimpleImputer(strategy="median")             # median: robust to the age=150 outlier
median_imputer.fit(train_df[num_cols])                          # FIT ON TRAIN ONLY
train_num = median_imputer.transform(train_df[num_cols])
test_num = median_imputer.transform(test_df[num_cols])
print("\nMedian-imputed train numeric cols:\n", train_num)

# add a missing-indicator flag (keeps the "was this missing?" signal)
train_df = train_df.copy()
train_df["age_was_missing"] = train_df["age"].isna().astype(int)

# KNN imputation alternative (uses similar rows to estimate the missing value)
knn_imputer = KNNImputer(n_neighbors=2)
knn_imputed = knn_imputer.fit_transform(train_df[num_cols])
print("KNN-imputed (alternative to median):\n", knn_imputed)

# -----------------------------------------------------------------
# 2. ENCODING
# -----------------------------------------------------------------
# One-Hot for NOMINAL "city" (no inherent order)
ohe = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
ohe.fit(train_df[["city"]])                                     # fit on train only
city_encoded = ohe.transform(train_df[["city"]])
print("\nOne-hot encoded city (drop_first avoids dummy trap):\n",
      pd.DataFrame(city_encoded, columns=ohe.get_feature_names_out()))

# Ordinal for "education" -- genuine order exists, so map it explicitly
edu_order = [["HS", "Bachelors", "Masters", "PhD"]]
ordinal_enc = OrdinalEncoder(categories=edu_order, handle_unknown="use_encoded_value",
                              unknown_value=-1)
ordinal_enc.fit(train_df[["education"]])
print("\nOrdinal encoded education (order preserved):\n",
      ordinal_enc.transform(train_df[["education"]]).ravel())

# Target/mean encoding (manual, with smoothing to reduce overfitting on rare categories)
y_train_demo = np.array([1, 0, 1, 1, 0])[:len(train_df)]
def target_encode(series, target, smoothing=5):
    global_mean = target.mean()
    stats = pd.DataFrame({"cat": series.values, "target": target}).groupby("cat")["target"]
    means, counts = stats.mean(), stats.count()
    smoothed = (counts * means + smoothing * global_mean) / (counts + smoothing)
    return series.map(smoothed)

print("\nTarget-encoded city (smoothed toward global mean for rare categories):\n",
      target_encode(train_df["city"], pd.Series(y_train_demo)))

# -----------------------------------------------------------------
# 3. SCALING -- compare Standard / MinMax / Robust on data WITH an outlier
# -----------------------------------------------------------------
ages = train_df[["age"]].fillna(train_df["age"].median())
print("\nRaw ages (note the outlier 150 if present in this split):\n", ages.values.ravel())
for name, scaler in [("StandardScaler", StandardScaler()),
                      ("MinMaxScaler", MinMaxScaler()),
                      ("RobustScaler (median/IQR)", RobustScaler())]:
    scaled = scaler.fit_transform(ages)
    print(f"{name:28s} -> {np.round(scaled.ravel(), 2)}")

print("\nKey talking points: fit preprocessing on train only (no leakage), "
      "median > mean imputation under skew/outliers, one-hot vs ordinal vs "
      "target encoding trade-offs, tree models don't need scaling, "
      "RobustScaler for outlier-heavy features.")
