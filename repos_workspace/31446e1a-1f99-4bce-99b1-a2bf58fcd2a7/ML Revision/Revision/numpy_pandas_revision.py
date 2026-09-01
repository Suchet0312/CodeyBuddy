"""
NUMPY + PANDAS QUICK REVISION
=============================
Interview angle: these are "warm-up" questions. Interviewers check that you're
fluent, not that you know exotic tricks. Know: vectorization (why loops are slow),
broadcasting rules, axis semantics (axis=0 -> down rows / per column,
axis=1 -> across columns / per row), and the difference between a view and a copy.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. NUMPY BASICS
# ---------------------------------------------------------------------------
print("=" * 70, "\nNUMPY\n", "=" * 70)

a = np.array([[1, 2, 3], [4, 5, 6]])          # shape (2,3)
print("array:\n", a)
print("shape:", a.shape, "dtype:", a.dtype, "ndim:", a.ndim)

# Vectorization: NEVER loop over numpy arrays element-by-element in an interview.
b = np.arange(6).reshape(2, 3)
print("elementwise a+b:\n", a + b)
print("matrix mult a @ b.T:\n", a @ b.T)        # (2,3) @ (3,2) -> (2,2)

# Broadcasting rule: dims are compared right-to-left; a dim of size 1 stretches
# to match. Shapes (2,3) and (3,) broadcast; (2,3) and (2,) do NOT.
row_means = a.mean(axis=1, keepdims=True)       # shape (2,1)
print("row-wise normalize (broadcast):\n", a - row_means)

# Axis cheat sheet:
print("sum axis=0 (col-wise, collapses rows):", a.sum(axis=0))
print("sum axis=1 (row-wise, collapses cols):", a.sum(axis=1))

# Boolean masking / fancy indexing (very common interview micro-task)
arr = np.array([5, -2, 3, -8, 0, 9])
print("mask > 0:", arr[arr > 0])
arr_clipped = np.where(arr < 0, 0, arr)          # vectorized if/else
print("clip negatives to 0:", arr_clipped)

# Views vs copies -- classic gotcha
sub = a[0, :]        # slicing -> VIEW, shares memory
sub[0] = 999
print("slicing is a view, original mutated:\n", a)
a[0, 0] = 1           # reset

copy_ex = a[a > 0]    # boolean/fancy indexing -> COPY
copy_ex[0] = -1
print("fancy-indexing is a copy, original unaffected:", a[0, 0] == 1)

# useful array ops
print("argmax:", arr.argmax(), "argsort:", arr.argsort())
print("unique + counts:", np.unique(arr, return_counts=True))
print("stack: vstack\n", np.vstack([a, a]))
print("concatenate axis=1:\n", np.concatenate([a, a], axis=1))

# random with seed (reproducibility matters in interviews/production)
rng = np.random.default_rng(42)
print("reproducible random ints:", rng.integers(0, 10, size=5))

# ---------------------------------------------------------------------------
# 2. PANDAS BASICS
# ---------------------------------------------------------------------------
print("\n" + "=" * 70, "\nPANDAS\n", "=" * 70)

df = pd.DataFrame({
    "id": range(1, 9),
    "team": ["A", "A", "B", "B", "C", "C", "A", "B"],
    "score": [55, 60, np.nan, 72, 90, 40, 65, np.nan],
    "region": ["N", "S", "N", "S", "N", "S", "N", "S"],
})
print(df)
print("\ninfo:")
df.info()
print("\ndescribe:\n", df.describe())

# selection: loc (label-based) vs iloc (position-based) -- common gotcha
print("\nloc by boolean:\n", df.loc[df["score"] > 60, ["team", "score"]])
print("\niloc first 2 rows, first 2 cols:\n", df.iloc[:2, :2])

# missing values
print("\nnull counts:\n", df.isna().sum())
df["score_filled"] = df["score"].fillna(df.groupby("team")["score"].transform("mean"))
print("\nfill NaN with group mean (avoids leaking global mean into a group):\n",
      df[["team", "score", "score_filled"]])

# groupby / agg -- extremely common interview task
agg = df.groupby("team").agg(
    avg_score=("score", "mean"),
    n=("id", "count"),
    max_score=("score", "max"),
).reset_index()
print("\ngroupby agg:\n", agg)

# pivot_table
pivot = df.pivot_table(values="score", index="team", columns="region", aggfunc="mean")
print("\npivot_table:\n", pivot)

# merge / join -- know inner/left/right/outer semantics
teams_meta = pd.DataFrame({"team": ["A", "B", "C"], "coach": ["Ravi", "Sam", "Lee"]})
merged = df.merge(teams_meta, on="team", how="left")
print("\nmerge (left join):\n", merged.head())

# apply / map / vectorized string ops -- prefer vectorized over .apply when possible
df["team_lower"] = df["team"].str.lower()          # vectorized (fast)
df["score_band"] = df["score"].apply(               # .apply only when logic is custom
    lambda x: "high" if pd.notna(x) and x >= 70 else "low"
)
print("\napply / str ops:\n", df[["team", "team_lower", "score", "score_band"]])

# sort, rank, duplicates
print("\nsorted by score desc:\n", df.sort_values("score", ascending=False).head(3))
print("\nduplicated rows:", df.duplicated(subset=["team", "region"]).sum())

# value_counts -- fast way to eyeball class balance
print("\nvalue_counts:\n", df["team"].value_counts())

# datetime handling (frequently asked)
dates = pd.date_range("2024-01-01", periods=5, freq="D")
ts = pd.Series(range(5), index=dates)
print("\nresample weekly sum:\n", ts.resample("W").sum())

print("\nDone. Key talking points: vectorization > loops, axis semantics, "
      "view vs copy, groupby+transform for leak-free imputation, merge types.")
