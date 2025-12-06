import pandas as pd
import numpy as np
from glob import glob

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
import joblib


TRAIN_FOLDER = "C:/Users/Noon/Documents/DSDE/projectTraffy/1/train_data"

files = glob(f"{TRAIN_FOLDER}/part-*.csv")
if not files:
    raise FileNotFoundError(f"No CSV part files found in {TRAIN_FOLDER}")

dfs = [pd.read_csv(f) for f in files]
train_data = pd.concat(dfs, ignore_index=True)


# columns that come from organization distance pivot in clean.py
org_dist_cols = [col for col in train_data.columns if col.startswith("dist_")]
print(f"Found {len(org_dist_cols)} organization distance columns")

def add_relevant_cols(df, org_dist_columns):
    """
    For each dist_* column, add a binary relevant_* column:
    relevant_* = 1 if distance > 0, else 0
    """
    if not org_dist_columns:
        return df

    relevant = (df[org_dist_columns] > 0).astype(int)
    new_names = [c.replace("dist_", "relevant_") for c in org_dist_columns]
    relevant.columns = new_names
    return df.join(relevant)

train_data = add_relevant_cols(train_data, org_dist_cols)

# Clip negative distances to 0 just in case
if org_dist_cols:
    train_data[org_dist_cols] = train_data[org_dist_cols].clip(lower=0)

# timestamp is string in ISO-like format from Spark
train_data["created_at"] = pd.to_datetime(train_data["timestamp"])
train_data["month"]      = train_data["created_at"].dt.month.astype("int16")
train_data["dayofweek"]  = train_data["created_at"].dt.dayofweek.astype("int8")  # 0=Mon
train_data["hour"]       = train_data["created_at"].dt.hour.astype("int8")
train_data["is_weekend"] = train_data["dayofweek"].isin([5, 6]).astype("int8")

coords = train_data[["latitude", "longitude"]].dropna()
print(f"KMeans training on {coords.shape[0]} coordinate rows")

# n_init="auto" is fine on recent sklearn; if it errors, remove n_init argument
kmeans = KMeans(n_clusters=50, random_state=42)
kmeans.fit(coords)

# Use kmeans to assign a region to every row (fill missing with mean coords)
coords_filled_all = train_data[["latitude", "longitude"]].fillna(coords.mean())
train_data["region"] = kmeans.predict(coords_filled_all)

# One-hot encode region
train_data = pd.get_dummies(train_data, columns=["region"], prefix="region")

region_dummy_cols = [c for c in train_data.columns if c.startswith("region_")]
print(f"Created {len(region_dummy_cols)} region dummy columns")

# Drop columns we don't want the model to use directly

drop_cols = ["ticket_id", "timestamp", "created_at", "coords"]
for c in drop_cols:
    if c in train_data.columns:
        train_data.drop(c, axis="columns", inplace=True)

# Define target fast_7d and feature columns

# Binary target: 1 = finishes within 7 days, 0 = takes longer than 7 days
hours_7 = 7 * 24
train_data["fast_7d"] = (train_data["completion_time_hours"] <= hours_7).astype(int)

# Columns we should NOT feed into the model
cols_to_exclude = {
    "completion_time_hours",
    "log_completion_hours",
    "time_bucket",
    "fast_7d",
}

feature_cols = [c for c in train_data.columns if c not in cols_to_exclude]
print(f"Using {len(feature_cols)} feature columns")

X = train_data[feature_cols]
y = train_data["fast_7d"]

# Train/validation split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("Train size:", X_train.shape, "Val size:", X_val.shape)

# Train HistGradientBoostingClassifier

clf_bin = HistGradientBoostingClassifier(
    max_depth=6,
    learning_rate=0.1,
    max_iter=800,
    min_samples_leaf=20,
    l2_regularization=1.0,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
)

clf_bin.fit(X_train, y_train)

# Evaluate on validation set

y_pred = clf_bin.predict(X_val)
y_proba = clf_bin.predict_proba(X_val)[:, 1]

acc = accuracy_score(y_val, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(
    y_val, y_pred, average="binary", pos_label=1
)
try:
    roc_auc = roc_auc_score(y_val, y_proba)
except Exception:
    roc_auc = float("nan")

print("\n=== Binary ≤7 days vs >7 days (validation) ===")
print("Accuracy:", acc)
print("Precision (fast<=7d):", prec)
print("Recall (fast<=7d):", rec)
print("F1 (fast<=7d):", f1)
print("ROC-AUC:", roc_auc)

print("\nClassification report:")
print(classification_report(
    y_val,
    y_pred,
    target_names=[">7 days", "≤7 days"],
    zero_division=0
))

print("Confusion matrix (rows=true, cols=pred):")
print(confusion_matrix(y_val, y_pred))

# Save model package for later use (Streamlit, prediction script)

model_package = {
    "model": clf_bin,
    "feature_cols": feature_cols,
    "org_dist_cols": org_dist_cols,
    "region_dummy_cols": region_dummy_cols,
    "kmeans": kmeans,
}

joblib.dump(model_package, "fast_7day_model.joblib")
print("\nSaved model package to fast_7day_model.joblib")
