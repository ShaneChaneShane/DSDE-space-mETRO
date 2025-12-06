import pandas as pd
import numpy as np
from glob import glob

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

import joblib
from preprocess_for_model import preprocess_for_model

# Load cleaned test data from Spark output

# Change this path if your folder is different
TEST_FOLDER = "C:/Users/Noon/Documents/DSDE/projectTraffy/1/test_data"

files = glob(f"{TEST_FOLDER}/part-*.csv")
if not files:
    raise FileNotFoundError(f"No CSV part files found in {TEST_FOLDER}")

dfs = [pd.read_csv(f) for f in files]
test_data = pd.concat(dfs, ignore_index=True)

print(f"Loaded test_data with shape: {test_data.shape}")

# Load model package

package = joblib.load("fast_7day_model.joblib")
model = package["model"]

# Build target fast_7d from completion_time_hours

hours_7 = 7 * 24
test_data["fast_7d"] = (test_data["completion_time_hours"] <= hours_7).astype(int)

y_test = test_data["fast_7d"].copy()

# Preprocess test data into feature matrix X_test

X_test = preprocess_for_model(test_data, package)
print(f"Prepared X_test with shape: {X_test.shape}")

# Predict and evaluate

y_proba = model.predict_proba(X_test)[:, 1]
y_pred  = (y_proba >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average="binary", pos_label=1
)
try:
    roc_auc = roc_auc_score(y_test, y_proba)
except Exception:
    roc_auc = float("nan")

print("\n=== Test set results: ≤7 days vs >7 days ===")
print("Accuracy:", acc)
print("Precision (fast<=7d):", prec)
print("Recall (fast<=7d):", rec)
print("F1 (fast<=7d):", f1)
print("ROC-AUC:", roc_auc)

print("\nClassification report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=[">7 days", "≤7 days"],
    zero_division=0
))

print("Confusion matrix (rows=true, cols=pred):")
print(confusion_matrix(y_test, y_pred))

# Simple baseline: always predict majority class

majority_class = y_test.mode()[0]
y_baseline = np.full_like(y_test, majority_class)

acc_base = accuracy_score(y_test, y_baseline)
print("\nBaseline (always predict class", majority_class, ") accuracy:", acc_base)
