import pandas as pd

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

def preprocess_for_model(df_raw, model_package):
    """
    df_raw:   pandas DataFrame from cleaned Spark output (train/test/new data)
    model_package: dict loaded from fast_7day_model.joblib

    returns: X (DataFrame) with same columns and order as model_package["feature_cols"]
    """
    df = df_raw.copy()

    org_dist_cols     = model_package.get("org_dist_cols", [])
    region_dummy_cols = model_package.get("region_dummy_cols", [])
    feature_cols      = model_package["feature_cols"]
    kmeans            = model_package["kmeans"]

    # Distance-related features
    if org_dist_cols:
        df = add_relevant_cols(df, org_dist_cols)
        df[org_dist_cols] = df[org_dist_cols].clip(lower=0)

    # Time features from timestamp
    df["created_at"] = pd.to_datetime(df["timestamp"])
    df["month"]      = df["created_at"].dt.month.astype("int16")
    df["dayofweek"]  = df["created_at"].dt.dayofweek.astype("int8")
    df["hour"]       = df["created_at"].dt.hour.astype("int8")
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype("int8")

    # Region clusters using TRAINED kmeans
    coords = df[["latitude", "longitude"]]
    coords_filled = coords.fillna(coords.mean())
    df["region"] = kmeans.predict(coords_filled)

    # One-hot encode region
    df = pd.get_dummies(df, columns=["region"], prefix="region")

    # Make sure all region dummy columns from training exist
    for col in region_dummy_cols:
        if col not in df.columns:
            df[col] = 0

    # Drop columns we don't want the model to see
    drop_cols = ["ticket_id", "timestamp", "created_at", "coords"]
    for c in drop_cols:
        if c in df.columns:
            df.drop(c, axis="columns", inplace=True)

    # Align to training feature columns
    # Any missing column will be filled with 0, extra columns will be dropped
    X = df.reindex(columns=feature_cols, fill_value=0)

    return X
