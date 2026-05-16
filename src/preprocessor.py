"""
Preprocessing Pipeline for CICIDS2017
  - Drop low-variance / constant / duplicate columns
  - Handle inf / NaN values
  - Binary and multiclass label encoding
  - Robust feature scaling
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
import warnings

warnings.filterwarnings("ignore")


# ── Columns known to be non-numeric / identifiers in CICIDS2017 ─────────────
# Includes both the original UNB MachineLearningCSV format AND the
# Hugging Face mirror format (which has src_ip_dec, attempted_category, etc.)
_DROP_COLS = [
    # original CICIDS2017 identifiers
    "flow_id", "source_ip", "source_port",
    "destination_ip", "destination_port",
    "timestamp", "src_ip", "dst_ip",
    "src_port", "dst_port",
    # extra columns from the HF mirror
    "src_ip_dec", "dst_ip_dec",
    "attempted_category",
]


def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove identifier / timestamp columns AND any non-numeric leftover
    columns (except 'label')."""
    # 1) Drop explicit identifier columns
    to_drop = [c for c in _DROP_COLS if c in df.columns]
    df = df.drop(columns=to_drop)

    # 2) Coerce remaining feature columns to numeric, drop those that fail
    feature_cols = [c for c in df.columns if c != "label"]
    non_numeric = []
    for c in feature_cols:
        if df[c].dtype == "object":
            coerced = pd.to_numeric(df[c], errors="coerce")
            if coerced.isna().mean() > 0.5:
                # Mostly non-numeric → drop
                non_numeric.append(c)
            else:
                df[c] = coerced
    if non_numeric:
        df = df.drop(columns=non_numeric)

    return df


def handle_inf_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Replace inf with NaN, then impute NaN with column median."""
    df = df.replace([np.inf, -np.inf], np.nan)
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    return df


def remove_constant_columns(df: pd.DataFrame,
                             threshold: float = 0.0) -> pd.DataFrame:
    """Drop columns with variance at or below *threshold*."""
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols]
    sel = VarianceThreshold(threshold=threshold)
    sel.fit(X)
    keep = list(X.columns[sel.get_support()])
    return df[keep + ["label"]]


def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    print(f"  Removed {before - len(df):,} duplicate rows.")
    return df


def encode_labels(df: pd.DataFrame,
                  binary: bool = False) -> tuple[pd.DataFrame, LabelEncoder]:
    """
    Encode the 'label' column.

    binary=True  → BENIGN=0, all attacks=1
    binary=False → integer-encode all unique classes
    """
    le = LabelEncoder()
    if binary:
        df = df.copy()
        df["label"] = (df["label"] != "BENIGN").astype(int)
        le.classes_ = np.array(["BENIGN", "ATTACK"])
    else:
        df = df.copy()
        df["label"] = le.fit_transform(df["label"])
    return df, le


def scale_features(X_train: np.ndarray,
                   X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, RobustScaler]:
    """Fit RobustScaler on training set, transform both splits."""
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    return X_train_s, X_test_s, scaler


# ── Master pipeline ──────────────────────────────────────────────────────────
def run_preprocessing(df: pd.DataFrame,
                      binary: bool = False,
                      variance_threshold: float = 0.0,
                      verbose: bool = True) -> tuple[pd.DataFrame, LabelEncoder]:
    """
    Full preprocessing pipeline.

    Returns cleaned DataFrame and the fitted LabelEncoder.
    """
    if verbose:
        print(f"[Preprocessing] Starting with {len(df):,} rows, "
              f"{len(df.columns)} columns …")

    df = drop_irrelevant_columns(df)
    df = handle_inf_nan(df)
    df = remove_duplicate_rows(df)
    df = remove_constant_columns(df, threshold=variance_threshold)
    df, le = encode_labels(df, binary=binary)

    if verbose:
        print(f"[Preprocessing] Done → {len(df):,} rows, "
              f"{len(df.columns) - 1} features, "
              f"{df['label'].nunique()} classes.")
    return df, le


# ── Train/test split helper ──────────────────────────────────────────────────
def split_features_labels(df: pd.DataFrame):
    """Return (X, y) numpy arrays from the preprocessed DataFrame."""
    X = df.drop(columns=["label"]).values.astype(np.float32)
    y = df["label"].values
    return X, y
