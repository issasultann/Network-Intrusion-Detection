"""
Data Loader for CICIDS2017 Dataset
Handles loading, merging, and initial inspection of all CSV files.
"""

import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path


# ── Column name normalizer ──────────────────────────────────────────────────
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("/", "_per_")
        .str.replace(r"[^a-z0-9_]", "", regex=True)
    )
    return df


# ── Label normalizer ────────────────────────────────────────────────────────
LABEL_MAP = {
    "benign":           "BENIGN",
    "bot":              "Bot",
    "portscan":         "PortScan",
    "infiltration":     "Infiltration",
    "heartbleed":       "Heartbleed",
    "dos hulk":         "DoS Hulk",
    "dos goldeneye":    "DoS GoldenEye",
    "dos slowloris":    "DoS Slowloris",
    "dos slowhttptest": "DoS SlowHTTPTest",
    "ddos":             "DDoS",
    "ftp-patator":      "FTP-Patator",
    "ssh-patator":      "SSH-Patator",
    "web attack \x96 brute force": "Web Attack Brute Force",
    "web attack \x96 xss":        "Web Attack XSS",
    "web attack \x96 sql injection": "Web Attack SQL Injection",
    "web attack – brute force": "Web Attack Brute Force",
    "web attack – xss":         "Web Attack XSS",
    "web attack – sql injection": "Web Attack SQL Injection",
    "web attack  brute force":  "Web Attack Brute Force",
    "web attack  xss":          "Web Attack XSS",
    "web attack  sql injection": "Web Attack SQL Injection",
}


def _clean_label(series: pd.Series) -> pd.Series:
    cleaned = series.str.strip()
    lower   = cleaned.str.lower()
    mapped  = lower.map(LABEL_MAP)
    return mapped.where(mapped.notna(), cleaned)


# ── Main loader ─────────────────────────────────────────────────────────────
def load_cicids2017(data_dir: str, sample_frac: float = 1.0,
                    random_state: int = 42) -> pd.DataFrame:
    """
    Load all CICIDS2017 CSV files from *data_dir* into a single DataFrame.

    Parameters
    ----------
    data_dir    : path to folder containing the CICIDS2017 CSV files
    sample_frac : fraction of rows to keep per file (useful for quick runs)
    random_state: reproducibility seed

    Returns
    -------
    Merged and lightly-cleaned DataFrame with a 'label' column.
    """
    data_dir = Path(data_dir)
    csv_files = sorted(glob.glob(str(data_dir / "*.csv")))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in '{data_dir}'.\n"
            "Download CICIDS2017 from https://www.unb.ca/cic/datasets/ids-2017.html "
            "and place the CSV files in that folder."
        )

    frames = []
    for fp in csv_files:
        print(f"  Loading {os.path.basename(fp)} …", end=" ")
        df = pd.read_csv(fp, low_memory=False, encoding="utf-8",
                         encoding_errors="replace")
        df = _normalize_columns(df)
        if sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=random_state)
        frames.append(df)
        print(f"{len(df):,} rows")

    merged = pd.concat(frames, ignore_index=True)

    # Standardize label column (may be named ' label' after strip)
    if "label" not in merged.columns:
        # Try to find a column that ends with 'label'
        label_col = [c for c in merged.columns if "label" in c]
        if label_col:
            merged.rename(columns={label_col[0]: "label"}, inplace=True)
        else:
            raise KeyError("No 'label' column found in dataset.")

    merged["label"] = _clean_label(merged["label"])

    print(f"\nTotal rows loaded : {len(merged):,}")
    print(f"Total columns     : {len(merged.columns)}")
    print("\nClass distribution:")
    vc = merged["label"].value_counts()
    for cls, cnt in vc.items():
        pct = cnt / len(merged) * 100
        print(f"  {cls:<35} {cnt:>8,}  ({pct:5.2f}%)")

    return merged


# ── Quick dataset summary ───────────────────────────────────────────────────
def dataset_summary(df: pd.DataFrame) -> dict:
    """Return a dict of key dataset statistics."""
    total = len(df)
    benign_count = (df["label"] == "BENIGN").sum()
    attack_count = total - benign_count
    num_classes  = df["label"].nunique()

    minority_class  = df["label"].value_counts().idxmin()
    minority_count  = df["label"].value_counts().min()
    imbalance_ratio = benign_count / max(minority_count, 1)

    return {
        "total_samples":    total,
        "benign_count":     int(benign_count),
        "attack_count":     int(attack_count),
        "num_classes":      num_classes,
        "minority_class":   minority_class,
        "minority_count":   int(minority_count),
        "imbalance_ratio":  round(imbalance_ratio, 1),
        "num_features":     len(df.columns) - 1,
    }
