"""
Resampling Strategies for Class Imbalance
  - SMOTE (Synthetic Minority Over-sampling Technique)
  - Tomek Links (under-sampling)
  - SMOTETomek (hybrid)
  - SMOTEENN (alternative hybrid)
  - RandomOverSampler / RandomUnderSampler baselines
"""

import numpy as np
from collections import Counter

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import TomekLinks, RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN


def _print_distribution(title: str, y: np.ndarray) -> None:
    print(f"\n{title}")
    counts = Counter(y)
    total  = len(y)
    for cls in sorted(counts):
        cnt = counts[cls]
        print(f"  Class {cls}: {cnt:>8,}  ({cnt/total*100:.2f}%)")


def apply_smote(X_train: np.ndarray, y_train: np.ndarray,
                k_neighbors: int = 5,
                random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Pure SMOTE over-sampling."""
    from collections import Counter
    min_count = min(Counter(y_train).values())
    k = min(k_neighbors, min_count - 1) if min_count > 1 else 1
    sm = SMOTE(k_neighbors=k, random_state=random_state)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    _print_distribution("After SMOTE:", y_res)
    return X_res, y_res


def apply_tomek(X_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pure Tomek Links under-sampling."""
    tl = TomekLinks()
    X_res, y_res = tl.fit_resample(X_train, y_train)
    _print_distribution("After Tomek Links:", y_res)
    return X_res, y_res


def apply_smote_tomek(X_train: np.ndarray, y_train: np.ndarray,
                      smote_k: int = 5,
                      random_state: int = 42,
                      max_samples_per_class: int = 50_000
                      ) -> tuple[np.ndarray, np.ndarray]:
    """
    Hybrid: SMOTE over-sampling followed by Tomek Links cleaning.
    This is the primary strategy recommended in the paper.

    Note on scalability: TomekLinks performs an O(n^2) nearest-neighbour
    pass, which is prohibitive on >100k samples. To stay tractable, we
    pre-undersample the majority class to `max_samples_per_class` BEFORE
    applying SMOTETomek. All minority samples are preserved.
    """
    _print_distribution("Before SMOTETomek:", y_train)

    counts = Counter(y_train)
    majority_class = max(counts, key=counts.get)
    majority_count = counts[majority_class]

    # Step 0: undersample majority class if too large (keeps Tomek tractable)
    if majority_count > max_samples_per_class:
        print(f"  Pre-undersampling majority class "
              f"({majority_class}: {majority_count:,} → {max_samples_per_class:,})")
        target_strategy = {
            cls: min(cnt, max_samples_per_class) for cls, cnt in counts.items()
        }
        rus = RandomUnderSampler(sampling_strategy=target_strategy,
                                 random_state=random_state)
        X_train, y_train = rus.fit_resample(X_train, y_train)
        _print_distribution("After pre-undersampling:", y_train)

    min_count = min(Counter(y_train).values())
    k = min(smote_k, min_count - 1) if min_count > 1 else 1
    smote = SMOTE(k_neighbors=k, random_state=random_state)
    st    = SMOTETomek(smote=smote, random_state=random_state)
    X_res, y_res = st.fit_resample(X_train, y_train)

    _print_distribution("After SMOTETomek:", y_res)
    return X_res, y_res


def apply_smote_enn(X_train: np.ndarray, y_train: np.ndarray,
                    random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Alternative hybrid: SMOTE + Edited Nearest Neighbours."""
    _print_distribution("Before SMOTEENN:", y_train)

    se = SMOTEENN(random_state=random_state)
    X_res, y_res = se.fit_resample(X_train, y_train)

    _print_distribution("After SMOTEENN:", y_res)
    return X_res, y_res


def apply_random_over(X_train: np.ndarray, y_train: np.ndarray,
                      random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Naive random over-sampling baseline."""
    ros = RandomOverSampler(random_state=random_state)
    X_res, y_res = ros.fit_resample(X_train, y_train)
    _print_distribution("After RandomOverSampler:", y_res)
    return X_res, y_res


def apply_random_under(X_train: np.ndarray, y_train: np.ndarray,
                       random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Naive random under-sampling baseline."""
    rus = RandomUnderSampler(random_state=random_state)
    X_res, y_res = rus.fit_resample(X_train, y_train)
    _print_distribution("After RandomUnderSampler:", y_res)
    return X_res, y_res


# ── Strategy registry ────────────────────────────────────────────────────────
RESAMPLING_STRATEGIES = {
    "none":           lambda X, y: (X, y),
    "smote":          apply_smote,
    "tomek":          apply_tomek,
    "smote_tomek":    apply_smote_tomek,   # ← paper's recommendation
    "smote_enn":      apply_smote_enn,
    "random_over":    apply_random_over,
    "random_under":   apply_random_under,
}


def resample(X_train: np.ndarray, y_train: np.ndarray,
             strategy: str = "smote_tomek",
             **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply the chosen resampling strategy.

    strategy : one of RESAMPLING_STRATEGIES keys
               default = 'smote_tomek' (paper recommendation)
    """
    if strategy not in RESAMPLING_STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. "
                         f"Choose from: {list(RESAMPLING_STRATEGIES.keys())}")
    fn = RESAMPLING_STRATEGIES[strategy]
    if strategy == "none":
        return fn(X_train, y_train)
    return fn(X_train, y_train, **kwargs)
