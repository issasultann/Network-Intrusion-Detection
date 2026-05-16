"""
Synthetic CICIDS2017-Like Demo Data Generator
----------------------------------------------
Generates realistic network flow features that mimic the statistical
properties of the real CICIDS2017 dataset.

Each attack class has distinct feature signatures so the classifiers
can actually learn and produce meaningful metrics.

Usage:
    from demo_data import generate_demo_dataset
    df = generate_demo_dataset(random_state=42)
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── 78 flow feature names (matching normalized CICIDS2017 column names) ──────
FEATURE_NAMES = [
    "flow_duration", "total_fwd_packets", "total_backward_packets",
    "total_length_of_fwd_packets", "total_length_of_bwd_packets",
    "fwd_packet_length_max", "fwd_packet_length_min",
    "fwd_packet_length_mean", "fwd_packet_length_std",
    "bwd_packet_length_max", "bwd_packet_length_min",
    "bwd_packet_length_mean", "bwd_packet_length_std",
    "flow_bytes_per_s", "flow_packets_per_s",
    "flow_iat_mean", "flow_iat_std", "flow_iat_max", "flow_iat_min",
    "fwd_iat_total", "fwd_iat_mean", "fwd_iat_std", "fwd_iat_max", "fwd_iat_min",
    "bwd_iat_total", "bwd_iat_mean", "bwd_iat_std", "bwd_iat_max", "bwd_iat_min",
    "fwd_psh_flags", "bwd_psh_flags", "fwd_urg_flags", "bwd_urg_flags",
    "fwd_header_length", "bwd_header_length",
    "fwd_packets_per_s", "bwd_packets_per_s",
    "min_packet_length", "max_packet_length",
    "packet_length_mean", "packet_length_std", "packet_length_variance",
    "fin_flag_count", "syn_flag_count", "rst_flag_count",
    "psh_flag_count", "ack_flag_count", "urg_flag_count",
    "cwe_flag_count", "ece_flag_count",
    "down_per_up_ratio", "average_packet_size",
    "avg_fwd_segment_size", "avg_bwd_segment_size",
    "fwd_header_length_1",
    "fwd_avg_bytes_per_bulk", "fwd_avg_packets_per_bulk", "fwd_avg_bulk_rate",
    "bwd_avg_bytes_per_bulk", "bwd_avg_packets_per_bulk", "bwd_avg_bulk_rate",
    "subflow_fwd_packets", "subflow_fwd_bytes",
    "subflow_bwd_packets", "subflow_bwd_bytes",
    "init_win_bytes_forward", "init_win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward",
    "active_mean", "active_std", "active_max", "active_min",
    "idle_mean", "idle_std", "idle_max", "idle_min",
]

# ── Per-class statistical signatures ─────────────────────────────────────────
# Each entry: (n_samples, mean_vector_scale, feature_noise_scale, special_overrides)
# Feature indices that matter most for discrimination:
#   0  = flow_duration       13 = flow_bytes_per_s
#   1  = total_fwd_packets   14 = flow_packets_per_s
#   43 = syn_flag_count      45 = psh_flag_count
#   46 = ack_flag_count      39 = packet_length_mean

_CLASS_PROFILES = {
    # name:  (n,   duration_µ, bytes_s_µ,  pkts_s_µ,  pkt_len_µ,  syn_µ, ack_µ, noise)
    "BENIGN":                   (80000, 50000, 1500,  12,  500, 0.1, 0.9, 0.20),
    "DoS Hulk":                 (10000,  1000, 85000, 800, 1300, 0.0, 1.0, 0.15),
    "PortScan":                 ( 5000,   120,   200, 900,   54, 1.0, 0.0, 0.10),
    "DDoS":                     ( 2000,   500, 70000, 600, 1200, 0.1, 0.9, 0.15),
    "DoS GoldenEye":            (  500,  2000, 60000, 400,  900, 0.0, 1.0, 0.18),
    "FTP-Patator":              (  300, 30000,   800,   4,  320, 0.2, 0.8, 0.25),
    "SSH-Patator":              (  300, 40000,   600,   3,  410, 0.2, 0.8, 0.25),
    "DoS Slowloris":            (  200,100000,   120,   1,   90, 0.0, 1.0, 0.20),
    "DoS SlowHTTPTest":         (  200, 90000,   150,   1,  130, 0.0, 1.0, 0.20),
    "Bot":                      (  100, 60000,   350,   2,  360, 0.1, 0.9, 0.30),
    "Web Attack Brute Force":   (  200, 20000,   900,   6,  260, 0.1, 0.9, 0.25),
    "Web Attack XSS":           (  100, 15000,   800,   5,  290, 0.1, 0.9, 0.25),
    "Web Attack SQL Injection":  (  20, 10000,   700,   5,  310, 0.1, 0.9, 0.30),
    "Infiltration":              (  10, 70000,   450,   3,  460, 0.0, 1.0, 0.35),
    "Heartbleed":                (  10,  5000,  4000,  10,  210, 0.0, 1.0, 0.30),
}


def _gen_class(label: str, profile: tuple, rng: np.random.Generator) -> pd.DataFrame:
    n, dur_mu, bps_mu, pps_mu, plen_mu, syn_mu, ack_mu, noise = profile
    n_feat = len(FEATURE_NAMES)

    # Base: all features random around 1.0 (will be scaled per-feature below)
    X = rng.exponential(scale=1.0, size=(n, n_feat)).astype(np.float32)

    # ── Inject class-specific patterns ───────────────────────────────────────
    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}

    def _set(feat, mu, scale=None):
        sc = scale if scale is not None else mu * noise
        X[:, idx[feat]] = np.abs(rng.normal(mu, max(sc, 1e-3), n))

    _set("flow_duration",           dur_mu,  dur_mu  * noise)
    _set("flow_bytes_per_s",        bps_mu,  bps_mu  * noise)
    _set("flow_packets_per_s",      pps_mu,  pps_mu  * noise)
    _set("packet_length_mean",      plen_mu, plen_mu * noise)
    _set("average_packet_size",     plen_mu, plen_mu * noise)

    _set("total_fwd_packets",       pps_mu * dur_mu / 1e6 * 0.6,
                                    pps_mu * dur_mu / 1e6 * 0.3)
    _set("total_backward_packets",  pps_mu * dur_mu / 1e6 * 0.4,
                                    pps_mu * dur_mu / 1e6 * 0.3)

    _set("fwd_packet_length_mean",  plen_mu * 0.6, plen_mu * noise)
    _set("bwd_packet_length_mean",  plen_mu * 0.4, plen_mu * noise)
    _set("max_packet_length",       plen_mu * 1.8, plen_mu * noise)
    _set("min_packet_length",       plen_mu * 0.1, plen_mu * 0.05)

    # Flag counts (discrete, clipped to [0,1] for single-packet flags)
    X[:, idx["syn_flag_count"]] = rng.binomial(1, syn_mu, n).astype(np.float32)
    X[:, idx["ack_flag_count"]] = rng.binomial(1, ack_mu, n).astype(np.float32)
    X[:, idx["psh_flag_count"]] = rng.binomial(1, 0.3,   n).astype(np.float32)
    X[:, idx["fin_flag_count"]] = rng.binomial(1, 0.2,   n).astype(np.float32)
    X[:, idx["rst_flag_count"]] = rng.binomial(1, 0.05,  n).astype(np.float32)

    # IAT features
    iat_mu = dur_mu / max(pps_mu * dur_mu / 1e6, 1)
    _set("flow_iat_mean", iat_mu, iat_mu * noise)
    _set("flow_iat_std",  iat_mu * 0.5, iat_mu * 0.3)
    _set("fwd_iat_mean",  iat_mu * 1.2, iat_mu * noise)
    _set("bwd_iat_mean",  iat_mu * 0.8, iat_mu * noise)

    # Header lengths
    X[:, idx["fwd_header_length"]]   = rng.normal(32, 4, n).clip(20, 60).astype(np.float32)
    X[:, idx["bwd_header_length"]]   = rng.normal(32, 4, n).clip(20, 60).astype(np.float32)
    X[:, idx["fwd_header_length_1"]] = X[:, idx["fwd_header_length"]]

    # Window bytes
    X[:, idx["init_win_bytes_forward"]]  = rng.choice([65535, 8192, 1024], n).astype(np.float32)
    X[:, idx["init_win_bytes_backward"]] = rng.choice([65535, 8192, 1024], n).astype(np.float32)

    # Segment size
    X[:, idx["min_seg_size_forward"]] = rng.normal(20, 2, n).clip(8, 40).astype(np.float32)

    # Packet length derived stats
    X[:, idx["packet_length_std"]]      = X[:, idx["packet_length_mean"]] * rng.uniform(0.1, 0.5, n)
    X[:, idx["packet_length_variance"]] = X[:, idx["packet_length_std"]] ** 2
    X[:, idx["fwd_packet_length_max"]]  = X[:, idx["fwd_packet_length_mean"]] * rng.uniform(1.5, 3.0, n)
    X[:, idx["fwd_packet_length_min"]]  = X[:, idx["fwd_packet_length_mean"]] * rng.uniform(0.05, 0.3, n)
    X[:, idx["fwd_packet_length_std"]]  = X[:, idx["fwd_packet_length_mean"]] * rng.uniform(0.1, 0.5, n)
    X[:, idx["bwd_packet_length_max"]]  = X[:, idx["bwd_packet_length_mean"]] * rng.uniform(1.5, 3.0, n)
    X[:, idx["bwd_packet_length_min"]]  = X[:, idx["bwd_packet_length_mean"]] * rng.uniform(0.05, 0.3, n)
    X[:, idx["bwd_packet_length_std"]]  = X[:, idx["bwd_packet_length_mean"]] * rng.uniform(0.1, 0.5, n)

    # Subflows
    X[:, idx["subflow_fwd_packets"]] = X[:, idx["total_fwd_packets"]]
    X[:, idx["subflow_fwd_bytes"]]   = (X[:, idx["total_fwd_packets"]] *
                                         X[:, idx["fwd_packet_length_mean"]])
    X[:, idx["subflow_bwd_packets"]] = X[:, idx["total_backward_packets"]]
    X[:, idx["subflow_bwd_bytes"]]   = (X[:, idx["total_backward_packets"]] *
                                         X[:, idx["bwd_packet_length_mean"]])

    # Active/idle (only relevant for long flows)
    active_mu = min(dur_mu * 0.7, 50000)
    _set("active_mean", active_mu, active_mu * 0.3)
    _set("active_std",  active_mu * 0.2, active_mu * 0.1)
    _set("active_max",  active_mu * 1.5, active_mu * 0.3)
    _set("active_min",  active_mu * 0.3, active_mu * 0.2)
    idle_mu = min(dur_mu * 0.3, 30000)
    _set("idle_mean", idle_mu, idle_mu * 0.4)
    _set("idle_std",  idle_mu * 0.2, idle_mu * 0.2)
    _set("idle_max",  idle_mu * 1.5, idle_mu * 0.4)
    _set("idle_min",  idle_mu * 0.1, idle_mu * 0.1)

    # Inject ~2% inf/NaN in flow_bytes_per_s (realistic CICIDS2017 quirk)
    nan_mask = rng.random(n) < 0.02
    X[nan_mask, idx["flow_bytes_per_s"]]    = np.inf
    X[nan_mask, idx["flow_packets_per_s"]]  = np.inf

    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["label"] = label
    return df


def generate_demo_dataset(random_state: int = 42,
                           scale: float = 1.0,
                           save_path: str = None) -> pd.DataFrame:
    """
    Generate a synthetic CICIDS2017-like DataFrame.

    Parameters
    ----------
    random_state : reproducibility seed
    scale        : multiply all class counts (0.5 = half-size for quick test)
    save_path    : if given, save as CSV to this path

    Returns
    -------
    DataFrame with 78 numeric feature columns + 'label' column
    """
    rng = np.random.default_rng(random_state)
    frames = []

    total = 0
    for label, profile in _CLASS_PROFILES.items():
        profile_scaled = (max(1, int(profile[0] * scale)),) + profile[1:]
        df_cls = _gen_class(label, profile_scaled, rng)
        frames.append(df_cls)
        total += len(df_cls)
        print(f"  Generated {label:<30}: {len(df_cls):>6,} samples")

    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    print(f"\nTotal synthetic samples : {total:,}")
    print(f"Feature columns         : {len(FEATURE_NAMES)}")
    print(f"Classes                 : {df['label'].nunique()}")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Saved to: {save_path}")

    return df
