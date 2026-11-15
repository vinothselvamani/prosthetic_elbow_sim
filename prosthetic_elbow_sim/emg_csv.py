"""Load two-channel EMG recordings from CSV (no ROS imports).

Expected columns (header row required, names configurable):
    biceps, triceps            raw sEMG samples, uniformly sampled
    label   (optional)         task phase per sample: rest / flex / extend / hold
                               (flex_40, extend_70, cocontraction ... are also understood)
"""

import csv

import numpy as np


def load_emg_csv(path, biceps_col='biceps', triceps_col='triceps', label_col='label'):
    """Return (emg (n,2) float array, labels list or None)."""
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        for c in (biceps_col, triceps_col):
            if c not in cols:
                raise ValueError(f'{path}: column "{c}" not found (columns: {cols})')
        has_label = label_col in cols
        emg, labels = [], []
        for row in reader:
            try:
                emg.append((float(row[biceps_col]), float(row[triceps_col])))
            except (TypeError, ValueError):
                continue  # skip blank / malformed rows
            if has_label:
                labels.append(row[label_col])
    if not emg:
        raise ValueError(f'{path}: no numeric samples found')
    return np.asarray(emg, dtype=float), (labels if has_label else None)


def moving_rms(x, window):
    """Moving-window RMS of a 1-D signal (same length, causal)."""
    x = np.asarray(x, dtype=float)
    c = np.cumsum(np.concatenate([[0.0], x * x]))
    idx = np.arange(len(x))
    a = np.maximum(0, idx + 1 - window)
    return np.sqrt(np.maximum(c[idx + 1] - c[a], 0.0) / (idx + 1 - a))
