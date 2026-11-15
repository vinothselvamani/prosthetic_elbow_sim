"""EMG time-domain features and intent labels (no ROS imports).

Hudgins time-domain feature set per channel:
    MAV  mean absolute value
    RMS  root mean square
    WL   waveform length
    ZC   zero crossings        (with amplitude threshold)
    SSC  slope sign changes    (with amplitude threshold)
"""

import numpy as np

FEATURE_NAMES = ['MAV', 'RMS', 'WL', 'ZC', 'SSC']

# Intent classes used by the classifier decoder and by the metrics
CLASSES = ['rest', 'flex', 'extend', 'hold']


def label_to_class(label: str) -> str:
    """Map task-phase labels (flex_40, extend_70, cocontraction, ...) to an intent class."""
    label = (label or '').strip().lower()
    if label.startswith('flex'):
        return 'flex'
    if label.startswith('ext'):
        return 'extend'
    if label.startswith('cocon') or label.startswith('hold'):
        return 'hold'
    return 'rest'


def mode_to_class(mode: str) -> str:
    """Map decoder modes (rest, flex, extend, cocontraction_hold) to an intent class."""
    mode = (mode or '').strip().lower()
    if mode.startswith('cocon') or mode == 'hold':
        return 'hold'
    if mode in ('flex', 'extend'):
        return mode
    return 'rest'


def td_features(window: np.ndarray, thresholds=(0.0, 0.0)) -> np.ndarray:
    """Time-domain features for one analysis window.

    window: array of shape (n_samples, n_channels)
    thresholds: per-channel amplitude threshold for ZC/SSC (suppresses noise counts)
    returns: 1-D array of length n_channels * 5, ordered [ch0 features..., ch1 features...]
    """
    window = np.asarray(window, dtype=float)
    if window.ndim != 2 or window.shape[0] < 3:
        raise ValueError('window must have shape (n_samples >= 3, n_channels)')
    feats = []
    for ch in range(window.shape[1]):
        x = window[:, ch]
        thr = thresholds[ch] if ch < len(thresholds) else 0.0
        dx = np.diff(x)
        mav = np.mean(np.abs(x))
        rms = np.sqrt(np.mean(x * x))
        wl = np.sum(np.abs(dx))
        zc = np.sum((x[:-1] * x[1:] < 0) & (np.abs(dx) >= thr))
        ssc = np.sum((dx[:-1] * dx[1:] < 0)
                     & ((np.abs(dx[:-1]) >= thr) | (np.abs(dx[1:]) >= thr)))
        # counts are normalised by window length so models transfer across window sizes
        n = len(x)
        feats.extend([mav, rms, wl / n, zc / n, ssc / n])
    return np.array(feats)


def sliding_windows(emg: np.ndarray, labels, window: int, step: int, thresholds=(0.0, 0.0)):
    """Cut a labelled recording into feature vectors.

    emg: (n_samples, 2) array; labels: sequence of per-sample labels (or None)
    returns X (n_windows, n_features), y (n_windows,) intent classes (label at window end)
    """
    emg = np.asarray(emg, dtype=float)
    X, y = [], []
    for end in range(window, len(emg) + 1, step):
        X.append(td_features(emg[end - window:end], thresholds))
        if labels is not None:
            y.append(label_to_class(labels[end - 1]))
    return np.array(X), np.array(y)
