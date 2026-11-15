"""Trial metrics for elbow-control experiments (no ROS imports).

All functions take time series already resampled onto one uniform time grid `t` (seconds).

Metrics
  intent_accuracy   share of samples where decoder mode == intended class
                    (samples within `settle_s` after each intended change are excluded)
  onset_delay_s     median time from an intended flex/extend onset until the decoder
                    first reports that class
  false_motion_deg  elbow travel (deg) during intended rest/hold phases = unwanted movement
  movement_deg      mean elbow excursion per intended flex/extend phase
  tracking_rmse_deg RMS difference commanded vs actual (simulated) elbow angle
  tracking_lag_s    lag of actual behind commanded angle (cross-correlation peak)
"""

import numpy as np

from prosthetic_elbow_sim.features import CLASSES, label_to_class, mode_to_class


def _segments(classes):
    """Yield (start_idx, end_idx_exclusive, class) runs of equal class."""
    if len(classes) == 0:
        return
    start = 0
    for i in range(1, len(classes) + 1):
        if i == len(classes) or classes[i] != classes[start]:
            yield start, i, classes[start]
            start = i


def confusion_matrix(intended, decoded, classes=CLASSES):
    idx = {c: i for i, c in enumerate(classes)}
    m = np.zeros((len(classes), len(classes)), dtype=int)
    for a, b in zip(intended, decoded):
        if a in idx and b in idx:
            m[idx[a], idx[b]] += 1
    return m


def compute_metrics(t, intended_labels, decoder_modes, command_deg, actual_deg=None,
                    settle_s=0.3):
    t = np.asarray(t, dtype=float)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.01
    intended = np.array([label_to_class(x) for x in intended_labels])
    decoded = np.array([mode_to_class(x) for x in decoder_modes])
    cmd = np.asarray(command_deg, dtype=float)
    out = {'duration_s': float(t[-1] - t[0]) if len(t) else 0.0}

    # --- intent accuracy (excluding transition windows) ---
    keep = np.ones(len(t), dtype=bool)
    settle_n = int(round(settle_s / dt))
    for s, _, _ in _segments(intended):
        keep[s:s + settle_n] = False
    out['intent_accuracy'] = float(np.mean(intended[keep] == decoded[keep])) if keep.any() else float('nan')
    out['confusion'] = confusion_matrix(intended[keep], decoded[keep])

    # --- onset delay for movement phases ---
    delays = []
    for s, e, c in _segments(intended):
        if c not in ('flex', 'extend'):
            continue
        hits = np.nonzero(decoded[s:e] == c)[0]
        if len(hits):
            delays.append(hits[0] * dt)
    out['onset_delay_s'] = float(np.median(delays)) if delays else float('nan')
    out['n_movements'] = len(delays)

    # --- unwanted motion during rest/hold, excursion during movement ---
    false_motion, excursions = 0.0, []
    for s, e, c in _segments(intended):
        seg = cmd[s:e]
        if len(seg) < 2:
            continue
        if c in ('rest', 'hold'):
            a = s + settle_n
            if a < e - 1:
                false_motion += float(np.sum(np.abs(np.diff(cmd[a:e]))))
        else:
            excursions.append(float(np.max(seg) - np.min(seg)))
    out['false_motion_deg'] = false_motion
    out['movement_deg'] = float(np.mean(excursions)) if excursions else float('nan')

    # --- tracking: commanded vs simulated joint ---
    if actual_deg is not None:
        act = np.asarray(actual_deg, dtype=float)
        ok = ~np.isnan(act) & ~np.isnan(cmd)
        out['tracking_rmse_deg'] = float(np.sqrt(np.mean((cmd[ok] - act[ok]) ** 2))) if ok.any() else float('nan')
        out['tracking_lag_s'] = estimate_lag(cmd[ok], act[ok], dt)
    else:
        out['tracking_rmse_deg'] = float('nan')
        out['tracking_lag_s'] = float('nan')
    return out


def estimate_lag(reference, response, dt, max_lag_s=0.5):
    """Lag (s) of `response` behind `reference` from the cross-correlation peak."""
    ref = np.asarray(reference, float) - np.mean(reference)
    res = np.asarray(response, float) - np.mean(response)
    if len(ref) < 3 or np.allclose(ref, 0) or np.allclose(res, 0):
        return float('nan')
    max_k = max(1, int(max_lag_s / dt))
    best_k, best_c = 0, -np.inf
    for k in range(0, max_k + 1):
        a, b = ref[:len(ref) - k], res[k:]
        if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
            continue
        c = np.corrcoef(a, b)[0, 1]  # normalised, so overlap length does not bias the peak
        if c > best_c:
            best_k, best_c = k, c
    return best_k * dt


def summary_row(name, m):
    return {
        'trial': name,
        'intent_accuracy_%': round(100 * m['intent_accuracy'], 1),
        'onset_delay_ms': round(1000 * m['onset_delay_s'], 0),
        'false_motion_deg': round(m['false_motion_deg'], 1),
        'movement_deg': round(m['movement_deg'], 1),
        'tracking_rmse_deg': round(m['tracking_rmse_deg'], 2),
        'tracking_lag_ms': round(1000 * m['tracking_lag_s'], 0),
        'duration_s': round(m['duration_s'], 1),
    }
