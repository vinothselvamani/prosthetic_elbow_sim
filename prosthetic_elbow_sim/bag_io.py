"""Read an elbow trial rosbag2 recording into aligned numpy arrays."""

import numpy as np

TRIAL_TOPICS = ['/emg/raw', '/emg/activation', '/emg/envelope', '/emg/intended_label',
                '/elbow/command_deg', '/elbow/mode', '/joint_states']


def read_bag(path, topics=TRIAL_TOPICS):
    """Return {topic: [(t_sec, msg), ...]} using the bag's receive timestamps."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(path), storage_id=''),
                rosbag2_py.ConverterOptions('', ''))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    wanted = [t for t in topics if t in types]
    msg_cls = {t: get_message(types[t]) for t in wanted}
    out = {t: [] for t in wanted}
    while reader.has_next():
        topic, data, stamp_ns = reader.read_next()
        if topic in msg_cls:
            out[topic].append((stamp_ns * 1e-9, deserialize_message(data, msg_cls[topic])))
    return out


def _hold(t_src, values, t_grid):
    """Zero-order-hold resample (last value before each grid time)."""
    if len(t_src) == 0:
        return None
    idx = np.searchsorted(np.asarray(t_src), t_grid, side='right') - 1
    idx = np.clip(idx, 0, len(t_src) - 1)
    values = np.asarray(values, dtype=object if isinstance(values[0], str) else float)
    res = values[idx]
    if res.dtype != object:
        res = res.astype(float)
        res[t_grid < t_src[0]] = np.nan
    return res


def trial_arrays(msgs, rate_hz=100.0, emg_rate_hz=1000.0):
    """Align topics on a uniform grid. Returns a dict of arrays (missing topics -> None)."""
    starts = [v[0][0] for v in msgs.values() if v]
    ends = [v[-1][0] for v in msgs.values() if v]
    if not starts:
        raise ValueError('bag contains none of the trial topics')
    t0, t1 = max(starts), min(ends)
    if t1 <= t0:
        t0, t1 = min(starts), max(ends)
    grid = np.arange(t0, t1, 1.0 / rate_hz)
    out = {'t': grid - t0}

    def series(topic, fn):
        items = msgs.get(topic) or []
        if not items:
            return None
        ts = [s for s, _ in items]
        vals = [fn(m) for _, m in items]
        return _hold(ts, vals, grid)

    out['mode'] = series('/elbow/mode', lambda m: m.data)
    out['intended'] = series('/emg/intended_label', lambda m: m.data)
    out['command_deg'] = series('/elbow/command_deg', lambda m: m.data)

    def elbow_deg(m):
        if 'elbow_joint' in m.name:
            return np.degrees(m.position[m.name.index('elbow_joint')])
        return np.nan
    out['actual_deg'] = series('/joint_states', elbow_deg)

    acts = msgs.get('/emg/activation') or []
    if acts:
        ts = [s for s, _ in acts]
        a_b = _hold(ts, [m.data[0] for _, m in acts], grid)
        a_t = _hold(ts, [m.data[1] for _, m in acts], grid)
        out['activation'] = np.column_stack([a_b, a_t])
    else:
        out['activation'] = None

    raws = msgs.get('/emg/raw') or []
    if raws:
        samples, times = [], []
        for stamp, m in raws:
            d = np.asarray(m.data).reshape(-1, 2)
            n = len(d)
            times.extend(stamp - (n - 1 - np.arange(n)) / emg_rate_hz)
            samples.append(d)
        times = np.asarray(times)
        emg = np.vstack(samples)
        keep = (times >= t0) & (times <= t1)
        out['emg_t'], out['emg'] = times[keep] - t0, emg[keep]
    else:
        out['emg_t'], out['emg'] = None, None
    return out
