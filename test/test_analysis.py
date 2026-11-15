"""Offline tests for features, classifier, metrics and bag alignment (no ROS runtime)."""

from types import SimpleNamespace

import numpy as np
import pytest

from prosthetic_elbow_sim.bag_io import trial_arrays
from prosthetic_elbow_sim.features import label_to_class, mode_to_class, td_features
from prosthetic_elbow_sim.metrics import compute_metrics, estimate_lag
from prosthetic_elbow_sim.offline import make_decoder, run_decoder, synthetic_recording


def test_label_mapping():
    assert label_to_class('flex_40') == 'flex'
    assert label_to_class('extend_70') == 'extend'
    assert label_to_class('cocontraction') == 'hold'
    assert label_to_class('rest') == 'rest'
    assert mode_to_class('cocontraction_hold') == 'hold'


def test_td_features_shape_and_values():
    w = np.column_stack([np.ones(100), -np.ones(100)])
    f = td_features(w)
    assert f.shape == (10,)
    assert f[0] == pytest.approx(1.0)  # MAV ch0
    assert f[3] == 0                   # no zero crossings


def test_lag_estimate():
    x = np.sin(np.linspace(0, 20, 2000))
    y = np.roll(x, 5)
    assert estimate_lag(x, y, 0.01) == pytest.approx(0.05)


def test_proportional_offline_metrics():
    emg, labels = synthetic_recording(20, seed=3)
    r = run_decoder(emg, labels, make_decoder('proportional'))
    m = compute_metrics(r['t'], r['intended'], r['mode'], r['command_deg'])
    assert m['intent_accuracy'] > 0.9
    assert m['n_movements'] >= 4
    assert m['false_motion_deg'] < 5


def test_classifier_decoder_end_to_end(tmp_path):
    pytest.importorskip('sklearn')
    from prosthetic_elbow_sim.features import sliding_windows
    from prosthetic_elbow_sim.train_decoder import build_model
    emg, labels = synthetic_recording(48, seed=11)
    X, y = sliding_windows(emg, labels, 150, 50)
    model = build_model('lda').fit(X, y)
    dec = make_decoder('classifier', {'model': model, 'thresholds': (0.0, 0.0)})
    emg2, labels2 = synthetic_recording(20, seed=12)
    r = run_decoder(emg2, labels2, dec)
    m = compute_metrics(r['t'], r['intended'], r['mode'], r['command_deg'])
    assert m['intent_accuracy'] > 0.85


def test_bag_alignment_with_fake_messages():
    S = SimpleNamespace
    msgs = {
        '/elbow/mode': [(10.0 + i * 0.01, S(data='flex' if i > 50 else 'rest')) for i in range(200)],
        '/emg/intended_label': [(10.0 + i * 0.01, S(data='flex_40' if i > 45 else 'rest')) for i in range(200)],
        '/elbow/command_deg': [(10.0 + i * 0.01, S(data=float(max(0, i - 50)))) for i in range(200)],
        '/joint_states': [(10.0 + i * 0.01, S(name=['elbow_joint'], position=[np.radians(max(0, i - 53))]))
                          for i in range(200)],
        '/emg/activation': [(10.0 + i * 0.01, S(data=[0.4, 0.0])) for i in range(200)],
        '/emg/raw': [(10.0 + i * 0.01, S(data=[0.1, 0.0] * 10)) for i in range(200)],
    }
    d = trial_arrays(msgs)
    assert len(d['t']) > 150
    assert d['emg'].shape[1] == 2
    m = compute_metrics(d['t'], d['intended'], d['mode'], d['command_deg'], d['actual_deg'])
    assert m['tracking_lag_s'] == pytest.approx(0.03, abs=0.011)


def test_simulated_subject_sessions_differ():
    from prosthetic_elbow_sim.emg_subjects import SubjectModel, simulate_session
    from prosthetic_elbow_sim.features import label_to_class
    s = SubjectModel(1)
    e1, l1 = simulate_session(s, 1, n_trials=3)
    e2, l2 = simulate_session(s, 2, n_trials=3)
    assert e1.shape[1] == 2 and len(e1) == len(l1)
    assert {label_to_class(x) for x in l1} >= {'rest', 'flex', 'extend'}
    # electrodes re-applied -> different signal level between sessions
    assert abs(np.std(e1[:, 0]) - np.std(e2[:, 0])) > 1e-4
