"""Offline tests of the EMG pipeline (no ROS needed): pytest test/"""

import math

import pytest

from prosthetic_elbow_sim.decoder_core import (
    ProportionalElbowDecoder, RmsEnvelope, normalise)
from prosthetic_elbow_sim.synthetic_emg import SyntheticEmg


def test_rms_of_constant():
    env = RmsEnvelope(10)
    for _ in range(20):
        v = env.add(-0.3)
    assert v == pytest.approx(0.3)


def test_normalise_clamps():
    assert normalise(0.01, 0.02, 0.5) == 0.0
    assert normalise(0.26, 0.02, 0.5) == pytest.approx(0.5)
    assert normalise(0.9, 0.02, 0.5) == 1.0


def test_deadband_and_cocontraction():
    d = ProportionalElbowDecoder()
    assert d.velocity_command(0.05, 0.0) == 0.0 and d.mode == 'rest'
    assert d.velocity_command(0.6, 0.6) == 0.0 and d.mode == 'cocontraction_hold'
    assert d.velocity_command(0.6, 0.0) > 0 and d.mode == 'flex'
    assert d.velocity_command(0.0, 0.6) < 0 and d.mode == 'extend'


def test_joint_limits_respected():
    d = ProportionalElbowDecoder(elbow_max=2.53)
    for _ in range(1000):
        d.step(1.0, 0.0, 0.01)
    assert d.angle == pytest.approx(2.53)
    for _ in range(1000):
        d.step(0.0, 1.0, 0.01)
    assert d.angle == pytest.approx(0.0)


def run_pipeline(seconds, fs=1000.0, control_hz=100.0, window_ms=150.0):
    gen = SyntheticEmg(fs=fs, seed=1)
    window = int(fs * window_ms / 1000)
    env = [RmsEnvelope(window), RmsEnvelope(window)]
    dec = ProportionalElbowDecoder()
    per_ctrl = int(fs / control_hz)
    trace = []
    for k in range(int(seconds * fs)):
        b, t, label = gen.sample()
        env[0].add(b)
        env[1].add(t)
        if (k + 1) % per_ctrl == 0:
            a_b = normalise(env[0].value, 0.02, 0.5)
            a_t = normalise(env[1].value, 0.02, 0.5)
            dec.step(a_b, a_t, 1.0 / control_hz)
            trace.append(((k + 1) / fs, label, dec.angle, dec.mode))
    return trace


def angle_at(trace, t):
    return min(trace, key=lambda r: abs(r[0] - t))[2]


def test_closed_loop_follows_scenario():
    tr = run_pipeline(16.0)
    # rest 0-2 s: elbow stays near 0
    assert angle_at(tr, 1.9) < math.radians(3)
    # flex_40 2-4.5 s: elbow flexes substantially
    assert angle_at(tr, 4.5) > math.radians(45)
    # rest 4.5-6 s: holds position (little drift)
    assert abs(angle_at(tr, 5.9) - angle_at(tr, 4.8)) < math.radians(3)
    # extend_40 6-8.5 s: comes back down
    assert angle_at(tr, 8.5) < angle_at(tr, 6.0) - math.radians(30)
    # cocontraction 11.5-13 s: elbow held (mode reported)
    modes = [r[3] for r in tr if 11.8 < r[0] < 12.9]
    assert modes.count('cocontraction_hold') > 0.8 * len(modes)
