"""Run a decoder over a recorded/synthetic EMG stream exactly as the ROS node does
(same envelope, same 100 Hz control loop) - without ROS. No ROS imports."""

import numpy as np

from prosthetic_elbow_sim.classifier_decoder import ClassifierElbowDecoder
from prosthetic_elbow_sim.decoder_core import ProportionalElbowDecoder, RmsEnvelope, normalise
from prosthetic_elbow_sim.synthetic_emg import SyntheticEmg


def synthetic_recording(seconds, fs=1000.0, seed=0, rest=(0.02, 0.02), mvc=(0.5, 0.5)):
    gen = SyntheticEmg(fs=fs, rest=rest, mvc=mvc, seed=seed)
    n = int(seconds * fs)
    emg = np.empty((n, 2))
    labels = []
    for i in range(n):
        b, t, lab = gen.sample()
        emg[i] = (b, t)
        labels.append(lab)
    return emg, labels


def run_decoder(emg, labels, decoder, fs=1000.0, control_hz=100.0, window_ms=150.0,
                rest=(0.02, 0.02), mvc=(0.5, 0.5)):
    """Returns dict with t, intended, mode, command_deg, activation (all at control_hz)."""
    window = max(3, int(round(fs * window_ms / 1000.0)))
    env = [RmsEnvelope(window), RmsEnvelope(window)]
    per_ctrl = int(round(fs / control_hz))
    dt = 1.0 / control_hz
    is_clf = isinstance(decoder, ClassifierElbowDecoder)
    t, intended, modes, cmd, act = [], [], [], [], []
    for k in range(len(emg)):
        env[0].add(emg[k, 0])
        env[1].add(emg[k, 1])
        if (k + 1) % per_ctrl or k + 1 < window:
            continue
        a_b = normalise(env[0].value, rest[0], mvc[0])
        a_t = normalise(env[1].value, rest[1], mvc[1])
        if is_clf:
            angle = decoder.step(emg[k + 1 - window:k + 1], a_b, a_t, dt)
        else:
            angle = decoder.step(a_b, a_t, dt)
        t.append((k + 1) / fs)
        intended.append(labels[k] if labels is not None else 'rest')
        modes.append(decoder.mode)
        cmd.append(np.degrees(angle))
        act.append((a_b, a_t))
    return {'t': np.asarray(t), 'intended': intended, 'mode': modes,
            'command_deg': np.asarray(cmd), 'activation': np.asarray(act)}


def make_decoder(kind, model_bundle=None, max_velocity=1.5, vote_length=5,
                 activation_threshold=0.10, cocontraction_threshold=0.25):
    if kind == 'proportional':
        return ProportionalElbowDecoder(activation_threshold=activation_threshold,
                                        cocontraction_threshold=cocontraction_threshold,
                                        max_velocity=max_velocity)
    return ClassifierElbowDecoder(model_bundle, max_velocity=max_velocity, vote_length=vote_length)
