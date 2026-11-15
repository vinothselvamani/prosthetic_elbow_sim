"""Realistic simulated EMG subjects for testing decoders without a sensor (no ROS imports).

Each subject has fixed physiology (signal gain, crosstalk, reaction time, tonic noise ...).
Each session follows the recording protocol in docs/WORKFLOW.md:

  rest 5 s | biceps MVC 3 s | rest 5 s | triceps MVC 3 s | rest 5 s |
  N task trials in random order: {flex, extend, hold} 3 s, each followed by 3 s rest

Effects modelled (what makes real EMG hard and decoders differ):
  * band-limited sEMG (20-450 Hz, peak ~80-120 Hz) whose amplitude follows activation
  * random effort per trial (25-80 % MVC), smooth ramp-up/down (muscle dynamics)
  * reaction time: the muscle starts 150-400 ms AFTER the cue (label) changes
  * crosstalk between biceps and triceps electrodes (subject-specific 8-30 %)
  * unbalanced co-contraction during "hold" (one muscle often dominates)
  * fatigue: amplitude rises and spectrum shifts to lower frequencies over the session
  * session 2 = electrodes re-applied: per-channel gain and crosstalk change
  * 50 Hz mains interference, baseline noise, occasional motion artefacts

Labels mark the CUE (what the subject was asked to do), as in a real experiment.
Units: millivolts.
"""

import numpy as np

FS = 1000.0


class SubjectModel:

    def __init__(self, subject_id: int, seed_base: int = 2026):
        r = np.random.default_rng(seed_base + 97 * subject_id)
        self.id = subject_id
        self.gain = r.uniform(0.25, 0.9, size=2)          # mV RMS at 100 % MVC (biceps, triceps)
        self.crosstalk = r.uniform(0.08, 0.30, size=2)     # leak into the other electrode
        self.reaction_s = r.uniform(0.15, 0.40)           # cue -> muscle onset
        self.rise_tau = r.uniform(0.06, 0.15)             # activation dynamics
        self.noise_mv = r.uniform(0.006, 0.02)            # amplifier/baseline noise
        self.mains_mv = r.uniform(0.0, 0.015)             # 50 Hz interference
        self.tonic = r.uniform(0.0, 0.05, size=2)         # resting muscle tone (fraction of MVC)
        self.peak_hz = r.uniform(80, 125, size=2)         # sEMG spectral peak
        self.fatigue = r.uniform(0.10, 0.35)              # amplitude rise / spectral drop by session end
        self.artefact_rate = r.uniform(0.02, 0.08)        # motion artefacts per second
        self.cocon_balance = r.uniform(0.55, 0.95)        # weaker muscle share during hold

    def session_offsets(self, session: int, r):
        if session == 1:
            return np.ones(2), np.zeros(2)
        # electrodes re-applied: gain changes +-30 %, crosstalk changes +-0.08
        return r.uniform(0.7, 1.3, size=2), r.uniform(-0.08, 0.08, size=2)


def _protocol(r, n_trials):
    segs = [('rest', 5.0, 0, 0), ('flex', 3.0, 1.0, 0), ('rest', 5.0, 0, 0),
            ('extend', 3.0, 0, 1.0), ('rest', 5.0, 0, 0)]
    kinds = ['flex', 'extend', 'hold'] * int(np.ceil(n_trials / 3))
    r.shuffle(kinds)
    for k in kinds[:n_trials]:
        effort = r.uniform(0.25, 0.80)
        dur = r.uniform(2.5, 3.5)
        if k == 'flex':
            segs.append(('flex', dur, effort, 0.0))
        elif k == 'extend':
            segs.append(('extend', dur, 0.0, effort))
        else:
            segs.append(('hold', dur, effort, effort))  # balance applied per subject later
        segs.append(('rest', r.uniform(2.5, 3.5), 0, 0))
    return segs


def _bandlimited_noise(n, fs, peak_hz, r):
    """Gaussian noise shaped like a sEMG spectrum (band-pass ~20-450 Hz), unit RMS."""
    white = r.standard_normal(n)
    f = np.fft.rfftfreq(n, 1 / fs)
    # smooth spectrum: rises from 20 Hz, peaks near peak_hz, decays towards 450 Hz
    shape = (f / peak_hz) ** 2 / (1 + (f / peak_hz) ** 4)
    shape[(f < 20) | (f > 450)] = 0
    x = np.fft.irfft(np.fft.rfft(white) * shape, n)
    return x / (np.std(x) + 1e-12)


def simulate_session(subject: SubjectModel, session: int, n_trials: int = 18, fs: float = FS):
    """Return (emg (n,2) mV, labels list) for one recording session."""
    r = np.random.default_rng(10_000 * subject.id + session)
    segs = _protocol(r, n_trials)
    gain_mult, ct_shift = subject.session_offsets(session, r)
    gain = subject.gain * gain_mult
    crosstalk = np.clip(subject.crosstalk + ct_shift, 0.02, 0.45)

    total = sum(s[1] for s in segs)
    n = int(total * fs)
    t = np.arange(n) / fs
    labels = np.empty(n, dtype=object)
    target = np.zeros((n, 2))
    i = 0
    for kind, dur, a_b, a_t in segs:
        k = int(dur * fs)
        labels[i:i + k] = kind
        if kind == 'hold':  # one muscle dominates, the other is weaker
            weaker = subject.cocon_balance
            if r.random() < 0.5:
                a_b, a_t = a_b, a_t * weaker
            else:
                a_b, a_t = a_b * weaker, a_t
        # muscle responds after the reaction time
        d = int(subject.reaction_s * fs + r.normal(0, 0.04) * fs)
        s0, s1 = min(n, max(0, i + d)), min(n, max(0, i + k + d))
        target[s0:s1] = (a_b, a_t)
        # within-trial effort fluctuation (humans don't hold perfectly steady)
        if s1 > s0 and (a_b or a_t):
            wobble = 1 + 0.12 * np.sin(2 * np.pi * r.uniform(0.3, 1.2) * t[s0:s1] + r.uniform(0, 6))
            target[s0:s1] *= wobble[:, None]
        i += k
    labels[i:] = 'rest'

    # first-order activation dynamics
    alpha = 1 - np.exp(-1 / (fs * subject.rise_tau))
    act = np.zeros_like(target)
    a = subject.tonic.copy()
    for j in range(n):
        a += alpha * (np.maximum(target[j], subject.tonic) - a)
        act[j] = a

    # fatigue: amplitude grows and spectrum shifts down over the session
    progress = t / t[-1]
    amp_fatigue = 1 + subject.fatigue * progress
    emg = np.zeros((n, 2))
    for ch in range(2):
        fresh = _bandlimited_noise(n, fs, subject.peak_hz[ch], r)
        tired = _bandlimited_noise(n, fs, subject.peak_hz[ch] * (1 - 0.3 * subject.fatigue), r)
        carrier = (1 - progress * subject.fatigue) * fresh + progress * subject.fatigue * tired
        carrier /= np.std(carrier)
        emg[:, ch] = gain[ch] * act[:, ch] * amp_fatigue * carrier

    # crosstalk: each electrode picks up part of the other muscle
    mixed = emg.copy()
    mixed[:, 0] += crosstalk[0] * emg[:, 1]
    mixed[:, 1] += crosstalk[1] * emg[:, 0]

    # mains, baseline noise, motion artefacts
    phase = r.uniform(0, 2 * np.pi, size=2)
    for ch in range(2):
        mixed[:, ch] += subject.mains_mv * np.sin(2 * np.pi * 50 * t + phase[ch])
        mixed[:, ch] += r.normal(0, subject.noise_mv, n)
    n_art = r.poisson(subject.artefact_rate * total)
    for _ in range(n_art):
        c = r.integers(0, n - 200)
        w = r.integers(40, 150)
        bump = r.uniform(0.1, 0.5) * np.hanning(w) * r.choice([-1, 1])
        mixed[c:c + w, r.integers(0, 2)] += bump
    return mixed, list(labels)


def write_csv(path, emg, labels):
    with open(path, 'w') as f:
        f.write('biceps,triceps,label\n')
        for (b, t), lab in zip(emg, labels):
            f.write(f'{b:.5f},{t:.5f},{lab}\n')


def main():
    import argparse
    import os
    ap = argparse.ArgumentParser(description='Generate simulated EMG subjects (2 sessions each).')
    ap.add_argument('--subjects', type=int, default=6)
    ap.add_argument('--trials', type=int, default=18, help='task trials per session')
    ap.add_argument('--out', default=os.path.expanduser('~/elbow_data'))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for s in range(1, args.subjects + 1):
        subj = SubjectModel(s)
        for sess in (1, 2):
            emg, labels = simulate_session(subj, sess, args.trials)
            path = os.path.join(args.out, f'S{s:02d}_session{sess}.csv')
            write_csv(path, emg, labels)
            print(f'{path}  ({len(emg) / FS:.0f} s)')
        print(f'   S{s:02d}: gain={np.round(subj.gain, 2)} mV, crosstalk={np.round(subj.crosstalk, 2)}, '
              f'reaction={subj.reaction_s * 1000:.0f} ms, fatigue={subj.fatigue:.2f}')


if __name__ == '__main__':
    main()
