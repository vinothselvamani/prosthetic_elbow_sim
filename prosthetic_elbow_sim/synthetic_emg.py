"""Synthetic two-channel surface-EMG generator (no ROS imports).

Model: amplitude-modulated zero-mean Gaussian noise, the standard simple sEMG model.
For activation a in [0, 1] the expected RMS is  rest + a * (mvc - rest),
so a decoder calibrated with the same rest/mvc values recovers a.
Includes first-order activation dynamics and biceps/triceps crosstalk.
"""

import math
import random

# (duration_s, biceps_activation, triceps_activation, label)
DEFAULT_SCENARIO = [
    (2.0, 0.0, 0.0, 'rest'),
    (2.5, 0.4, 0.0, 'flex_40'),
    (1.5, 0.0, 0.0, 'rest'),
    (2.5, 0.0, 0.4, 'extend_40'),
    (1.5, 0.0, 0.0, 'rest'),
    (1.5, 0.7, 0.0, 'flex_70'),
    (1.5, 0.5, 0.5, 'cocontraction'),
    (1.0, 0.0, 0.0, 'rest'),
    (2.0, 0.0, 0.7, 'extend_70'),
]


class SyntheticEmg:

    def __init__(self, fs=1000.0, rest=(0.02, 0.02), mvc=(0.5, 0.5),
                 crosstalk=0.08, tau_s=0.05, scenario=None, seed=None):
        self.fs = fs
        self.rest = rest
        self.mvc = mvc
        self.crosstalk = crosstalk
        self.alpha = 1.0 - math.exp(-1.0 / (fs * tau_s))  # activation low-pass
        self.scenario = scenario or DEFAULT_SCENARIO
        self.cycle_len = sum(seg[0] for seg in self.scenario)
        self.rng = random.Random(seed)
        self.t = 0.0
        self.act = [0.0, 0.0]

    def target(self, t):
        """Intended (b, t, label) at time t; the scenario loops forever."""
        tc = t % self.cycle_len
        for dur, a_b, a_t, label in self.scenario:
            if tc < dur:
                return a_b, a_t, label
            tc -= dur
        last = self.scenario[-1]
        return last[1], last[2], last[3]

    def sample(self):
        """Return (biceps_sample, triceps_sample, label) and advance one sample."""
        a_b, a_t, label = self.target(self.t)
        self.act[0] += self.alpha * (a_b - self.act[0])
        self.act[1] += self.alpha * (a_t - self.act[1])
        eff_b = min(1.0, self.act[0] + self.crosstalk * self.act[1])
        eff_t = min(1.0, self.act[1] + self.crosstalk * self.act[0])
        sd_b = self.rest[0] + eff_b * (self.mvc[0] - self.rest[0])
        sd_t = self.rest[1] + eff_t * (self.mvc[1] - self.rest[1])
        self.t += 1.0 / self.fs
        return self.rng.gauss(0.0, sd_b), self.rng.gauss(0.0, sd_t), label
