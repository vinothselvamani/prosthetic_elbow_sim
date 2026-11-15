"""Pure-Python EMG -> elbow decoding logic (no ROS imports, so it can be unit-tested
and reused offline, e.g. in a Jupyter notebook on recorded data).

Conventional two-site proportional myoelectric control:
    biceps activation  -> elbow flexion velocity
    triceps activation -> elbow extension velocity
    both strongly active (co-contraction) -> hold position
"""

from collections import deque
import math


class RmsEnvelope:
    """Moving-window RMS envelope of one EMG channel."""

    def __init__(self, window_samples: int):
        if window_samples < 1:
            raise ValueError('window_samples must be >= 1')
        self._buf = deque(maxlen=window_samples)
        self._sum_sq = 0.0

    def add(self, sample: float) -> float:
        if len(self._buf) == self._buf.maxlen:
            old = self._buf[0]
            self._sum_sq -= old * old
        self._buf.append(sample)
        self._sum_sq += sample * sample
        # guard against tiny negative values from floating-point drift
        return math.sqrt(max(self._sum_sq, 0.0) / len(self._buf))

    @property
    def value(self) -> float:
        if not self._buf:
            return 0.0
        return math.sqrt(max(self._sum_sq, 0.0) / len(self._buf))


def normalise(rms: float, rest: float, mvc: float) -> float:
    """Map RMS to 0..1 activation using resting level and MVC calibration."""
    span = mvc - rest
    if span <= 0:
        raise ValueError('mvc must be greater than rest level')
    return min(max((rms - rest) / span, 0.0), 1.0)


class ProportionalElbowDecoder:
    """Turns two normalised activations into an elbow angle trajectory."""

    def __init__(self, activation_threshold=0.10, cocontraction_threshold=0.25,
                 max_velocity=1.5, elbow_min=0.0, elbow_max=2.53, initial_angle=0.0):
        self.activation_threshold = activation_threshold
        self.cocontraction_threshold = cocontraction_threshold
        self.max_velocity = max_velocity
        self.elbow_min = elbow_min
        self.elbow_max = elbow_max
        self.angle = min(max(initial_angle, elbow_min), elbow_max)
        self.velocity = 0.0
        self.mode = 'rest'

    def velocity_command(self, a_biceps: float, a_triceps: float) -> float:
        """Velocity (rad/s) from normalised activations; also sets self.mode."""
        if (a_biceps >= self.cocontraction_threshold
                and a_triceps >= self.cocontraction_threshold):
            self.mode = 'cocontraction_hold'
            return 0.0
        diff = a_biceps - a_triceps
        if abs(diff) < self.activation_threshold:
            self.mode = 'rest'
            return 0.0
        # remove the deadband so velocity starts from zero at the threshold
        sign = 1.0 if diff > 0 else -1.0
        scaled = (abs(diff) - self.activation_threshold) / (1.0 - self.activation_threshold)
        self.mode = 'flex' if sign > 0 else 'extend'
        return sign * scaled * self.max_velocity

    def step(self, a_biceps: float, a_triceps: float, dt: float) -> float:
        """Integrate one control step; returns the new commanded angle (rad)."""
        self.velocity = self.velocity_command(a_biceps, a_triceps)
        new_angle = self.angle + self.velocity * dt
        clamped = min(max(new_angle, self.elbow_min), self.elbow_max)
        if clamped != new_angle:
            self.velocity = 0.0
        self.angle = clamped
        return self.angle
