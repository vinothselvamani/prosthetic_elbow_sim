"""Pattern-recognition elbow decoder (no ROS imports).

A trained classifier (LDA, ANN/MLP, SVM ... any scikit-learn pipeline saved with joblib)
predicts the intent class from TD features of the latest EMG window:
    rest -> 0,  flex -> +v,  extend -> -v,  hold -> 0
Speed is proportional to the activation of the muscle that drives the movement
("pattern recognition + proportional speed"), so stronger contractions move faster.
A majority vote over the last few decisions removes single-window misclassifications.
"""

from collections import Counter, deque

import numpy as np

from prosthetic_elbow_sim.features import td_features


def load_model(path):
    import joblib
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or 'model' not in bundle:
        raise ValueError(f'{path} is not a model file created by train_decoder')
    return bundle


class ClassifierElbowDecoder:

    def __init__(self, model_bundle, max_velocity=1.5, elbow_min=0.0, elbow_max=2.53,
                 initial_angle=0.0, vote_length=5, min_activation=0.05):
        self.model = model_bundle['model']
        self.thresholds = model_bundle.get('thresholds', (0.0, 0.0))
        self.kind = model_bundle.get('kind', 'classifier')
        self.max_velocity = max_velocity
        self.elbow_min = elbow_min
        self.elbow_max = elbow_max
        self.min_activation = min_activation
        self.angle = min(max(initial_angle, elbow_min), elbow_max)
        self.velocity = 0.0
        self.mode = 'rest'
        self.raw_class = 'rest'
        self._votes = deque(maxlen=max(1, vote_length))

    def classify(self, window: np.ndarray) -> str:
        feats = td_features(window, self.thresholds).reshape(1, -1)
        self.raw_class = str(self.model.predict(feats)[0])
        self._votes.append(self.raw_class)
        return Counter(self._votes).most_common(1)[0][0]

    def step(self, window: np.ndarray, a_biceps: float, a_triceps: float, dt: float) -> float:
        cls = self.classify(window)
        if cls == 'flex':
            speed = max(a_biceps, self.min_activation)
            self.velocity = speed * self.max_velocity
        elif cls == 'extend':
            speed = max(a_triceps, self.min_activation)
            self.velocity = -speed * self.max_velocity
        else:
            self.velocity = 0.0
        self.mode = 'cocontraction_hold' if cls == 'hold' else cls
        new_angle = self.angle + self.velocity * dt
        clamped = min(max(new_angle, self.elbow_min), self.elbow_max)
        if clamped != new_angle:
            self.velocity = 0.0
        self.angle = clamped
        return self.angle
