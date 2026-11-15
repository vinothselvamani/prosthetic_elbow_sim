"""Train an LDA / ANN (MLP) / SVM intent classifier for the elbow decoder.

Usage:
  # quick start with synthetic EMG (120 s)
  ros2 run prosthetic_elbow_sim train_decoder --synthetic 120 --model lda --out ~/elbow_models/lda.joblib
  ros2 run prosthetic_elbow_sim train_decoder --synthetic 120 --model mlp --out ~/elbow_models/mlp.joblib

  # your own labelled recording(s): columns biceps, triceps, label
  ros2 run prosthetic_elbow_sim train_decoder subject01_train.csv --fs 1000 --model lda \
      --out ~/elbow_models/s01_lda.joblib

Features: Hudgins TD set (MAV, RMS, WL, ZC, SSC) per channel, 150 ms windows, 50 ms step.
Reports 5-fold cross-validated accuracy and a confusion matrix, then fits on all data.
Labels: rest / flex / extend / hold (flex_40, extend_70, cocontraction ... also accepted).
"""

import argparse
import os

import numpy as np

from prosthetic_elbow_sim.calibrate_emg import calibrate
from prosthetic_elbow_sim.emg_csv import load_emg_csv
from prosthetic_elbow_sim.features import CLASSES, sliding_windows
from prosthetic_elbow_sim.offline import synthetic_recording


def build_model(kind, seed=0):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    if kind == 'lda':
        clf = LinearDiscriminantAnalysis()
    elif kind == 'mlp':
        # early_stopping is off: some scikit-learn builds crash with text labels when it is on
        clf = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=600, alpha=1e-3,
                            early_stopping=False, random_state=seed)
    elif kind == 'svm':
        clf = SVC(kernel='rbf', C=10.0, gamma='scale')
    else:
        raise ValueError(f'unknown model type {kind}')
    return make_pipeline(StandardScaler(), clf)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('csv', nargs='*', help='labelled EMG CSV file(s)')
    ap.add_argument('--synthetic', type=float, default=0.0,
                    help='seconds of synthetic EMG to train on instead of CSV')
    ap.add_argument('--model', choices=['lda', 'mlp', 'svm'], default='lda')
    ap.add_argument('--out', required=True, help='output model file (.joblib)')
    ap.add_argument('--fs', type=float, default=1000.0)
    ap.add_argument('--window-ms', type=float, default=150.0)
    ap.add_argument('--step-ms', type=float, default=50.0)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    import joblib
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    if args.synthetic > 0:
        recordings = [synthetic_recording(args.synthetic, args.fs, seed=1000 + args.seed)]
        source = f'synthetic {args.synthetic:.0f} s'
    elif args.csv:
        recordings = [load_emg_csv(p) for p in args.csv]
        source = ', '.join(os.path.basename(p) for p in args.csv)
    else:
        ap.error('give CSV file(s) or --synthetic SECONDS')

    window = int(args.fs * args.window_ms / 1000)
    step = int(args.fs * args.step_ms / 1000)
    all_emg = np.vstack([r[0] for r in recordings])
    all_lab = sum((list(r[1]) for r in recordings if r[1] is not None), [])
    if any(r[1] is None for r in recordings):
        raise SystemExit('Every CSV needs a `label` column for training.')
    rest, mvc, thresholds = calibrate(all_emg, all_lab, args.fs, args.window_ms)

    X, y = [], []
    for emg, labels in recordings:
        Xi, yi = sliding_windows(emg, labels, window, step, thresholds)
        X.append(Xi)
        y.append(yi)
    X, y = np.vstack(X), np.concatenate(y)
    present = [c for c in CLASSES if c in set(y)]
    print(f'Data: {source} -> {len(X)} windows, classes: ' +
          ', '.join(f'{c}={int(np.sum(y == c))}' for c in present))

    model = build_model(args.model, args.seed)
    n_splits = min(5, min(int(np.sum(y == c)) for c in present))
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
        pred = cross_val_predict(model, X, y, cv=cv)
        acc = float(np.mean(pred == y))
        print(f'{n_splits}-fold cross-validated accuracy ({args.model.upper()}): {100 * acc:.1f} %')
        cm = confusion_matrix(y, pred, labels=present)
        print('confusion (rows = intended, cols = predicted):')
        print('          ' + ''.join(f'{c:>9s}' for c in present))
        for c, row in zip(present, cm):
            print(f'  {c:8s}' + ''.join(f'{v:9d}' for v in row))
    else:
        acc = float('nan')
        print('Not enough windows per class for cross-validation; fitting anyway.')

    model.fit(X, y)
    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    joblib.dump({'model': model, 'kind': args.model, 'thresholds': thresholds,
                 'window_ms': args.window_ms, 'fs': args.fs, 'cv_accuracy': acc,
                 'rest_level': rest, 'mvc': mvc, 'source': source}, out)
    print(f'\nSaved {out}')
    print('Run it in simulation:\n  ros2 launch prosthetic_elbow_sim sim.launch.py '
          f'decoder_type:=classifier model_file:={out}')


if __name__ == '__main__':
    main()
