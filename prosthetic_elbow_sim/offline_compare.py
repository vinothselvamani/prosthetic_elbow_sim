"""Compare decoders offline on the same EMG (no Gazebo, no ROS runtime needed).

Usage:
  # synthetic test data (different random seed from the training data)
  ros2 run prosthetic_elbow_sim offline_compare --synthetic 64 \
      --models ~/elbow_models/lda.joblib ~/elbow_models/mlp.joblib --out ~/elbow_results/offline

  # your own labelled test recording (use a different session from training!)
  ros2 run prosthetic_elbow_sim offline_compare subject01_test.csv --fs 1000 \
      --rest 0.012 0.010 --mvc 0.41 0.37 --models ~/elbow_models/s01_lda.joblib

Always includes the conventional proportional decoder as the baseline.
Saves per-decoder figures, comparison.png, confusion.png and metrics.csv.
"""

import argparse
import os

import numpy as np

from prosthetic_elbow_sim.analyze_bags import print_table, write_table
from prosthetic_elbow_sim.classifier_decoder import load_model
from prosthetic_elbow_sim.emg_csv import load_emg_csv
from prosthetic_elbow_sim.metrics import compute_metrics, summary_row
from prosthetic_elbow_sim.offline import make_decoder, run_decoder, synthetic_recording
from prosthetic_elbow_sim.plotting import comparison_figure, confusion_figure, trial_figure


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('csv', nargs='?', help='labelled test CSV (biceps, triceps, label)')
    ap.add_argument('--synthetic', type=float, default=0.0, help='seconds of synthetic test EMG')
    ap.add_argument('--seed', type=int, default=7, help='seed for synthetic test data')
    ap.add_argument('--models', nargs='*', default=[], help='.joblib models from train_decoder')
    ap.add_argument('--fs', type=float, default=1000.0)
    ap.add_argument('--rest', type=float, nargs=2, default=None, help='rest RMS biceps triceps')
    ap.add_argument('--mvc', type=float, nargs=2, default=None, help='MVC RMS biceps triceps')
    ap.add_argument('--max-velocity', type=float, default=1.5)
    ap.add_argument('--out', default='elbow_results/offline')
    args = ap.parse_args()

    if args.synthetic > 0:
        emg, labels = synthetic_recording(args.synthetic, args.fs, seed=args.seed)
        rest, mvc = (0.02, 0.02), (0.5, 0.5)
        src = f'synthetic {args.synthetic:.0f} s (seed {args.seed})'
    elif args.csv:
        emg, labels = load_emg_csv(args.csv)
        if labels is None:
            raise SystemExit('The test CSV needs a `label` column to score decoders.')
        if args.rest is None or args.mvc is None:
            from prosthetic_elbow_sim.calibrate_emg import calibrate
            rest, mvc, _ = calibrate(emg, labels, args.fs)
            print('Calibration taken from the test file itself; pass --rest/--mvc from your '
                  'training session for a stricter evaluation.')
        else:
            rest, mvc = tuple(args.rest), tuple(args.mvc)
        src = os.path.basename(args.csv)
    else:
        ap.error('give a test CSV or --synthetic SECONDS')

    os.makedirs(args.out, exist_ok=True)
    decoders = [('proportional', make_decoder('proportional', max_velocity=args.max_velocity))]
    for path in args.models:
        bundle = load_model(os.path.expanduser(path))
        name = os.path.splitext(os.path.basename(path))[0]
        decoders.append((name, make_decoder('classifier', bundle, max_velocity=args.max_velocity)))

    print(f'Test data: {src}')
    rows, confs = [], []
    for name, dec in decoders:
        r = run_decoder(emg, labels, dec, fs=args.fs, rest=rest, mvc=mvc)
        m = compute_metrics(r['t'], r['intended'], r['mode'], r['command_deg'])
        rows.append(summary_row(name, m))
        confs.append((name, m['confusion']))
        k0 = int(min(len(emg), 16 * args.fs))  # plot first 16 s of raw EMG for readability
        trial_figure(os.path.join(args.out, f'{name}_figure.png'), f'{name} decoder ({src})',
                     r['t'], r['intended'], r['mode'], r['activation'], r['command_deg'],
                     None, (1 + np.arange(k0)) / args.fs, emg[:k0])
    for row in rows:  # no simulated joint offline -> drop tracking columns
        row.pop('tracking_rmse_deg', None)
        row.pop('tracking_lag_ms', None)
    write_table(os.path.join(args.out, 'metrics.csv'), rows)
    confusion_figure(os.path.join(args.out, 'confusion.png'), confs)
    comparison_figure(os.path.join(args.out, 'comparison.png'), rows)
    print()
    print_table(rows)
    print(f'\nSaved figures and metrics.csv in {os.path.abspath(args.out)}')


if __name__ == '__main__':
    main()
