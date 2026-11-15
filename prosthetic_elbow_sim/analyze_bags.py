"""Analyse one or more trial recordings (rosbag2) -> figures + metrics table.

Usage:
  ros2 run prosthetic_elbow_sim analyze_bags ~/elbow_recordings/trial_01
  ros2 run prosthetic_elbow_sim analyze_bags ~/elbow_recordings/prop_* ~/elbow_recordings/lda_* \
      --out ~/elbow_results

Outputs (in --out, default ./elbow_results):
  <trial>_figure.png / .pdf   EMG, activation, decoded intent, commanded vs simulated angle
  metrics.csv                 one row per trial
  comparison.png              bar charts across trials (when more than one trial)
  confusion.png               intended-vs-decoded confusion matrices
"""

import argparse
import csv
import os

import numpy as np

from prosthetic_elbow_sim.bag_io import read_bag, trial_arrays
from prosthetic_elbow_sim.metrics import compute_metrics, summary_row
from prosthetic_elbow_sim.plotting import comparison_figure, confusion_figure, trial_figure


def analyse_trial(path, out_dir, settle_s=0.3):
    name = os.path.basename(os.path.normpath(path))
    data = trial_arrays(read_bag(path))
    if data['mode'] is None or data['command_deg'] is None:
        raise RuntimeError(f'{name}: /elbow/mode or /elbow/command_deg missing from the bag')
    intended = data['intended']
    if intended is None:
        print(f'  [{name}] no /emg/intended_label in bag -> intent metrics are not meaningful')
        intended = np.array(['rest'] * len(data['t']), dtype=object)
    m = compute_metrics(data['t'], intended, data['mode'], data['command_deg'],
                        data['actual_deg'], settle_s=settle_s)
    for ext in ('png', 'pdf'):
        trial_figure(os.path.join(out_dir, f'{name}_figure.{ext}'), name, data['t'], intended,
                     data['mode'], data['activation'], data['command_deg'], data['actual_deg'],
                     data['emg_t'], data['emg'])
    return name, m


def write_table(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def print_table(rows):
    keys = list(rows[0].keys())
    widths = [max(len(k), *(len(str(r[k])) for r in rows)) for k in keys]
    print('  '.join(k.ljust(w) for k, w in zip(keys, widths)))
    for r in rows:
        print('  '.join(str(r[k]).ljust(w) for k, w in zip(keys, widths)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bags', nargs='+', help='rosbag2 folders')
    ap.add_argument('--out', default='elbow_results', help='output folder')
    ap.add_argument('--settle', type=float, default=0.3,
                    help='seconds after each intended change excluded from accuracy')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows, confs = [], []
    for bag in args.bags:
        print(f'Analysing {bag} ...')
        name, m = analyse_trial(bag, args.out, args.settle)
        rows.append(summary_row(name, m))
        confs.append((name, m['confusion']))

    write_table(os.path.join(args.out, 'metrics.csv'), rows)
    confusion_figure(os.path.join(args.out, 'confusion.png'), confs)
    if len(rows) > 1:
        comparison_figure(os.path.join(args.out, 'comparison.png'), rows)
    print()
    print_table(rows)
    print(f'\nFigures and metrics.csv saved in {os.path.abspath(args.out)}')


if __name__ == '__main__':
    main()
