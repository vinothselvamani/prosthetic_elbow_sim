"""Multi-subject decoder comparison: calibrate + train on session 1, test on session 2.

Usage:
  ros2 run prosthetic_elbow_sim simulate_subjects --subjects 6          # or record real ones
  ros2 run prosthetic_elbow_sim run_study --data ~/elbow_data --out ~/elbow_results/study

Data folder: S01_session1.csv, S01_session2.csv, S02_session1.csv ... (biceps, triceps, label)

For each subject
  1. calibrate rest/MVC and ZC/SSC thresholds on session 1
  2. train LDA and MLP (ANN) intent classifiers on session 1
  3. run proportional, LDA and MLP decoders on session 2 (unseen data, re-applied electrodes)
  4. compute intent accuracy, onset delay, unwanted motion, movement per phase

Outputs: per_subject.csv, summary.csv (mean ± SD, paired Wilcoxon vs proportional),
study_summary.png/.pdf, confusion_all.png, S01_<decoder>_figure.png
"""

import argparse
import csv
import glob
import os
import re

import numpy as np

from prosthetic_elbow_sim.calibrate_emg import calibrate
from prosthetic_elbow_sim.emg_csv import load_emg_csv
from prosthetic_elbow_sim.features import sliding_windows
from prosthetic_elbow_sim.metrics import compute_metrics
from prosthetic_elbow_sim.offline import make_decoder, run_decoder
from prosthetic_elbow_sim.plotting import confusion_figure, trial_figure

METRICS = [('intent_accuracy', 'Intent accuracy (%)', 100.0, True),
           ('onset_delay_s', 'Onset delay from cue (ms)', 1000.0, False),
           ('false_motion_deg', 'Unwanted motion (deg)', 1.0, False),
           ('movement_deg', 'Movement per phase (deg)', 1.0, None)]


def find_subjects(folder):
    out = {}
    for p in glob.glob(os.path.join(folder, 'S*_session1.csv')):
        sid = re.match(r'(S\d+)_session1\.csv', os.path.basename(p)).group(1)
        p2 = os.path.join(folder, f'{sid}_session2.csv')
        if os.path.exists(p2):
            out[sid] = (p, p2)
    return dict(sorted(out.items()))


def wilcoxon_p(a, b):
    try:
        from scipy.stats import wilcoxon
        a, b = np.asarray(a), np.asarray(b)
        if np.allclose(a, b):
            return 1.0
        return float(wilcoxon(a, b).pvalue)
    except Exception:
        return float('nan')


def summary_figure(path, decoders, per_subject):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.dpi': 300})
    colours = {'proportional': '#34495e', 'lda': '#c0392b', 'mlp': '#2471a3', 'svm': '#27ae60'}
    fig, axes = plt.subplots(1, len(METRICS), figsize=(2.3 * len(METRICS) + 0.6, 3.0))
    subjects = sorted({r['subject'] for r in per_subject})
    for ax, (key, label, scale, higher) in zip(axes, METRICS):
        for i, d in enumerate(decoders):
            vals = np.array([r[key] * scale for r in per_subject if r['decoder'] == d])
            ax.bar(i, np.nanmean(vals), yerr=np.nanstd(vals), color=colours.get(d, '#888'),
                   alpha=0.85, capsize=3, width=0.65)
            ax.scatter(np.full(len(vals), i) + np.linspace(-0.15, 0.15, len(vals)), vals,
                       s=9, color='black', zorder=3)
        # connect each subject across decoders
        for s in subjects:
            ys = [next(r[key] * scale for r in per_subject if r['decoder'] == d and r['subject'] == s)
                  for d in decoders]
            ax.plot(range(len(decoders)), ys, color='grey', lw=0.5, alpha=0.5, zorder=2)
        arrow = '' if higher is None else (' ↑' if higher else ' ↓')
        ax.set_title(label + arrow, fontsize=8)
        ax.set_xticks(range(len(decoders)))
        ax.set_xticklabels([d.upper() if d != 'proportional' else 'Prop.' for d in decoders])
        ax.grid(axis='y', alpha=0.25)
    fig.suptitle(f'Decoder comparison, train session 1 → test session 2 (n = {len(subjects)} subjects, mean ± SD)',
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default=os.path.expanduser('~/elbow_data'))
    ap.add_argument('--out', default=os.path.expanduser('~/elbow_results/study'))
    ap.add_argument('--models', nargs='+', default=['lda', 'mlp'], choices=['lda', 'mlp', 'svm'])
    ap.add_argument('--fs', type=float, default=1000.0)
    ap.add_argument('--settle', type=float, default=0.5,
                    help='s after each cue excluded from accuracy (covers reaction time)')
    ap.add_argument('--max-velocity', type=float, default=1.5)
    ap.add_argument('--vote', type=int, default=5, help='classifier majority-vote length')
    args = ap.parse_args()

    from prosthetic_elbow_sim.train_decoder import build_model
    subjects = find_subjects(args.data)
    if not subjects:
        raise SystemExit(f'No S??_session1.csv / S??_session2.csv pairs in {args.data}')
    os.makedirs(args.out, exist_ok=True)
    window = int(args.fs * 0.150)
    step = int(args.fs * 0.050)
    decoders = ['proportional'] + args.models
    per_subject, confs = [], {d: 0 for d in decoders}

    for sid, (p1, p2) in subjects.items():
        emg1, lab1 = load_emg_csv(p1)
        emg2, lab2 = load_emg_csv(p2)
        rest, mvc, thr = calibrate(emg1, lab1, args.fs)
        X, y = sliding_windows(emg1, lab1, window, step, thr)
        print(f'{sid}: rest={np.round(rest, 4)} mvc={np.round(mvc, 3)}  train windows={len(X)}')
        for d in decoders:
            if d == 'proportional':
                dec = make_decoder('proportional', max_velocity=args.max_velocity)
            else:
                model = build_model(d).fit(X, y)
                dec = make_decoder('classifier', {'model': model, 'thresholds': thr},
                                   max_velocity=args.max_velocity, vote_length=args.vote)
            r = run_decoder(emg2, lab2, dec, fs=args.fs, rest=rest, mvc=mvc)
            m = compute_metrics(r['t'], r['intended'], r['mode'], r['command_deg'], settle_s=args.settle)
            per_subject.append({'subject': sid, 'decoder': d,
                                'intent_accuracy': m['intent_accuracy'],
                                'onset_delay_s': m['onset_delay_s'],
                                'false_motion_deg': m['false_motion_deg'],
                                'movement_deg': m['movement_deg']})
            confs[d] = confs[d] + m['confusion']
            print(f'   {d:12s} acc={100 * m["intent_accuracy"]:5.1f}%  onset={1000 * m["onset_delay_s"]:5.0f} ms  '
                  f'unwanted={m["false_motion_deg"]:6.1f} deg')
            if sid == next(iter(subjects)):
                k0 = int(min(len(emg2), 60 * args.fs))
                sel = r['t'] <= 60
                trial_figure(os.path.join(args.out, f'{sid}_{d}_figure.png'),
                             f'{sid} session 2: {d} decoder (first 60 s)', r['t'][sel],
                             [x for x, s in zip(r['intended'], sel) if s],
                             [x for x, s in zip(r['mode'], sel) if s],
                             r['activation'][sel], r['command_deg'][sel], None,
                             (1 + np.arange(k0)) / args.fs, emg2[:k0])

    # per-subject table
    with open(os.path.join(args.out, 'per_subject.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(per_subject[0].keys()))
        w.writeheader()
        w.writerows(per_subject)

    # summary mean ± SD + Wilcoxon vs proportional
    rows = []
    for d in decoders:
        row = {'decoder': d}
        for key, label, scale, _ in METRICS:
            vals = np.array([r[key] * scale for r in per_subject if r['decoder'] == d])
            row[label] = f'{np.nanmean(vals):.1f} ± {np.nanstd(vals):.1f}'
            if d != 'proportional':
                base = np.array([r[key] * scale for r in per_subject if r['decoder'] == 'proportional'])
                row[label + ' p'] = f'{wilcoxon_p(vals, base):.3f}'
        rows.append(row)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(args.out, 'summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    for ext in ('png', 'pdf'):
        summary_figure(os.path.join(args.out, f'study_summary.{ext}'), decoders, per_subject)
    confusion_figure(os.path.join(args.out, 'confusion_all.png'), [(d, confs[d]) for d in decoders])

    print(f'\nSummary (n = {len(subjects)} subjects, mean ± SD; p = paired Wilcoxon vs proportional)')
    for r in rows:
        print('  ' + r['decoder'].ljust(13) + '  '.join(f'{k}: {v}' for k, v in r.items() if k != 'decoder'))
    print(f'\nSaved in {os.path.abspath(args.out)}')


if __name__ == '__main__':
    main()
