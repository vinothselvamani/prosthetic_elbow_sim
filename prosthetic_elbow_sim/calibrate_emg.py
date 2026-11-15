"""Calibrate the decoder (rest level, MVC, ZC/SSC thresholds) from a recording.

Usage:
  ros2 run prosthetic_elbow_sim calibrate_emg my_session.csv --fs 1000 \
      --out ~/ros2_ws/src/prosthetic_elbow_sim/config/decoder_subject01.yaml

Recording protocol (one CSV, ~60 s):
  rest 5 s -> max biceps (elbow flexion) 3 s -> rest 5 s -> max triceps (extension) 3 s
  -> rest 5 s -> then the normal task movements you want to decode.
If the CSV has a `label` column, rest is taken from rest-labelled samples;
otherwise from the quietest 10 % of the recording. MVC = 99th percentile of the RMS envelope.
"""

import argparse
import os

import numpy as np

from prosthetic_elbow_sim.emg_csv import load_emg_csv, moving_rms
from prosthetic_elbow_sim.features import label_to_class

TEMPLATE = """# Decoder parameters calibrated from {src}
emg_decoder:
  ros__parameters:
    use_sim_time: true
    # decoder_type / model_file are chosen on the launch command line
    emg_rate_hz: {fs}
    window_ms: {window_ms}
    control_rate_hz: 100.0
    rest_level: [{rest_b:.6g}, {rest_t:.6g}]
    mvc: [{mvc_b:.6g}, {mvc_t:.6g}]
    activation_threshold: 0.10
    cocontraction_threshold: 0.25
    max_velocity: 1.5
    vote_length: 5
    elbow_min: 0.0
    elbow_max: 2.53
    command_topic: /elbow_position_controller/commands
"""


def calibrate(emg, labels, fs, window_ms=150.0):
    window = max(1, int(fs * window_ms / 1000))
    env = np.column_stack([moving_rms(emg[:, 0], window), moving_rms(emg[:, 1], window)])
    env = env[window:]  # drop warm-up
    if labels is not None:
        cls = np.array([label_to_class(x) for x in labels])[window:]
        rest_mask = cls == 'rest'
    else:
        rest_mask = None
    rest, mvc = [], []
    for ch in range(2):
        e = env[:, ch]
        if rest_mask is not None and rest_mask.sum() > fs:
            rest.append(float(np.median(e[rest_mask])))
        else:
            rest.append(float(np.percentile(e, 10)))
        mvc.append(float(np.percentile(e, 99)))
    # ZC/SSC threshold: a bit above resting noise amplitude
    thresholds = [0.5 * r for r in rest]
    return rest, mvc, thresholds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('csv')
    ap.add_argument('--fs', type=float, default=1000.0, help='sampling rate (Hz)')
    ap.add_argument('--window-ms', type=float, default=150.0)
    ap.add_argument('--biceps', default='biceps')
    ap.add_argument('--triceps', default='triceps')
    ap.add_argument('--out', default='', help='write a decoder YAML here')
    args = ap.parse_args()

    emg, labels = load_emg_csv(args.csv, args.biceps, args.triceps)
    rest, mvc, thr = calibrate(emg, labels, args.fs, args.window_ms)
    print(f'{len(emg)} samples ({len(emg) / args.fs:.1f} s), labels: {"yes" if labels else "no"}')
    for i, ch in enumerate(('biceps', 'triceps')):
        ratio = mvc[i] / rest[i] if rest[i] > 0 else float('inf')
        flag = '' if ratio > 4 else '   <-- weak signal: check electrode placement / gain'
        print(f'  {ch:8s} rest RMS = {rest[i]:.5g}   MVC RMS = {mvc[i]:.5g}   MVC/rest = {ratio:.1f}{flag}')
    if args.out:
        txt = TEMPLATE.format(src=os.path.basename(args.csv), fs=args.fs, window_ms=args.window_ms,
                              rest_b=rest[0], rest_t=rest[1], mvc_b=mvc[0], mvc_t=mvc[1])
        with open(os.path.expanduser(args.out), 'w') as f:
            f.write(txt)
        print(f'\nWrote {args.out}\nUse it with:  ros2 launch prosthetic_elbow_sim sim.launch.py '
              f'emg_source:=csv csv_file:={os.path.abspath(args.csv)} decoder_config:={args.out}')
    else:
        print('\nPut these in config/decoder.yaml:')
        print(f'    rest_level: [{rest[0]:.6g}, {rest[1]:.6g}]')
        print(f'    mvc: [{mvc[0]:.6g}, {mvc[1]:.6g}]')


if __name__ == '__main__':
    main()
