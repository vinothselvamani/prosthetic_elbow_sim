"""Record a labelled two-channel EMG session from an Arduino (or any serial device).

The subject follows on-screen cues; every sample is labelled with the cue that was shown,
and the session is saved in the format used by calibrate_emg / train_decoder / run_study.

Usage:
  ros2 run prosthetic_elbow_sim record_emg --port /dev/ttyACM0 --subject 1 --session 1
  ros2 run prosthetic_elbow_sim record_emg --dry-run            # rehearse the cues, no hardware

Device output: one line per sample "biceps,triceps" (ADC counts), e.g. from
arduino/emg_stream/emg_stream.ino at 1000 Hz, 230400 baud.

Protocol (about 2 min): rest 5 s | MAX biceps 3 s | rest 5 s | MAX triceps 3 s | rest 5 s |
then --trials cues in random order (FLEX / EXTEND / HOLD = co-contract, ~3 s) each followed by rest.

Saved: <out>/S<subject>_session<session>.csv   columns biceps, triceps (mV at the ADC,
DC removed, 20 Hz high-pass), label. Raw counts are kept in ..._raw.csv.
"""

import argparse
import os
import random
import sys
import threading
import time

import numpy as np

CUES = {
    'rest': ('REST', 'Relax your arm completely'),
    'flex_mvc': ('MAX FLEX', 'Bend the elbow / tense the BICEPS as hard as you can'),
    'extend_mvc': ('MAX EXTEND', 'Straighten / tense the TRICEPS as hard as you can'),
    'flex': ('FLEX', 'Bend the elbow (biceps), medium effort'),
    'extend': ('EXTEND', 'Straighten the elbow (triceps), medium effort'),
    'hold': ('HOLD', 'Tense BOTH biceps and triceps (stiff arm)'),
}


def build_protocol(trials, seed):
    r = random.Random(seed)
    seq = [('rest', 5.0), ('flex_mvc', 3.0), ('rest', 5.0), ('extend_mvc', 3.0), ('rest', 5.0)]
    kinds = (['flex', 'extend', 'hold'] * (trials // 3 + 1))[:trials]
    r.shuffle(kinds)
    for k in kinds:
        seq.append((k, round(r.uniform(2.5, 3.5), 1)))
        seq.append(('rest', round(r.uniform(2.5, 3.5), 1)))
    return seq


def label_for_csv(cue):
    return {'flex_mvc': 'flex', 'extend_mvc': 'extend'}.get(cue, cue)


class SerialReader(threading.Thread):
    """Reads 'b,t' lines; stores (host_time, b, t)."""

    def __init__(self, port, baud):
        super().__init__(daemon=True)
        import serial
        self.ser = serial.Serial(port, baud, timeout=1)
        self.samples = []
        self.bad = 0
        self.running = True

    def run(self):
        self.ser.reset_input_buffer()
        while self.running:
            line = self.ser.readline()
            if not line:
                continue
            try:
                b, t = line.decode('ascii', 'ignore').strip().split(',')[:2]
                self.samples.append((time.monotonic(), float(b), float(t)))
            except ValueError:
                self.bad += 1

    def stop(self):
        self.running = False
        self.join(timeout=2)
        self.ser.close()


def show_cue(cue, seconds_left, next_cue, total_left):
    word, hint = CUES[cue]
    nxt = CUES[next_cue][0] if next_cue else 'END'
    sys.stdout.write('\x1b[2J\x1b[H')  # clear screen
    sys.stdout.write('\n\n' + ' ' * 6 + f'\x1b[1;7m  {word:^14s}  \x1b[0m   {seconds_left:4.1f} s\n\n')
    sys.stdout.write(' ' * 6 + hint + '\n\n')
    sys.stdout.write(' ' * 6 + f'next: {nxt}      session time left: {total_left:5.0f} s\n')
    sys.stdout.flush()


def highpass(x, fs, fc=20.0):
    """First-order high-pass (removes DC offset and slow drift)."""
    a = 1.0 / (1.0 + 2 * np.pi * fc / fs)
    y = np.zeros_like(x)
    for i in range(1, len(x)):
        y[i] = a * (y[i - 1] + x[i] - x[i - 1])
    return y


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', default='/dev/ttyACM0', help='e.g. /dev/ttyACM0 or /dev/ttyUSB0')
    ap.add_argument('--baud', type=int, default=230400)
    ap.add_argument('--subject', type=int, default=1)
    ap.add_argument('--session', type=int, default=1)
    ap.add_argument('--trials', type=int, default=18)
    ap.add_argument('--vref', type=float, default=5.0, help='ADC reference voltage (3.3 for ESP32)')
    ap.add_argument('--adc-max', type=int, default=1023, help='1023 (Arduino) or 4095 (ESP32)')
    ap.add_argument('--out', default=os.path.expanduser('~/elbow_data'))
    ap.add_argument('--dry-run', action='store_true', help='show the cues only, no hardware')
    args = ap.parse_args()

    protocol = build_protocol(args.trials, seed=100 * args.subject + args.session)
    total = sum(d for _, d in protocol)
    name = f'S{args.subject:02d}_session{args.session}'
    print(f'{name}: {len(protocol)} cues, {total:.0f} s. Sit comfortably, arm relaxed by your side.')

    reader = None
    if not args.dry_run:
        try:
            reader = SerialReader(args.port, args.baud)
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f'Cannot open {args.port}: {e}\n'
                             'Check the cable, `ls /dev/ttyACM* /dev/ttyUSB*`, and that your user is in '
                             'the dialout group:  sudo usermod -aG dialout $USER  (then log out/in).')
        reader.start()
        time.sleep(2.0)  # Arduino resets when the port opens
        n0 = len(reader.samples)
        time.sleep(1.0)
        rate = len(reader.samples) - n0
        print(f'Receiving about {rate} samples/s on {args.port}')
        if rate < 500:
            reader.stop()
            raise SystemExit('Too few samples. Is emg_stream.ino uploaded and --baud matching (230400)?')
    input('Press Enter to start the session...')

    cue_times = []  # (start_time, cue)
    t_start = time.monotonic()
    try:
        for i, (cue, dur) in enumerate(protocol):
            t0 = time.monotonic()
            cue_times.append((t0, cue))
            nxt = protocol[i + 1][0] if i + 1 < len(protocol) else None
            while (left := dur - (time.monotonic() - t0)) > 0:
                show_cue(cue, left, nxt, total - (time.monotonic() - t_start))
                time.sleep(0.1)
    except KeyboardInterrupt:
        print('\nStopped early; saving what was recorded.')
    t_end = time.monotonic()
    sys.stdout.write('\x1b[2J\x1b[H')
    print('Session finished. Relax.')

    if reader is None:
        return
    reader.stop()
    data = [s for s in reader.samples if t_start <= s[0] <= t_end]
    if len(data) < 1000:
        raise SystemExit('Almost no data recorded.')
    times = np.array([s[0] for s in data])
    raw = np.array([[s[1], s[2]] for s in data])
    fs = len(data) / (times[-1] - times[0])
    starts = np.array([c[0] for c in cue_times])
    idx = np.clip(np.searchsorted(starts, times, side='right') - 1, 0, len(cue_times) - 1)
    labels = [label_for_csv(cue_times[i][1]) for i in idx]

    mv = raw * (args.vref / args.adc_max) * 1000.0
    emg = np.column_stack([highpass(mv[:, 0], fs), highpass(mv[:, 1], fs)])

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, name + '.csv')
    with open(path, 'w') as f:
        f.write('biceps,triceps,label\n')
        for (b, t), lab in zip(emg, labels):
            f.write(f'{b:.4f},{t:.4f},{lab}\n')
    with open(os.path.join(args.out, name + '_raw.csv'), 'w') as f:
        f.write('host_time,biceps_adc,triceps_adc,label\n')
        for (ht, b, t), lab in zip(data, labels):
            f.write(f'{ht - t_start:.4f},{b:.0f},{t:.0f},{lab}\n')

    print(f'Saved {path}: {len(emg)} samples, {len(emg) / fs:.0f} s, measured rate {fs:.0f} Hz'
          f'{"  (bad lines: %d)" % reader.bad if reader.bad else ""}')
    if abs(fs - 1000) > 50:
        print(f'NOTE: rate is {fs:.0f} Hz, so pass --fs {fs:.0f} to calibrate_emg / train_decoder / run_study.')
    for ch, nm in enumerate(('biceps', 'triceps')):
        rest_rms = np.sqrt(np.mean(emg[[l == 'rest' for l in labels], ch] ** 2))
        act_rms = np.sqrt(np.mean(emg[[l != 'rest' for l in labels], ch] ** 2))
        flag = '' if act_rms > 3 * rest_rms else '   <-- weak: check electrodes / sensor gain'
        print(f'  {nm:8s} rest RMS {rest_rms:.3f} mV   active RMS {act_rms:.3f} mV{flag}')
    print('Check it:  ros2 run prosthetic_elbow_sim calibrate_emg ' + path)


if __name__ == '__main__':
    main()
