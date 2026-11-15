"""Record one trial of the running simulation into a rosbag2 folder.

Usage (while sim.launch.py is running):
  ros2 run prosthetic_elbow_sim record_trial --name prop_v1.5 --duration 32
Creates ~/elbow_recordings/<name>_<YYYYmmdd_HHMMSS>/ and stops after --duration seconds
(default 32 s = two synthetic EMG cycles). Press Ctrl+C to stop early.
"""

import argparse
import datetime
import os
import signal
import subprocess
import time

from prosthetic_elbow_sim.bag_io import TRIAL_TOPICS


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--name', default='trial', help='short condition name, e.g. prop_v1.5 or lda')
    ap.add_argument('--duration', type=float, default=32.0, help='seconds to record')
    ap.add_argument('--folder', default=os.path.expanduser('~/elbow_recordings'))
    args = ap.parse_args()

    os.makedirs(args.folder, exist_ok=True)
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out = os.path.join(args.folder, f'{args.name}_{stamp}')
    cmd = ['ros2', 'bag', 'record', '-o', out, '--topics'] + TRIAL_TOPICS
    print(f'Recording {args.duration:.0f} s -> {out}')
    proc = subprocess.Popen(cmd)
    try:
        t_end = time.time() + args.duration
        while time.time() < t_end and proc.poll() is None:
            left = int(t_end - time.time())
            print(f'\r  {left:3d} s left ', end='', flush=True)
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=15)
    if not os.path.isdir(out):
        print(f'\nRecording FAILED - nothing saved (see the error above).')
        return
    print(f'\nSaved: {out}')
    print(f'Analyse with:  ros2 run prosthetic_elbow_sim analyze_bags {out}')


if __name__ == '__main__':
    main()
