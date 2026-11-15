# Research workflow: recording, analysis, real EMG, decoder comparison

Install the Python tools once:
```bash
sudo apt install python3-sklearn python3-joblib python3-matplotlib python3-numpy
cd ~/ros2_ws && colcon build --symlink-install --packages-select prosthetic_elbow_sim
source install/setup.bash
```

---

## 1. Record a trial

Terminal 1, start the simulation:
```bash
ros2 launch prosthetic_elbow_sim sim.launch.py
```
Terminal 2, record (stops by itself after 32 s, which is two synthetic EMG cycles):
```bash
ros2 run prosthetic_elbow_sim record_trial --name prop_v1.5 --duration 32
```
This saves `~/elbow_recordings/prop_v1.5_<date_time>/`, a rosbag with raw EMG, activation,
decoder mode, intended task phase, commanded angle and the simulated joint angle.

## 2. Analyse it (figures + metrics)

```bash
ros2 run prosthetic_elbow_sim analyze_bags ~/elbow_recordings/prop_v1.5_* --out ~/elbow_results
```
Outputs in `~/elbow_results/`:

| File | Content |
|---|---|
| `<trial>_figure.png/.pdf` | raw sEMG, activation, decoded intent, commanded vs simulated elbow angle, with the intended task phase shaded in the background |
| `metrics.csv` | one row per trial |
| `confusion.png` | intended vs decoded class (%) |
| `comparison.png` | bar charts across trials (when you pass more than one bag) |

| Metric | Meaning |
|---|---|
| `intent_accuracy_%` | % of time the decoder mode matches the intended phase (the first 0.3 s after each change is excluded) |
| `onset_delay_ms` | median time from intended flex/extend onset to the decoder reacting |
| `false_motion_deg` | elbow travel during intended rest/hold, i.e. unwanted movement (lower is better) |
| `movement_deg` | mean elbow excursion per movement phase |
| `tracking_rmse_deg` | commanded vs simulated joint angle (controller + dynamics fidelity) |
| `tracking_lag_ms` | delay of the simulated joint behind the command |

## 3. Use real EMG

**Recording protocol** (about 60 s per file, sampled uniformly, e.g. 1000 Hz):
rest 5 s → max biceps (elbow flexion) 3 s → rest 5 s → max triceps (extension) 3 s → rest 5 s →
then repeat the task: rest / flex / extend / co-contract (hold), about 3 s each, several times.

**CSV columns:** `biceps, triceps, label`. Label each sample `rest`, `flex`, `extend` or `hold`.
A label column is optional for driving the simulation, but you need it to train classifiers and score accuracy.

**Calibrate** (rest level, MVC):
```bash
ros2 run prosthetic_elbow_sim calibrate_emg subject01_s1.csv --fs 1000 \
    --out ~/ros2_ws/src/prosthetic_elbow_sim/config/decoder_s01.yaml
```
**Drive the simulated elbow with it:**
```bash
ros2 launch prosthetic_elbow_sim sim.launch.py emg_source:=csv csv_file:=$PWD/subject01_s1.csv \
    decoder_config:=$HOME/ros2_ws/src/prosthetic_elbow_sim/config/decoder_s01.yaml
```
Sensor note: if your sensor outputs an **envelope** (for example MyoWare "ENV" output) rather than raw EMG,
the proportional decoder still works after calibration. The classifier features (ZC, SSC, WL) need **raw** EMG.

## 4. Compare decoders (proportional vs LDA vs ANN)

**Train** on one session:
```bash
ros2 run prosthetic_elbow_sim train_decoder subject01_s1.csv --fs 1000 --model lda --out ~/elbow_models/s01_lda.joblib
ros2 run prosthetic_elbow_sim train_decoder subject01_s1.csv --fs 1000 --model mlp --out ~/elbow_models/s01_mlp.joblib
```
(To try it without real data, use `--synthetic 120` instead of the CSV.)

**A. Offline comparison** (fast; same EMG for every decoder, no Gazebo). Test on a **different session**:
```bash
ros2 run prosthetic_elbow_sim offline_compare subject01_s2.csv --fs 1000 \
    --rest <rest_b> <rest_t> --mvc <mvc_b> <mvc_t> \
    --models ~/elbow_models/s01_lda.joblib ~/elbow_models/s01_mlp.joblib --out ~/elbow_results/offline_s01
```
**B. In-simulation comparison** (includes controller and joint dynamics). Run each decoder and record:
```bash
ros2 launch prosthetic_elbow_sim sim.launch.py emg_source:=csv csv_file:=... decoder_config:=...
ros2 run prosthetic_elbow_sim record_trial --name s01_prop

ros2 launch prosthetic_elbow_sim sim.launch.py emg_source:=csv csv_file:=... decoder_config:=... \
    decoder_type:=classifier model_file:=$HOME/elbow_models/s01_lda.joblib
ros2 run prosthetic_elbow_sim record_trial --name s01_lda

ros2 run prosthetic_elbow_sim analyze_bags ~/elbow_recordings/s01_* --out ~/elbow_results/sim_s01
```

### Good practice for the paper
- Train and test on **different sessions** (or at least different trials). The cross-validation accuracy printed by
  `train_decoder` is optimistic, so report the held-out session results.
- Repeat for several subjects and report the mean ± SD of each metric. Use a paired test (e.g. Wilcoxon) between decoders.
- The synthetic EMG is idealised: every decoder scores close to 100 % on it. It shows that the pipeline works,
  **not** which decoder is better. Draw conclusions only from real recordings.

---

## 5. No sensor yet? Use simulated subjects

`simulate_subjects` creates realistic two-session recordings per subject. They include
band-limited sEMG, random effort, a 150–400 ms reaction time after each cue, biceps/triceps crosstalk,
unbalanced co-contraction, fatigue, electrode shift between sessions, 50 Hz mains and motion artefacts.
```bash
ros2 run prosthetic_elbow_sim simulate_subjects --subjects 6 --out ~/elbow_data
ros2 run prosthetic_elbow_sim run_study --data ~/elbow_data --out ~/elbow_results/study
xdg-open ~/elbow_results/study/study_summary.png
```
`run_study` calibrates and trains on session 1, tests proportional / LDA / MLP on session 2, and writes
`per_subject.csv`, `summary.csv` (mean ± SD, paired Wilcoxon vs proportional), `study_summary.png/.pdf`,
`confusion_all.png` and example trial figures for S01.

Drive the Gazebo elbow with a simulated subject:
```bash
ros2 run prosthetic_elbow_sim calibrate_emg ~/elbow_data/S01_session1.csv --out ~/elbow_data/S01_decoder.yaml
ros2 run prosthetic_elbow_sim train_decoder ~/elbow_data/S01_session1.csv --model mlp --out ~/elbow_models/S01_mlp.joblib
ros2 launch prosthetic_elbow_sim sim.launch.py emg_source:=csv csv_file:=$HOME/elbow_data/S01_session2.csv \
    decoder_config:=$HOME/elbow_data/S01_decoder.yaml decoder_type:=classifier model_file:=$HOME/elbow_models/S01_mlp.joblib
```
Simulated subjects are for developing and testing the pipeline. Report decoder conclusions only from real recordings.

---

## 6. Record real EMG with an Arduino

Hardware: an Arduino Uno/Nano (or ESP32) with two EMG sensor boards (e.g. Upside Down Labs BioAmp EXG Pill /
Muscle BioAmp, or MyoWare RAW output). Biceps sensor → **A0**, triceps sensor → **A1**, reference electrode
on a bony spot (elbow). **Run the laptop on battery (charger unplugged) while electrodes are on a person.**

1. Upload `arduino/emg_stream/emg_stream.ino` with the Arduino IDE (it streams `biceps,triceps` at 1000 Hz, 230400 baud).
2. Allow serial access once: `sudo usermod -aG dialout $USER`, then log out and back in. Install: `sudo apt install python3-serial`.
3. Rehearse the cues: `ros2 run prosthetic_elbow_sim record_emg --dry-run`
4. Record two sessions per subject (take the electrodes off and re-apply them between sessions):
```bash
ros2 run prosthetic_elbow_sim record_emg --port /dev/ttyACM0 --subject 1 --session 1
ros2 run prosthetic_elbow_sim record_emg --port /dev/ttyACM0 --subject 1 --session 2
```
For an ESP32 add `--vref 3.3 --adc-max 4095`. Files go to `~/elbow_data/S01_session1.csv` and so on, in the same format
as the simulated subjects, so `run_study --data ~/elbow_data` works unchanged. Keep real and simulated data in
**separate folders** (use `--out ~/elbow_data_real`).

Ethics: recording people (including yourself, for a thesis) normally needs approval from your institution's ethics committee
and informed consent. Check this with your supervisor before collecting data for publication.
