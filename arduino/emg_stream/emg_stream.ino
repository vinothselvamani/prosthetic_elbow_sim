// Two-channel EMG streamer for prosthetic_elbow_sim (record_emg tool).
// Board: Arduino Uno / Nano / Mega (5 V, 10-bit ADC). Also works on ESP32 (12-bit ADC).
//
// Wiring (e.g. Upside Down Labs BioAmp EXG Pill / Muscle BioAmp, MyoWare RAW output):
//   biceps sensor OUT  -> A0
//   triceps sensor OUT -> A1
//   sensor GND -> GND, sensor VCC -> 5V (or 3.3V, as the sensor requires)
//   Reference electrode on a bony spot (elbow / wrist).
//   Use the RAW (unrectified) output if your sensor has RAW and ENV outputs.
//
// Output: one line per sample, "biceps,triceps\n", raw ADC counts, at 1000 Hz, 230400 baud.
// Safety: power the laptop from its BATTERY (unplug the charger) while electrodes are on a person.

const unsigned long SAMPLE_US = 1000;   // 1000 Hz
const long BAUD = 230400;
unsigned long next_t;

void setup() {
  Serial.begin(BAUD);
  while (!Serial) { }
  next_t = micros();
}

void loop() {
  if ((long)(micros() - next_t) >= 0) {
    next_t += SAMPLE_US;
    int b = analogRead(A0);
    int t = analogRead(A1);
    Serial.print(b);
    Serial.print(',');
    Serial.println(t);
  }
}
