"""ROS 2 node: replays recorded two-channel EMG from a CSV file onto /emg/raw.

Use this to drive the simulated elbow with your own recordings
(e.g. from an EMG armband, BioAmp/Myoware sensor, or a lab amplifier export).

CSV format (header row required, extra columns ignored):
    biceps,triceps,label
    0.0132,-0.0041,rest
    ...
`label` is optional. If present it is published on /emg/intended_label so that
analyze_bags can score the decoder (intent accuracy, onset delay).
Samples are assumed to be uniformly sampled at emg_rate_hz.
Calibrate rest_level / mvc with:  ros2 run prosthetic_elbow_sim calibrate_emg <csv>
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String

from prosthetic_elbow_sim.emg_csv import load_emg_csv


class EmgCsvPlayer(Node):

    def __init__(self):
        super().__init__('emg_csv_player')
        path = self.declare_parameter('csv_file', '').value
        fs = self.declare_parameter('emg_rate_hz', 1000.0).value
        chunk_ms = self.declare_parameter('chunk_ms', 10.0).value
        b_col = self.declare_parameter('biceps_column', 'biceps').value
        t_col = self.declare_parameter('triceps_column', 'triceps').value
        l_col = self.declare_parameter('label_column', 'label').value
        self.loop = self.declare_parameter('loop', True).value

        if not path:
            raise RuntimeError('Set the csv_file parameter to your EMG recording.')
        emg, labels = load_emg_csv(path, b_col, t_col, l_col)
        self.samples = emg.tolist()
        self.labels = labels

        self.idx = 0
        self.n_per_chunk = max(1, int(round(fs * chunk_ms / 1000.0)))
        self.pub = self.create_publisher(Float64MultiArray, '/emg/raw', 50)
        self.label_pub = self.create_publisher(String, '/emg/intended_label', 10)
        self.timer = self.create_timer(chunk_ms / 1000.0, self.on_timer)
        self.get_logger().info(
            f'Replaying {len(self.samples)} samples ({len(self.samples) / fs:.1f} s) from {path}'
            f'{" with labels" if labels else ""}')

    def on_timer(self):
        data = []
        label = None
        for _ in range(self.n_per_chunk):
            if self.idx >= len(self.samples):
                if not self.loop:
                    self.get_logger().info('End of recording.')
                    self.timer.cancel()
                    break
                self.idx = 0
            data.extend(self.samples[self.idx])
            if self.labels:
                label = self.labels[self.idx]
            self.idx += 1
        if data:
            self.pub.publish(Float64MultiArray(data=data))
        if label is not None:
            self.label_pub.publish(String(data=label))


def main(args=None):
    rclpy.init(args=args)
    try:
        node = EmgCsvPlayer()
    except (RuntimeError, ValueError) as err:
        print(f'[emg_csv_player] {err}')
        rclpy.shutdown()
        return
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
