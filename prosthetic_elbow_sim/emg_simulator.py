"""ROS 2 node: publishes synthetic biceps/triceps sEMG following a looping task scenario.

Publishes:
  /emg/raw              (std_msgs/Float64MultiArray) interleaved [b0, t0, b1, t1, ...],
                        one chunk every chunk_ms (default 10 samples @ 1 kHz)
  /emg/intended_label   (std_msgs/String)  ground-truth task phase (flex_40, rest, ...)
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String

from prosthetic_elbow_sim.synthetic_emg import SyntheticEmg


class EmgSimulator(Node):

    def __init__(self):
        super().__init__('emg_simulator')
        fs = self.declare_parameter('emg_rate_hz', 1000.0).value
        chunk_ms = self.declare_parameter('chunk_ms', 10.0).value
        rest = list(self.declare_parameter('rest_level', [0.02, 0.02]).value)
        mvc = list(self.declare_parameter('mvc', [0.50, 0.50]).value)
        crosstalk = self.declare_parameter('crosstalk', 0.08).value
        seed = self.declare_parameter('seed', 42).value

        self.gen = SyntheticEmg(fs=fs, rest=rest, mvc=mvc, crosstalk=crosstalk, seed=seed)
        self.n_per_chunk = max(1, int(round(fs * chunk_ms / 1000.0)))
        self.raw_pub = self.create_publisher(Float64MultiArray, '/emg/raw', 50)
        self.label_pub = self.create_publisher(String, '/emg/intended_label', 10)
        self.create_timer(chunk_ms / 1000.0, self.on_timer)
        self.get_logger().info(
            f'Synthetic EMG: {fs:.0f} Hz, {self.n_per_chunk} samples/msg, '
            f'scenario cycle {self.gen.cycle_len:.1f} s')

    def on_timer(self):
        data = []
        label = 'rest'
        for _ in range(self.n_per_chunk):
            b, t, label = self.gen.sample()
            data.extend((b, t))
        self.raw_pub.publish(Float64MultiArray(data=data))
        self.label_pub.publish(String(data=label))


def main(args=None):
    rclpy.init(args=args)
    node = EmgSimulator()
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
