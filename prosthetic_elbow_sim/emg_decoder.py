"""ROS 2 node: two-channel EMG -> elbow position commands.

decoder_type (parameter):
  proportional  conventional two-site proportional myoelectric control (default)
  classifier    pattern recognition (LDA / ANN / SVM model from train_decoder) with
                proportional speed; set model_file to the .joblib file

Subscribes:
  /emg/raw  (std_msgs/Float64MultiArray)  interleaved samples [b0, t0, b1, t1, ...]
Publishes:
  <command_topic>     (std_msgs/Float64MultiArray)  [elbow_angle_rad]  -> ros2_control
  /emg/envelope       (std_msgs/Float64MultiArray)  [biceps_rms, triceps_rms]
  /emg/activation     (std_msgs/Float64MultiArray)  [a_biceps, a_triceps]  (0..1)
  /elbow/command_deg  (std_msgs/Float64)            commanded angle in degrees
  /elbow/mode         (std_msgs/String)             rest | flex | extend | cocontraction_hold
"""

from collections import deque
import math

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray, String

from prosthetic_elbow_sim.decoder_core import (
    ProportionalElbowDecoder, RmsEnvelope, normalise)


class EmgDecoder(Node):

    def __init__(self):
        super().__init__('emg_decoder')
        p = self.declare_parameter
        decoder_type = p('decoder_type', 'proportional').value
        model_file = p('model_file', '').value
        emg_rate = p('emg_rate_hz', 1000.0).value
        window_ms = p('window_ms', 150.0).value
        control_rate = p('control_rate_hz', 100.0).value
        self.rest = list(p('rest_level', [0.02, 0.02]).value)
        self.mvc = list(p('mvc', [0.50, 0.50]).value)
        command_topic = p('command_topic', '/elbow_position_controller/commands').value
        max_velocity = p('max_velocity', 1.5).value
        elbow_min = p('elbow_min', 0.0).value
        elbow_max = p('elbow_max', 2.53).value
        activation_threshold = p('activation_threshold', 0.10).value
        cocontraction_threshold = p('cocontraction_threshold', 0.25).value
        vote_length = p('vote_length', 5).value

        window = max(3, int(round(emg_rate * window_ms / 1000.0)))
        self.env = [RmsEnvelope(window), RmsEnvelope(window)]
        self.raw = deque(maxlen=window)  # (biceps, triceps) samples for classifier features
        self.window = window

        if decoder_type == 'classifier':
            if not model_file:
                raise RuntimeError('decoder_type=classifier needs model_file (from train_decoder)')
            from prosthetic_elbow_sim.classifier_decoder import ClassifierElbowDecoder, load_model
            bundle = load_model(model_file)
            self.decoder = ClassifierElbowDecoder(
                bundle, max_velocity=max_velocity, elbow_min=elbow_min, elbow_max=elbow_max,
                vote_length=vote_length)
            label = f'classifier ({bundle.get("kind", "?")}: {model_file})'
        else:
            self.decoder = ProportionalElbowDecoder(
                activation_threshold=activation_threshold,
                cocontraction_threshold=cocontraction_threshold,
                max_velocity=max_velocity, elbow_min=elbow_min, elbow_max=elbow_max)
            label = 'proportional'
        self.is_classifier = decoder_type == 'classifier'

        self.dt = 1.0 / control_rate
        self.synced_to_joint = False

        self.create_subscription(Float64MultiArray, '/emg/raw', self.on_emg, 50)
        self.create_subscription(JointState, '/joint_states', self.on_joint_state, 10)
        self.cmd_pub = self.create_publisher(Float64MultiArray, command_topic, 10)
        self.env_pub = self.create_publisher(Float64MultiArray, '/emg/envelope', 10)
        self.act_pub = self.create_publisher(Float64MultiArray, '/emg/activation', 10)
        self.deg_pub = self.create_publisher(Float64, '/elbow/command_deg', 10)
        self.mode_pub = self.create_publisher(String, '/elbow/mode', 10)
        self.create_timer(self.dt, self.on_control)

        self.get_logger().info(
            f'EMG decoder ready: {label}, window={window} samples, '
            f'control={control_rate:.0f} Hz, commands -> {command_topic}')

    def on_joint_state(self, msg: JointState):
        # start integrating from the elbow's actual angle (once)
        if self.synced_to_joint or 'elbow_joint' not in msg.name:
            return
        self.decoder.angle = msg.position[msg.name.index('elbow_joint')]
        self.synced_to_joint = True

    def on_emg(self, msg: Float64MultiArray):
        data = msg.data
        for i in range(0, len(data) - 1, 2):
            self.env[0].add(data[i])
            self.env[1].add(data[i + 1])
            self.raw.append((data[i], data[i + 1]))

    def on_control(self):
        b_rms, t_rms = self.env[0].value, self.env[1].value
        a_b = normalise(b_rms, self.rest[0], self.mvc[0])
        a_t = normalise(t_rms, self.rest[1], self.mvc[1])
        if self.is_classifier:
            if len(self.raw) < self.window:
                return  # wait for a full feature window
            angle = self.decoder.step(np.asarray(self.raw), a_b, a_t, self.dt)
        else:
            angle = self.decoder.step(a_b, a_t, self.dt)

        self.cmd_pub.publish(Float64MultiArray(data=[angle]))
        self.env_pub.publish(Float64MultiArray(data=[b_rms, t_rms]))
        self.act_pub.publish(Float64MultiArray(data=[a_b, a_t]))
        self.deg_pub.publish(Float64(data=math.degrees(angle)))
        self.mode_pub.publish(String(data=self.decoder.mode))


def main(args=None):
    rclpy.init(args=args)
    try:
        node = EmgDecoder()
    except (RuntimeError, ValueError, OSError) as err:
        print(f'[emg_decoder] {err}')
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
