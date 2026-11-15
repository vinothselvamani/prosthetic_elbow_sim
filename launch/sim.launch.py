"""Gazebo simulation of the EMG-controlled prosthetic elbow.

Usage:
  ros2 launch prosthetic_elbow_sim sim.launch.py                       # synthetic EMG
  ros2 launch prosthetic_elbow_sim sim.launch.py emg_source:=csv \
      csv_file:=/path/to/recording.csv                                 # your recording
  ros2 launch prosthetic_elbow_sim sim.launch.py emg_source:=none      # elbow only
  ros2 launch prosthetic_elbow_sim sim.launch.py gui:=false            # headless
  ros2 launch prosthetic_elbow_sim sim.launch.py decoder_type:=classifier \
      model_file:=$HOME/elbow_models/lda.joblib                        # LDA / ANN decoder
  ros2 launch prosthetic_elbow_sim sim.launch.py decoder_config:=/path/to/decoder_subject01.yaml
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            RegisterEventHandler)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (Command, LaunchConfiguration, PathJoinSubstitution,
                                  PythonExpression)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg = get_package_share_directory('prosthetic_elbow_sim')
    xacro_file = os.path.join(pkg, 'urdf', 'elbow.urdf.xacro')
    controllers = os.path.join(pkg, 'config', 'controllers.yaml')
    default_decoder_cfg = os.path.join(pkg, 'config', 'decoder.yaml')

    emg_source = LaunchConfiguration('emg_source')
    csv_file = LaunchConfiguration('csv_file')
    gui = LaunchConfiguration('gui')
    decoder_cfg = LaunchConfiguration('decoder_config')
    decoder_type = LaunchConfiguration('decoder_type')
    model_file = LaunchConfiguration('model_file')

    robot_description = ParameterValue(
        Command(['xacro ', xacro_file, ' controllers_file:=', controllers]),
        value_type=str)

    gz_args = PythonExpression(["'-r empty.sdf' if '", gui, "' == 'true' else '-r -s empty.sdf'"])
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py'])),
        launch_arguments={'gz_args': gz_args, 'on_exit_shutdown': 'true'}.items())

    clock_bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
        output='screen')

    robot_state_publisher = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description, 'use_sim_time': True}],
        output='screen')

    spawn = Node(
        package='ros_gz_sim', executable='create',
        arguments=['-topic', 'robot_description', '-name', 'prosthetic_elbow'],
        output='screen')

    jsb_spawner = Node(
        package='controller_manager', executable='spawner',
        arguments=['joint_state_broadcaster', '--param-file', controllers], output='screen')
    elbow_spawner = Node(
        package='controller_manager', executable='spawner',
        arguments=['elbow_position_controller', '--param-file', controllers], output='screen')

    decoder = Node(
        package='prosthetic_elbow_sim', executable='emg_decoder',
        parameters=[decoder_cfg, {'decoder_type': decoder_type, 'model_file': model_file}],
        condition=IfCondition(PythonExpression(["'", emg_source, "' != 'none'"])),
        output='screen')

    emg_sim = Node(
        package='prosthetic_elbow_sim', executable='emg_simulator',
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(PythonExpression(["'", emg_source, "' == 'sim'"])),
        output='screen')

    emg_csv = Node(
        package='prosthetic_elbow_sim', executable='emg_csv_player',
        parameters=[{'use_sim_time': True, 'csv_file': csv_file}],
        condition=IfCondition(PythonExpression(["'", emg_source, "' == 'csv'"])),
        output='screen')

    # start controllers only after the model exists, and EMG only after the controller
    start_controllers = RegisterEventHandler(OnProcessExit(
        target_action=spawn, on_exit=[jsb_spawner, elbow_spawner]))
    start_emg = RegisterEventHandler(OnProcessExit(
        target_action=elbow_spawner, on_exit=[decoder, emg_sim, emg_csv]))

    return LaunchDescription([
        DeclareLaunchArgument('emg_source', default_value='sim',
                              description='sim | csv | none'),
        DeclareLaunchArgument('csv_file', default_value='',
                              description='EMG recording for emg_source:=csv'),
        DeclareLaunchArgument('gui', default_value='true',
                              description='Show the Gazebo GUI'),
        DeclareLaunchArgument('decoder_config', default_value=default_decoder_cfg,
                              description='decoder YAML (e.g. from calibrate_emg)'),
        DeclareLaunchArgument('decoder_type', default_value='proportional',
                              description='proportional | classifier'),
        DeclareLaunchArgument('model_file', default_value='',
                              description='.joblib model from train_decoder (classifier)'),
        gazebo, clock_bridge, robot_state_publisher, spawn,
        start_controllers, start_emg,
    ])
