from glob import glob
import os

from setuptools import setup

package_name = 'prosthetic_elbow_sim'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.xacro')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'data'), glob('data/*.csv')),
        (os.path.join('share', package_name, 'arduino', 'emg_stream'), glob('arduino/emg_stream/*.ino')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='PhD Student',
    maintainer_email='you@example.com',
    description='Simulated EMG-controlled transhumeral prosthetic elbow (ROS 2 + Gazebo).',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'emg_simulator = prosthetic_elbow_sim.emg_simulator:main',
            'emg_csv_player = prosthetic_elbow_sim.emg_csv_player:main',
            'emg_decoder = prosthetic_elbow_sim.emg_decoder:main',
            'record_trial = prosthetic_elbow_sim.record_trial:main',
            'analyze_bags = prosthetic_elbow_sim.analyze_bags:main',
            'calibrate_emg = prosthetic_elbow_sim.calibrate_emg:main',
            'train_decoder = prosthetic_elbow_sim.train_decoder:main',
            'offline_compare = prosthetic_elbow_sim.offline_compare:main',
            'simulate_subjects = prosthetic_elbow_sim.emg_subjects:main',
            'run_study = prosthetic_elbow_sim.run_study:main',
            'record_emg = prosthetic_elbow_sim.record_emg:main',
        ],
    },
)
