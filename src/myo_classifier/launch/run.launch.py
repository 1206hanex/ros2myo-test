import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_share = get_package_share_directory('myo_classifier')
    params = os.path.join(pkg_share, 'config', 'cfg.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=params,
            description="~/ros2_myo/src/myo_classifier/config/cfg.yaml"
        ),
        Node(
            package='myo_classifier',
            executable='myo_stream',
            name='myo_stream',
            output='screen',
            parameters=[params],
        ),
        Node(
            package='myo_classifier',
            executable='manager',   # or 'myo_cnn_seq'
            name='manager',
            output='screen',
            parameters=[params],
        ),
    ])
