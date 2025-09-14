import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnShutdown
from launch_ros.actions import Node

def generate_launch_description():
    # Optionally expose the package share dir via ENV if needed by your nodes
    pkg_share = get_package_share_directory('myo_space_trigger')
    os.environ['MYO_SPACE_TRIGGER_SHARE'] = pkg_share

    return LaunchDescription([
        Node(
            package='myo_space_trigger',
            executable='myo_stream',
            name='myo_emg_publisher',
            output='screen',
        ),
        Node(
            package='myo_space_trigger',
            executable='myo_rf',
            name='myo_emg_classifier',
            output='screen',
        ),
        # Ensure clean shutdown on Ctrl+C
        RegisterEventHandler(
            OnShutdown(on_shutdown=lambda event, context: context.emit_event(event.__class__()))
        ),
    ])