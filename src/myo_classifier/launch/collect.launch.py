from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Parameter file (default to myo_classifier/share/myo_classifier/config/cfg.yaml)
    params_file = LaunchConfiguration('params_file')
    default_params = PathJoinSubstitution([
        FindPackageShare('myo_classifier'),
        'config',
        'cfg.yaml'
    ])

    declare_params = DeclareLaunchArgument(
        'params_file',
        default_value=default_params,
        description='YAML file with defaults for stream/recorder/manager'
    )

    # 1) Publisher: myo_stream
    myo_stream = Node(
        package='myo_classifier',
        executable='myo_stream',
        name='myo_stream',
        output='screen',
        parameters=[params_file],
    )

    # 2) Recorder: gesture_recorder (starts AFTER myo_stream is up)
    gesture_recorder = Node(
        package='myo_classifier',
        executable='gesture_recorder',
        name='gesture_recorder',
        output='screen',
        parameters=[params_file],
    )
    start_recorder_when_myo_starts = RegisterEventHandler(
        OnProcessStart(
            target_action=myo_stream,
            on_start=[gesture_recorder],
        )
    )
    return LaunchDescription([
        declare_params,
        myo_stream,
        start_recorder_when_myo_starts,
    ])
