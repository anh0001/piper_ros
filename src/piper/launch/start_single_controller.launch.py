from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os

os.environ["RCUTILS_COLORIZED_OUTPUT"] = "1"   # 强制彩色日志

def generate_launch_description():
    log_level_arg = DeclareLaunchArgument(
        'log_level',
        default_value='info',
        description='Logging level (debug, info, warn, error, fatal).'
    )
    # Declare the launch arguments
    can_port_arg = DeclareLaunchArgument(
        'can_port',
        default_value='can0',
        description='CAN port to be used by the Piper node.'
    )
    auto_enable_arg = DeclareLaunchArgument(
        'auto_enable',
        default_value='true',
        description='Automatically enable the Piper node.'
    )

    rviz_ctrl_flag_arg = DeclareLaunchArgument(
        'rviz_ctrl_flag',
        default_value='false',
        description='Start rviz flag.'
    )

    gripper_exist_arg = DeclareLaunchArgument(
        'gripper_exist',
        default_value='true',
        description='gripper'
    )

    gripper_val_mutiple_arg = DeclareLaunchArgument(
        'gripper_val_mutiple',
        default_value='1',
        description='gripper'
    )

    joint_states_topic_arg = DeclareLaunchArgument(
        'joint_states_topic',
        default_value='/joint_states',
        description='JointState feedback topic published by the PiPER driver.'
    )

    joint_cmd_topic_arg = DeclareLaunchArgument(
        'joint_cmd_topic',
        default_value='/piper/joint_cmd',
        description='JointState command topic consumed by the PiPER driver.'
    )

    joint_name_prefix_arg = DeclareLaunchArgument(
        'joint_name_prefix',
        default_value='piper_',
        description='Prefix applied to published joint names (e.g., piper_joint1).'
    )

    # Define the node
    piper_node = Node(
        package='piper',
        executable='piper_single_ctrl',
        name='piper_ctrl_single_node',
        output='screen',
        ros_arguments=['--log-level', LaunchConfiguration('log_level')],
        parameters=[{
            'can_port': LaunchConfiguration('can_port'),
            'auto_enable': LaunchConfiguration('auto_enable'),
            'gripper_val_mutiple': LaunchConfiguration('gripper_val_mutiple'),
            'gripper_exist': LaunchConfiguration('gripper_exist'),
            'joint_name_prefix': LaunchConfiguration('joint_name_prefix'),
        }],
        remappings=[
            ('joint_states_single', LaunchConfiguration('joint_states_topic')),
            ('joint_ctrl_single', LaunchConfiguration('joint_cmd_topic')),
            # ('joint_states_feedback', '/joint_states'),
        ]
    )

    # Return the LaunchDescription
    return LaunchDescription([
        log_level_arg,
        can_port_arg,
        auto_enable_arg,
        gripper_exist_arg,
        gripper_val_mutiple_arg,
        joint_states_topic_arg,
        joint_cmd_topic_arg,
        joint_name_prefix_arg,
        piper_node
    ])
