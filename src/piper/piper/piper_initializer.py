#!/usr/bin/env python3
"""Initialize PiPER arm control mode if it gets stuck."""

import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Bool

from piper_msgs.msg import PiperStatusMsg, PosCmd
from piper_msgs.srv import Enable


class PiperInitializer(Node):
    def __init__(self):
        super().__init__('piper_initializer')

        self.declare_parameter('arm_status_topic', 'arm_status')
        self.declare_parameter('pos_cmd_topic', 'pos_cmd')
        self.declare_parameter('enable_flag_topic', 'enable_flag')
        self.declare_parameter('enable_service', 'enable_srv')
        self.declare_parameter('standby_mode', 0)
        self.declare_parameter('can_command_mode', 1)
        self.declare_parameter('status_timeout_sec', 5.0)

        arm_status_topic = self.get_parameter('arm_status_topic').value
        pos_cmd_topic = self.get_parameter('pos_cmd_topic').value
        enable_flag_topic = self.get_parameter('enable_flag_topic').value
        enable_service = self.get_parameter('enable_service').value
        self.standby_mode = int(self.get_parameter('standby_mode').value)
        self.can_command_mode = int(self.get_parameter('can_command_mode').value)
        self.status_timeout_sec = float(self.get_parameter('status_timeout_sec').value)

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.status_sub = self.create_subscription(
            PiperStatusMsg,
            arm_status_topic,
            self.arm_status_callback,
            qos_profile,
        )

        self.pos_cmd_pub = self.create_publisher(PosCmd, pos_cmd_topic, qos_profile)
        self.enable_pub = self.create_publisher(Bool, enable_flag_topic, qos_profile)

        self.enable_client = self.create_client(Enable, enable_service)

        self.current_mode = None

        self.get_logger().info('Piper initializer node started')

    def arm_status_callback(self, msg: PiperStatusMsg) -> None:
        self.current_mode = msg.ctrl_mode

    def send_mode_cmd(self, mode_value: int) -> None:
        msg = PosCmd()
        msg.x = 0.0
        msg.y = 0.0
        msg.z = 0.0
        msg.roll = 0.0
        msg.pitch = 0.0
        msg.yaw = 0.0
        msg.gripper = 0.0
        msg.mode1 = int(mode_value)
        msg.mode2 = 0

        self.pos_cmd_pub.publish(msg)
        self.get_logger().info(f'Sent mode command: mode1={mode_value}')

    def enable_arm(self, enable_state: bool) -> bool:
        while not self.enable_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Enable service not available, waiting...')

        request = Enable.Request()
        request.enable_request = bool(enable_state)

        future = self.enable_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is None:
            self.get_logger().error(f'Service call failed: {future.exception()}')
            return False

        if future.result().enable_response:
            self.get_logger().info(
                f"Successfully {'enabled' if enable_state else 'disabled'} the arm"
            )
            return True

        self.get_logger().warning(
            f"Failed to {'enable' if enable_state else 'disable'} the arm"
        )
        return False

    def publish_enable_flag(self, enable_state: bool) -> None:
        msg = Bool()
        msg.data = bool(enable_state)
        self.enable_pub.publish(msg)
        self.get_logger().info(f'Published enable_flag: {enable_state}')

    def wait_for_status_update(self, timeout: float) -> bool:
        start_time = time.time()
        while self.current_mode is None:
            rclpy.spin_once(self, timeout_sec=0.1)
            time.sleep(0.1)
            if time.time() - start_time > timeout:
                self.get_logger().warning('Timeout waiting for status update')
                return False
        return True

    def initialize_control_mode(self) -> bool:
        self.get_logger().info('Starting Piper control mode initialization...')

        if not self.wait_for_status_update(timeout=self.status_timeout_sec):
            self.get_logger().error('Failed to get arm status. Is the driver running?')
            return False

        self.get_logger().info(f'Current control mode: {self.current_mode}')

        if self.current_mode == self.can_command_mode:
            self.get_logger().info('Arm already in CAN command control mode')
            return True

        self.get_logger().info('Setting to standby mode...')
        self.send_mode_cmd(self.standby_mode)
        time.sleep(1.0)
        rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().info('Setting to CAN command control mode...')
        for _ in range(5):
            self.send_mode_cmd(self.can_command_mode)
            time.sleep(0.5)
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.current_mode == self.can_command_mode:
                self.get_logger().info('Successfully set to CAN command control mode')
                return True

        self.get_logger().info('First attempt failed. Trying disable/enable cycle...')

        if not self.enable_arm(False):
            self.get_logger().error('Failed to disable the arm')
            return False

        time.sleep(2.0)

        if not self.enable_arm(True):
            self.get_logger().error('Failed to enable the arm')
            return False

        time.sleep(2.0)

        self.get_logger().info('Setting to standby mode...')
        self.send_mode_cmd(self.standby_mode)
        time.sleep(1.0)
        rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().info('Setting to CAN command control mode...')
        for _ in range(10):
            self.send_mode_cmd(self.can_command_mode)
            time.sleep(0.5)
            rclpy.spin_once(self, timeout_sec=0.5)
            if self.current_mode == self.can_command_mode:
                self.get_logger().info(
                    'Successfully set to CAN command control mode after reset'
                )
                return True

        self.get_logger().error(
            'Failed to set CAN command control mode after multiple attempts'
        )
        return False


def main(args=None) -> None:
    rclpy.init(args=args)
    initializer = PiperInitializer()

    success = initializer.initialize_control_mode()

    if success:
        initializer.get_logger().info('Enabling the arm...')
        initializer.publish_enable_flag(True)
        time.sleep(0.5)
        initializer.enable_arm(True)
        initializer.get_logger().info(
            'Piper arm initialized to CAN command control mode and enabled.'
        )
    else:
        initializer.get_logger().error(
            'Failed to initialize Piper arm to CAN command control mode.'
        )

    initializer.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
