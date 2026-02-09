#!/usr/bin/env python3
"""
FollowJointTrajectory bridge for PiPER.

Exposes FollowJointTrajectory action servers and publishes JointState commands
to the PiPER driver while using real joint state feedback from hardware.
"""

import math
from typing import Dict, List, Tuple

import rclpy
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.node import Node

from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState


def _duration_to_sec(duration_msg: Duration) -> float:
    return float(duration_msg.sec) + float(duration_msg.nanosec) / 1e9


class PiperFollowJointTrajectoryBridge(Node):
    def __init__(self) -> None:
        super().__init__("piper_follow_joint_trajectory_bridge")

        self.declare_parameter(
            "arm_action_name",
            "/piper_arm_controller/follow_joint_trajectory",
        )
        self.declare_parameter(
            "gripper_action_name",
            "/piper_gripper_controller/follow_joint_trajectory",
        )
        self.declare_parameter("command_topic", "/piper/joint_cmd")
        self.declare_parameter("state_topic", "/joint_states")
        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("default_speed", 30)
        self.declare_parameter("joint_name_prefix", "piper_")
        self.declare_parameter(
            "joint_names",
            [],
        )

        self.arm_action_name = (
            self.get_parameter("arm_action_name").get_parameter_value().string_value
        )
        self.gripper_action_name = (
            self.get_parameter("gripper_action_name").get_parameter_value().string_value
        )
        self.command_topic = (
            self.get_parameter("command_topic").get_parameter_value().string_value
        )
        self.state_topic = (
            self.get_parameter("state_topic").get_parameter_value().string_value
        )
        self.publish_rate_hz = max(
            1.0, self.get_parameter("publish_rate_hz").get_parameter_value().double_value
        )
        self.default_speed = int(
            self.get_parameter("default_speed").get_parameter_value().integer_value
        )
        self.default_speed = max(1, min(self.default_speed, 100))
        self.joint_name_prefix = (
            self.get_parameter("joint_name_prefix").get_parameter_value().string_value
        )
        provided_names = list(
            self.get_parameter("joint_names").get_parameter_value().string_array_value
        )
        if provided_names:
            self.full_joint_names = provided_names
        else:
            prefix = self.joint_name_prefix or ""
            self.full_joint_names = [
                f"{prefix}joint1",
                f"{prefix}joint2",
                f"{prefix}joint3",
                f"{prefix}joint4",
                f"{prefix}joint5",
                f"{prefix}joint6",
                f"{prefix}joint7",
            ]

        if len(self.full_joint_names) != 7:
            self.get_logger().warn(
                "Expected 7 joint names for PiPER; using provided list anyway."
            )

        self.last_positions: Dict[str, float] = {
            name: 0.0 for name in self.full_joint_names
        }
        self.last_command_positions: Dict[str, float] = {
            name: 0.0 for name in self.full_joint_names
        }

        self.command_pub = self.create_publisher(JointState, self.command_topic, 10)
        self.state_sub = self.create_subscription(
            JointState, self.state_topic, self._state_callback, 10
        )

        self.arm_action_server = ActionServer(
            self,
            FollowJointTrajectory,
            self.arm_action_name,
            execute_callback=self._execute_arm,
            goal_callback=self._goal_callback_arm,
            cancel_callback=self._cancel_callback,
        )

        self.gripper_action_server = ActionServer(
            self,
            FollowJointTrajectory,
            self.gripper_action_name,
            execute_callback=self._execute_gripper,
            goal_callback=self._goal_callback_gripper,
            cancel_callback=self._cancel_callback,
        )

        self.get_logger().info(
            f"PiPER FollowJointTrajectory bridge ready: arm={self.arm_action_name}, "
            f"gripper={self.gripper_action_name}"
        )

    def _state_callback(self, msg: JointState) -> None:
        for idx, name in enumerate(msg.name):
            if name in self.last_positions and idx < len(msg.position):
                self.last_positions[name] = msg.position[idx]

    def _cancel_callback(self, _goal_handle) -> CancelResponse:
        return CancelResponse.ACCEPT

    def _goal_callback_arm(self, goal_request) -> GoalResponse:
        allowed = self.full_joint_names[:6]
        return self._validate_goal(goal_request, allowed)

    def _goal_callback_gripper(self, goal_request) -> GoalResponse:
        allowed = [self.full_joint_names[6]]
        return self._validate_goal(goal_request, allowed)

    def _validate_goal(self, goal_request, allowed_joints: List[str]) -> GoalResponse:
        traj = goal_request.trajectory
        if not traj.joint_names:
            self.get_logger().warn("Rejected goal: empty joint_names.")
            return GoalResponse.REJECT
        if not traj.points:
            self.get_logger().warn("Rejected goal: no trajectory points.")
            return GoalResponse.REJECT
        for joint_name in traj.joint_names:
            if joint_name not in allowed_joints:
                self.get_logger().warn(
                    f"Rejected goal: joint {joint_name} not allowed for this action."
                )
                return GoalResponse.REJECT
        last_t = -math.inf
        for point in traj.points:
            if len(point.positions) != len(traj.joint_names):
                self.get_logger().warn(
                    "Rejected goal: positions length does not match joint_names."
                )
                return GoalResponse.REJECT
            t = _duration_to_sec(point.time_from_start)
            if t < last_t:
                self.get_logger().warn(
                    "Rejected goal: time_from_start is not nondecreasing."
                )
                return GoalResponse.REJECT
            last_t = t
        return GoalResponse.ACCEPT

    def _execute_arm(self, goal_handle):
        allowed = self.full_joint_names[:6]
        return self._execute(goal_handle, allowed)

    def _execute_gripper(self, goal_handle):
        allowed = [self.full_joint_names[6]]
        return self._execute(goal_handle, allowed)

    def _execute(self, goal_handle, allowed_joints: List[str]):
        goal = goal_handle.request
        traj = goal.trajectory
        joint_names = list(traj.joint_names)
        points = self._build_points(traj.points, joint_names)
        total_duration = points[-1][0] if points else 0.0

        rate = self.create_rate(self.publish_rate_hz)
        start_time = self.get_clock().now()

        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result = FollowJointTrajectory.Result()
                result.error_code = getattr(
                    FollowJointTrajectory.Result, "INVALID_GOAL", -1
                )
                result.error_string = "Goal canceled."
                return result

            now = self.get_clock().now()
            elapsed = (now - start_time).nanoseconds / 1e9
            if elapsed >= total_duration:
                target = points[-1][1] if points else {}
                self._publish_command(now, target)
                break

            target = self._interpolate(points, elapsed)
            self._publish_command(now, target)
            rate.sleep()

        goal_handle.succeed()
        result = FollowJointTrajectory.Result()
        result.error_code = getattr(
            FollowJointTrajectory.Result, "SUCCESSFUL", 0
        )
        result.error_string = "Succeeded."
        return result

    def _build_points(
        self, points, joint_names: List[str]
    ) -> List[Tuple[float, Dict[str, float]]]:
        built: List[Tuple[float, Dict[str, float]]] = []
        for point in points:
            t = _duration_to_sec(point.time_from_start)
            positions = {name: pos for name, pos in zip(joint_names, point.positions)}
            built.append((t, positions))
        if not built:
            built.append((0.0, {}))
        return built

    def _interpolate(
        self, points: List[Tuple[float, Dict[str, float]]], t: float
    ) -> Dict[str, float]:
        if t <= points[0][0]:
            return points[0][1]
        for idx in range(len(points) - 1):
            t0, p0 = points[idx]
            t1, p1 = points[idx + 1]
            if t0 <= t <= t1:
                if t1 <= t0:
                    return p1
                ratio = (t - t0) / (t1 - t0)
                interpolated = {}
                for joint_name, pos0 in p0.items():
                    pos1 = p1.get(joint_name, pos0)
                    interpolated[joint_name] = pos0 + (pos1 - pos0) * ratio
                return interpolated
        return points[-1][1]

    def _publish_command(self, now, target: Dict[str, float]) -> None:
        positions: List[float] = []
        for joint_name in self.full_joint_names:
            if joint_name in target:
                value = target[joint_name]
            elif joint_name in self.last_command_positions:
                value = self.last_command_positions[joint_name]
            else:
                value = self.last_positions.get(joint_name, 0.0)
            positions.append(value)

        cmd = JointState()
        cmd.header.stamp = now.to_msg()
        cmd.name = list(self.full_joint_names)
        cmd.position = positions
        cmd.velocity = [0.0] * len(self.full_joint_names)
        if cmd.velocity:
            cmd.velocity[-1] = float(self.default_speed)
        cmd.effort = [0.0] * len(self.full_joint_names)

        self.command_pub.publish(cmd)

        for idx, name in enumerate(self.full_joint_names):
            self.last_command_positions[name] = positions[idx]


def main() -> None:
    rclpy.init()
    node = PiperFollowJointTrajectoryBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
