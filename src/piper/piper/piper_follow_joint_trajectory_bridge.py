#!/usr/bin/env python3
"""
FollowJointTrajectory bridge for PiPER.

Exposes FollowJointTrajectory action servers and a GripperCommand action server,
then publishes JointState commands to the PiPER driver while using real joint
state feedback from hardware.
"""

import math
from typing import Dict, List, Tuple

import rclpy
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory, GripperCommand
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
        self.declare_parameter(
            "gripper_cmd_action_name",
            "/piper_gripper_controller/gripper_cmd",
        )
        self.declare_parameter("command_topic", "/piper/joint_cmd")
        self.declare_parameter("state_topic", "/joint_states")
        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("default_speed", 30)
        self.declare_parameter("gripper_goal_tolerance", 0.01)
        self.declare_parameter("gripper_progress_epsilon", 0.002)
        self.declare_parameter("gripper_no_progress_timeout_sec", 0.75)
        self.declare_parameter("gripper_max_execution_sec", 2.0)
        self.declare_parameter("gripper_republish_period_sec", 0.25)
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
        self.gripper_cmd_action_name = (
            self.get_parameter("gripper_cmd_action_name").get_parameter_value().string_value
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
        # Allow live tuning of the PiPER speed (1-100%) via `ros2 param set`.
        from rcl_interfaces.msg import SetParametersResult

        def _on_set_params(params):
            for p in params:
                if p.name == "default_speed":
                    self.default_speed = max(1, min(int(p.value), 100))
                    self.get_logger().info(f"default_speed -> {self.default_speed}%")
            return SetParametersResult(successful=True)
        self.add_on_set_parameters_callback(_on_set_params)
        self.gripper_goal_tolerance = max(
            1e-4,
            self.get_parameter("gripper_goal_tolerance")
            .get_parameter_value()
            .double_value,
        )
        self.gripper_progress_epsilon = max(
            1e-5,
            self.get_parameter("gripper_progress_epsilon")
            .get_parameter_value()
            .double_value,
        )
        self.gripper_no_progress_timeout_sec = max(
            0.1,
            self.get_parameter("gripper_no_progress_timeout_sec")
            .get_parameter_value()
            .double_value,
        )
        self.gripper_max_execution_sec = max(
            self.gripper_no_progress_timeout_sec,
            self.get_parameter("gripper_max_execution_sec")
            .get_parameter_value()
            .double_value,
        )
        self.gripper_republish_period_sec = max(
            0.05,
            self.get_parameter("gripper_republish_period_sec")
            .get_parameter_value()
            .double_value,
        )
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
        self.last_efforts: Dict[str, float] = {
            name: 0.0 for name in self.full_joint_names
        }
        self.last_command_positions: Dict[str, float] = {
            name: 0.0 for name in self.full_joint_names
        }
        self._state_received = False

        # Use ReentrantCallbackGroup so the state subscription callback
        # can fire while the action execute callback is waiting.
        self._cb_group = ReentrantCallbackGroup()

        self.command_pub = self.create_publisher(JointState, self.command_topic, 10)
        self.state_sub = self.create_subscription(
            JointState, self.state_topic, self._state_callback, 10,
            callback_group=self._cb_group,
        )

        self.arm_action_server = ActionServer(
            self,
            FollowJointTrajectory,
            self.arm_action_name,
            execute_callback=self._execute_arm,
            goal_callback=self._goal_callback_arm,
            cancel_callback=self._cancel_callback,
            callback_group=self._cb_group,
        )

        self.gripper_action_server = ActionServer(
            self,
            FollowJointTrajectory,
            self.gripper_action_name,
            execute_callback=self._execute_gripper,
            goal_callback=self._goal_callback_gripper,
            cancel_callback=self._cancel_callback,
            callback_group=self._cb_group,
        )

        self.gripper_cmd_action_server = ActionServer(
            self,
            GripperCommand,
            self.gripper_cmd_action_name,
            execute_callback=self._execute_gripper_cmd,
            goal_callback=lambda _: GoalResponse.ACCEPT,
            cancel_callback=self._cancel_callback,
            callback_group=self._cb_group,
        )

        self.get_logger().info(
            f"PiPER bridge ready: arm={self.arm_action_name}, "
            f"gripper_fjt={self.gripper_action_name}, "
            f"gripper_cmd={self.gripper_cmd_action_name}"
        )

    def _state_callback(self, msg: JointState) -> None:
        self._state_received = True
        for idx, name in enumerate(msg.name):
            if name in self.last_positions and idx < len(msg.position):
                self.last_positions[name] = msg.position[idx]
            if name in self.last_efforts and idx < len(msg.effort):
                self.last_efforts[name] = msg.effort[idx]

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

    def _execute_gripper_cmd(self, goal_handle):
        """Handle GripperCommand action — send position directly, no planning."""
        goal = goal_handle.request
        target_position = goal.command.position
        gripper_joint = self.full_joint_names[6]
        start_pos = (
            self.last_positions.get(gripper_joint, None) if self._state_received else None
        )
        self.get_logger().info(
            f"GripperCommand: moving {gripper_joint} from {start_pos} to {target_position:.4f}"
        )

        now = self.get_clock().now()
        self._publish_command(now, {gripper_joint: target_position})
        start_time = self.get_clock().now()
        last_progress_time = start_time
        last_progress_pos = start_pos
        last_publish_time = start_time
        significant_motion_threshold = max(
            self.gripper_goal_tolerance,
            self.gripper_progress_epsilon * 5.0,
        )
        rate = self.create_rate(20.0)

        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return self._build_gripper_result(
                    gripper_joint,
                    stalled=False,
                    reached_goal=False,
                )

            current_time = self.get_clock().now()
            elapsed = (current_time - start_time).nanoseconds / 1e9
            current_pos = (
                self.last_positions.get(gripper_joint, None)
                if self._state_received
                else None
            )
            delta = None if current_pos is None else abs(current_pos - target_position)

            if delta is not None and delta <= self.gripper_goal_tolerance:
                self.get_logger().info(
                    "GripperCommand succeeded: "
                    f"start={self._format_float(start_pos)} "
                    f"current={self._format_float(current_pos)} "
                    f"target={target_position:.4f} "
                    f"delta={delta:.4f} elapsed={elapsed:.3f}s"
                )
                goal_handle.succeed()
                return self._build_gripper_result(
                    gripper_joint,
                    stalled=False,
                    reached_goal=True,
                )

            if current_pos is not None:
                if last_progress_pos is None:
                    last_progress_pos = current_pos
                    last_progress_time = current_time
                elif abs(current_pos - last_progress_pos) >= self.gripper_progress_epsilon:
                    last_progress_pos = current_pos
                    last_progress_time = current_time

            since_progress = (current_time - last_progress_time).nanoseconds / 1e9
            made_directional_progress = self._made_directional_progress(
                start_pos,
                current_pos,
                target_position,
                significant_motion_threshold,
            )

            if elapsed >= self.gripper_max_execution_sec:
                if made_directional_progress:
                    self.get_logger().warn(
                        "GripperCommand timed out after directional progress: "
                        f"start={self._format_float(start_pos)} "
                        f"current={self._format_float(current_pos)} "
                        f"target={target_position:.4f} "
                        f"delta={self._format_delta(delta)} elapsed={elapsed:.3f}s; "
                        "treating as success at mechanical limit or model mismatch"
                    )
                    goal_handle.succeed()
                    return self._build_gripper_result(
                        gripper_joint,
                        stalled=True,
                        reached_goal=False,
                    )
                self.get_logger().warn(
                    "GripperCommand timeout: "
                    f"start={self._format_float(start_pos)} "
                    f"current={self._format_float(current_pos)} "
                    f"target={target_position:.4f} "
                    f"delta={self._format_delta(delta)} elapsed={elapsed:.3f}s; "
                    "target may be out of hardware range or blocked"
                )
                goal_handle.abort()
                return self._build_gripper_result(
                    gripper_joint,
                    stalled=False,
                    reached_goal=False,
                )

            if since_progress >= self.gripper_no_progress_timeout_sec:
                if made_directional_progress:
                    self.get_logger().warn(
                        "GripperCommand stalled after directional progress: "
                        f"start={self._format_float(start_pos)} "
                        f"current={self._format_float(current_pos)} "
                        f"target={target_position:.4f} "
                        f"delta={self._format_delta(delta)} elapsed={elapsed:.3f}s; "
                        "treating as success at mechanical limit or model mismatch"
                    )
                    goal_handle.succeed()
                    return self._build_gripper_result(
                        gripper_joint,
                        stalled=True,
                        reached_goal=False,
                    )
                self.get_logger().warn(
                    "GripperCommand stalled: "
                    f"start={self._format_float(start_pos)} "
                    f"current={self._format_float(current_pos)} "
                    f"target={target_position:.4f} "
                    f"delta={self._format_delta(delta)} elapsed={elapsed:.3f}s; "
                    "target may be out of hardware range or blocked"
                )
                goal_handle.abort()
                return self._build_gripper_result(
                    gripper_joint,
                    stalled=True,
                    reached_goal=False,
                )

            republish_elapsed = (
                current_time - last_publish_time
            ).nanoseconds / 1e9
            if republish_elapsed >= self.gripper_republish_period_sec:
                self._publish_command(current_time, {gripper_joint: target_position})
                last_publish_time = current_time

            rate.sleep()

        goal_handle.abort()
        return self._build_gripper_result(
            gripper_joint,
            stalled=False,
            reached_goal=False,
        )

    def _build_gripper_result(
        self,
        gripper_joint: str,
        *,
        stalled: bool,
        reached_goal: bool,
    ):
        result = GripperCommand.Result()
        result.position = self.last_positions.get(gripper_joint, 0.0)
        result.effort = self.last_efforts.get(gripper_joint, 0.0)
        result.stalled = stalled
        result.reached_goal = reached_goal
        return result

    def _format_float(self, value) -> str:
        if value is None:
            return "None"
        return f"{value:.4f}"

    def _format_delta(self, value) -> str:
        if value is None:
            return "None"
        return f"{value:.4f}"

    def _made_directional_progress(
        self,
        start_pos,
        current_pos,
        target_position: float,
        threshold: float,
    ) -> bool:
        if start_pos is None or current_pos is None:
            return False

        direction = target_position - start_pos
        if abs(direction) < threshold:
            return False

        moved = current_pos - start_pos
        if direction > 0.0:
            return moved >= threshold
        return moved <= -threshold

    def _execute(self, goal_handle, allowed_joints: List[str]):
        goal = goal_handle.request
        traj = goal.trajectory
        joint_names = list(traj.joint_names)
        points = self._build_points(traj.points, joint_names)
        total_duration = points[-1][0] if points else 0.0

        self.get_logger().info(
            f"Executing trajectory: {len(traj.points)} points, "
            f"duration={total_duration:.2f}s, joints={joint_names}"
        )
        if points:
            self.get_logger().info(f"First point: {points[0]}")
            self.get_logger().info(f"Last point: {points[-1]}")

        # PiPER uses CAN-bus position commands with its own internal motion
        # controller. Send the final target position once; the arm's firmware
        # handles the trajectory internally.  Then poll joint_states until the
        # arm reaches the target (or a timeout expires).
        final_target = points[-1][1] if points else {}

        # Send the target position command once
        now = self.get_clock().now()
        self._publish_command(now, final_target)
        self.get_logger().info("Sent target position to PiPER hardware.")

        # Wait for hardware to reach target.
        # Requires MultiThreadedExecutor so _state_callback runs while we wait.
        settle_timeout = max(total_duration * 4.0, 10.0)
        settle_tolerance = 0.05  # radians
        settle_start = self.get_clock().now()
        rate = self.create_rate(5.0)

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

            # Check if hardware reached target
            reached = True
            for joint_name, target_pos in final_target.items():
                current_pos = self.last_positions.get(joint_name, None)
                if current_pos is None or abs(current_pos - target_pos) > settle_tolerance:
                    reached = False
                    break

            if reached:
                self.get_logger().info("Hardware reached target position.")
                break

            settle_elapsed = (now - settle_start).nanoseconds / 1e9
            if settle_elapsed >= settle_timeout:
                self.get_logger().warn(
                    f"Settle timeout ({settle_timeout:.1f}s) reached, "
                    "hardware may not have fully reached target."
                )
                break

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

    _pub_log_count = 0

    def _publish_command(self, now, target: Dict[str, float]) -> None:
        # Build positions for all 7 joints (arm + gripper)
        all_positions: List[float] = []
        for joint_name in self.full_joint_names:
            if joint_name in target:
                value = target[joint_name]
            elif self._state_received and joint_name in self.last_positions:
                value = self.last_positions[joint_name]
            else:
                value = self.last_command_positions.get(joint_name, 0.0)
            all_positions.append(value)

        # Always publish all 7 joints. The driver (piper_ctrl_single_node_new)
        # runs GripperCtrl(joint_6) on EVERY command, and joint_6 defaults to 0
        # when the command carries < 7 positions -- so a 6-joint arm-only command
        # slams the gripper shut mid-motion. By always sending the gripper at its
        # held position (last_positions[gripper], filled above), the driver gets
        # joint_6 = current gripper angle and holds it open during an arm move.
        publish_gripper = True
        cmd_names = list(self.full_joint_names)
        cmd_positions = all_positions

        cmd = JointState()
        cmd.header.stamp = now.to_msg()
        cmd.name = cmd_names
        cmd.position = cmd_positions
        # Publish the PiPER global speed (1-100%) in velocity[6] so the driver
        # uses it: empty velocity makes the driver default to MotionCtrl_2(...,100)
        # = full speed, which is why MoveIt velocity_scaling had no effect. We
        # always publish all 7 joints (gripper held), so index 6 is valid.
        cmd.velocity = [0.0] * len(cmd_positions)
        if len(cmd.velocity) >= 7:
            cmd.velocity[6] = float(self.default_speed)

        if self._pub_log_count < 3:
            pos_str = ", ".join(f"{p:.4f}" for p in cmd_positions)
            self.get_logger().info(
                f"CMD#{self._pub_log_count}: names={cmd_names}, "
                f"pos=[{pos_str}]"
            )
            self._pub_log_count += 1

        self.command_pub.publish(cmd)

        for idx, name in enumerate(self.full_joint_names):
            self.last_command_positions[name] = all_positions[idx]


def main() -> None:
    rclpy.init()
    node = PiperFollowJointTrajectoryBridge()
    # MultiThreadedExecutor allows joint state callbacks to run
    # while the action execute callback is waiting for the arm to settle.
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
