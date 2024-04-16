"""
A basic dm environment for transporter data collection on the Franka robot.
"""
import time
from typing import Dict
from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation as R
import dm_env

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.logging import get_logger

from moveit.planning import MoveItPy
from moveit_configs_utils import MoveItConfigsBuilder
from ament_index_python.packages import get_package_share_directory

from geometry_msgs.msg import PoseStamped
from control_msgs.action import GripperCommand

def plan_and_execute(
    robot,
    planning_component,
    single_plan_parameters=None,
    multi_plan_parameters=None,
    sleep_time=0.0,
):
    """Helper function to plan and execute a motion."""
    # plan to goal
    if multi_plan_parameters is not None:
        plan_result = planning_component.plan(
            multi_plan_parameters=multi_plan_parameters
        )
    elif single_plan_parameters is not None:
        plan_result = planning_component.plan(
            single_plan_parameters=single_plan_parameters
        )
    else:
        plan_result = planning_component.plan()

    # execute the plan
    if plan_result:
        robot_trajectory = plan_result.trajectory
        robot.execute(robot_trajectory, controllers=[])
    else:
        raise RuntimeError("Failed to plan trajectory")

    time.sleep(sleep_time)

class GripperClient(Node):

    def __init__(self, gripper_controller):
        super().__init__("gripper_client")
        self.gripper_action_client = ActionClient(
            self,
            GripperCommand, 
            gripper_controller,
        )
    
    def close_gripper(self):
        goal = GripperCommand.Goal()
        goal.command.position = 0.8
        goal.command.max_effort = 3.0
        self.gripper_action_client.wait_for_server()
        return self.gripper_action_client.send_goal_async(goal)

    def open_gripper(self):
        goal = GripperCommand.Goal()
        goal.command.position = 0.0
        goal.command.max_effort = 3.0
        self.gripper_action_client.wait_for_server()
        return self.gripper_action_client.send_goal_async(goal)

class FrankaTable(dm_env.Environment):
    """
    This dm_env is intended to be used in conjunction with PyQt data collection application.
    The management of ROS communication is handled by the data collection application.
    This application is intended to simplify moving the arm during camera calibration procedure.
    """

    def __init__(self, args):
        robot_ip = args.robot_ip
        use_gripper = args.use_gripper
        use_fake_hardware = args.use_fake_hardware
        
        moveit_config = (
            MoveItConfigsBuilder(robot_name="panda", package_name="franka_robotiq_moveit_config")
            .robot_description(file_path=get_package_share_directory("franka_robotiq_description") + "/urdf/robot.urdf.xacro",
                mappings={
                    "robot_ip": robot_ip,
                    "robotiq_gripper": use_gripper,
                    "use_fake_hardware": use_fake_hardware,
                    })
            .robot_description_semantic("config/panda.srdf.xacro", 
                mappings={
                    "robotiq_gripper": use_gripper,
                    })
            .trajectory_execution("config/moveit_controllers.yaml")
            .moveit_cpp(
                file_path=get_package_share_directory("panda_motion_planning_demos")
                + "/config/moveit_cpp.yaml"
            )
            .to_moveit_configs()
            ).to_dict()

        self.panda = MoveItPy(config_dict=moveit_config)
        self.panda_arm = self.panda.get_planning_component("panda_arm") 
        self.gripper_client = GripperClient(args.gripper_controller)

        self.dummy_observation = {
                                "dummy_output": np.zeros(7),                 
                                }

        self.workspace_params = None

    def reset(self) -> dm_env.TimeStep:
        return dm_env.TimeStep(
                step_type=dm_env.StepType.FIRST,
                reward=0.0,
                discount=0.0,
                observation=self.dummy_observation,
                )

    def step(self, dummy_action=np.zeros(7)) -> dm_env.TimeStep:
        """
        Samples and moves to a random pose within the calibration workspace.
        """
        # sample pose
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = "panda_link0"
        pose_msg.pose.position.x = self.starting_pose[0] + np.random.uniform(
                self.workspace_params["x_min"], 
                self.workspace_params["x_max"]
                )
        pose_msg.pose.position.y = self.starting_pose[1] + np.random.uniform(
                self.workspace_params["y_min"],
                self.workspace_params["y_max"]
                )
        pose_msg.pose.position.z = self.starting_pose[2] + np.random.uniform(
                self.workspace_params["z_min"],
                self.workspace_params["z_max"]
                )
        
        # Create a rotation object from Euler angles specifying axes of rotation
        rot_x = np.random.uniform(
                self.workspace_params["rot_x_min"],
                self.workspace_params["rot_x_max"]
                )
        rot_y = np.random.uniform(
                self.workspace_params["rot_y_min"],
                self.workspace_params["rot_y_max"]
                )
        rot_z = np.random.uniform(
                self.workspace_params["rot_z_min"],
                self.workspace_params["rot_z_max"]
                )
        rot = R.from_euler("XYZ", [rot_x, rot_y, rot_z], degrees=True)

        existing_orientation = R.from_euler("XYZ", [
            self.starting_pose[3],
            self.starting_pose[4],
            self.starting_pose[5],
        ], degrees=True)

        # Apply the rotation to the existing orientation
        new_orientation = existing_orientation * rot
        new_orientation_quat = new_orientation.as_quat()
        pose_msg.pose.orientation.x = new_orientation_quat[0]
        pose_msg.pose.orientation.y = new_orientation_quat[1]
        pose_msg.pose.orientation.z = new_orientation_quat[2]
        pose_msg.pose.orientation.w = new_orientation_quat[3]

        # move to pose
        self.panda_arm.set_start_state_to_current_state()
        self.panda_arm.set_goal_state(pose_stamped_msg=pose_msg, pose_link="panda_link8")
        plan_and_execute(self.panda, self.panda_arm, sleep_time=3.0)

        return dm_env.TimeStep(
                step_type=dm_env.StepType.MID,
                reward=0.0,
                discount=0.0,
                observation=self.dummy_observation,
                )


    # both observation and action are dummy vectors as we are not using them
    def observation_spec(self) -> Dict[str, dm_env.specs.Array]:
        return {
                "dummy_output": dm_env.specs.Array(shape=(7), dtype=np.float32),
                }

    def action_spec(self) -> dm_env.specs.Array:
        return dm_env.specs.Array(
                shape=(7,),
                dtype=np.float32,
                )

    def close(self):
        raise NotImplementedError

    def set_workspace(self, params):
        self.starting_pose = self.gripper_pose()
        self.workspace_params = params

    def gripper2base(self):
        """Get the transform from the gripper coordinate frame to the base coordinate frame"""
        self.panda_arm.set_start_state_to_current_state()
        robot_state = self.panda_arm.get_start_state()
        return robot_state.get_frame_transform("panda_link8")

    def gripper_pose(self):
        """Get the pose of the gripper"""
        self.panda_arm.set_start_state_to_current_state()
        robot_state = self.panda_arm.get_start_state()
        
        pose = robot_state.get_pose("panda_link8")

        pose_pos = np.array([
            pose.position.x,
            pose.position.y,
            pose.position.z,
        ])

        pos_euler = R.from_quat([
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ]).as_euler("xyz", degrees=True)

        return np.concatenate([pose_pos, pos_euler])

