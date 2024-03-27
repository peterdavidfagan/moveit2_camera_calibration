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

    def __init__(self):
        super().__init__("gripper_client")
        self.gripper_action_client = ActionClient(
            self,
            GripperCommand, 
            "/robotiq_position_controller/gripper_cmd"
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
    This application is intended to make data collection compatible with env_logger.
    """

    def __init__(self):
        robot_ip = "" # not applicable for fake hardware
        use_gripper = "true" 
        use_fake_hardware = "true" 
        
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
        self.gripper_client = GripperClient()

        self.mode="pick"
        self.current_observation = None

    def reset(self) -> dm_env.TimeStep:
        return dm_env.TimeStep(
                step_type=dm_env.StepType.FIRST,
                reward=0.0,
                discount=0.0,
                observation=self.current_observation,
                )

    def step(self, pose) -> dm_env.TimeStep:
        if self.mode == "pick":
            self.pick(pose)
        else:
            self.place(pose)

        return dm_env.TimeStep(
                step_type=dm_env.StepType.MID,
                reward=0.0,
                discount=0.0,
                observation=self.current_observation,
                )

    def set_observation(self, obs):
        self.current_observation = obs

    def observation_spec(self) -> Dict[str, dm_env.specs.Array]:
        return {
                #"overhead_camera/depth": dm_env.specs.Array(shape=(640,640), dtype=np.float32),
                "overhead_camera/rgb": dm_env.specs.Array(shape=(640, 640, 3), dtype=np.float32),
                }

    def action_spec(self) -> dm_env.specs.Array:
        return dm_env.specs.Array(
                shape=(7,), # [x, y, z, qx, qy, qz, qw]
                dtype=np.float32,
                )

    def close(self):
        raise NotImplementedError

    def pick(self, pose):
        pick_pose_msg = PoseStamped()
        pick_pose_msg.header.frame_id = "panda_link0"
        pick_pose_msg.pose.position.x = pose[0]
        pick_pose_msg.pose.position.y = pose[1]
        pick_pose_msg.pose.position.z = pose[2]
        pick_pose_msg.pose.orientation.x = pose[3]
        pick_pose_msg.pose.orientation.y = pose[4]
        pick_pose_msg.pose.orientation.z = pose[5]
        pick_pose_msg.pose.orientation.w = pose[6]
        
        # prepick pose
        self.panda_arm.set_start_state_to_current_state()
        pre_pick_pose_msg = deepcopy(pick_pose_msg)
        pre_pick_pose_msg.pose.position.z += 0.1
        self.panda_arm.set_goal_state(pose_stamped_msg=pre_pick_pose_msg, pose_link="panda_link8")
        plan_and_execute(self.panda, self.panda_arm, sleep_time=3.0)

        # pick pose
        self.panda_arm.set_start_state_to_current_state()
        self.panda_arm.set_goal_state(pose_stamped_msg=pick_pose_msg, pose_link="panda_link8")
        plan_and_execute(self.panda, self.panda_arm, sleep_time=3.0)

        # close gripper
        self.gripper_client.close_gripper()
        time.sleep(2.0)
        
        # raise arm
        self.panda_arm.set_start_state_to_current_state()
        pre_pick_pose_msg.pose.position.z += 0.2
        self.panda_arm.set_goal_state(pose_stamped_msg=pre_pick_pose_msg, pose_link="panda_link8")
        plan_and_execute(self.panda, self.panda_arm, sleep_time=3.0)

        self.mode = "place"

    def place(self, pose):
        place_pose_msg = PoseStamped()
        place_pose_msg.header.frame_id = "panda_link0"
        place_pose_msg.pose.position.x = pose[0]
        place_pose_msg.pose.position.y = pose[1]
        place_pose_msg.pose.position.z = pose[2]
        place_pose_msg.pose.orientation.x = pose[3]
        place_pose_msg.pose.orientation.y = pose[4]
        place_pose_msg.pose.orientation.z = pose[5]
        place_pose_msg.pose.orientation.w = pose[6]
        
        # preplace pose
        self.panda_arm.set_start_state_to_current_state()
        pre_place_pose_msg = deepcopy(place_pose_msg)
        pre_place_pose_msg.pose.position.z += 0.1
        self.panda_arm.set_goal_state(pose_stamped_msg=pre_place_pose_msg, pose_link="panda_link8")
        plan_and_execute(self.panda, self.panda_arm, sleep_time=3.0)

        # place pose
        self.panda_arm.set_start_state_to_current_state()
        self.panda_arm.set_goal_state(pose_stamped_msg=place_pose_msg, pose_link="panda_link8")
        plan_and_execute(self.panda, self.panda_arm, sleep_time=3.0)

        # open gripper
        self.gripper_client.open_gripper()
        time.sleep(2.0)
        
        # raise arm
        self.panda_arm.set_start_state_to_current_state()
        pre_place_pose_msg.pose.position.z += 0.2
        self.panda_arm.set_goal_state(pose_stamped_msg=pre_place_pose_msg, pose_link="panda_link8")
        plan_and_execute(self.panda, self.panda_arm, sleep_time=3.0)

        self.mode = "pick"
