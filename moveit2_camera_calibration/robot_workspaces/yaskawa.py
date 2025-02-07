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

from moveit.planning import MoveItPy, PlanRequestParameters, MultiPipelinePlanRequestParameters
from moveit_configs_utils import MoveItConfigsBuilder
from ament_index_python.packages import get_package_share_directory

from geometry_msgs.msg import PoseStamped, Pose
from control_msgs.action import GripperCommand
from industrial_msgs.msg import RobotStatus
from motoros2_interfaces.srv import StartTrajMode, ResetError
from std_srvs.srv import Trigger

from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive

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


class ControllerClient(Node):
    def __init__(self):
        super().__init__("yc1000_client")
        
        # topic for robot status
        self._robot_status = self.create_subscription(
            RobotStatus, 
            "/robot_status", 
            self.robot_status_callback, 
        )

        # clear error
        self.reset_error = self.create_client(
            ResetError, 
            '/reset_error', 
            )

        # start trajectory mode
        self.start_trajectory_mode = self.create_client(
            StartTrajMode,
            '/start_traj_mode',
        )

        # stop trajectory mode
        self.stop_trajectory_mode = self.create_client(
            Trigger,
            '/stop_traj_mode',
        )

    def robot_status_callback(self, msg):
        self._robot_status = msg

    def prerun_checks(self):
        # check for errors if they exist return
        if self._robot_status.in_error.val != 0:
            return False

        # check if robot is in motion
        elif self._robot_status.in_motion.val != 0:
            return False
        
        return True

    def start_traj_mode(self):

        if self.prerun_checks() == False:
            return
        else:
            req = StartTrajMode.Request()
            future = self.start_trajectory_mode.call_async(req)
            result = rclpy.spin_until_future_complete(self, future)
            return result

    def stop_traj_mode(self):
        req = Trigger.Request()
        future = self.stop_trajectory_mode.call_async(req)
        rclpy.spin_until_future_complete(self, future)

    def close(self):
        self.destroy_node()
        rclpy.shutdown()


class Yaskawa(dm_env.Environment):
    """
    This dm_env is intended to be used in conjunction with PyQt data collection application.
    The management of ROS communication is handled by the data collection application.
    This application is intended to simplify moving the arm during camera calibration procedure.
    """

    def __init__(self, args):
        
        moveit_config = (
        MoveItConfigsBuilder(
            robot_name="gp88",
        )
        .robot_description(
            file_path=get_package_share_directory("gp88_description") + "/urdf/gp88.xacro",
            mappings={"use_fake_hardware": "false", "attach_tool": "false"}
        )
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .moveit_cpp(
                    file_path=get_package_share_directory("gp88_moveit_config") + "/config/moveit_cpp.yaml"
                )
        .to_moveit_configs()
        ).to_dict()


        self.gp88 = MoveItPy(config_dict=moveit_config)
        self.planning_scene_monitor = self.gp88.get_planning_scene_monitor()
        self.gp88_arm = self.gp88.get_planning_component("arm") 
        self.controller_client = ControllerClient()

         # add ground plane to planning scene
        with self.planning_scene_monitor.read_write() as scene:
            collision_object = CollisionObject()
            collision_object.header.frame_id = "base_link"
            collision_object.id = "ground_plane"

            box_pose = Pose()
            box_pose.position.x = 0.0
            box_pose.position.y = 0.0
            box_pose.position.z = 0.0

            box = SolidPrimitive()
            box.type = SolidPrimitive.BOX
            box.dimensions = [5.0, 5.0, 0.001]

            collision_object.primitives.append(box)
            collision_object.primitive_poses.append(box_pose)
            collision_object.operation = CollisionObject.ADD

            scene.apply_collision_object(collision_object)
        
            # finally handle the allowed collisions for the object
            scene.allowed_collision_matrix.set_entry("ground_plane", "base_link", True)
            scene.current_state.update()  # Important to ensure the scene is updated

        self.dummy_observation = {
                                "dummy_output": np.zeros(7),                 
                                }

        self.workspace_params = {
            "x_min": 0.88,
            "x_max": 1.5,
            "y_min": -0.8,
            "y_max": 2.4,
            "z_min": 0.0, # conservative for now
            "z_max": 0.89,
        }


    def reset(self) -> dm_env.TimeStep:
        self.controller_client.start_traj_mode()
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
        multi_pipeline_plan_request_params = MultiPipelinePlanRequestParameters(
            self.gp88, ["pilz_lin", "pilz_ptp", "ompl_rrtc"]
        )
        
        # perform prerun check
        if self.controller_client.prerun_checks() == False:
            self.controller_client.stop_traj_mode()
            return dm_env.TimeStep(
                step_type=dm_env.StepType.LAST,
                reward=0.0,
                discount=0.0,
                observation=self.dummy_observation,
                )

        # sample pose
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = "base_link"
        pose_msg.pose.position.x = np.random.uniform(
                self.workspace_params["x_min"], 
                self.workspace_params["x_max"]
                )
        pose_msg.pose.position.y = np.random.uniform(
                self.workspace_params["y_min"],
                self.workspace_params["y_max"]
                )
        pose_msg.pose.position.z = np.random.uniform(
                self.workspace_params["z_min"],
                self.workspace_params["z_max"]
                )
        
        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.707
        pose_msg.pose.orientation.z = 0.0
        pose_msg.pose.orientation.w = 0.707

        # move to pose
        self.gp88_arm.set_start_state_to_current_state()
        self.gp88_arm.set_goal_state(pose_stamped_msg=pose_msg, pose_link="link_6_t")
        plan_and_execute(self.gp88, self.gp88_arm, multi_plan_parameters=multi_pipeline_plan_request_params, sleep_time=0.5)

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
        self.controller_client.stop_traj_mode()

    def set_workspace(self, params):
        self.starting_pose = self.gripper_pose()
        self.workspace_params = params

    def gripper2base(self):
        """Get the transform from the gripper coordinate frame to the base coordinate frame"""
        self.gp88_arm.set_start_state_to_current_state()
        robot_state = self.gp88_arm.get_start_state()
        return robot_state.get_frame_transform("link_6_t")

    def gripper_pose(self):
        """Get the pose of the gripper"""
        self.gp88_arm.set_start_state_to_current_state()
        robot_state = self.gp88_arm.get_start_state()
        
        pose = robot_state.get_pose("link_6_t")

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

def main(args=None):
    rclpy.init(args=None)
    env = Yaskawa(args)
    env.reset()
    env.step()
    env.close()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()