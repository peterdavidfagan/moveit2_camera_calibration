import os
import time
import yaml
import sys
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import scipy.spatial.transform as st
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QRadioButton, QButtonGroup
from PyQt6.QtGui import QPixmap, QImage, QMouseEvent
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo

import cv2

from robot_workspaces.mj_calibration_franka_table import FrankaTable
import envlogger


class ImageSubscriber(QThread):
    new_image = pyqtSignal(object)

    def __init__(self, camera_topic):
        super().__init__()
        self.cv_bridge = CvBridge()
        self.camera_topic = camera_topic

    def run(self):
        node = rclpy.create_node('image_subscriber')
        subscription = node.create_subscription(Image, self.camera_topic, self.image_callback, 10)
        rclpy.spin(node)

    def image_callback(self, msg):
        rgb_img = self.cv_bridge.imgmsg_to_cv2(msg, "rgb8")
        self.new_image.emit(rgb_img)


class CameraInfoSubscriber(QThread):
    new_camera_info = pyqtSignal(object)

    def __init__(self, camera_info_topic):
        super().__init__()
        self.camera_info_topic = camera_info_topic

    def run(self):
        node = rclpy.create_node('camera_info_subscriber')
        subscription = node.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)
        rclpy.spin(node)

    def camera_info_callback(self, msg):
        self.new_camera_info.emit(msg)


class MainWindow(QMainWindow):
    def __init__(self, env):
        super().__init__()

        # environment for execution of task
        self.env = env

        # GUI application parameters
        self.calibration_status = "None"
        self.current_image = None
        self.camer_info = None
        
        # initialize the GUI
        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_screen = QVBoxLayout(central_widget)
        horizontal_panes = QHBoxLayout()
        left_pane = QVBoxLayout()
        right_pane = QVBoxLayout()
        right_pane.setAlignment(Qt.AlignmentFlag.AlignTop)
        right_pane.setSpacing(20)

        # Left Pane
        self.label_image = QLabel()
        left_pane.addWidget(self.label_image)
        horizontal_panes.addLayout(left_pane)

        # Right Pane
        self.camera_topic_label = QLabel("Camera Topics:")
        self.camera_topic_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.camera_topic_label.setFixedHeight(20)

        self.camera_topic_name = QLineEdit()
        self.camera_topic_name.setPlaceholderText("Enter image topic")
        self.camera_topic_name.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.camera_topic_name.setFixedHeight(20)
        self.camera_topic_name.returnPressed.connect(self.update_camera_topic)

        self.camera_info_topic_name = QLineEdit()
        self.camera_info_topic_name.setPlaceholderText("Enter camera info topic")
        self.camera_info_topic_name.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.camera_info_topic_name.setFixedHeight(20)
        self.camera_info_topic_name.returnPressed.connect(self.update_camera_info_topic)

        self.upload_aruco_parameters_button = QPushButton("Upload Aruco Parameters")
        self.upload_aruco_parameters_button.setFixedHeight(20)
        self.upload_aruco_parameters_button.clicked.connect(self.upload_aruco_parameters)

        # checkbox for calibration type
        self.calibration_type_label = QLabel("Calibration Type:")
        self.calibration_type_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.calibration_type_label.setFixedHeight(20)
        
        self.calibration_type_button_group = QButtonGroup()
        self.calibration_type_eye_in_hand = QRadioButton("Eye-in-Hand")
        self.calibration_type_hand_eye = QRadioButton("Hand-eye")
        self.calibration_type_button_group.addButton(self.calibration_type_eye_in_hand)
        self.calibration_type_button_group.addButton(self.calibration_type_hand_eye)   

        self.start_calibration_button = QPushButton("Start Calibration")
        self.start_calibration_button.setFixedHeight(20)
        self.start_calibration_button.clicked.connect(self.start_calibration)

        self.calibration_status_label = QLabel(f"Calibration Status: {self.calibration_status}")
        self.calibration_status_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.calibration_status_label.setFixedHeight(20)

        # add all widgets to right pane
        right_pane.addWidget(self.camera_topic_label)
        right_pane.addWidget(self.camera_topic_name)
        right_pane.addWidget(self.camera_info_topic_name)
        right_pane.addWidget(self.upload_aruco_parameters_button)
        right_pane.addWidget(self.calibration_type_label)
        right_pane.addWidget(self.calibration_type_eye_in_hand)
        right_pane.addWidget(self.calibration_type_hand_eye)
        right_pane.addWidget(self.start_calibration_button)
        right_pane.addWidget(self.calibration_status_label)

        # add to main screen
        horizontal_panes.addLayout(right_pane)
        main_screen.addLayout(horizontal_panes)
        central_widget.setLayout(main_screen)

        self.setWindowTitle('Camera Calibration')
        self.show()

    def update_camera_topic(self):
        self.image_subscriber = ImageSubscriber(self.camera_topic_name.text())
        self.image_subscriber.new_image.connect(self.update_image)
        self.image_subscriber.start()

    def update_camera_info_topic(self):
        self.camera_info_subscriber = CameraInfoSubscriber(self.camera_info_topic_name.text())
        self.camera_info_subscriber.new_camera_info.connect(self.update_camera_info)
        self.camera_info_subscriber.start()

    def update_image(self, rgb_img):
        # store the current image
        self.current_image = rgb_img.copy() 

        # display the image
        height, width, channel = rgb_img.shape
        bytes_per_line = channel * width
        qimg = QImage(rgb_img.data, width, height, QImage.Format(13))
        pixmap = QPixmap.fromImage(qimg)
        self.label_image.setPixmap(pixmap)
    
    def upload_aruco_parameters(self):
        # get the path to the aruco parameters file
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Aruco Parameters File", "", "YAML Files (*.yaml)")
        if file_path:
            # read the yaml file
            with open(file_path, "r") as file:
                self.calib_config = yaml.load(file, Loader=yaml.FullLoader)

            self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)
            self.charuco_board = cv2.aruco.CharucoBoard_create(
                self.calib_config["charuco"]["squares_x"],
                self.calib_config["charuco"]["squares_y"],
                self.calib_config["charuco"]["square_length"],
                self.calib_config["charuco"]["marker_length"],
                self.aruco_dict
            )
            self.detector_params = cv2.aruco.DetectorParameters_create()
            self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            self.calib_flags = cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_PRINCIPAL_POINT + cv2.CALIB_FIX_FOCAL_LENGTH
            self.cv_bridge = CvBridge()
        self.env.set_workspace(self.calib_config["workspace"]) 

    def detect_charuco_board(self, image):
        """
        Detect charuco board in image

        Adapted from: https://github.com/AlexanderKhazatsky/R2D2/blob/1aa471ae35cd9b11e20cc004c15ad4c74e92605d/r2d2/calibration/calibration_utils.py#L122
        """
        # detect aruco markers
        corners, ids, rejectedImgPoints = aruco.detectMarkers(
            image, 
            self.aruco_dict, 
            parameters=self.detector_params,
        )
        
        # find undetected markers
        corners, ids, _, _ = cv2.aruco.refineDetectedMarkers(
            image,
            self.charuco_board,
            corners,
            ids,
            rejectedImgPoints,
            parameters=self.detector_params,
            cameraMatrix=np.array(self._camera_info.k).reshape(3,3),
            distCoeffs=np.array(self._camera_info.d),
            )

        # if no markers found, return
        if ids is None:
            return None, None

        # detect charuco board
        num_corners_found, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            corners, 
            ids, 
            image, 
            self.charuco_board, 
            cameraMatrix=np.array(self._camera_info.k).reshape(3,3),
            distCoeffs=np.array(self._camera_info.d),
        )

        # if no charuco board found, return
        if num_corners_found < 5:
            return None, None

        # draw detected charuco board
        image = aruco.drawDetectedCornersCharuco(
            image, charuco_corners,
        )

        return image, charuco_corners, charuco_ids, image.shape[:2]


    def calc_target_to_camera(self, readings):
        """
        Calculate target to camera transform

        Adapted from: https://github.com/AlexanderKhazatsky/R2D2/blob/1aa471ae35cd9b11e20cc004c15ad4c74e92605d/r2d2/calibration/calibration_utils.py#L164
        """
        init_corners_all = []  # Corners discovered in all images processed
        init_ids_all = []  # Aruco ids corresponding to corners discovered
        fixed_image_size = readings[0][3]

        # Proccess Readings #
        init_successes = []
        for i in range(len(readings)):
            corners, charuco_corners, charuco_ids, img_size = readings[i]
            assert img_size == fixed_image_size
            init_corners_all.append(charuco_corners)
            init_ids_all.append(charuco_ids)
            init_successes.append(i)

        # First Pass: Find Outliers #
        threshold = 10
        if len(init_successes) < threshold:
            return None

        calibration_error, cameraMatrix, distCoeffs, rvecs, tvecs, stdIntrinsics, stdExtrinsics, perViewErrors = (
            aruco.calibrateCameraCharucoExtended(
                charucoCorners=init_corners_all,
                charucoIds=init_ids_all,
                board=self.charuco_board,
                imageSize=fixed_image_size,
                flags=self.calib_flags,
                cameraMatrix=np.array(self._camera_info.k).reshape(3,3),
                distCoeffs=np.array(self._camera_info.d),
            )
        )

        # Remove Outliers #
        final_corners_all = [
                init_corners_all[i] for i in range(len(perViewErrors)) if perViewErrors[i] <= 3.0 # TODO: read from params
        ]
        final_ids_all = [
            init_ids_all[i] for i in range(len(perViewErrors)) if perViewErrors[i] <= 3.0
        ]
        final_successes = [
            init_successes[i] for i in range(len(perViewErrors)) if perViewErrors[i] <= 3.0
        ]
        if len(final_successes) < threshold:
            return None

        # Second Pass: Calculate Finalized Extrinsics #
        calibration_error, cameraMatrix, distCoeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
            charucoCorners=final_corners_all,
            charucoIds=final_ids_all,
            board=self.charuco_board,
            imageSize=fixed_image_size,
            flags=self.calib_flags,
            cameraMatrix=np.array(self._camera_info.k).reshape(3,3),
            distCoeffs=np.array(self._camera_info.d),
        )
        
        # Return Transformation #
        if calibration_error > 3.0:
            return None

        rmats = [Rotation.from_rotvec(rvec.flatten()).as_matrix() for rvec in rvecs]
        tvecs = [tvec.flatten() for tvec in tvecs]

        return rmats, tvecs, final_successes

    def run_eye_2_hand_calibration(self):
        """Calibrate third person camera to robot base"""
        # check if we have both camera info and image
        if self.current_image is None: #or self.camera_info is None:
            raise Exception("No image or camera info received yet")
        
        # collect samples
        workspace_config = self.calib_config["workspace"]

        images = []
        gripper2base_vals = []
        gripper_poses = []
        for i in range(self.calib_config["eye_to_hand_calibration"]["num_samples"]):
            self.env.step() 
            time.sleep(0.5) # required to ensure latest image is captured
            
            # capture image
            img = self.current_image.copy()
            images.append(img)

            # capture gripper pose
            print("Gripper pose: {}".format(self.env.gripper_pose()))
            gripper_poses.append(self.env.gripper_pose())

            # capture base to ee transform
            gripper2base = self.env.gripper2base()
            gripper2base_vals.append(gripper2base)
            
            # sleep
            time.sleep(self.calib_config["eye_to_hand_calibration"]["sample_delay"])
        
        # process captured images
        readings = []
        for image in images:
            readings.append(self.detect_charuco_board(image))
        
        # calculate target to camera transform
        R_target2cam, t_target2cam, successes = self.calc_target_to_camera(readings)
        
        # filter gripper2base by successes
        gripper2base_vals = [gripper2base_vals[i] for i in successes]
        R_base2gripper = [t[:3,:3].T for t in gripper2base_vals]
        t_base2gripper = [-R @ t[:3,3] for R, t in zip(R_base2gripper, gripper2base_vals)]

        # run calibration for cam2base
        rmat, pos = cv2.calibrateHandEye(
            R_gripper2base=R_base2gripper,
            t_gripper2base=t_base2gripper,
            R_target2cam=R_target2cam,
            t_target2cam=t_target2cam,
            method=4,
        )

        return rmat, pos

    def run_eye_in_hand_calibration(self):
        """Calibrate hand-mounted camera to robot gripper"""
        # check if we have both camera info and image
        if self.current_image is None: #or self.camera_info is None:
            raise Exception("No image or camera info received yet")
        
        # collect samples
        workspace_config = self.calib_config["workspace"]

        images = []
        gripper2base_vals = []
        gripper_poses = []
        for i in range(self.calib_config["eye_to_hand_calibration"]["num_samples"]):
            self.env.step() 
            time.sleep(0.5) # required to ensure latest image is captured
            
            # capture image
            img = self.current_image.copy()
            images.append(img)

            # capture gripper pose
            print("Gripper pose: {}".format(self.gripper_pose()))
            gripper_poses.append(self.env.gripper_pose())

            # capture base to ee transform
            gripper2base = self.env.gripper2base()
            gripper2base_vals.append(gripper2base)
            
            # sleep
            time.sleep(self.calib_config["eye_to_hand_calibration"]["sample_delay"])

        # process captured images
        readings = []
        for image in images:
            #self.visualize_image(image)
            readings.append(self.detect_charuco_board(image))
        
        # calculate target to camera transform
        R_target2cam, t_target2cam, successes = self.calc_target_to_camera(readings)
        
        # filter gripper2base by successes
        gripper2base_vals = [gripper2base_vals[i] for i in successes]
        R_gripper2base = [t[:3,:3] for t in gripper2base_vals]
        t_gripper2base = [t[:3,3] for t in gripper2base_vals]

        # run calibration for cam2base
        rmat, pos = cv2.calibrateHandEye(
            R_gripper2base=R_gripper2base,
            t_gripper2base=t_gripper2base,
            R_target2cam=R_target2cam,
            t_target2cam=t_target2cam,
            method=4,
        )
        
        return rmat, pos

    def start_calibration(self):
        if self.calibration_type_button_group.checkedId() == 0:
            self.run_eye_in_hand_calibration()
        else:
            self.run_eye_2_hand_calibration()


def main(args=None):
    rclpy.init(args=None)
    env = FrankaTable()
    os.makedirs(os.path.join(os.path.dirname(__file__), "data"), exist_ok=True)
    app = QApplication(sys.argv)
    ex = MainWindow(env)
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
