import os
import sys
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import scipy.spatial.transform as st
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog
from PyQt6.QtGui import QPixmap, QImage, QMouseEvent
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

import cv2

from robot_workspaces.mj_transporter_franka_table import FrankaTable
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


class MainWindow(QMainWindow):
    def __init__(self, env):
        super().__init__()

        # initialize the GUI
        self.initUI()

        # start ROS image subscriber

        # environment for recording data
        self.env = env

        # GUI application parameters
        self.mode = "pick"
        self.table_height = 0.0
        self.x = 0.0
        self.y = 0.0
        self.gripper_rot_z = 0.0
        self.camera_intrinsics = None
        self.camera_extrinsics = None

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_screen = QVBoxLayout(central_widget)
        horizontal_panes = QHBoxLayout()
        left_pane = QVBoxLayout()
        right_pane = QVBoxLayout()
        right_pane.setSpacing(2)

        # Left Pane
        self.label_image = QLabel()
        left_pane.addWidget(self.label_image)
        horizontal_panes.addLayout(left_pane)


        # Right Pane
        self.camera_topic_label = QLabel("Camera Topic:")
        self.camera_topic_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.camera_topic_label.setFixedHeight(20)

        self.camera_topic_name = QLineEdit()
        self.camera_topic_name.setPlaceholderText("Enter topic name")
        self.camera_topic_name.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.camera_topic_name.setFixedHeight(20)
        self.camera_topic_name.returnPressed.connect(self.update_camera_topic)

        # add all widgets to right pane
        right_pane.addWidget(self.camera_topic_label)
        right_pane.addWidget(self.camera_topic_name)

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

    def update_image(self, rgb_img):
        # store the current image
        self.current_image = rgb_img.copy() 

        # display the image
        height, width, channel = rgb_img.shape
        bytes_per_line = channel * width
        qimg = QImage(rgb_img.data, width, height, QImage.Format(13))
        pixmap = QPixmap.fromImage(qimg)
        self.label_image.setPixmap(pixmap)

    def env_step(self):
        # map pixel coords to world coords
        ## TODO: get depth value from camera 
        depth_val = 0.4

        ## convert current pixels coordinates to camera frame coordinates
        pixel_coords = np.array([self.x, self.y])
        image_coords = np.concatenate([pixel_coords, np.ones(1)])
        camera_coords =  np.linalg.inv(self.camera_intrinsics) @ image_coords
        camera_coords *= -depth_val # negative sign due to mujoco camera convention (for debug only!)

        ## convert camera coordinates to world coordinates
        camera_coords = np.concatenate([camera_coords, np.ones(1)])
        world_coords = np.linalg.inv(self.camera_extrinsics) @ camera_coords
        world_coords = world_coords[:3] / world_coords[3]

        print("World Coordinates:", world_coords)
        
        world_coords = np.array([0.25, 0.25, 0.5]) # hardcode while debugging
        quat = R.from_euler('xyz', [0, 180, self.gripper_rot_z], degrees=True).as_quat()
        pose = np.concatenate([world_coords, quat])

        if self.mode == "pick":
            self.env.step(pose)
        else:
            self.env.step(pose)
    
    def env_reset(self):
        self.env.set_observation(self.current_image)
        self.env.reset()


def main(args=None):
    rclpy.init(args=None)
    env = FrankaTable()
    os.makedirs(os.path.join(os.path.dirname(__file__), "data"), exist_ok=True)
    app = QApplication(sys.argv)
    ex = MainWindow(env)
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
