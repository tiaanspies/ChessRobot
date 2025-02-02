from IK_Solvers.traditional import MotionPlanner
from Positioning.motor_commands import MotorCommandsSerial
from IK_Solvers.traditional import MotionPlanner
from Chessboard_detection import Aruco
from Camera import Camera_Manager

import numpy as np

class Robot:
    def __init__(self):
        self.init_aruco_tracker()
        self.init_camera()
        self.init_motion_planner()
        self.init_motor_commands()

    def init_aruco_tracker(self):
        """
        Initialize Aurco tracker and load parameters for patterns being used.
        """
        #Initialize the aruco tracker
        self.aruco_tracker = Aruco.ArucoTracker()

        # generate new pattern and save
        self.aruco_tracker.load_marker_pattern_positions(22, 30, 20, 15)

    def init_camera(self):
        # create camera object
        self.cam = Camera_Manager.RPiCamera(loadSavedFirst=False, storeImgHist=False)

    def init_motion_planner(self):
        self.motion_planner = MotionPlanner()

    def init_motor_commands(self):
        self.motor_commands = MotorCommandsSerial()

    def get_rcs_pos_aruco(self):
        """
        Returns position of the gripper control point in the robot coordinate system.

        Camera is used to locate position.
        """
        _, image = self.cam.read()
        camera_matrix, dist_matrix = self.cam.camera_matrix, self.cam.dist_matrix

        ccs_current_pos = self.aruco_tracker.estimate_camera_pose(image, camera_matrix, dist_matrix)
        print("ccs_current_pos: ", ccs_current_pos)
        ccs_control_pt_pos = self.motion_planner.camera_to_control_pt_pos(ccs_current_pos)
        rcs_control_pt_pos = self.motion_planner.ccs_to_rcs(ccs_control_pt_pos)

        return rcs_control_pt_pos
    
    def move_to_single(self, pos_xyz, gripper_state, apply_compensation):
        """
        Move robot to position in robot coordinate system.

        Parameters:
        pos_xyz (np.array): position to move to in robot coordinate system
        gripper_state: defined in motor commands
        apply_compensation (bool): whether to apply position compensation
        """
        thetas = self.motion_planner.inverse_kinematics(pos_xyz, apply_compensation)
        self.motor_commands.filter_go_to(thetas, np.array([gripper_state]))