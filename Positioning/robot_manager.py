from IK_Solvers.traditional import MotionPlanner
from Positioning.motor_commands import MotorCommandsSerial
from Chessboard_detection import Aruco, Chess_Vision
from Camera import Camera_Manager
import yaml, os
import numpy as np

class Robot:
    def __init__(self):
        self.init_aruco_tracker()
        self.init_camera()
        self.init_motion_planner()
        self.init_motor_commands()
        self.load_configs()

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
        dir_path = os.path.dirname(os.path.realpath(__file__))
        abs_path = dir_path + "/Chessboard_detection/TestImages/Temp"
        self.cam = Camera_Manager.RPiCamera(abs_path,loadSavedFirst=False, storeImgHist=False)

        if not self.cam.isOpened():
            raise("Cannot open camera.")

    def init_motion_planner(self):
        self.motion_planner = MotionPlanner()

    def init_motor_commands(self):
        self.motor_commands = MotorCommandsSerial()

    def load_configs(self):
        """
        Load yaml config files.
        """

        self.config_kinematics = yaml.safe_load(open("config/kinematics.yml"))["IK_CONFIG"]

    def get_rcs_pos_aruco(self):
        """
        Returns position of the gripper control point in the robot coordinate system.

        Camera is used to locate position.
        """
        _, image = self.cam.read()
        camera_matrix, dist_matrix = self.cam.camera_matrix, self.cam.dist_matrix

        ccs_current_pos = self.aruco_tracker.estimate_camera_pose(image, camera_matrix, dist_matrix)
        ccs_control_pt_pos = self.motion_planner.camera_to_control_pt_pos(ccs_current_pos)
        rcs_control_pt_pos = self.motion_planner.ccs_to_rcs(ccs_control_pt_pos)

        return rcs_control_pt_pos
    
    def move_to_single(self, pos_xyz, gripper_state, apply_compensation):
        """
        Move robot to a single position in robot coordinate system.

        Parameters:
        pos_xyz (np.array): position to move to in robot coordinate system [3x1] shape
        gripper_state: defined in motor commands
        apply_compensation (bool): whether to apply position compensation
        """
        thetas = self.motion_planner.inverse_kinematics(pos_xyz, apply_compensation)
        self.motor_commands.filter_go_to(thetas, np.array([gripper_state]))

    def move_to_path(self, path_xyz, gripper_commands, apply_compensation):
        """
        Move robot along a path in robot coordinate system.

        Parameters:
        path_xyz (np.array): path to move along in robot coordinate system [Nx3] shape
        gripper_commands: defined in motor commands
        apply_compensation (bool): whether to apply position compensation
        """
        joint_angles = self.motion_planner.inverse_kinematics(path_xyz, apply_compensation) # convert waypoints to joint angles
        self.motor_commands.filter_run(joint_angles, gripper_commands)
    
    def move_home(self):
        """
        Move robot to home position.
        """
        base = self.config_kinematics["home_position_joint_angles"]["base"]
        shoulder = self.config_kinematics["home_position_joint_angles"]["shoulder"]
        elbow = self.config_kinematics["home_position_joint_angles"]["elbow"]
        angles = np.array([base, shoulder, elbow]).reshape(3,1)

        self.motor_commands.filter_go_to(angles, self.motor_commands.GRIPPER_OPEN)
