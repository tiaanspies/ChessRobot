import numpy as np

try:
    from board import SCL, SDA
    import busio
    from adafruit_pca9685 import PCA9685
    from adafruit_servokit import ServoKit
except ModuleNotFoundError:
    print("COULD NOT FIND BOARD MODULE!!!")
from time import sleep
import logging
import datetime
from pathlib import Path
import path_directories
import Positioning.hiwonder_control as hw
import Positioning.gripperElec as gripper_elec

class MotorCommandsPWM:
    def __init__(self):
        try:
            # Create the I2C bus interface.
            i2c_bus = busio.I2C(SCL, SDA)

            # Create a simple PCA9685 class instance.
            pca = PCA9685(i2c_bus)

            # Set the PWM frequency to 50hz.
            pca.frequency = 50

            # Create a servokit class instance with the total number of channels needed
            kit = ServoKit(channels=16)

            # define shortcuts and set pin numbers for each motor
            self.base = kit.servo[1]
            self.shoulder = kit.servo[2]
            self.elbow = kit.servo[3]
            self.grip = kit.servo[4]

            # set all pulsewidth ranges
            self.base.set_pulse_width_range(500,2500)
            self.shoulder.set_pulse_width_range(500,2500)
            self.elbow.set_pulse_width_range(500,2500)
            self.grip.set_pulse_width_range(500,2500) # TODO: this one is probably different
        except NameError as e:
            logging.error(f"error creating MotorCommands Object in __init__ in motor_commands.py: {e}")

        self.OPEN = np.pi/4 # TODO: replace this with the angle needed for it to be open (in radians)
        self.CLOSED = 3*np.pi/4 # TODO: replace this with the angle needed for it to be closed (in radians)

        # variables for run_once function
        self.angles = None
        self.path_progress = 0
        self.plan_points = None

    def go_to(self, theta, angletype='rad'):
        """moves directly to provided theta configuration"""
        if angletype == 'rad':
            angle = np.rad2deg(theta)
        elif angletype == 'deg':
            angle = theta
        else:
            raise ValueError("angletype argument must be either 'rad' or 'deg'")
        
        self.base.angle = angle[0]
        self.shoulder.angle = angle[1]
        self.elbow.angle = angle[2]
        self.grip.angle = angle[3]

    def run(self, thetas, angletype='rad'):
        """runs the full set of theta commands"""
        # save data
        prefix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        logging.info("Saving recorded data.")
        np.save(Path(path_directories.RUN_PATH, prefix + "_measured.npy"), thetas)
        
        if angletype == 'rad':
            angles = np.rad2deg(thetas)
        elif angletype == 'deg':
            angles = thetas
        else:
            raise ValueError("angletype argument must be either 'rad' or 'deg'")
        
        try:
            for angle in angles.T:
                self.base.angle = angle[0]
                self.shoulder.angle = angle[1]
                self.elbow.angle = angle[2]
                self.grip.angle = angle[3]
                sleep(.3) # will need to decrease eventually

        except KeyboardInterrupt:
            self.grip.angle = np.rad2deg(self.OPEN)
            pass # TODO: make sure this means gripper is open

    def load_path(self, thetas, plan_points = None, angletype='rad'):
        """
        Load new set of angles into class, allows for run_once command to function.
        """

        # convert angles to degrees if necessary
        if angletype == 'rad':
            angles = np.rad2deg(thetas)
        elif angletype == 'deg':
            angles = thetas
        else:
            raise ValueError("angletype argument must be either 'rad' or 'deg'")
        
        # Load the plan if it is provided
        self.plan_points = plan_points
        
        self.angles = angles
        self.path_progress = 0
        self.path_len = angles.shape[1]

        self.plan_start_offset = angles.shape[1] - plan_points.shape[1]

    def run_once(self):
        """
        Move one step in the path that is saved. Load new paths with load_path
        """

        # if self.path_progress >= self.path_len:
        #     return False, None
        # 0 1 2 3 4 5 6 7 8 9 10
        # ------------>

        # check if next pos contains np.nan
        if np.any(np.isnan(self.angles[:, self.path_progress])):
            logging.debug(f"skipping {self.angles[:, self.path_progress]}")
            self.path_progress += 1
            return True, self.angles[:, self.path_progress]
        
        logging.debug(f"moving to {self.angles[:, self.path_progress]}")
        self.base.angle = self.angles[0, self.path_progress]
        self.shoulder.angle = self.angles[1, self.path_progress] 
        self.elbow.angle = self.angles[2, self.path_progress]
        self.grip.angle = self.angles[3, self.path_progress]
        
        if self.path_progress >= self.plan_start_offset:
            plan_points = self.plan_points[:, self.path_progress - self.plan_start_offset].reshape(3,1)
        else:
            plan_points = None
            
        self.path_progress += 1
        path_in_progress = self.path_progress < (self.path_len-1)

        return path_in_progress, plan_points
    
    def correct_limits(self, thetas, pos, lim_map):
        """Sets values out of limits equal to NAN"""
        thetas[:, lim_map == 1] = np.nan

        if pos is not None:
            pos[:, lim_map == 1] = np.nan

        return thetas, pos

    def filter_run(self, thetas, grip_commands):
        """Sort commands -> correct_lims -> run"""
        joint_angles, exceeds_lim = self.sort_commands(thetas, grip_commands)

        logging.info("Correcting joint angles")
        joint_angles, _ = self.correct_limits(joint_angles, None, exceeds_lim)
        
        self.run(joint_angles)

    
    def sort_commands(self, thetas, grip_commands):
        thetas[3,:] = grip_commands
        thetas = thetas % (2 * np.pi)
        thetas[0,:] = ((np.pi - thetas[0,:]) - np.pi/4) * 2 # fix the base angle by switching rot direction, shifting to the front slice, then handling the gear ratio
        thetas[1,:] = thetas[1,:] # make any necessary changes to the shoulder angles
        thetas[2,:] = 2*np.pi - thetas[2,:] # make any necessary changes to the elbow angles
        thetas =  thetas % (2 * np.pi)

        exceeds_pi = np.any(thetas > np.pi, axis=0)
        # if any(thetas.ravel() > np.pi):
        #     raveled = thetas.ravel()
        #     logging.error(f'Thetas greater than pi: {raveled[raveled > np.pi]}')
        #     raise ValueError(f'IK solution requires angles greater than the 180-degree limits of motors\n'\
        #         f'thetas: {thetas}')
        
        return thetas, exceeds_pi

class MotorCommandsSerial:
    def __init__(self):
        try:
            self.motor = hw.SerialServoCtrl()
            self.gripper = gripper_elec.GripperMotor()
        except NameError as e:
            logging.error(f"error creating MotorCommands Object in __init__ in motor_commands.py: {e}")

        self.OPEN = np.pi/4 # TODO: replace this with the angle needed for it to be open (in radians)
        self.CLOSED = 3*np.pi/4 # TODO: replace this with the angle needed for it to be closed (in radians)

        # variables for run_once function
        self.angles = None
        self.path_progress = 0
        self.plan_points = None
        
        self.GRIPPER_CLOSE = 1
        self.GRIPPER_MED = 2
        self.GRIPPER_OPEN = 3


    def go_to(self, theta, angletype='rad'):
        """moves directly to provided theta configuration"""
        print(theta)
        if angletype == 'rad':
            angle = np.rad2deg(theta)
        elif angletype == 'deg':
            angle = theta
        else:
            raise ValueError("angletype argument must be either 'rad' or 'deg'")
        
        pos_dict = {
            'base': angle[0],
            'shoulder': angle[1],
            'elbow': angle[2]
        }

        self.motor.move_to_multi_angle_pos(5000, pos_dict)

    def run(self, thetas, angletype='rad'):
        """runs the full set of theta commands"""
        # save data
        prefix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        logging.info("Saving recorded data.")
        # np.save(Path(path_directories.RUN_PATH, prefix + "_measured.npy"), thetas)
        
        if angletype == 'rad':
            angles = np.rad2deg(thetas[:3, :])
            angles = np.vstack((angles, thetas[3, :]))
        elif angletype == 'deg':
            angles = thetas
        else:
            raise ValueError("angletype argument must be either 'rad' or 'deg'")
        
        self.go_to(angles[:,0], 'deg')
        for angle in angles.T:
            pos_dict = {
                'base': angle[0],
                'shoulder': angle[1],
                'elbow': angle[2]
            }

            gripper_command = angle[3]

            if gripper_command == 1:
                self.gripper.close_gripper_force()
            elif gripper_command == 2:
                self.gripper.gripper_medium_open()
            elif gripper_command == 3:
                self.gripper.open_gripper()

            self.motor.move_to_multi_angle_pos(200, pos_dict)
            # sleep(0.2)


    def load_path(self, thetas, plan_points = None, angletype='rad'):
        """
        Load new set of angles into class, allows for run_once command to function.
        """

        # convert angles to degrees if necessary
        if angletype == 'rad':
            angles = np.rad2deg(thetas)
        elif angletype == 'deg':
            angles = thetas
        else:
            raise ValueError("angletype argument must be either 'rad' or 'deg'")
        
        # Load the plan if it is provided
        self.plan_points = plan_points
        
        self.angles = angles
        self.path_progress = 0
        self.path_len = angles.shape[1]

        self.plan_start_offset = angles.shape[1] - plan_points.shape[1]

    def run_once(self, move_time=2000):
        """
        Move one step in the path that is saved. Load new paths with load_path
        """

        # if self.path_progress >= self.path_len:
        #     return False, None
        # 0 1 2 3 4 5 6 7 8 9 10
        # ------------>

        # check if next pos contains np.nan
        if np.any(np.isnan(self.angles[:, self.path_progress])):
            logging.debug(f"skipping {self.angles[:, self.path_progress]}")
            self.path_progress += 1
            return True, self.angles[:, self.path_progress]
        
        # logging.debug(f"moving to {self.angles[:, self.path_progress]}")

        pos_dict = {
            'base': self.angles[0, self.path_progress],
            'shoulder': self.angles[1, self.path_progress],
            'elbow': self.angles[2, self.path_progress]
        }

        gripper_command = self.angles[3, self.path_progress]

        if gripper_command == 1:
            self.gripper.close_gripper_force()
        elif gripper_command == 0:
            self.gripper.open_gripper()
        
        self.motor.move_to_multi_angle_pos(move_time, pos_dict)
        
        if self.path_progress >= self.plan_start_offset:
            plan_points = self.plan_points[:, self.path_progress - self.plan_start_offset].reshape(3,1)
        else:
            plan_points = None
            
        self.path_progress += 1
        path_in_progress = self.path_progress < (self.path_len-1)

        return path_in_progress, plan_points
    
    def correct_limits(self, thetas, pos, lim_map):
        """Sets values out of limits equal to NAN"""
        thetas[:, lim_map == 1] = np.nan

        if pos is not None:
            pos[:, lim_map == 1] = np.nan

        return thetas, pos

    def filter_run(self, thetas, grip_commands):
        """Sort commands -> correct_lims -> run"""
        joint_angles, exceeds_lim = self.sort_commands(thetas, grip_commands)

        logging.info("Correcting joint angles")
        joint_angles, _ = self.correct_limits(joint_angles, None, exceeds_lim)
        
        self.run(joint_angles)

    def filter_go_to(self, thetas, gripp_commands):
        """Sort commands -> correct_lims -> run"""
        joint_angles, exceeds_lim = self.sort_commands(thetas, gripp_commands)

        logging.info("Correcting joint angles")
        joint_angles, _ = self.correct_limits(joint_angles, None, exceeds_lim)
        
        self.go_to(joint_angles[:, 0])

    
    def sort_commands(self, thetas, grip_commands):
        """Add the gripper commands and adjust joint angle to be in correct range.
        Pass None to grip_commands to give default open gripper command."""
        
        thetas = thetas % (2 * np.pi)
        thetas[0,:] = ((np.pi - thetas[0,:]) - np.pi/4) * 2 # fix the base angle by switching rot direction, shifting to the front slice, then handling the gear ratio
        thetas[1,:] = thetas[1,:] # make any necessary changes to the shoulder angles
        thetas[2,:] = 2*np.pi - thetas[2,:] # make any necessary changes to the elbow angles

        if grip_commands == None:
            thetas[3,:] = self.GRIPPER_OPEN
        else:
            thetas[3,:] = grip_commands
        # thetas =  thetas % (2 * np.pi)

        exceeds_pi = np.any(thetas > 210/180*np.pi, axis=0)
        exceeds_neg_30 = np.any(thetas < -30/180*np.pi, axis=0)
        out_of_limits = exceeds_pi | exceeds_neg_30

        if np.any(out_of_limits):
            raise ValueError(f'IK solution requires angles greater than the 180-degree limits of motors\n'\
                f'thetas: {thetas[:, out_of_limits]}')        
        return thetas, exceeds_pi