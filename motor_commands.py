import numpy as np

try:
    from board import SCL, SDA
    import busio
    from adafruit_pca9685 import PCA9685
    from adafruit_servokit import ServoKit
except ModuleNotFoundError:
    print("COULD NOT FIND BOARD MODULE!!!")
from time import sleep


class MotorCommands:
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
        except NameError:
            print("COULD NOT LOAD")

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
                sleep(.01) # will need to decrease eventually

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

    def run_once(self):
        """
        Move one step in the path that is saved. Load new paths with load_path
        """

        if self.path_progress >= self.path_len:
            return False, None
        
        self.base.angle = self.angles[0, self.path_progress]
        self.shoulder.angle = self.angles[1, self.path_progress]
        self.elbow.angle = self.angles[2, self.path_progress]
        self.grip.angle = self.angles[3, self.path_progress]
        
        plan_points = self.plan_points[:, self.path_progress].reshape(3,1)
        
        self.path_progress += 1

        return True, plan_points
    
    def sort_commands(self, thetas, grip_commands):
        thetas[3,:] = grip_commands
        thetas = thetas % (2 * np.pi)
        thetas[0,:] = ((np.pi - thetas[0,:]) - np.pi/4) * 2 # fix the base angle by switching rot direction, shifting to the front slice, then handling the gear ratio
        thetas[1,:] = thetas[1,:] # make any necessary changes to the shoulder angles
        thetas[2,:] = 2*np.pi - thetas[2,:] # make any necessary changes to the elbow angles
        thetas =  thetas % (2 * np.pi)

        if any(thetas.ravel() > np.pi):
            raise ValueError('IK solution requires angles greater than the 180-degree limits of motors')
        
        return thetas
