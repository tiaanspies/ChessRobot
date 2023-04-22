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

    def run(self, thetas, angletype='rad', record=False):
        count = 0
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
                sleep(.1) # will need to decrease eventually
                # input(f"count: {count}. move to next pos?")
                count += 1
        except KeyboardInterrupt:
            self.grip.angle = np.rad2deg(self.OPEN)
            pass # TODO: make sure this means gripper is open
    
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
