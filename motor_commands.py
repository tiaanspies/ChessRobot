import numpy as np
from board import SCL, SDA
import busio
from adafruit_pca9685 import PCA9685
from adafruit_servokit import ServoKit
from time import sleep


class MotorCommands:
    def __init__(self):
        
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

    def go_to(self, theta, angletype='rad'):
        """moves directly to provided theta configuration"""
        if angletype == 'rad':
            angle = np.rad2deg(theta)
        elif angletype == 'deg':
            angle = theta
        else:
            raise ValueError("angletype argument must be either 'rad' or 'deg'")
        
        self.base.angle(angle[0])
        self.shoulder.angle(angle[1])
        self.elbow.angle(angle[2])
        self.grip.angle(angle[3])

    def run(self, thetas, angletype='rad'):
        """runs the full set of theta commands"""
        if angletype == 'rad':
            angle = np.rad2deg(theta)
        elif angletype == 'deg':
            angle = theta
        else:
            raise ValueError("angletype argument must be either 'rad' or 'deg'")
        
        try:
            for theta in thetas.T:
                angle = np.rad2deg(theta)
                self.base.angle(angle[0])
                self.shoulder.angle(angle[1])
                self.elbow.angle(angle[2])
                self.grip.angle(angle[3])
                sleep(1) # will need to decrease eventually
        except KeyboardInterrupt:
            pass # TODO: make sure this means gripper is open