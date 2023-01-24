import numpy as np
from board import SCL, SDA
import busio
from adafruit_pca9685 import PCA9685
from adafruit_servokit import ServoKit
from time import sleep


class MotorCommands:
    def __init__(self, thetas):
        # Pin numbers
        self.BASE_CHANNEL = 0
        self.SHOULDER_CHANNEL = 1
        self.ELBOW_CHANNEL = 2
        self.GRIP_CHANNEL = 3

        # Create the I2C bus interface.
        i2c_bus = busio.I2C(SCL, SDA)

        # Create a simple PCA9685 class instance.
        pca = PCA9685(i2c_bus)

        # Set the PWM frequency to 50hz.
        pca.frequency = 50

        # Create a servokit class instance with the total number of channels needed
        kit = ServoKit(channels=4)

        # define shortcuts for each
        self.base = kit.servo[self.BASE_CHANNEL]
        self.shoulder = kit.servo[self.SHOULDER_CHANNEL]
        self.elbow = kit.servo[self.ELBOW_CHANNEL]
        self.grip = kit.servo[self.GRIP_CHANNEL]

    def go_to(self, theta):
        """moves directly to provided theta configuration"""
        angle = np.rad2deg(theta)
        self.base.angle(angle[0])
        self.shoulder.angle(angle[1])
        self.elbow.angle(angle[2])
        self.grip.angle(angle[3])

    def run(self, thetas):
        """runs the full set of theta commands"""
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