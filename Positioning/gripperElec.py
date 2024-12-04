
from board import SCL, SDA
import busio
from adafruit_pca9685 import PCA9685
from adafruit_servokit import ServoKit
import smbus

class GripperEncoder:
    def __init__(self):
        self.DEVICE_AS5600 = 0x36 # Default device I2C address
        self.bus = smbus.SMBus(1)

    def ReadRawAngle(self): # Read angle (0-360 represented as 0-4096)
        read_bytes = self.bus.read_i2c_block_data(self.DEVICE_AS5600, 0x0C, 2)
        return (read_bytes[0]<<8) | read_bytes[1]


    def ReadMagnitude(self): # Read magnetism magnitude
        read_bytes = self.bus.read_i2c_block_data(self.DEVICE_AS5600, 0x1B, 2)
        return (read_bytes[0]<<8) | read_bytes[1]

    def ReadAngle(self):
        return self.ReadRawAngle() * 360.0 / 4096.0

class GripperMotor:
    def __init__(self):
        # Create the I2C bus interface.
        i2c_bus = busio.I2C(SCL, SDA)

        # Create a simple PCA9685 class instance.
        pca = PCA9685(i2c_bus)

        # Set the PWM frequency to 50hz.
        pca.frequency = 50

        # Create a servokit class instance with the total number of channels needed
        kit = ServoKit(channels=16)

        self.gripper_servo = kit.servo[0]
        self.gripper_servo.set_pulse_width_range(500,2500)

    def set_angle(self, angle):
        self.gripper_servo.angle = angle