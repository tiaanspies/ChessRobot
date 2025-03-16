
from board import SCL, SDA
import busio
from adafruit_pca9685 import PCA9685
from adafruit_servokit import ServoKit
import smbus
import yaml
import time

class GripperEncoder:
    def __init__(self):
        self.DEVICE_AS5600 = 0x36 # Default device I2C address
        self.bus = smbus.SMBus(1)

        config = yaml.safe_load(open("config/servo_config.yml", 'r'))
        self.config = config["gripper"]

    def readRawAngle(self): # Read angle (0-360 represented as 0-4096)
        read_bytes = self.bus.read_i2c_block_data(self.DEVICE_AS5600, 0x0C, 2)
        return (read_bytes[0]<<8) | read_bytes[1]


    def readMagnitude(self): # Read magnetism magnitude
        read_bytes = self.bus.read_i2c_block_data(self.DEVICE_AS5600, 0x1B, 2)
        return (read_bytes[0]<<8) | read_bytes[1]

    def readAngleUncorrected(self):
        return self.readRawAngle() * 360.0 / 4096.0
    
    def readAngle(self):
        angle = self.readAngleUncorrected()
        return angle + self.config["offset_enc_to_motor"]

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

        # ELec settings
        self.gripper_servo = kit.servo[12]
        self.gripper_servo.set_pulse_width_range(500,2500)

        # Config
        config = yaml.safe_load(open("config/servo_config.yml", 'r'))
        self.gripper_config = config["gripper"]

        self.controller_modes = {"ANGLE":1, "FORCE":2}
        self.controller_mode = self.controller_modes["ANGLE"]

        # Encoder
        self.gripperEncoder = GripperEncoder()

    def __del__(self):
        self.gripper_servo.angle = self.gripper_config["angle_init"]
        time.sleep(0.5)

    def set_angle(self, angle):
        """Set angle of gripper. Angle in degrees."""

        if angle < self.gripper_config["angle_min_motor"]:
            angle = self.gripper_config["angle_min_motor"]
        elif angle > self.gripper_config["angle_max_motor"]:
            angle = self.gripper_config["angle_max_motor"]

        self.gripper_servo.angle = angle

        #check that gripper is within threshold of open
        count = 0
        while abs(self.gripperEncoder.readAngle() - self.gripper_config["angle_init"]) > self.gripper_config["angle_threshold"]:
            time.sleep(0.1)
            count += 1

            if count > 10:
                print("Gripper did not reach desired angle.")
                break

    def open_gripper(self):
        """Open gripper."""
        self.set_controller_mode("ANGLE")
        self.set_angle(self.gripper_config["angle_init"])

    def gripper_medium_open(self):
        """Open gripper to medium position."""
        self.set_controller_mode("ANGLE")
        self.set_angle(self.gripper_config["angle_medium_open"])

    def close_gripper_force(self):
        """ CLose gripper with force control."""
        self.set_controller_mode("FORCE")
        self.set_force(self.gripper_config["pickup_force"])

    def save_config(self):
        config = yaml.safe_load(open("config/servo_config.yml", 'r'))
        config["gripper"] = self.gripper_config
        with open("config/servo_config.yml", 'w') as file:
            yaml.dump(config, file)

    def set_controller_mode(self, mode):
        """Change control mode. Options: ANGLE, FORCE"""
        self.controller_mode = self.controller_modes[mode]

        if self.controller_mode == self.controller_modes["FORCE"]:
            self.set_force(0)

    def set_force(self, force):
        """Set force to apply to gripper."""
        
        if force > self.gripper_config["force_max"]:
            force = self.gripper_config["force_max"]

        current_angle = self.gripperEncoder.readAngle()
        current_force = self.gripper_servo.angle - current_angle

        # update at least once.
        self.gripper_servo.angle = self.gripper_servo.angle + min(force-current_force, self.gripper_config["force_step_dist"])

        # Update force
        time.sleep(self.gripper_config["force_time_step"])
        current_angle = self.gripperEncoder.readAngle()
        current_force = self.gripper_servo.angle - current_angle

        while abs(force - current_force) > self.gripper_config["force_threshold"]:
            self.gripper_servo.angle = self.gripper_servo.angle + min(force-current_force, self.gripper_config["force_step_dist"])

            time.sleep(self.gripper_config["force_time_step"])
            current_angle = self.gripperEncoder.readAngle()
            current_force = self.gripper_servo.angle - current_angle

            print(f"Force: {current_force:.2f}, Angle: {current_angle:.2f}, Motor angle: {self.gripper_servo.angle:.2f}")

def main():
    gripper = GripperMotor()
    gripper.open_gripper()
    gripper.set_controller_mode("FORCE")
    gripper.set_force(40)
    time.sleep(2)
    gripper.set_force(0)

if __name__ == "__main__":
    main()