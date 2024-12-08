import serial
import time
import yaml

# COMMANDS supported by dev board.
COMMAND_MOVE_MULTI = 3
COMMAND_READ_POS_MULTI = 21

class SerialServoCtrl:
    def __init__(self, config_file="config/servo_config.yml"):
        self.serial_con = serial.Serial('/dev/ttyS0', 9600, timeout=1)
        self.config_file = config_file

        # Load configuration from YAML file
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

        # Set min and max values for servo angles from config
        self.servo_config = config   

    def __del__(self):
        self.serial_con.close()

    def send_command(self, ser, command, parameters):
        """Create packet to send to serial servo master.
        Packets always contain 2 bytes of 0x55, the length that will be transmitted (excluding 0x55 bytes).
        then the command and its respective parameters.
        """
        length = 2 + len(parameters)
        packet = [0x55, 0x55, length, command] + parameters
        ser.write(bytearray(packet))

    def move_to_multi_serial_pos(self, move_time, motor_ids, positions):
        """
        Input: - time over which to complete the move (milli seconds)
        - List of motor IDs
        - List of corresponding positions to move to
        Output: None, will send serial command to serial dev board to complete the move.
        """
        # make sure min move time is 200ms
        move_time = max(200, move_time)

        # clamp position values between 0 and 1000
        for pos in positions:
            pos = max(0, pos)
            pos = min(1000, pos)

        num_servos = len(motor_ids) # Parameter 1
        time_hex_low = move_time & 0xFF # Parameter 2 (Low 8 bits of time)
        time_hex_high = (move_time >> 8) & 0xFF # parameter 3 (High 8 bits of time)

        position_parameters = [[id, position & 0xFF, (position >> 8) & 0xFF] for id, position in zip(motor_ids, positions)]
        flat_position = [x for xs in position_parameters for x in xs] # Flatten the list
        
        parameters = [num_servos, time_hex_low, time_hex_high] + flat_position

        self.send_command(self.serial_con, COMMAND_MOVE_MULTI, parameters)

    def move_to_multi_angle_pos(self, move_time, pos_req_dict):
        """
        Input:  - move time: time over which to complete the move (millseconds)
                - pos_req_dict: Dict of axis name and angle to move to.

        output: None. Sends serial data over uart to move to tthe selected location/        
        """
        ids = []
        serial_positions = []

        for axis in pos_req_dict:
            ids.append(self.servo_config[axis]["servo_id"])

            angle_req = pos_req_dict[axis]
            if angle_req < self.servo_config[axis]["angle_min"]:
                angle_req = self.servo_config[axis]["angle_min"]
                print(f"Requested angle for {axis} is below minimum. Setting to minimum.")

            if angle_req > self.servo_config[axis]["angle_max"]:
                angle_req = self.servo_config[axis]["angle_max"]
                print(f"Requested angle for {axis} is above maximum. Setting to maximum.")

            serial_position = self.interpolate_angles(angle_req, self.servo_config[axis])
            serial_positions.append(serial_position)

        self.move_to_multi_serial_pos(move_time, ids, serial_positions)

    def interpolate_angles(self, target_angle, limit_dict):
        serial_pos_range = (limit_dict["serial_pos_max"]-limit_dict["serial_pos_min"])
        angle_pos_range = limit_dict["servo_range_max"]-limit_dict["servo_range_min"]
        serial_angle = target_angle/angle_pos_range*serial_pos_range
        return round(serial_angle)

    def read_pos_multi(self, servo_ids):
        """
        Input: Array of servo motor positions to read.
        Output: Dict of servo positions of the servos (integers 0-1000)
        """
        num_servos = len(servo_ids) + 3
        parameters = [num_servos] + servo_ids

        self.send_command(self.serial_con, COMMAND_READ_POS_MULTI, parameters)

        response = self.serial_con.read_all()

        if len(response) < 5:
            return None  # Invalid response

        if response[0] != 0x55 or response[1] != 0x55:
            return None  # Invalid header

        num_servos_returned = response[4]
        positions = {}

        index = 5
        for _ in range(num_servos_returned):
            servo_id = response[index]
            pos_low = response[index + 1]
            pos_high = response[index + 2]
            position = (pos_high << 8) | pos_low
            positions[servo_id] = position
            index += 3

        return positions

    def calibrate_home_positions(self):
        calib_pos = {}
        for axis in ["base", "shoulder", "elbow"]:

            bool_save_axis = input(f"Do you want to save a new calibration position for {axis} joint? Y/N")
            if bool_save_axis != "Y":
                continue

            input(f"Move {axis} to {self.servo_config[axis]['angle_min']} degrees. Press enter when complete.")
            serial_position = self.read_pos_multi([self.servo_config[axis]["servo_id"]])

            calib_pos[axis]["servo_pos_min"] = serial_position

            input(f"Move {axis} to {self.servo_config[axis]['angle_max']} degrees. Press enter when complete.")
            serial_position = self.read_pos_multi([self.servo_config[axis]["servo_id"]])

            calib_pos[axis]["servo_pos_max"] = serial_position

        self.save_calibration_to_file(calib_pos)

    def save_calibration_to_file(self, calib_pos):

        for axis in calib_pos:
            self.servo_config[axis]["servo_pos_min"] = calib_pos[axis]["servo_pos_min"]
            self.servo_config[axis]["servo_pos_max"] = calib_pos[axis]["servo_pos_max"]

        with open(self.config_file, 'w') as file:
            yaml.safe_dump(self.servo_config, file)

# move_time = 3000
# motor_ids = [5, 6, 7]
# positions = [600, 1000, 500]
# positions = [400, 1000, 1000]
# position_params = parameters_for_position(move_time, motor_ids, positions)

# send_command(ser, 3, position_params)  # Send the command
