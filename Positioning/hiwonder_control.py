import serial
import time
import yaml

# COMMANDS supported by dev board.
COMMAND_MOVE_MULTI = 3
COMMAND_READ_POS_MULTI = 21
COMMAND_READ_VOLTAGE = 15

class SerialServoCtrl:
    def __init__(self, config_file="config/servo_config.yml"):
        self.serial_con = serial.Serial('/dev/ttyS0', 9600, timeout=1)
        self.config_file = config_file

        # Load configuration from YAML file
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

        # Set min and max values for servo angles from config
        self.servo_config = config

        self.config_ids = {}
        for axis in self.servo_config:
            if axis == "general":
                continue

            self.config_ids[self.servo_config[axis]["servo_id"]] = axis

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

        assert self.voltage_ok(), "Voltage too low to move servos"

        # make sure min move time is 200ms
        move_time = max(self.servo_config["general"]["min_move_time_ms"], move_time)

        # clamp position values between 0 and 1000
        for id, pos in zip(motor_ids, positions):
            axis = self.config_ids[id]
            pos = max(self.servo_config[axis]["serial_pos_min"], pos)
            pos = min(self.servo_config[axis]["serial_pos_max"], pos)

        num_servos = len(motor_ids) # Parameter 1
        time_hex_low = move_time & 0xFF # Parameter 2 (Low 8 bits of time)
        time_hex_high = (move_time >> 8) & 0xFF # parameter 3 (High 8 bits of time)

        position_parameters = [[id, position & 0xFF, (position >> 8) & 0xFF] for id, position in zip(motor_ids, positions)]
        flat_position = [x for xs in position_parameters for x in xs] # Flatten the list
        
        parameters = [num_servos, time_hex_low, time_hex_high] + flat_position

        self.send_command(self.serial_con, COMMAND_MOVE_MULTI, parameters)
        time.sleep(move_time/1000)

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
            
            if self.servo_config[axis]["reverse"]:
                angle_req = self.servo_config[axis]["zero_offset"] - pos_req_dict[axis]
            else:
                angle_req = self.servo_config[axis]["zero_offset"] + pos_req_dict[axis]

            serial_position = self.interpolate_angles(angle_req, self.servo_config[axis])
            serial_positions.append(serial_position)

        self.move_to_multi_serial_pos(move_time, ids, serial_positions)
        

    def interpolate_angles(self, target_angle, limit_dict):
        serial_pos_range = limit_dict["serial_pos_max"]-limit_dict["serial_pos_min"]
        angle_pos_range = limit_dict["servo_range_max"]-limit_dict["servo_range_min"]

        target_angle = min(limit_dict["angle_max_overtravel"], target_angle)
        target_angle = max(limit_dict["angle_min_overtravel"], target_angle)
        
        serial_pos = target_angle/angle_pos_range*serial_pos_range
        return round(serial_pos)

    def read_pos_multi(self, servo_ids):
        """
        Input: Array of servo motor positions to read.
        Output: Dict of servo positions of the servos (integers 0-1000)
        """
        num_servos = len(servo_ids)
        parameters = [num_servos] + servo_ids
    
        self.send_command(self.serial_con, COMMAND_READ_POS_MULTI, parameters)

        # 5 + x*3 bytes are returned, where x is the number of servos
        response = self.serial_con.read(5)

        if len(response) < 5:
            print(f"Invalid response received from servo master: {response}")
            return None  # Invalid response

        if response[0] != 0x55 or response[1] != 0x55:
            print(f"Invalid header received from servo master: {response[0:2]}")
            return None  # Invalid header

        num_servos_returned = response[4]

        response = self.serial_con.read(3 * num_servos_returned)
        positions = {}

        index = 0
        for _ in range(num_servos_returned):
            servo_id = response[index]
            pos_low = response[index + 1]
            pos_high = response[index + 2]
            position = (pos_high << 8) | pos_low
            positions[servo_id] = position
            index += 3
        
        return positions

    def voltage_ok(self):
        """
        Input: None
        Output: Bool, True if voltage is within limits, False otherwise
        """
        voltage = self.read_voltage()
        return voltage > self.servo_config["general"]["min_voltage"] 
    
    def read_voltage(self):
        """
        Input: None
        Output: Voltage (V) input to the servo master board.
        """
        parameters = []
    
        self.send_command(self.serial_con, COMMAND_READ_VOLTAGE, parameters)

        # 5 + x*3 bytes are returned, where x is the number of servos
        response = self.serial_con.read(6)

        if len(response) < 6:
            print(f"Invalid response received from servo master: {response}")
            return 0  # Invalid response

        if response[0] != 0x55 or response[1] != 0x55:
            print(f"Invalid header received from servo master: {response[0:2]}")
            return None  # Invalid header

        pos_low = response[4]
        pos_high = response[5]
        voltage = (pos_high << 8) | pos_low
        
        return float(voltage)/1000.0
    
    def interpolate_serial(self, serial_pos, limit_dict):
        """
        Input: Serial position of servo motor
        Output: Angle in degrees
        """
        serial_pos_range = (limit_dict["serial_pos_max"]-limit_dict["serial_pos_min"])
        angle_pos_range = limit_dict["servo_range_max"]-limit_dict["servo_range_min"]
        return round(serial_pos/serial_pos_range*angle_pos_range)
    
    def read_pos_multi_angle_raw(self, servo_ids):
        """
        Input: Array of servo motor positions to read.
        Output: Dict of servo positions in degrees.
        """
        positions = self.read_pos_multi(servo_ids)
        angle_positions = {}

        for servo_id in positions:
            servo_config = self.servo_config[self.config_ids[servo_id]]
            angle_positions[servo_id] = self.interpolate_serial(positions[servo_id], servo_config)

        return angle_positions
    
    def read_pos_angle_raw(self, servo_id):
        """
        Input: Servo motor position to read.
        Output: Position of servo motor in degrees.
        """
        position = self.read_pos_multi_angle_raw([servo_id])
        return position[servo_id]

    def read_pos_multi_angle(self, input) -> dict[str, int]:
        """ Read Position of servo motors. Add zero offset and reverse direction if needed."""

        if isinstance(input[0], str):
            servo_ids = [self.servo_config[axis]["servo_id"] for axis in input]
        elif isinstance(input[0], int):
            servo_ids = input

        positions = self.read_pos_multi_angle_raw(servo_ids)
        angle_positions = {}

        for servo_id in positions:
            axis = self.config_ids[servo_id]
            servo_config = self.servo_config[axis]
            if servo_config["reverse"]:
                angle_positions[axis] = servo_config["zero_offset"] - positions[servo_id]
            else:
                angle_positions[axis] = servo_config["zero_offset"] + positions[servo_id]

        return angle_positions
    
    def update_config(self, axis, variable, value):
        """Clamp values within range for that axis"""

        value_new = min(self.servo_config[axis]["servo_range_max"], value)
        value_new = max(self.servo_config[axis]["servo_range_min"], value_new)

        self.servo_config[axis][variable] = value_new

    def calibrate_home_positions(self):

        for axis in ["base", "shoulder", "elbow"]:

            bool_save_axis = input(f"Do you want to save a new calibration position for {axis} joint? Y/N")
            if bool_save_axis != "Y":
                continue
            
            ## Calibrate direction
            input(f"Move {axis} closest to 0 degrees. This is to detect if direction is reversed. Press enter when complete.\n")
            serial_position = self.read_pos_angle_raw(self.servo_config[axis]["servo_id"])
            halfway_angle = (self.servo_config[axis]["servo_range_max"] - self.servo_config[axis]["servo_range_min"])/2
            self.servo_config[axis]["reverse"] = serial_position > halfway_angle

            ## Calibrate zero offset
            reply = input(f"Move {axis} exactly to 0 or 180 degrees. This is to set the zero offset. Enter 0 or 180.\n")
            
            serial_position = self.read_pos_angle_raw(self.servo_config[axis]["servo_id"])
            if reply == "0":
                self.update_config(axis, "zero_offset", serial_position)
            elif reply == "180":
                if not self.servo_config[axis]["reverse"]:
                    self.update_config(axis, "zero_offset", serial_position - 180)
                else:
                    self.update_config(axis, "zero_offset", serial_position + 180)
            else:
                print("Invalid input. Please enter 0 or 180.")
                return None
            
            ## Calibrate overtravel limits
            input(f"Move {axis} to lower overtravel limit. Press enter when complete.\n")
            serial_position = self.read_pos_angle_raw(self.servo_config[axis]["servo_id"])
            self.update_config(axis, "angle_min_overtravel", serial_position)

            input(f"Move {axis} to upper overtravel limit. Press enter when complete.\n")
            serial_position = self.read_pos_angle_raw(self.servo_config[axis]["servo_id"])
            self.update_config(axis, "angle_max_overtravel", serial_position)

            if self.servo_config[axis]["reverse"]:
                max = self.servo_config[axis]["angle_max_overtravel"]
                min = self.servo_config[axis]["angle_min_overtravel"]
                self.update_config(axis, "angle_min_overtravel", max)
                self.update_config(axis, "angle_max_overtravel", min)
        self.save_calibration_to_file()

    def save_calibration_to_file(self):
        with open(self.config_file, 'w') as file:
            yaml.safe_dump(self.servo_config, file)

if __name__ == "__main__":
    controller = SerialServoCtrl()
    # controller.move_to_multi_angle_pos(2000, {"shoulder": 90, "elbow":0})
    # controller.move_to_multi_angle_pos(3000, {"base": 180, "shoulder": 30, "elbow":40})
    # time.sleep(3)
    # controller.move_to_multi_angle_pos(2000, {"base": 90, "shoulder": 90, "elbow":90})
    # time.sleep(2)
    # controller.move_to_multi_angle_pos(2000, {"base": 0, "shoulder": 30, "elbow":40})

    # print("Angle")
    # pos = controller.read_pos_multi_angle([
    #     controller.servo_config["base"]["servo_id"], 
    #     controller.servo_config["shoulder"]["servo_id"], 
    #     controller.servo_config["elbow"]["servo_id"]
    # ])

    # print(pos)
    # controller.calibrate_home_positions()

    v = controller.read_voltage()

    print("Voltage: ", v)

