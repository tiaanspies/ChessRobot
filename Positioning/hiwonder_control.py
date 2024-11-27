import serial
import time

def create_command_packet(command, parameters):
    length = 2 + len(parameters)
    packet = [0x55, 0x55, length, command] + parameters
    return bytearray(packet)

def send_command(ser, command, parameters):
    packet = create_command_packet(command, parameters)
    ser.write(packet)
    time.sleep(0.1)  # Wait for the servo to process the command

def parameters_for_position(time, motor_ids, positions):
    
    servo_num = len(motor_ids) # Parameter 1
    time_hex_low = time & 0xFF # Parameter 2 (Low 8 bits of time)
    time_hex_high = (time >> 8) & 0xFF # parameter 3 (High 8 bits of time)

    position_parameters = [[id, position & 0xFF, (position >> 8) & 0xFF] for id, position in zip(motor_ids, positions)]
    flat_position = [x for xs in position_parameters for x in xs] # Flatten the list
    
    parameters = [servo_num, time_hex_low, time_hex_high] + flat_position

    return parameters

time = 1000
motor_ids = [1, 2, 3]
positions = [1000, 500, 750]
position_params = parameters_for_position(time, motor_ids, positions)

# Open the serial port
ser = serial.Serial('/dev/ttyS0', 9600, timeout=1)

send_command(ser, 3, position_params)  # Send the command

# Close the serial port
ser.close()