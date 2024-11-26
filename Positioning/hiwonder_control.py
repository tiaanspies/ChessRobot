import serial
import time

def calculate_checksum(data):
    checksum = sum(data) & 0xFF
    return ~checksum & 0xFF

def create_command_packet(servo_id, command, parameters):
    length = 3 + len(parameters)
    packet = [0x55, 0x55, servo_id, length, command] + parameters
    checksum = calculate_checksum(packet[2:])
    packet.append(checksum)
    return bytearray(packet)

def send_command(ser, servo_id, command, parameters):
    packet = create_command_packet(servo_id, command, parameters)
    ser.write(packet)
    time.sleep(0.1)  # Wait for the servo to process the command
    response = ser.read_all()
    return response

# Open the serial port
ser = serial.Serial('/dev/ttyS0', 115200, timeout=1)

# Example: Send a command to servo with ID 1 to move to position 512
servo_id = 1
command = 0x01  # Example command for position control
parameters = [0x02, 0x00]  # Example parameters for position 512 (0x0200)
response = send_command(ser, servo_id, command, parameters)

print("Response:", response)

# Close the serial port
ser.close()
def receive_data(ser):
    data = ser.read_all()
    return data