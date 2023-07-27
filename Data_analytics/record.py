import time
import socket
import struct
import numpy as np
import msvcrt

# LOCAL IP to communicate with NodeJS server
UDP_IP = '127.0.0.1'
UDP_PORT_REC = 5001
UDP_PORT_SEND = 5000

#POSITION TOLERANCE
POS_TOL = 100 #mm

# Message IDs
MSG_ID_DEST = bytes([0x10]) #message id for sending new destination
MSG_ID_REQ_LIST = bytes([0x11]) # request list of bot IDs
MSG_ID_POS_REQ = bytes([0x12]) # request bot position
MSG_ID_SEND_WAYPOINTS = bytes([0x13]) # send waypoints for path planning
MSG_ID_QUEUE_STATE_REQ = bytes([0x14]) # req state of bot queue
MSG_ID_QUEUE_CLEAR = bytes([0x15]) # clear bot waypoint queue
MSG_INVALID = bytes([0x55]) # something went wrong

def req_bot_pos(botID:int):
    """Requests the position of the bot corresponding to botID.
    Bot id = int between 0 and 254
    """
    # Sends message to NODEJS server to request bot position

    head = (MSG_ID_POS_REQ,bytes([botID]))
    data = ()
    send_udp_packet(head, data)

    # wait for response
    # Create a UDP socket and bind it to the specified IP and port
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(2)

    sock.bind((UDP_IP, UDP_PORT_REC))

    try:
        # Receive UDP packet
        data, addr = sock.recvfrom(1024)
    except TimeoutError:
        print("Rec timed out")
        sock.close()
        raise("RECEIVE MISSED")

    msg = struct.unpack('c', data[:1])[0]
    espID = struct.unpack('c', data[1:2])[0]

    #check that data was valid
    if msg == MSG_INVALID:
        sock.close()
        raise("something went wrong requesting position data")

    if espID != bytes([botID]):
        sock.close()
        raise(f"Received bot ID {espID} does not match requested ID {botID}.")

    #Unpack data from packet
    x = struct.unpack('f', data[2:6])[0]*1000
    y = struct.unpack('f', data[6:10])[0]*1000
    z = struct.unpack('f', data[10:14])[0]*1000
    is_tracked = struct.unpack('B', data[30:31])[0] == 1

    sock.close()
    return x, y, z, is_tracked

def send_udp_packet(head, data):
    """
    Reads the head of the message then builds the data
    accordiningly.

    When adding new commands they must be added here.
    """
    # Create socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Initialize head na dpos_data
    head_new = bytes([0x00])
    data_new = b''

    message_id = head[0]
    # build message depending on message type
    if message_id==(MSG_ID_DEST): # set destination
        data_new = struct.pack('ffff', *data)
        head_new = struct.pack('cc', *head)

    elif message_id == (MSG_ID_REQ_LIST): # request bot list
        data_new = bytes([0x00])
        head_new = struct.pack('c', *head)

    elif message_id == (MSG_ID_POS_REQ): #req position of bot
        data_new = bytes([0x00])
        head_new = struct.pack('cc', *head)

    elif message_id == (MSG_ID_SEND_WAYPOINTS): # send waypoints for bot
        data_new = b''
        for point in data:
            data_new += struct.pack("ff", point[0], point[1])
            # print(f"waypoint: {point[0]}, {point[1]}")
        head_new = struct.pack('cc', *head)

    elif message_id == (MSG_ID_QUEUE_STATE_REQ): # req bot stack state
        data_new = bytes([0x00])
        head_new = struct.pack('cc', *head)

    elif message_id == (MSG_ID_QUEUE_CLEAR): # request queue clear
        data_new = bytes([0x00])
        head_new = struct.pack('cc', *head)

    else:
        sock.close()
        raise(f"INVALID MESSAGE HEAD {head}")

    # send message
    sock.sendto(head_new+data_new, (UDP_IP, UDP_PORT_SEND))
    sock.close()


BOT_ID = 9

positions = np.zeros((0,3))
pack_count = 0
x, y, z, _ = req_bot_pos(BOT_ID)
start = np.array([x,y,z])
start_pos = np.array([230, 500, 0])

print("\n Press anything to stop recording")
a = "n"
count = 0
while a != "y":
    try:
        x, y, z, _ = req_bot_pos(BOT_ID)
    except TimeoutError:
        continue
    
    curr_pos = np.array([x, y, z])
    a = curr_pos-start+start_pos
    positions = np.vstack([positions, a])
    time.sleep(0.02)

    pack_count += 1
    print(f"Packets received {pack_count}", end="\r")

    # Check for user input without blocking
    a = input(f"Count: {count}. Enter anything to record next pos \r")
    count += 1
    # if msvcrt.kbhit():
    #     input_text = msvcrt.getche().decode()
    #     if input_text:
    #         break

np.save("positions.npy", positions)