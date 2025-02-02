"""
Hardware required:
 - Servo gripper motor
 - Gripper motor encoder. (range should not overflow within gripper range)

Script will move the motor to both ends of gripper range to calibrate open and close positions.

If results are good the script will save the results to "sevo_config.yml" file.

"""

import time
import Positioning.gripperElec as gripperElec

MUC_THRESHOLD = 10
MOTOR_ANGLE_INIT = 60
def calGripper():
    print("Calibrating gripper")    

    gripperMotor = gripperElec.GripperMotor()
    gripperEncoder = gripperElec.GripperEncoder()
    
    #MUC in negative direction
    gripperMotor.set_angle(MOTOR_ANGLE_INIT)
    time.sleep(1)

    # read angle
    angle = gripperEncoder.readAngleUncorrected()
    enc_angle_init = angle
    
    min_motor_angle = MOTOR_ANGLE_INIT
    min_encoder_angle = angle

    while min_motor_angle > gripperMotor.gripper_config["angle_min_motor"]:
        min_motor_angle -= 1
        gripperMotor.set_angle(min_motor_angle)
        time.sleep(0.5)
        
        min_encoder_angle = gripperEncoder.readAngleUncorrected()
        motor_angle_diff = MOTOR_ANGLE_INIT - min_motor_angle
        enc_angle_diff = enc_angle_init - min_encoder_angle

        print(f"Motor angle diff: {motor_angle_diff}, Encoder angle diff: {enc_angle_diff}")

        if abs(motor_angle_diff - enc_angle_diff) > MUC_THRESHOLD:
            print("MUC REACHED")
            break

    print(f"Min motor angle: {min_motor_angle}, Min encoder angle: {min_encoder_angle}")
            
    #MUC in positive direction
    gripperMotor.set_angle(MOTOR_ANGLE_INIT)
    time.sleep(1)

    # read angle
    angle = gripperEncoder.readAngleUncorrected()
    enc_angle_init = angle
    
    max_motor_angle = MOTOR_ANGLE_INIT
    max_encoder_angle = angle

    while max_motor_angle < gripperMotor.gripper_config["angle_max_motor"]:
        max_motor_angle += 1
        gripperMotor.set_angle(max_motor_angle)
        time.sleep(0.5)
        
        max_encoder_angle = gripperEncoder.readAngleUncorrected()
        motor_angle_diff = MOTOR_ANGLE_INIT - max_motor_angle
        enc_angle_diff = enc_angle_init - max_encoder_angle

        print(f"Motor angle diff: {motor_angle_diff}, Encoder angle diff: {enc_angle_diff}")

        if abs(motor_angle_diff - enc_angle_diff) > MUC_THRESHOLD:
            print("MUC REACHED")
            break

    # Check motor and encoder offset
    print("Checking motor and encoder offset")
    min_pos = min_motor_angle+MUC_THRESHOLD*1.5
    max_pos = max_motor_angle-MUC_THRESHOLD*1.5
    positions_motor = []
    positions_encoder = []
    for x in range(10):
        position_motor = min_pos + x * (max_pos - min_pos) / 9
        gripperMotor.set_angle(position_motor)
        time.sleep(0.5)
        position_encoder = gripperEncoder.readAngleUncorrected()

        positions_motor.append(position_motor)
        positions_encoder.append(position_encoder)
        print(f"Motor pos: {position_motor}, Encoder pos: {position_encoder}, Diff: {position_motor - position_encoder}")

    average_offset = sum([positions_motor[i] - positions_encoder[i] for i in range(10)]) / 10
    print(f"Average offset: {average_offset}")
    

    # PRINT RESULTS AND SAVE
    print(f"max motor angle: {max_motor_angle}, max encoder angle: {max_encoder_angle}")
    print("Gripper calibration complete")
    print("")
    print("=====================================")
    print("Results:")
    print(f"Min Encoder Angle: {min_encoder_angle}")
    print(f"Max Encoder Angle: {max_encoder_angle}")
    print(f"Average Offset: {average_offset}")
    print("Do you want to save these values? Y/N")
    save = input()
    gripperMotor.gripper_config["angle_min_enc"] = min_encoder_angle
    gripperMotor.gripper_config["angle_max_enc"] = max_encoder_angle 
    gripperMotor.gripper_config["offset_enc_to_motor"] = average_offset
    if save == "Y":
        gripperMotor.save_config()
        print("Values saved")
    else:
        print("Values not saved")  

if __name__ == "__main__":
    calGripper()