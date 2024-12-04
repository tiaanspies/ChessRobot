import time
import Positioning.gripperElec as gripperElec

MUC_THRESHOLD = 10
MOTOR_ANGLE_INIT = 80
def calGripper():
    print("Calibrating gripper")    

    gripperMotor = gripperElec.GripperMotor()
    gripperEncoder = gripperElec.GripperEncoder()

    # Move gripper to 90 degrees
    
    #MUC in negative direction
    gripperMotor.set_angle(MOTOR_ANGLE_INIT)
    time.sleep(1)

    # read angle
    angle = gripperEncoder.ReadAngle()
    enc_angle_init = angle
    
    min_motor_angle = MOTOR_ANGLE_INIT
    min_encoder_angle = angle

    while min_motor_angle > 5:
        min_motor_angle -= 1
        gripperMotor.set_angle(min_motor_angle)
        time.sleep(0.5)
        
        min_encoder_angle = gripperEncoder.ReadAngle()
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
    angle = gripperEncoder.ReadAngle()
    enc_angle_init = angle
    
    max_motor_angle = MOTOR_ANGLE_INIT
    max_encoder_angle = angle

    while max_motor_angle < 175:
        max_motor_angle += 1
        gripperMotor.set_angle(max_motor_angle)
        time.sleep(0.5)
        
        max_encoder_angle = gripperEncoder.ReadAngle()
        motor_angle_diff = MOTOR_ANGLE_INIT - max_motor_angle
        enc_angle_diff = enc_angle_init - max_encoder_angle

        print(f"Motor angle diff: {motor_angle_diff}, Encoder angle diff: {enc_angle_diff}")

        if abs(motor_angle_diff - enc_angle_diff) > MUC_THRESHOLD:
            print("MUC REACHED")
            break

    print(f"max motor angle: {max_motor_angle}, max encoder angle: {max_encoder_angle}")

calGripper()
    
