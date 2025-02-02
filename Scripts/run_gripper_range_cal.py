"""
Hardware required:
 - Servo gripper motor
 - Gripper motor encoder. (range should not overflow within gripper range)

Script will move the motor to both ends of gripper range to calibrate open and close positions.

If results are good the script will save the results to "sevo_config.yml" file.

"""

import Positioning.gripper_range_cal

Positioning.gripper_range_cal.calGripper()