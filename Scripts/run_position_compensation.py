"""
Calibration file to calculate tranformation matrix and calculate home position.

All calibrations use the camera to compare the actual position vs the expected postition from Inverse Kinematics.

"""

import Positioning.calibrate_position_compensation as pos_comp

pos_comp.run("Debug")