import numpy as np
import scipy.spatial.transform as tf

# Static transform between hand xsens frame and spoon frame defined in README
_rot_hand_to_prespoon = np.array([[0, 1, 0],
                                  [-1, 0, 0],
                                  [0, 0, 1]]) # Intermediate hand frame flipping operation
_spoon_tilt_angles = np.array([np.pi/6, -np.pi/4, 0]) # yaw roll pitch
_rot_prespoon_to_spoon = tf.Rotation.from_euler('ZXY', _spoon_tilt_angles).as_matrix()
ROT_HAND_TO_SPOON = _rot_hand_to_prespoon @ _rot_prespoon_to_spoon

# Spoon length
_spoon_length = 0.30
POS_HAND_TO_SPOON_S = np.array([_spoon_length, 0, 0])

# Off of the ground
PAN_Z_OFFSET = 0.01
PLATE_Z_OFFSET = 0.01

# Plotting shapes
PAN_RADIUS = 0.15
PAN_HEIGHT = 0.02
PLATE_RADIUS = 0.15
PLATE_HEIGHT = 0.02
HAND_BOX = np.array([.048, .03, .013]) # wrt hand frame
SPOON_BOX = np.array([.325, .07, .02]) # wrt spoon frame