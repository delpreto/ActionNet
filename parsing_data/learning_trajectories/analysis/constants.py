import numpy as np
import scipy.spatial.transform as tf

# Static transform between hand xsens frame and spoon frame defined in README
_rot_hand_to_prespoon = np.array([[0, 1, 0],
                                  [-1, 0, 0],
                                  [0, 0, 1]]) # Intermediate hand frame flipping operation
_spoon_tilt_angles = np.array([np.pi/6, -np.pi/4, 0]) # yaw roll pitch
_rot_prespoon_to_spoon = tf.Rotation.from_euler('ZXY', _spoon_tilt_angles).as_matrix()
ROT_HAND_TO_SPOON = _rot_hand_to_prespoon @ _rot_prespoon_to_spoon

# Static transform from spoon frame defined in README and side spoon configuration
_rot_prespoon_to_side_spoon = tf.Rotation.from_euler('XYZ', np.array([0, 0, np.pi/2])).as_matrix() # 90deg yaw
ROT_HAND_TO_SIDE_SPOON = _rot_hand_to_prespoon @ _rot_prespoon_to_side_spoon

# Static transform from spoon frame defined in README and straight spoon configuration
ROT_HAND_TO_STRAIGHT_SPOON = _rot_hand_to_prespoon

# Spoon length
_spoon_length = 0.30
POS_HAND_TO_SPOON_S = np.array([_spoon_length, 0, 0])

# Static transform between hand xsens frame and jug frame defined in README
ROT_HAND_TO_JUG = np.array([
    [0, 0, 1],
    [0, 1, 0],
    [-1, 0, 0],
])
POS_HAND_TO_JUG_J = np.array([0, 0, -0.05])

# Reference object offsets of the ground
PAN_Z_OFFSET = 0.01
PLATE_Z_OFFSET = 0.01

# Object shapes
PAN_RADIUS = 0.15
PAN_HEIGHT = 0.02
PLATE_RADIUS = 0.15
PLATE_HEIGHT = 0.02
GLASS_RADIUS = 0.025
GLASS_HEIGHT = 0.15
HAND_BOX = np.array([.048, .03, .013]) # wrt hand frame
SPOON_BOX = np.array([.325, .07, .02]) # wrt spoon frame
JUG_BOX = np.array([0.175, 0.0625, 0.25]) # wrt jug frame
TABLE_BOX = np.array([1, 1, 0.2]) # wrt table frame
TABLE_ORIGIN = np.array([-0.5, 0, -0.1]) # this is the table frame

# 'Scooping region' -- height above plate/pan
PICKUP_HEIGHT = 0.15
DROPOFF_HEIGHT = 0.15

# Random sampling for out of distribution cases
TABLE_BBOX = [[-0.75, -0.5], [-0.25, 0.5]] # min_x, min_y, max_x, max_y
HAND_Z_OFFSET = 0.125
XYZ_NOISE_STD_DEV = [0.05, 0.05, 0.05]
GLASS_Z_OFFSET = 0.158