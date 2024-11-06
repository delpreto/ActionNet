import numpy as np
import scipy.spatial.transform as tf

# Static transform between hand xsens frame and spoon frame defined in README
rot_prespoon_to_hand = np.array([[0, 1, 0],
                            [-1, 0, 0],
                            [0, 0, 1]]) # Intermediate hand frame flipping operation
tilt_angles = np.array([np.pi/6, -np.pi/4, 0]) # rad, yaw then roll
rot_spoon_to_prespoon = tf.Rotation.from_euler('ZXY', tilt_angles).as_matrix()
rot_spoon_to_hand = rot_prespoon_to_hand @ rot_spoon_to_prespoon

# Spoon length
spoon_length = 0.30
pos_hand_to_spoon_S = np.array([spoon_length, 0, 0])

# Off of the ground
pan_z_offset = 0.01
plate_z_offset = 0.01

# Plotting shapes
pan_radius = 0.15
pan_height = 0.02
plate_radius = 0.15
plate_height = 0.02
hand_box = np.array([.048, .03, .013]) # wrt hand frame
spoon_box = np.array([.325, .07, .02]) # wrt spoon frame