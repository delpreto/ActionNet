"""
Given an input HDF5 file of hand pose trajectories
Converts coordinate frames from XSENS to Controller KDL library frame 
    (necessary for running offline control pipeline)
Outputs numpy matrices for time, position, quaternion
"""
import os
import numpy as np
import h5py

import utils
from constants import *


def convert_position(xyz_xsens_m):
  # A sample pitcher home for a human demonstration was [-0.3, 0.25, 0.18]
  #  and a sample glass home was [-0.38	-0.20	  0.18]
  #  with x point backwards, y pointing right, and z pointing up.
  # Baxter has a pitcher home position of about [0.9, -0.4, 0]
  #  with x pointing forwards, y pointing left, and z pointing down.
  # Human and Baxter are rotated 180 about z (x and y are negated)
  return [
    -xyz_xsens_m[0] + 0.6,
    -xyz_xsens_m[1] - 0.15,
     xyz_xsens_m[2] - 0.18,
  ]


# - Main - #

def convert_xsens_to_controller(
    input_trajectory_file,
    output_directory,
):
    with h5py.File(input_trajectory_file, 'r') as in_file:

        # Iterate through input trajectories
        for in_traj_key, in_traj_group in in_file.items():

            time = np.array(in_traj_group['data']['time'])
            pos_world_to_hand_W = np.array(in_traj_group['data']['pos_world_to_hand_W'])

            # Get quaternions in wijk
            if 'quat_world_to_hand_wijk' in in_traj_group['data'].keys():
                quat_world_to_hand_wijk = np.array(in_traj_group['data']['quat_world_to_hand_wijk'])
            elif 'quat_world_to_hand_ijkw' in in_traj_group['data'].keys():
                quat_world_to_hand_ijkw = np.array(in_traj_group['data']['quat_world_to_hand_ijkw'])
                quat_world_to_hand_wijk = np.roll(quat_world_to_hand_ijkw, shift=1, axis=1)
            elif 'rot_world_to_hand' in in_traj_group['data'].keys():
                rot_world_to_hand = np.array(in_traj_group['data']['rot_world_to_hand'])
                quat_world_to_hand_wijk = np.array([utils.rot_matrix_to_quat(R, scalar_first=True) for R in rot_world_to_hand])
            else:
                raise KeyError('No orientation fields found')
            
            # Static transforms
            rot_world_to_nominal_hand = tf.Rotation.from_euler('XYZ', np.array([0, 0, -np.pi/2])).as_matrix()
            rot_world_to_nominal_kdl = tf.Rotation.from_euler('XYZ', np.array([0, np.pi/2, 0])).as_matrix()
            z_swap = tf.Rotation.from_euler('XYZ', np.array([0, 0, np.pi])).as_matrix()
            
            # Transform to KDL coordinate system
            pos_world_to_hand_W = np.array([convert_position(p) for p in pos_world_to_hand_W])
            rot_world_to_hand = np.array([utils.quat_to_rot_matrix(quat, scalar_first=True) for quat in quat_world_to_hand_wijk])

            # After much trial and error..
            A = rot_world_to_hand @ rot_world_to_nominal_hand.T
            B = z_swap @ A @ z_swap
            rot_world_to_kdl = B @ rot_world_to_nominal_kdl

            # Convert to quaternions
            quat_world_to_hand_wijk = np.array([utils.rot_matrix_to_quat(R, scalar_first=True) for R in rot_world_to_kdl])

            trajectory_dir = output_directory + in_traj_key + '/'
            os.makedirs(trajectory_dir, exist_ok=True)
            np.save(trajectory_dir + 'time.npy', arr=time)
            np.save(trajectory_dir + 'pos_world_to_hand_W.npy', arr=pos_world_to_hand_W)
            np.save(trajectory_dir + 'quat_world_to_hand_wijk.npy', arr=quat_world_to_hand_wijk)


if __name__ == '__main__':
    # Script inputs
    input_trajectory_file = os.path.expanduser('~/data/scooping/inference_LinOSS_train_scooping_5678_side.hdf5')
    output_directory = os.path.expanduser('~/data/scooping/controller/inference_LinOSS_train_scooping_5678_side/')

    # Main
    convert_xsens_to_controller(
        input_trajectory_file,
        output_directory,
    )