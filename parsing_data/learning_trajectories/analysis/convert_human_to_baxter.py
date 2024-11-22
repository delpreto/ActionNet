"""
Given an input HDF5 file of hand pose trajectories
Converts coordinate frames from XSENS to Baxter
Outputs numpy matrices for time, position, quaternion
"""
import os
import numpy as np
import h5py
from scipy.spatial.transform import Rotation

import utils
from constants import *


# - Helpers - #

def convert_quaternion(quat_xsens_wijk):
  # Do an initial rotation, to make the xsens quat match the example quat used during testing.
  quat_wijk = quat_xsens_wijk
  rotates_by_deg = [
    [0, 0, -180],
    ]
  # Apply the rotations.
  rotation_quat = Rotation.from_quat([quat_wijk[1], quat_wijk[2], quat_wijk[3], quat_wijk[0]])
  for i in range(len(rotates_by_deg)-1, -1, -1):
    rotate_by_deg = rotates_by_deg[i]
    rotation_toApply = Rotation.from_rotvec(np.radians(rotate_by_deg))
    rotation_quat = rotation_quat * rotation_toApply
  ijkw = rotation_quat.as_quat()
  quat_wijk = [ijkw[3], ijkw[0], ijkw[1], ijkw[2]]
  # print(quat_wijk)
  # print()
  
  # Negate the i and j components.
  quat_wijk = [quat_wijk[0], -quat_wijk[1], -quat_wijk[2], quat_wijk[3]]
  
  # Apply the rotations determined during testing.
  rotates_by_deg = [
      [0, 0, 180],
      [0, -90, 0],
      [0, 0, 90],
      [0, 180, 0],
      ]
  # Apply the rotations.
  rotation_quat = Rotation.from_quat([quat_wijk[1], quat_wijk[2], quat_wijk[3], quat_wijk[0]])
  for i in range(len(rotates_by_deg)-1, -1, -1):
    rotate_by_deg = rotates_by_deg[i]
    rotation_toApply = Rotation.from_rotvec(np.radians(rotate_by_deg))
    rotation_quat = rotation_quat * rotation_toApply
  ijkw = rotation_quat.as_quat()
  quat_wijk = [ijkw[3], ijkw[0], ijkw[1], ijkw[2]]
  
  # Return the result.
  return quat_wijk

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

def convert_position_trajectoryHand(xyz_xsens_m):
  # Lift it a bit to be safe.
  xyz_baxter_m = convert_position(xyz_xsens_m)
  return [
    xyz_baxter_m[0],
    xyz_baxter_m[1],
    xyz_baxter_m[2] + (2)/100,
  ]


# - Main - #

def convert_human_to_baxter(
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

            # Convert to Baxter frame
            pos_world_to_hand_W = np.array([convert_position_trajectoryHand(p) for p in pos_world_to_hand_W])
            quat_world_to_hand_wijk = np.array([convert_quaternion(q) for q in quat_world_to_hand_wijk])

            trajectory_dir = output_directory + in_traj_key + '/'
            os.makedirs(trajectory_dir, exist_ok=True)
            np.save(trajectory_dir + 'time.npy', arr=time)
            np.save(trajectory_dir + 'pos_world_to_hand_W.npy', arr=pos_world_to_hand_W)
            np.save(trajectory_dir + 'quat_world_to_hand_wijk.npy', arr=quat_world_to_hand_wijk)


if __name__ == '__main__':
    # Script inputs
    input_trajectory_file = os.path.expanduser('~/data/scooping/inference_LinOSS_train_scooping_5678.hdf5')
    output_directory = os.path.expanduser('~/data/scooping/baxter/inference_LinOSS_train_scooping_5678/')

    # Main
    convert_human_to_baxter(
        input_trajectory_file,
        output_directory,
    )