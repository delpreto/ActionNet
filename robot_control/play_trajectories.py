
from __future__ import print_function

from BaxterController import BaxterController

import h5py
import numpy as np
import time

################################################

# Specify the file with trajectory data.
data_filepath = 'pouring_training_data.hdf5'

# Specify columns in the data file.
data_file_gripper_position_columns_xyz_m = [0,1,2]
data_file_gripper_quaternion_columns_wijk = [9,10,11,12]

# Specify which trajectories to run.
trajectory_indexes_toRun = [0] # None to run all

# Specify trajectory durations.
trajectory_duration_s = 10

################################################

# Convert from Xsens quaternion to Baxter quaternion.
def xsens_to_baxter_quaternion(xsens_quaternion_wijk):
  return xsens_quaternion_wijk

# Translate a position from Xsens human frame to Baxter frame.
def xens_to_baxter_position(xsens_position_xyz_m):
  return xsens_position_xyz_m

################################################

# Create a Baxter controller.
controller = BaxterController(limb_name='right', print_debug=True)
controller.move_to_neutral()
time.sleep(5)

# Extract the data.
h5_file = h5py.File(data_filepath, 'r')
feature_matrices = np.squeeze(h5_file['feature_matrices'])
labels = np.array([str(x) for x in h5_file['labels']])
feature_matrices = feature_matrices[np.where(labels == 'human')[0], :, :]

# Loop through each trajectory.
for trajectory_index in feature_matrices.shape[0]:
  if trajectory_indexes_toRun is not None and trajectory_index not in trajectory_indexes_toRun:
    continue
  print('='*50)
  print('TRAJECTORY %02d' % trajectory_index)
  
  # Get the position and quaternion for each timestep.
  feature_matrix = np.squeeze(feature_matrices[trajectory_index, :, :])
  gripper_positions_xyz_m = feature_matrix[data_file_gripper_position_columns_xyz_m, :]
  gripper_quaternions_wijk = feature_matrix[data_file_gripper_quaternion_columns_wijk, :]
  
  # Solve inverse kinematics to compute joint angles for each timestep.
  # At each timestep, seed the solver with the previous solution to help make motion more continuous/smooth.
  joint_angles_rad = []
  seed_joint_angles_rad = None
  for time_index in range(feature_matrix.shape[0]):
    if time_index > 0:
      seed_joint_angles_rad = joint_angles_rad[time_index-1]
    joint_angles_rad.append(controller.get_joint_angles_rad_for_gripper_pose(
                            gripper_position_m=gripper_positions_xyz_m[time_index,:],
                            gripper_orientation_quaternion_wijk=gripper_quaternions_wijk[time_index,:],
                            seed_joint_angles_rad=seed_joint_angles_rad)
    )
  if None in joint_angles_rad:
    print('ERROR computing inverse kinematics for at least one timestep.')
    continue
  
  # Get a time vector.
  time_s = np.linspace(start=0, stop=trajectory_duration_s, num=feature_matrix.shape[0])
  Fs = (len(time_s)-1)/(time_s[-1] - time_s[0])
  
  # Build and run the trajectory.
  controller.build_trajectory(times_from_start_s=time_s,
                              joint_angles_rad=joint_angles_rad,
                              goal_time_tolerance_s=Fs)
  controller.run_trajectory(wait_for_completion=True)
  
  # Wait and then return to neutral.
  time.sleep(5)
  controller.move_to_neutral()
  time.sleep(5)
  

  
  
  
  







