
from __future__ import print_function

from BaxterController import BaxterController
from BaxterHeadController import BaxterHeadController

import numpy as np
import os
import time

from threading import Thread

################################################

# Specify the file with trajectory data.
data_filepath = 'pouring_training_data_S00_forBaxter.npy'

# Specify columns in the data file.
data_file_gripper_position_columns_xyz_m = [1,2,3]
data_file_gripper_quaternion_columns_wijk = [4,5,6,7]
data_file_times_s_column = 0 # None to use the duration specified below

# Specify default trajectory durations to use if no times_s column was provided.
trajectory_duration_s = 10

# Specify which trajectories to run.
trajectory_indexes_toRun = [0] # None to run all


################################################

# # Convert from Xsens quaternion to Baxter quaternion.
# def xsens_to_baxter_quaternion(xsens_quaternion_wijk):
#   return [
#     xsens_quaternion_wijk[0],
#     -xsens_quaternion_wijk[1],
#     -xsens_quaternion_wijk[2],
#     -xsens_quaternion_wijk[3]
#   ]
#
# # Translate a position from Xsens human frame to Baxter frame.
# def xsens_to_baxter_position(xsens_position_xyz_m):
#   # A sample home for a human demonstration was [-0.3, 0.26, 0.17]
#   #  with x point backwards, y pointing right, and z pointing up.
#   # Baxter has a home position of about [1, -0.25, 0]
#   #  with x pointing forwards, y pointing left, and z pointing down.
#   return [
#     -xsens_position_xyz_m[0] + 0.7,
#     -xsens_position_xyz_m[1] + 0.0,
#     xsens_position_xyz_m[2] - 0.2,
#     ]

################################################

# Create a Baxter controller.
controller = BaxterController(limb_name='right', print_debug=True)
headController = BaxterHeadController()

# Move the resting position.
# controller.move_to_resting()
# time.sleep(5)

# Extract the data.
if 'hdf' in os.path.splitext(data_filepath)[-1].lower():
  import h5py
  h5_file = h5py.File(data_filepath, 'r')
  feature_matrices = np.squeeze(h5_file['feature_matrices'])
  labels = np.array([str(x) for x in h5_file['labels']])
  feature_matrices = feature_matrices[np.where(labels == 'human')[0], :, :]
elif 'npy' in os.path.splitext(data_filepath)[-1].lower():
  feature_matrices = np.load(data_filepath)
else:
  raise AssertionError('Unknown input data file extension [%s]' % os.path.splitext(data_filepath)[-1].lower())

# Loop through each trajectory.
for trajectory_index in range(feature_matrices.shape[0]):
  if trajectory_indexes_toRun is not None and trajectory_index not in trajectory_indexes_toRun:
    continue
  print('='*50)
  print('TRAJECTORY %02d' % trajectory_index)
  
  # Get the position and quaternion for each timestep.
  feature_matrix = np.squeeze(feature_matrices[trajectory_index, :, :])
  gripper_positions_xyz_m = feature_matrix[:, data_file_gripper_position_columns_xyz_m]
  gripper_quaternions_wijk = feature_matrix[:, data_file_gripper_quaternion_columns_wijk]
  
  # Get a time vector.
  if data_file_times_s_column is not None:
    times_s = np.squeeze(feature_matrix[:, data_file_times_s_column])
  else:
    times_s = np.linspace(start=0, stop=trajectory_duration_s, num=feature_matrix.shape[0])
  Fs = (len(times_s)-1)/(times_s[-1] - times_s[0])
  Ts = 1/Fs
  
  # # Convert to Baxter coordinates.
  # for timestep_index in range(len(times_s)):
  #   gripper_positions_xyz_m[timestep_index] = xsens_to_baxter_position(gripper_positions_xyz_m[timestep_index])
  #   gripper_quaternions_wijk[timestep_index] = xsens_to_baxter_quaternion(gripper_quaternions_wijk[timestep_index])
  
  # Build the trajectory.
  success = controller.build_trajectory_from_gripper_poses(
    times_from_start_s=[times_s[i] for i in range(0,len(times_s))],
    gripper_positions_m=[gripper_positions_xyz_m[i] for i in range(0,len(times_s))],
    gripper_orientations_quaternion_wijk=[gripper_quaternions_wijk[i] for i in range(0,len(times_s))],
    goal_time_tolerance_s=0.1,
    initial_seed_joint_angles_rad=controller.get_resting_joint_angles_rad(should_print=False),
    should_print=False)

  if success:
    # Start the video animation if one exists.
    animation_video_filepath = 'pouring_animation_S00_%02d.mp4' % trajectory_index
    if os.path.exists(animation_video_filepath):
      video_thread = None
      done_playing_video = True
      def showVideo_thread():
        global done_playing_video
        done_playing_video = False
        headController.showVideo(videoFile=animation_video_filepath)
        done_playing_video = True
      video_thread = Thread(target=showVideo_thread)
      video_thread.start()
    else:
      video_thread = None
    # Run the trajectory.
    controller.run_trajectory(wait_for_completion=True)
    # Nod and wait.
    headController.nod(times=3)
    time.sleep(2)
    # Wait for the video to finish.
    if video_thread is not None and not done_playing_video:
      video_thread.join()
  else:
    print('Error building trajectory')

  # Wait and then return to resting.
  # time.sleep(5)
  # controller.move_to_resting()

controller.quit()
headController.quit()

  
  
  
  







