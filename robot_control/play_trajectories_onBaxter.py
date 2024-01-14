
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

# Create a Baxter controller.
controller = BaxterController(limb_name='right', print_debug=True)
headController = BaxterHeadController()

# Move the resting position.
# controller.move_to_resting()
# time.sleep(5)

# Extract the data.
feature_matrices = np.load(data_filepath)

# Loop through each trajectory.
for trajectory_index in range(feature_matrices.shape[0]):
  if trajectory_indexes_toRun is not None and trajectory_index not in trajectory_indexes_toRun:
    continue
  print('='*50)
  print('TRAJECTORY %02d' % trajectory_index)
  
  headController.showColor('black')
  headController.setHaloLED(green_percent=0, red_percent=0)
  
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
  
  # Build the trajectory.
  success = controller.build_trajectory_from_gripper_poses(
    times_from_start_s=[times_s[i] for i in range(0,len(times_s))],
    gripper_positions_m=[gripper_positions_xyz_m[i] for i in range(0,len(times_s))],
    gripper_orientations_quaternion_wijk=[gripper_quaternions_wijk[i] for i in range(0,len(times_s))],
    goal_time_tolerance_s=0.1,
    initial_seed_joint_angles_rad=controller.get_resting_joint_angles_rad(should_print=False),
    should_print=False)

  if success:
    headController.showColor('black')
    headController.setHaloLED(green_percent=100, red_percent=100)
    # Move to the starting position.
    controller.move_to_trajectory_start(wait_for_completion=True)
    # Start the video animation if one exists.
    animation_video_filepath = 'pouring_animation_S00_%02d.mp4' % trajectory_index
    if os.path.exists(animation_video_filepath):
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
      done_playing_video = True
    # Run the trajectory.
    controller.run_trajectory(wait_for_completion=True)
    # Nod.
    headController.setHaloLED(green_percent=100, red_percent=0)
    headController.nod(times=2)
    # Wait for the video to finish.
    if video_thread is not None and not done_playing_video:
      video_thread.join()
  else:
    print('Error building trajectory')
    headController.setHaloLED(green_percent=0, red_percent=100)

  # Wait
  time.sleep(3)
  
  # Return to resting.
  # controller.move_to_resting()

headController.showColor('black')
headController.setHaloLED(green_percent=0, red_percent=0)
controller.quit()
headController.quit()

  
  
  
  







