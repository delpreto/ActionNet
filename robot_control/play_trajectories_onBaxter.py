
from __future__ import print_function

from BaxterController import BaxterController
from BaxterHeadController import BaxterHeadController

import numpy as np
import os
import time
from collections import OrderedDict

from threading import Thread

################################################

# Specify the file with trajectory data.
trajectoryData_filepath = 'pouring_training_data_S00_forBaxter.npy'
referenceHandData_filepath = 'pouring_training_referenceObject_positions_S00_forBaxter.npy'
def get_animation_filepath(trajectory_index):
  return os.path.join('trajectory_animations_front',
                      'trajectory_animation_S00_trial%02d.mp4' % trajectory_index)

# Specify columns in the data file.
data_file_gripper_position_columns_xyz_m = [1,2,3]
data_file_gripper_quaternion_columns_wijk = [4,5,6,7]
data_file_times_s_column = 0 # None to use the duration specified below

# Specify default trajectory durations to use if no times_s column was provided.
trajectory_duration_s = 10

# Specify which trajectories to run.
trajectory_indexes_toRun = [0] # None to run all

# Specify the reference hand orientation.
# referenceHand_quaternion_wijk = [-0.5, -0.5, -0.5, -0.5]
# referenceHand_quaternion_wijk = [-0.6123724356957947, -0.6123724356957945, -0.35355339059327384, -0.3535533905932738] # gripper vertical, forearm rotated out by 30 degrees
referenceHand_quaternion_wijk = [-0.18301270189221933, -0.6830127018922194, 0.18301270189221922, -0.6830127018922194] # gripper horizontal, forearm rotated out by 30 degrees

# Specify resting joint angles for each arm.
resting_joint_angles_rad = {
  'left': OrderedDict([('left_s0', -0.2849369313497156), ('left_s1', -0.023009711818281205), ('left_e0', -0.388480634531981), ('left_e1', 1.2620826932327243), ('left_w0', 0.7689078699275637), ('left_w1', -1.1213399559442374), ('left_w2', 1.5558400141127808)]),
  'right': OrderedDict([('right_s0', 0.7025632008515195), ('right_s1', -0.146495165243057), ('right_e0', 0.15109710760671324), ('right_e1', 1.525543893552044), ('right_w0', 0.4421699621079705), ('right_w1', -1.565810889234036), ('right_w2', -1.7495050885833143)]),
}

################################################

# Create a Baxter controller.
controller_right = BaxterController(limb_name='right', print_debug=True)
controller_right.set_resting_joint_angles_rad(resting_joint_angles_rad['right'])
controller_right.open_gripper()
if referenceHandData_filepath is not None:
  controller_left = BaxterController(limb_name='left', print_debug=True)
  controller_left.set_resting_joint_angles_rad(resting_joint_angles_rad['left'])
  controller_left.open_gripper()
else:
  controller_left = None
headController = BaxterHeadController()

# # Move to the resting position.
# controller_right.move_to_resting()
# if controller_left is not None:
#   controller_left.move_to_resting()
# time.sleep(5)

# Extract the data.
feature_matrices = np.load(trajectoryData_filepath)
if referenceHandData_filepath is not None:
  referenceHand_positions_m = np.load(referenceHandData_filepath)
else:
  referenceHand_positions_m = None

# Loop through each trajectory.
for trajectory_index in range(feature_matrices.shape[0]):
  if trajectory_indexes_toRun is not None and trajectory_index not in trajectory_indexes_toRun:
    continue
  print('='*50)
  print('TRAJECTORY %02d' % trajectory_index)
  
  headController.showColor('black')
  headController.setHaloLED(green_percent=0, red_percent=0)
  
  # Get the reference hand position.
  if referenceHand_positions_m is not None:
    referenceHand_position_m = referenceHand_positions_m[trajectory_index, :]
  
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
  success = controller_right.build_trajectory_from_gripper_poses(
    times_from_start_s=[times_s[i] for i in range(0,len(times_s))],
    gripper_positions_m=[gripper_positions_xyz_m[i] for i in range(0,len(times_s))],
    gripper_orientations_quaternion_wijk=[gripper_quaternions_wijk[i] for i in range(0,len(times_s))],
    goal_time_tolerance_s=0.1,
    initial_seed_joint_angles_rad=controller_right.get_resting_joint_angles_rad(should_print=False),
    should_print=False)

  if success:
    headController.showColor('black')
    headController.setHaloLED(green_percent=100, red_percent=100)
    # Move to the starting position.
    if controller_left is not None:
      controller_left.open_gripper()
    abort = raw_input('Press enter to move to the starting position or q to quit: ').strip().lower() == 'q'
    if abort:
      break
    controller_right.move_to_trajectory_start(wait_for_completion=True)
    if controller_left is not None:
      controller_left.move_to_gripper_pose(gripper_position_m=referenceHand_position_m,
                                           gripper_orientation_quaternion_wijk=referenceHand_quaternion_wijk,
                                           wait_for_completion=True,
                                           seed_joint_angles_rad=controller_left.get_resting_joint_angles_rad())
      abort = raw_input('Press enter to close the left hand or q to quit: ').strip().lower() == 'q'
      if abort:
        break
      controller_left.close_gripper()
    # Wait for confirmation.
    abort = raw_input('Press enter to start the trajectory or q to quit: ').strip().lower() == 'q'
    if abort:
      break
    # Start the video animation if one exists.
    animation_video_filepath = get_animation_filepath(trajectory_index)
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
    controller_right.run_trajectory(wait_for_completion=True)
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
headController.quit()
controller_right.quit()
if controller_left is not None:
  controller_left.quit()



  
  
  
  







