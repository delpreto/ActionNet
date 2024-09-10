
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
input_data_dir = 'trajectory_data'

trajectoryData_filepaths = {
  'S00': os.path.join(input_data_dir, 'pouring_trainingData_S00_forBaxter.npy'),
  'S11': os.path.join(input_data_dir, 'pouring_trainingData_S11_forBaxter.npy'),
  'MODEL': os.path.join(input_data_dir, 'data_to_evaluate_forBaxter.npy'),
}
referenceHandData_filepaths = {
  'S00': os.path.join(input_data_dir, 'pouring_trainingData_S00_forBaxter_referenceObject.npy'),
  'S11': os.path.join(input_data_dir, 'pouring_trainingData_S11_forBaxter.npy'),
  'MODEL': os.path.join(input_data_dir, 'data_to_evaluate_forBaxter_referenceObject.npy'),
}
# trajectoryData_filepath = os.path.join(input_data_dir, 'pouring_trainingData_S00_forBaxter.npy')
# referenceHandData_filepath = os.path.join(input_data_dir, 'pouring_trainingData_S00_forBaxter_referenceObject.npy')

# trajectoryData_filepath = os.path.join(input_data_dir, 'pouring_trainingData_S11_forBaxter.npy')
# referenceHandData_filepath = os.path.join(input_data_dir, 'pouring_trainingData_S11_forBaxter_referenceObject.npy')

# trajectoryData_filepath = os.path.join(input_data_dir, 'data_to_evaluate_forBaxter.npy')
# referenceHandData_filepath = os.path.join(input_data_dir, 'data_to_evaluate_forBaxter_referenceObject.npy')

def get_animation_filepath(trajectory_index):
  # return os.path.join('trajectory_animations_front',
  #                     'trajectory_animation_S00_trial%02d.mp4' % trajectory_index)
  # return os.path.join('trajectory_animations_front',
  #                     'trajectory_animation_S11_trial%02d.mp4' % trajectory_index)
  return os.path.join('trajectory_animations_front',
                      'trajectory_animation_model_trial%02d.mp4' % trajectory_index)

# Specify columns in the data file.
data_file_gripper_position_columns_xyz_m = [1,2,3]
data_file_gripper_quaternion_columns_wijk = [4,5,6,7]
data_file_times_s_column = 0 # None to use the duration specified below

# Specify an offset to add to positions.
gripper_position_offset_m = np.array([0, 0, 0], dtype=float)/100
referenceHand_position_offset_m = np.array([0, 0, 2], dtype=float)/100 # negative y towards center

# Specify default trajectory durations to use if no times_s column was provided.
trajectory_duration_s = 10

# Specify which trajectories to run.
trajectory_indexes_toRun = None # None to run all
# trajectory_indexes_toRun = [0] # None to run all

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

# Prompt user for trajectories to run.
while True:
  command = raw_input('Enter a command of the form "S00/S11/model trial_index start/pour/run" or q to quit: ').strip().lower()
  if command == 'q':
    break
  if command == 'open' or command == 'o':
    if controller_left is not None:
      controller_left.open_gripper()
    continue
  elif command == 'close' or command == 'c':
    if controller_left is not None:
      controller_left.close_gripper()
    continue
  elif 'offset gripper' in command or 'og' == command.split()[0].strip():
    try:
      offsets = [int(x) for x in command.split()[1:]]
      gripper_position_offset_m = np.array(offsets, dtype=float)/100
      print('Set gripper_position_offset_m to', gripper_position_offset_m)
    except:
      pass
    continue
  elif 'offset reference' in command or 'or' == command.split()[0].strip():
    try:
      offsets = [int(x) for x in command.split()[1:]]
      referenceHand_position_offset_m = np.array(offsets, dtype=float)/100
      print('Set referenceHand_position_offset_m to', referenceHand_position_offset_m)
    except:
      pass
    continue
  try:
    example_type = command.split(' ')[0].strip().upper()
    trajectory_index = int(command.split(' ')[1].strip())
    command = command.split(' ')[2:]
  except:
    continue
  if trajectory_indexes_toRun is not None and trajectory_index not in trajectory_indexes_toRun:
    continue
  print('='*50)
  print('TRAJECTORY INDEX %02d' % trajectory_index)
  
  headController.showColor('black')
  headController.setHaloLED(green_percent=0, red_percent=0)

  # Extract the data.
  feature_matrices = np.load(trajectoryData_filepaths[example_type])
  if referenceHandData_filepaths is not None:
    referenceHand_positions_m = np.load(referenceHandData_filepaths[example_type])
  else:
    referenceHand_positions_m = None
  
  # Get the reference hand position.
  if referenceHand_positions_m is not None:
    referenceHand_position_m = referenceHand_positions_m[trajectory_index, :]
  
  # Get the position and quaternion for each timestep.
  feature_matrix = np.squeeze(feature_matrices[trajectory_index, :, :])
  gripper_positions_xyz_m = feature_matrix[:, data_file_gripper_position_columns_xyz_m]
  gripper_quaternions_wijk = feature_matrix[:, data_file_gripper_quaternion_columns_wijk]
  
  # Add an offset if desired.
  gripper_positions_xyz_m = gripper_positions_xyz_m + gripper_position_offset_m
  if referenceHand_position_m is not None:
    referenceHand_position_m = referenceHand_position_m + referenceHand_position_offset_m
  
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
  if not success:
    print('Error building trajectory')
    headController.setHaloLED(green_percent=0, red_percent=100)
    continue
  
  # Move to the starting position.
  if 'start' in command:
    if 'ref' not in command or 'pitcher' in command:
      controller_right.move_to_trajectory_start(wait_for_completion=True)
    if 'ref' in command or 'pitcher' not in command:
      if controller_left is not None:
        controller_left.open_gripper()
        controller_left.move_to_gripper_pose(gripper_position_m=referenceHand_position_m,
                                             gripper_orientation_quaternion_wijk=referenceHand_quaternion_wijk,
                                             wait_for_completion=True,
                                             seed_joint_angles_rad=controller_left.get_resting_joint_angles_rad())
  
  # Move to the pouring position.
  elif 'pour' in command:
    if 'ref' not in command or 'pitcher' in command:
      controller_right.move_to_trajectory_index(step_index=int(len(times_s)/2), wait_for_completion=True)
    if 'ref' in command or 'pitcher' not in command:
      if controller_left is not None:
        controller_left.open_gripper()
        controller_left.move_to_gripper_pose(gripper_position_m=referenceHand_position_m,
                                             gripper_orientation_quaternion_wijk=referenceHand_quaternion_wijk,
                                             wait_for_completion=True,
                                             seed_joint_angles_rad=controller_left.get_resting_joint_angles_rad())
  
  # Run the trajectory
  elif command == 'run':
    headController.showColor('black')
    headController.setHaloLED(green_percent=100, red_percent=100)
    # Move to the starting position.
    if controller_left is not None:
      controller_left.open_gripper()
    abort = raw_input('  Press enter to move to the starting position or q to quit: ').strip().lower() == 'q'
    if abort:
      break
    controller_right.move_to_trajectory_start(wait_for_completion=True)
    if controller_left is not None:
      controller_left.move_to_gripper_pose(gripper_position_m=referenceHand_position_m,
                                           gripper_orientation_quaternion_wijk=referenceHand_quaternion_wijk,
                                           wait_for_completion=True,
                                           seed_joint_angles_rad=controller_left.get_resting_joint_angles_rad())
      abort = raw_input('  Press enter to close the left hand or q to quit: ').strip().lower() == 'q'
      if abort:
        break
      controller_left.close_gripper()
    # Wait for confirmation.
    abort = raw_input('  Press enter to start the trajectory or q to quit: ').strip().lower() == 'q'
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
  
  
  # # Wait
  # time.sleep(3)
  
  # Return to resting.
  # controller.move_to_resting()

headController.showColor('black')
headController.setHaloLED(green_percent=0, red_percent=0)
headController.quit()
controller_right.quit()
if controller_left is not None:
  controller_left.quit()



  
  
  
  







