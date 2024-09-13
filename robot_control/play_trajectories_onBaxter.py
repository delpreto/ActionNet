
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
  'model': os.path.join(input_data_dir, 'pouring_modelData_forBaxter.npy'),
}
referenceHandData_filepaths = {
  'S00': os.path.join(input_data_dir, 'pouring_trainingData_S00_forBaxter_referenceObject.npy'),
  'S11': os.path.join(input_data_dir, 'pouring_trainingData_S11_forBaxter_referenceObject.npy'),
  'model': os.path.join(input_data_dir, 'pouring_modelData_forBaxter_referenceObject.npy'),
}
# trajectoryData_filepath = os.path.join(input_data_dir, 'pouring_trainingData_S00_forBaxter.npy')
# referenceHandData_filepath = os.path.join(input_data_dir, 'pouring_trainingData_S00_forBaxter_referenceObject.npy')

# trajectoryData_filepath = os.path.join(input_data_dir, 'pouring_trainingData_S11_forBaxter.npy')
# referenceHandData_filepath = os.path.join(input_data_dir, 'pouring_trainingData_S11_forBaxter_referenceObject.npy')

# trajectoryData_filepath = os.path.join(input_data_dir, 'data_to_evaluate_forBaxter.npy')
# referenceHandData_filepath = os.path.join(input_data_dir, 'data_to_evaluate_forBaxter_referenceObject.npy')

def get_example_type(user_input):
  if user_input.lower().strip() in ['model', 'm']:
    return 'model'
  try:
    subject_index = int(user_input.lower().split('s')[1])
    return 'S%02d' % subject_index
  except:
    pass
  return None
  
def get_animation_filepath(example_type, trajectory_index):
  return os.path.join('trajectory_animations_front',
                      'trajectory_animation_%s_trial%02d.mp4' % (example_type, trajectory_index))

# Specify columns in the data file.
data_file_gripper_position_columns_xyz_m = [1,2,3]
data_file_gripper_quaternion_columns_wijk = [4,5,6,7]
data_file_times_s_column = 0 # None to use the duration specified below

# Specify an offset to add to positions.
pitcher_position_offset_m = np.array([0, -4, 8], dtype=float) / 100
glass_position_offset_m = np.array([-3, -14, 6], dtype=float) / 100 # negative y towards center

# Specify default trajectory durations to use if no times_s column was provided.
trajectory_duration_s = 10

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

speed_factor = 1

# Create Baxter controllers.
controller_right = BaxterController(limb_name='right', print_debug=True)
controller_right.set_resting_joint_angles_rad(resting_joint_angles_rad['right'])
controller_right.open_gripper()
controller_left = BaxterController(limb_name='left', print_debug=True)
controller_left.set_resting_joint_angles_rad(resting_joint_angles_rad['left'])
controller_left.open_gripper()
headController = BaxterHeadController()

# # Move to the resting position.
# controller_right.move_to_resting()
# if controller_left is not None:
#   controller_left.move_to_resting()
# time.sleep(5)

menu_descriptions_commands = [
  ['Move to starting position', 's00/s11/model trial_index start [g/p]'],
  ['Move to pouring position', 's00/s11/model trial_index pour [g/p]'],
  ['Run a trajectory', 's00/s11/model trial_index run'],
  ['', ''],
  ['Open the left gripper', 'open/o'],
  ['Close the left gripper', 'close/c'],
  ['', ''],
  ['Set the pitcher offset [cm]', '"offset-pitcher"/op # # #'],
  ['Set the glass offset [cm]', '"offset-glass"/og # # #'],
  ['Adjust the pitcher offset [cm]', '"offset pitcher"/op x/y/z'],
  ['Adjust the glass offset [cm]', '"offset glass"/og x/y/z'],
  ['', ''],
  ['Set the speed factor', 'speed/s #'],
  ['', ''],
  ['Show the menu', 'menu/m'],
  ['Quit', 'q/quit'],
  ]
max_description_length = max([len(x[0]) for x in menu_descriptions_commands])
menu_str = '\n' + '='*50 + '\n'
for (description, command) in menu_descriptions_commands:
  menu_str += '  %s | %s\n' % (description.ljust(max_description_length), command)
menu_str += '='*50 + '\n'
print(menu_str)

next_command = None
previous_trajectory_command = None
previous_command = ''

# Prompt user for commands or trajectories to run.
while True:
  print()
  print('='*50)
  if next_command is None:
    command = raw_input('>> Enter a command: ').strip().lower()
    if command == '.':
      print('Rerunning the previous command: %s' % previous_command)
      command = previous_command
    else:
      previous_command = command
  else:
    print('Rerunning the previous trajectory command: %s' % next_command)
    command = next_command
    next_command = None
  # Show the menu
  if len(command) == 0:
    continue
  if command.lower() in ['m', 'menu']:
    print(menu_str)
    continue
  # Quit
  if command.lower() in ['q', 'quit']:
    print('Quitting')
    break
  # Control the left gripper.
  if command == 'open' or command == 'o':
    print('Opening the left gripper')
    if controller_left is not None:
      controller_left.open_gripper()
    continue
  elif command == 'close' or command == 'c':
    print('Closing the left gripper')
    if controller_left is not None:
      controller_left.close_gripper()
    continue
  elif command.split()[0] == 'clear':
    try:
      if command.split()[1].strip() in ['p', 'pitcher']:
        print('Clearing the pitcher arm')
        controller_right.move_to_joint_angles_rad({
          'right_s0': np.deg2rad(11.1181640625),
          'right_s1': np.deg2rad(-39.7705078125),
          'right_e0': np.deg2rad(3.251953125),
          'right_e1': np.deg2rad(118.498535156),
          'right_w0': np.deg2rad(-24.3017578125),
          'right_w1': np.deg2rad(-79.9145507812),
          'right_w2': np.deg2rad(-82.6611328125),
        }, wait_for_completion=True)
      if command.split()[1].strip() in ['g', 'glass']:
        print('Clearing the glass arm')
        if controller_left is not None:
          # joint_angles_rad['right_s0'] = np.deg2rad(15)
          controller_left.move_to_joint_angles_rad({
            'left_s0': np.deg2rad(0),
            'left_s1': np.deg2rad(3.58154296875),
            'left_e0': np.deg2rad(-0.81298828125),
            'left_e1': np.deg2rad(88.76953125),
            'left_w0': np.deg2rad(30.8276367187),
            'left_w1': np.deg2rad(-89.9560546875),
            'left_w2': np.deg2rad(174.594726562),
          }, wait_for_completion=True)
    except:
      print('Error parsing or executing the command')
    continue
  # Set the gripper or pitcher offsets.
  elif 'offset-pitcher' in command or 'op' == command.split()[0].strip() \
      or 'offset-glass' in command or 'og' == command.split()[0].strip():
    updating_pitcher = 'offset-pitcher' in command or 'op' == command.split()[0].strip()
    updating_glass = 'offset-glass' in command or 'og' == command.split()[0].strip()
    success = False
    if updating_pitcher:
      print('Current pitcher offset [cm]: ', 100*pitcher_position_offset_m)
    if updating_glass:
      print('Current glass offset [cm]: ', 100*glass_position_offset_m)
    if len(command.split()) <= 1 or len(command.split()[1].strip()) == 0:
      continue
    # Set the absolute offsets.
    try:
      offsets_m = np.array([int(x) for x in command.split()[1:]], dtype=float)/100
      if updating_pitcher:
        pitcher_position_offset_m = offsets_m
        print('Set pitcher_position_offset_cm to', 100*pitcher_position_offset_m)
      if updating_glass:
        glass_position_offset_m = offsets_m
        print('Set glass_position_offset_cm to', 100*glass_position_offset_m)
      success = True
    except:
      pass
    # Relatively adjust the offsets.
    if not success:
      try:
        relative_offsets_str = command.split()[1]
        x_count = sum([c == 'x' for c in relative_offsets_str])
        y_count = sum([c == 'y' for c in relative_offsets_str])
        z_count = sum([c == 'z' for c in relative_offsets_str])
        relative_offsets_cm = np.array([x_count, y_count, z_count], dtype=float)
        if '-' in relative_offsets_str:
          relative_offsets_cm *= -1
        if updating_pitcher:
          pitcher_position_offset_m += relative_offsets_cm / 100
          print('Set pitcher_position_offset_cm to', 100*pitcher_position_offset_m)
        if updating_glass:
          glass_position_offset_m += relative_offsets_cm / 100
          print('Set glass_position_offset_cm to', 100*glass_position_offset_m)
        success = True
      except:
        pass
    if success and previous_trajectory_command is not None:
      rerun_trajectory_command = raw_input('Press enter to rerun "%s" (or q to abort) ' % previous_trajectory_command)
      if rerun_trajectory_command.lower().strip() not in ['q', 'quit']:
        next_command = previous_trajectory_command
    if not success:
      print('Invalid offset command')
    continue
  # Adjust the speed factor.
  if command.split()[0] in ['speed', 's']:
    try:
      speed_factor = float(command.split()[1].strip())
      print('Set the speed factor to %g' % speed_factor)
    except:
      print('Invalid speed command')
    continue
  # Parse a trajectory-based command.
  try:
    command_split = [x.strip() for x in command.split(' ')]
    example_type = get_example_type(command_split[0])
    trajectory_index = int(command_split[1])
    trajectory_command = command_split[2].lower()
    if len(command_split) >= 4:
      trajectory_arm = command_split[3].lower()
      trajectory_move_pitcher = trajectory_arm in ['pitcher', 'p']
      trajectory_move_glass = trajectory_arm in ['glass', 'g']
    else:
      trajectory_move_pitcher = True
      trajectory_move_glass = True
    if controller_left is None:
      trajectory_move_glass = False
    previous_trajectory_command = command
  except:
    print('Invalid trajectory command')
    continue
  
  # Clear the screen and LEDs.
  headController.showColor('black')
  headController.setHaloLED(green_percent=0, red_percent=0)

  # Extract the data.
  if example_type not in trajectoryData_filepaths:
    print('Unknown example type [%s]' % example_type)
    continue
  feature_matrices = np.load(trajectoryData_filepaths[example_type])
  if referenceHandData_filepaths is not None:
    referenceHand_positions_m = np.load(referenceHandData_filepaths[example_type])
  else:
    referenceHand_positions_m = None
  
  # Get the reference hand position.
  if referenceHand_positions_m is not None:
    referenceHand_position_m = referenceHand_positions_m[trajectory_index, :]
  else:
    referenceHand_position_m = None
  
  # Get the position and quaternion for each timestep.
  feature_matrix = np.squeeze(feature_matrices[trajectory_index, :, :])
  gripper_positions_xyz_m = feature_matrix[:, data_file_gripper_position_columns_xyz_m]
  gripper_quaternions_wijk = feature_matrix[:, data_file_gripper_quaternion_columns_wijk]
  
  # Add an offset if desired.
  gripper_positions_xyz_m = gripper_positions_xyz_m + pitcher_position_offset_m
  if referenceHand_position_m is not None:
    referenceHand_position_m = referenceHand_position_m + glass_position_offset_m
  
  # Get a time vector.
  if data_file_times_s_column is not None:
    times_s = np.squeeze(feature_matrix[:, data_file_times_s_column])
  else:
    times_s = np.linspace(start=0, stop=trajectory_duration_s, num=feature_matrix.shape[0])
  times_s = times_s/speed_factor
  Fs = (len(times_s)-1)/(times_s[-1] - times_s[0])
  Ts = 1/Fs
  
  # Build the trajectory.
  success = controller_right.build_trajectory_from_gripper_poses(
    times_from_start_s=[times_s[i] for i in range(0, len(times_s))],
    gripper_positions_m=[gripper_positions_xyz_m[i] for i in range(0, len(times_s))],
    gripper_orientations_quaternion_wijk=[gripper_quaternions_wijk[i] for i in range(0, len(times_s))],
    goal_time_tolerance_s=0.1,
    initial_seed_joint_angles_rad=controller_right.get_resting_joint_angles_rad(should_print=False),
    should_print=False)
  if not success:
    print('Error building trajectory')
    headController.setHaloLED(green_percent=0, red_percent=100)
    continue
  
  # Move to the starting position.
  if 'start' == trajectory_command:
    print('Moving to starting position of %s index %d | pitcher %s glass %s' % (example_type, trajectory_index, trajectory_move_pitcher, trajectory_move_glass))
    if trajectory_move_pitcher:
      controller_right.move_to_trajectory_start(wait_for_completion=True)
    if trajectory_move_glass:
      controller_left.move_to_gripper_pose(gripper_position_m=referenceHand_position_m,
                                           gripper_orientation_quaternion_wijk=referenceHand_quaternion_wijk,
                                           wait_for_completion=True,
                                           seed_joint_angles_rad=controller_left.get_resting_joint_angles_rad())
  
  # Move to the pouring position.
  elif 'pour' == trajectory_command:
    print('Moving to pouring position of %s index %d | pitcher %s glass %s' % (example_type, trajectory_index, trajectory_move_pitcher, trajectory_move_glass))
    if trajectory_move_pitcher:
      controller_right.move_to_trajectory_index(step_index=int(len(times_s)/2), wait_for_completion=True)
    if trajectory_move_glass:
      controller_left.move_to_gripper_pose(gripper_position_m=referenceHand_position_m,
                                           gripper_orientation_quaternion_wijk=referenceHand_quaternion_wijk,
                                           wait_for_completion=True,
                                           seed_joint_angles_rad=controller_left.get_resting_joint_angles_rad())
  
  # Run the trajectory
  elif 'run' == trajectory_command:
    print('Running trajectory for %s index %d | pitcher %s glass %s' % (example_type, trajectory_index, trajectory_move_pitcher, trajectory_move_glass))
    headController.showColor('black')
    headController.setHaloLED(green_percent=100, red_percent=100)
    # Move to the starting position.
    if trajectory_move_glass:
      controller_left.open_gripper()
    abort = raw_input('  Press enter to move to the starting position or q to quit: ').strip().lower() in ['q', 'quit']
    if abort:
      continue
    if trajectory_move_pitcher:
      controller_right.move_to_trajectory_start(wait_for_completion=True)
    if trajectory_move_glass:
      controller_left.move_to_gripper_pose(gripper_position_m=referenceHand_position_m,
                                           gripper_orientation_quaternion_wijk=referenceHand_quaternion_wijk,
                                           wait_for_completion=True,
                                           seed_joint_angles_rad=controller_left.get_resting_joint_angles_rad())
      abort = raw_input('  Press enter to close the left hand or q to quit: ').strip().lower() in ['q', 'quit']
      if abort:
        continue
      controller_left.close_gripper()
    video_thread = None
    done_playing_video = True
    if trajectory_move_pitcher:
      # Wait for confirmation.
      abort = raw_input('  Press enter to start the trajectory or q to quit: ').strip().lower() in ['q', 'quit']
      if abort:
        continue
      # Start the video animation if one exists.
      animation_video_filepath = get_animation_filepath(example_type, trajectory_index)
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



  
  
  
  







