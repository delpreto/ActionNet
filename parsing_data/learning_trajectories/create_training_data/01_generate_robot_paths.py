
############
#
# Copyright (c) 2024 MIT CSAIL and Joseph DelPreto
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
# IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# See https://action-sense.csail.mit.edu for more usage information.
# Created 2021-2024 for the MIT ActionSense project by Joseph DelPreto [https://josephdelpreto.com].
#
############

import h5py
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import os
script_dir = os.path.dirname(os.path.realpath(__file__))
actionsense_root_dir = script_dir
while os.path.split(actionsense_root_dir)[-1] != 'ActionSense':
  actionsense_root_dir = os.path.realpath(os.path.join(actionsense_root_dir, '..'))

from learning_trajectories.helpers.plot_animations import *

###################################################################
# Configuration
###################################################################

# Choose the activity to process.
activity_mode = 'pouring' # 'pouring', 'scooping'

# Specify the subjects to consider.
subject_ids_toProcess = ['S00', 'S10', 'S11'] # S00, S10, S11, ted_S00

# Specify the input file of extracted trajectory data,
# and the output file for robot trajectory data.
results_dir = os.path.realpath(os.path.join(actionsense_root_dir, 'results', 'learning_trajectories'))
trajectory_data_filepaths_humans = [os.path.join(results_dir, '%s_paths_humans_%s.hdf5' % (activity_mode, subject_id_toProcess)) for subject_id_toProcess in subject_ids_toProcess]
trajectory_data_filepaths_robots = [os.path.join(results_dir, '%s_paths_robots_%s.hdf5' % (activity_mode, subject_id_toProcess)) for subject_id_toProcess in subject_ids_toProcess]

# Specify outputs.
show_plot_human_and_robot_hand_paths = False
save_plot_human_and_robot_hand_paths = False
stationary_duration_s = 2 if activity_mode == 'pouring' else 1 # might want to match stationary_position_buffer_duration_s?

###################################################################
# Generate robot paths
###################################################################

for subject_index in range(len(subject_ids_toProcess)):
  trajectory_data_filepath_humans = trajectory_data_filepaths_humans[subject_index]
  trajectory_data_filepath_robots = trajectory_data_filepaths_robots[subject_index]
  
  # Open the files of extracted trajectory data.
  h5_file_humans = h5py.File(trajectory_data_filepath_humans, 'r')
  h5_file_robots = h5py.File(trajectory_data_filepath_robots, 'w')
  
  # Get the list of body segments, which can be used to index data matrices below.
  body_segment_names_humans = h5_file_humans['body_segment_names']
  body_segment_names_humans = [name.decode('utf-8') for name in body_segment_names_humans]
  print()
  print('See the following list of body segments.')
  print('The starred segments should be used as the hand/elbow/shoulder chain.')
  for (body_segment_index, body_segment_name) in enumerate(body_segment_names_humans):
    if body_segment_name in ['RightUpperArm', 'RightForeArm', 'RightHand']:
      print('*', end='')
    else:
      print(' ', end='')
    print(' %02d: %s' % (body_segment_index, body_segment_name))
  # Highlight segment indexes useful for the activity trajectories:
  hand_index_humans = body_segment_names_humans.index('RightHand')
  elbow_index_humans = body_segment_names_humans.index('RightForeArm')
  shoulder_index_humans = body_segment_names_humans.index('RightUpperArm')
  
  # Save a sample list of body segments for the robot, which will correspond to the data matrices below.
  # TODO Update if needed
  body_segment_names_robots = ['RightHand', 'RightForeArm', 'RightUpperArm']
  hand_index_robots = body_segment_names_robots.index('RightHand')
  elbow_index_robots = body_segment_names_robots.index('RightForeArm')
  shoulder_index_robots = body_segment_names_robots.index('RightUpperArm')
  h5_file_robots.create_dataset('body_segment_names', data=body_segment_names_robots)
  
  # Loop through each subject and trial.
  print()
  fig = None
  for group_name in h5_file_humans.keys():
    # Check if this group is for a subject's data (or something else such as metadata)
    try:
      subject_id = int(group_name.split('subject_')[-1])
    except:
      continue
    print('See data for subject %02d' % subject_id)
    subject_group_human = h5_file_humans[group_name]
    
    # Make an analogous group for the robot trajectories.
    subject_group_robot = h5_file_robots.create_group(group_name)
    
    # Loop through each trial for this subject.
    for (trial_name, trial_group_human) in subject_group_human.items():
      trial_id = int(trial_name.split('trial_')[-1])
      
      # Get the timestamp for each sample, as seconds since trial start.
      time_s = np.squeeze(np.array(trial_group_human['time_s']))
      # Get the global xyz position of each body segment.
      # It is a Tx23x3 matrix, indexed as [timestep][body_segment_index][xyz].
      body_segment_position_m = np.squeeze(np.array(trial_group_human['body_segment_position_m']))
      # Get the orientation of each body segment as a quaternion.
      # It is a Tx23x4 matrix, indexed as [timestep][body_segment_index][wxyz].
      body_segment_quaternion_wijk = np.squeeze(np.array(trial_group_human['body_segment_quaternion_wijk']))
      # Get the global xyz positions, orientations, and timestamp
      #  of the inferred stationary position (when the person holds the pitcher or scooper relatively still).
      stationary_position_m = np.squeeze(np.array(trial_group_human['stationary']['body_segment_position_m']))
      stationary_quaternion_wijk = np.squeeze(np.array(trial_group_human['stationary']['body_segment_quaternion_wijk']))
      stationary_time_s = np.squeeze(np.array(trial_group_human['stationary']['time_s']))
      stationary_index = np.where(abs(stationary_time_s - time_s) == np.min(abs(stationary_time_s - time_s)))[0][-1]
      # Get the inferred reference object position.
      referenceObject_position_m = np.squeeze(np.array(trial_group_human['reference_object']['position_m']))
  
      # Highlight data that should be used to generate a robot path.
      # The path should have the following characteristics:
      #   A sample at each timestamp in sampling_times_s
      #   At start_time_s (probably 0):
      #     The hand/elbow/shoulder should be at start_positions_m[hand_index/elbow_index/shoulder_index]
      #     The hand orientation should be given by start_quaternions_wijk[hand_index]
      #   At stationary_time_s:
      #     The hand/elbow/shoulder should be at stationary_positions_m[hand_index/elbow_index/shoulder_index]
      #     The hand orientation should be given by stationary_quaternions_wijk[hand_index]
      #   At end_time_s (the last sample):
      #     The hand/elbow/shoulder should be at end_positions_m[hand_index/elbow_index/shoulder_index]
      #     The hand orientation should be given by end_quaternions_wijk[hand_index]
      start_positions_m = np.squeeze(body_segment_position_m[0, :, :])
      stationary_positions_m = np.squeeze(body_segment_position_m[stationary_index, :, :])
      end_positions_m = np.squeeze(body_segment_position_m[-1, :, :])
      start_quaternions_wijk = np.squeeze(body_segment_quaternion_wijk[0, :, :])
      stationary_quaternions_wijk = np.squeeze(body_segment_quaternion_wijk[stationary_index, :, :])
      end_quaternions_wijk = np.squeeze(body_segment_quaternion_wijk[-1, :, :])
      start_time_s = time_s[0]
      stationary_time_s = time_s[stationary_index]
      end_time_s = time_s[-1]
      sampling_times_s = time_s
      
      # Use linear interpolation for now.
      # TODO Use another planning method?
      time_s_robot = np.array([
          start_time_s,
          stationary_time_s - stationary_duration_s/2,
          stationary_time_s + stationary_duration_s/2,
          end_time_s
        ])
      body_segment_position_m_robot = np.array([
          start_positions_m,
          stationary_positions_m,
          stationary_positions_m,
          end_positions_m
        ])
      body_segment_quaternion_wijk_robot = np.array([
          start_quaternions_wijk,
          stationary_quaternions_wijk,
          stationary_quaternions_wijk,
          end_quaternions_wijk
        ])
      fn_interpolate_data = interpolate.interp1d(
          time_s_robot, # x values
          body_segment_position_m_robot,   # y values
          axis=0,        # axis of the data along which to interpolate
          kind='linear', # interpolation method, such as 'linear', 'zero', 'nearest', 'quadratic', 'cubic', etc.
          fill_value='extrapolate' # how to handle x values outside the original range
      )
      body_segment_position_m_robot = fn_interpolate_data(sampling_times_s)
      fn_interpolate_data = interpolate.interp1d(
          time_s_robot, # x values
          body_segment_quaternion_wijk_robot,   # y values
          axis=0,        # axis of the data along which to interpolate
          kind='linear', # interpolation method, such as 'linear', 'zero', 'nearest', 'quadratic', 'cubic', etc.
          fill_value='extrapolate' # how to handle x values outside the original range
      )
      body_segment_quaternion_wijk_robot = fn_interpolate_data(sampling_times_s)
      
      time_s_robot = sampling_times_s
  
      # Only use select body segments for the interpolated robot paths.
      # Should match body_segment_names_robots defined above.
      body_segment_position_m_robot = body_segment_position_m_robot[:, [hand_index_humans, elbow_index_humans, shoulder_index_humans], :]
      body_segment_quaternion_wijk_robot = body_segment_quaternion_wijk_robot[:, [hand_index_humans, elbow_index_humans, shoulder_index_humans], :]
  
      if show_plot_human_and_robot_hand_paths or save_plot_human_and_robot_hand_paths:
        if fig is None:
          fig = plt.figure()
        plt.clf()
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        fig.add_subplot(1, 1, 1, projection='3d')
        ax = fig.get_axes()[0]
        ax.view_init(16, 44)
        ax.set_xlabel('X [cm]')
        ax.set_ylabel('Y [cm]')
        ax.set_zlabel('Z [cm]')
        human_hand_path_cm = 100*np.squeeze(body_segment_position_m[:, hand_index_humans, :])
        robot_hand_path_cm = 100*np.squeeze(body_segment_position_m_robot[:, hand_index_robots, :])
        ax.plot3D(human_hand_path_cm[:, 0], human_hand_path_cm[:, 1], human_hand_path_cm[:, 2], alpha=1)
        ax.plot3D(robot_hand_path_cm[:, 0], robot_hand_path_cm[:, 1], robot_hand_path_cm[:, 2], alpha=1)
        ax.plot3D(100*referenceObject_position_m[0], 100*referenceObject_position_m[1], 100*referenceObject_position_m[2], 'o', alpha=1)
        ax.set_box_aspect([ub - lb for lb, ub in (ax.get_xlim(), ax.get_ylim(), ax.get_zlim())])
        ax.set_title('Subject %02d Trial %02d' % (subject_id, trial_id))
        if show_plot_human_and_robot_hand_paths:
          plt.draw()
          plt_wait_for_keyboard_press()
        if save_plot_human_and_robot_hand_paths:
          plot_dir = os.path.join(results_dir, '%s_paths_humanRobot')
          os.makedirs(plot_dir, exist_ok=True)
          output_filepath = os.path.join(plot_dir, '%s_paths_humanRobot_S%02d_%02d.jpg' %
                                         (activity_mode, subject_id, trial_id))
          plt.savefig(output_filepath, dpi=300)
  
      # Save robot data.
      # Make an analogous group for the robot trajectories.
      trial_group_robot = subject_group_robot.create_group(trial_name)
      trial_group_robot.create_dataset('body_segment_position_m', data=body_segment_position_m_robot)
      trial_group_robot.create_dataset('body_segment_quaternion_wijk', data=body_segment_quaternion_wijk_robot)
      trial_group_robot.create_dataset('time_s', data=time_s_robot)
      
  print()
  
  # Clean up.
  h5_file_humans.close()
  h5_file_robots.close()
  
  
  
  
  
  
  
  
  
