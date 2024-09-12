
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
from collections import OrderedDict
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

# Specify the input files of trajectory data,
# and the output file for feature matrices and labels.
results_dir = os.path.realpath(os.path.join(actionsense_root_dir, 'results', 'learning_trajectories', 'humans'))
trajectory_data_filepaths_humans = [os.path.join(results_dir, '%s_paths_humans_%s.hdf5' % (activity_mode, subject_id_toProcess)) for subject_id_toProcess in subject_ids_toProcess]
trajectory_data_filepaths_robots = [os.path.join(results_dir, '%s_paths_robots_%s.hdf5' % (activity_mode, subject_id_toProcess)) for subject_id_toProcess in subject_ids_toProcess]
training_data_filepaths = [os.path.join(results_dir, '%s_trainingData_%s.hdf5' % (activity_mode, subject_id_toProcess)) for subject_id_toProcess in subject_ids_toProcess]

# Specify outputs.
include_robot_examples = False
plot_hand_path_features = False

# Specify how to standardize training data segment times and lengths.
num_resampled_timesteps = 100
normalize_time = False

###################################################################
# Process the files and create training data
###################################################################

for subject_index in range(len(subject_ids_toProcess)):
  trajectory_data_filepath_humans = trajectory_data_filepaths_humans[subject_index]
  training_data_filepath = training_data_filepaths[subject_index]
  if include_robot_examples:
    trajectory_data_filepath_robots = trajectory_data_filepaths_robots[subject_index]
  
  # Open the files of extracted trajectory data.
  data_file_humans = h5py.File(trajectory_data_filepath_humans, 'r')
  if include_robot_examples:
    data_file_robots = h5py.File(trajectory_data_filepath_robots, 'r')
  
  # Get the list of body segments and joints, which can be used to index data matrices below.
  body_segment_names_humans = data_file_humans['body_segment_names']
  body_segment_names_humans = [name.decode('utf-8') for name in body_segment_names_humans]
  body_joint_names_humans = data_file_humans['body_joint_names']
  body_joint_names_humans = [name.decode('utf-8') for name in body_joint_names_humans]
  if include_robot_examples:
    body_segment_names_robots = data_file_robots['body_segment_names']
    body_segment_names_robots = [name.decode('utf-8') for name in body_segment_names_robots]
  print()
  print('See the following list of body segments (in the human data).')
  print('The starred segments should be used as the hand/elbow/shoulder chain.')
  for (body_segment_index, body_segment_name) in enumerate(body_segment_names_humans):
    if body_segment_name in ['RightUpperArm', 'RightForeArm', 'RightHand']:
      print('*', end='')
    else:
      print(' ', end='')
    print(' %02d: %s' % (body_segment_index, body_segment_name))
  print('See the following list of body joints (in the human data).')
  for (body_joint_index, body_joint_name) in enumerate(body_joint_names_humans):
    print(' %02d: %s' % (body_joint_index, body_joint_name))
  # Highlight segment indexes useful for the activity trajectories, which will be extracted as features:
  hand_segment_index_humans = body_segment_names_humans.index('RightHand')
  elbow_segment_index_humans = body_segment_names_humans.index('RightForeArm')
  shoulder_segment_index_humans = body_segment_names_humans.index('RightUpperArm')
  wrist_joint_index_humans = body_joint_names_humans.index('RightWrist')
  elbow_joint_index_humans = body_joint_names_humans.index('RightElbow')
  shoulder_joint_index_humans = body_joint_names_humans.index('RightShoulder')
  if include_robot_examples:
    hand_segment_index_robots = body_segment_names_robots.index('RightHand')
    elbow_segment_index_robots = body_segment_names_robots.index('RightForeArm')
    shoulder_segment_index_robots = body_segment_names_robots.index('RightUpperArm')
  
  # Create feature matrices for each trial, labeled as human or robot.
  # The current matrices will concatenate shoulder, elbow, and hand positions and hand orientation.
  # For humans, will also concatenate joint angles.
  feature_matrices = {}
  labels = []
  
  # Helper to create a feature matrix from the processed trajectory data.
  def add_training_segment(time_s, body_segment_position_m, body_segment_quaternion_wijk,
                           joint_angle_eulerZXY_xyz_rad, joint_angle_eulerXZY_xyz_rad,
                           referenceObject_position_m, hand_to_pitcher_angles_rad, stationary_time_s,
                           is_human, subject_id=-1, trial_id=-1):
    
    # Collect position and orientation features.
    features = OrderedDict()
    num_timesteps = body_segment_position_m.shape[0]
    if is_human:
      features['hand_position_m'] = np.reshape(body_segment_position_m[:, hand_segment_index_humans, :], (num_timesteps, -1)) # unwrap xyz from each segment
      features['elbow_position_m'] = np.reshape(body_segment_position_m[:, elbow_segment_index_humans, :], (num_timesteps, -1)) # unwrap xyz from each segment
      features['shoulder_position_m'] = np.reshape(body_segment_position_m[:, shoulder_segment_index_humans, :], (num_timesteps, -1)) # unwrap xyz from each segment
      features['hand_quaternion_wijk'] = np.reshape(body_segment_quaternion_wijk[:, hand_segment_index_humans, :], (num_timesteps, -1))
      features['elbow_quaternion_wijk'] = np.reshape(body_segment_quaternion_wijk[:, elbow_segment_index_humans, :], (num_timesteps, -1))
      features['shoulder_quaternion_wijk'] = np.reshape(body_segment_quaternion_wijk[:, shoulder_segment_index_humans, :], (num_timesteps, -1))
      features['wrist_joint_angle_xyz_rad'] = np.reshape(joint_angle_eulerZXY_xyz_rad[:, wrist_joint_index_humans, :], (num_timesteps, -1))
      features['elbow_joint_angle_xyz_rad'] = np.reshape(joint_angle_eulerZXY_xyz_rad[:, elbow_joint_index_humans, :], (num_timesteps, -1))
      features['shoulder_joint_angle_xyz_rad'] = np.reshape(joint_angle_eulerXZY_xyz_rad[:, shoulder_joint_index_humans, :], (num_timesteps, -1))
    else:
      features['hand_position_m'] = np.reshape(body_segment_position_m[:, hand_segment_index_robots, :], (num_timesteps, -1)) # unwrap xyz from each segment
      features['elbow_position_m'] = np.reshape(body_segment_position_m[:, elbow_segment_index_robots, :], (num_timesteps, -1)) # unwrap xyz from each segment
      features['shoulder_position_m'] = np.reshape(body_segment_position_m[:, shoulder_segment_index_robots, :], (num_timesteps, -1)) # unwrap xyz from each segment
      features['hand_quaternion_wijk'] = np.reshape(body_segment_quaternion_wijk[:, hand_segment_index_robots, :], (num_timesteps, -1))
      features['elbow_quaternion_wijk'] = np.reshape(body_segment_quaternion_wijk[:, elbow_segment_index_robots, :], (num_timesteps, -1))
      features['shoulder_quaternion_wijk'] = np.reshape(body_segment_quaternion_wijk[:, shoulder_segment_index_robots, :], (num_timesteps, -1))
      features['wrist_joint_angle_xyz_rad'] = np.nan*np.ones(shape=(num_timesteps, 3))
      features['elbow_joint_angle_xyz_rad'] = np.nan*np.ones(shape=(num_timesteps, 3))
      features['shoulder_joint_angle_xyz_rad'] = np.nan*np.ones(shape=(num_timesteps, 3))
    # Add time.
    time_s = time_s - time_s[0]
    if normalize_time:
      time_s = time_s/time_s[-1]
    features['time_s'] = np.reshape(time_s, (-1,1))
  
    # Resample the data.
    target_timestamps = np.linspace(time_s[0], time_s[-1], num_resampled_timesteps)
    for (key, data) in features.items():
      fn_interpolate_data = interpolate.interp1d(
          time_s,         # x values
          data,           # y values
          axis=0,         # axis of the data along which to interpolate
          kind='linear',  # interpolation method, such as 'linear', 'zero', 'nearest', 'quadratic', 'cubic', etc.
          fill_value='extrapolate' # how to handle x values outside the original range
      )
      features[key] = fn_interpolate_data(target_timestamps)
    
    # Add the reference object position.
    features['referenceObject_position_m'] = np.atleast_2d(referenceObject_position_m)
    
    # Add the pitcher holding angle.
    features['hand_to_pitcher_angles_rad'] = np.atleast_2d(hand_to_pitcher_angles_rad)
    
    # Add the stationary pouring index.
    stationary_index = np.squeeze(features['time_s']).searchsorted(stationary_time_s)
    features['stationary_index'] = np.atleast_2d(stationary_index)
    
    # Add to the main lists.
    for key in features:
      feature_matrices.setdefault(key, [])
      feature_matrices[key].append(features[key])
    labels.append('human' if is_human else 'robot')
    
    # Plot if desired.
    if plot_hand_path_features:
      fig = plt.figure()
      figManager = plt.get_current_fig_manager()
      figManager.window.showMaximized()
      fig.add_subplot(1, 1, 1, projection='3d')
      ax = fig.get_axes()[0]
      ax.view_init(16, 44)
      ax.set_xlabel('X [cm]')
      ax.set_ylabel('Y [cm]')
      ax.set_zlabel('Z [cm]')
      hand_path_cm = 100*features['hand_position_m']
      ax.plot3D(hand_path_cm[:, 0], hand_path_cm[:, 1], hand_path_cm[:, 2], alpha=1)
      ax.set_box_aspect([ub - lb for lb, ub in (ax.get_xlim(), ax.get_ylim(), ax.get_zlim())])
      ax.set_title('Hand Path Features: '
                   'Subject %s Trial %s' % (subject_id, trial_id))
      plt.draw()
      plt_wait_for_keyboard_press()
      plt.close(fig)
    
    
  
  # Loop through each subject and trial.
  print()
  for group_name in data_file_humans.keys():
    # Check if this group is for a subject's data (or something else such as metadata)
    try:
      subject_id = int(group_name.split('subject_')[-1])
    except:
      continue
    print('See data for subject %02d' % subject_id)
    subject_group_human = data_file_humans[group_name]
    if include_robot_examples:
      subject_group_robot = data_file_robots[group_name]
    
    # Loop through each trial for this subject.
    for (trial_name, trial_group_human) in subject_group_human.items():
      trial_id = int(trial_name.split('trial_')[-1])
      if include_robot_examples:
        trial_group_robot = subject_group_robot[trial_name]
      
      # Get the timestamp for each sample, as seconds since trial start.
      time_s = np.squeeze(np.array(trial_group_human['time_s']))
      if include_robot_examples:
        assert(np.array_equal(np.squeeze(np.array(trial_group_robot['time_s'])), time_s))
      
      # Get the global xyz position of each body segment for the human demonstration.
      # It is a Tx23x3 matrix, indexed as [timestep][body_segment_index][xyz].
      body_segment_position_m = np.squeeze(np.array(trial_group_human['body_segment_position_m']))
      # Get the orientation of each body segment as a quaternion.
      # It is a Tx23x4 matrix, indexed as [timestep][body_segment_index][wijk].
      body_segment_quaternion_wijk = np.squeeze(np.array(trial_group_human['body_segment_quaternion_wijk']))
      # Get the joint angles as radians.
      # It is a Tx9x3 matrix, indexed as [timestep][body_joint_index][xyz].
      body_joint_angle_eulerZXY_xyz_rad = np.squeeze(np.array(trial_group_human['joint_angle_eulerZXY_xyz_rad']))
      body_joint_angle_eulerXZY_xyz_rad = np.squeeze(np.array(trial_group_human['joint_angle_eulerXZY_xyz_rad']))
      # Get the global xyz positions, orientations, and timestamp
      #  of the inferred stationary position (when the person holds the pitcher or scooper relatively still).
      # NOTE: Not currently used in the features, but shown here in case it becomes useful.
      stationary_position_m = np.squeeze(np.array(trial_group_human['stationary']['body_segment_position_m']))
      stationary_quaternion_wijk = np.squeeze(np.array(trial_group_human['stationary']['body_segment_quaternion_wijk']))
      stationary_time_s = np.squeeze(np.array(trial_group_human['stationary']['time_s']))
      # Get the reference object position.
      referenceObject_position_m = np.array(trial_group_human['reference_object_position_m'])
      # Get the pitcher holding angle.
      hand_to_pitcher_angles_rad = np.array(trial_group_human['hand_to_pitcher_angles_rad'])
      
      # Add a labeled feature matrix for this trial.
      add_training_segment(time_s, body_segment_position_m, body_segment_quaternion_wijk,
                           body_joint_angle_eulerZXY_xyz_rad, body_joint_angle_eulerXZY_xyz_rad,
                           referenceObject_position_m, hand_to_pitcher_angles_rad, stationary_time_s,
                           is_human=True, subject_id=subject_id, trial_id=trial_name)
      
      # Do the same for the robot path based on this trial.
      if include_robot_examples:
        body_segment_position_m = np.squeeze(np.array(trial_group_robot['body_segment_position_m']))
        body_segment_quaternion_wijk = np.squeeze(np.array(trial_group_robot['body_segment_quaternion_wijk']))
        add_training_segment(time_s, body_segment_position_m, body_segment_quaternion_wijk, None, None, None, None, None, is_human=False)
  
  print()
  
  # Clean up.
  data_file_humans.close()
  if include_robot_examples:
    data_file_robots.close()
  
  # Save the data if desired
  if training_data_filepath is not None:
    training_data_file = h5py.File(training_data_filepath, 'w')
    training_data_file.create_dataset('labels', data=labels)
    for key in feature_matrices:
      training_data_file.create_dataset(key, data=np.array(feature_matrices[key]))
    training_data_file.close()
  
  
  
  
  
  
  
  
  
  
  
  
