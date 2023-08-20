
############
#
# Copyright (c) 2022 MIT CSAIL and Joseph DelPreto
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
# See https://action-net.csail.mit.edu for more usage information.
# Created 2021-2022 for the MIT ActionNet project by Joseph DelPreto [https://josephdelpreto.com].
#
############

import h5py
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import os
script_dir = os.path.dirname(os.path.realpath(__file__))

# CHOOSE THE ACTIVITY TO PROCESS
configure_for_pouring = False # otherwise will be scooping

# Specify the folder of experiments to parse.
data_dir = os.path.realpath(os.path.join(script_dir, '..', '..', '..', 'results', 'learning_trajectories'))
# Specify the input files of extracted trajectory data.
trajectory_data_filepath_humans = os.path.join(data_dir, '%s_paths_humans.hdf5' % ('pouring' if configure_for_pouring else 'scooping'))
trajectory_data_filepath_robots = os.path.join(data_dir, '%s_paths_robots.hdf5' % ('pouring' if configure_for_pouring else 'scooping'))
# Specify the output file for the feature matrices and labels.
training_data_filepath = os.path.join(data_dir, '%s_training_data.hdf5' % ('pouring' if configure_for_pouring else 'scooping')) # None to not save
referenceOjbect_positions_filepath = os.path.join(data_dir, '%s_training_referenceObject_positions.hdf5' % ('pouring' if configure_for_pouring else 'scooping')) # None to not save

# Specify how to standardize training data segment lengths.
# Will currently normalize time of each trial to be from 0 to 1,
#  then resample it to have the following number of samples.
num_normalized_timesteps = 100
normalized_timestamps = np.linspace(0, 1, num_normalized_timesteps)

plot_hand_path_features = False

###################################################################
###################################################################
###################################################################

# Open the files of extracted trajectory data.
data_file_humans = h5py.File(trajectory_data_filepath_humans, 'r')
data_file_robots = h5py.File(trajectory_data_filepath_robots, 'r')

# Get the list of body segments, which can be used to index data matrices below.
body_segment_names_humans = data_file_humans['body_segment_names']
body_segment_names_humans = [name.decode('utf-8') for name in body_segment_names_humans]
body_segment_names_robots = data_file_robots['body_segment_names']
body_segment_names_robots = [name.decode('utf-8') for name in body_segment_names_robots]
print()
print('See the following list of body segments (in the human data).')
print('The starred segments should be used as the hand/elbow/shoulder chain.')
for (body_segment_index, body_segment_name) in enumerate(body_segment_names_humans):
  if body_segment_name in ['Right Upper Arm', 'Right Forearm', 'Right Hand']:
    print('*', end='')
  else:
    print(' ', end='')
  print(' %02d: %s' % (body_segment_index, body_segment_name))
# Highlight segment indexes useful for the activity trajectories:
hand_index_humans = body_segment_names_humans.index('Right Hand')
elbow_index_humans = body_segment_names_humans.index('Right Forearm')
shoulder_index_humans = body_segment_names_humans.index('Right Upper Arm')
hand_index_robots = body_segment_names_robots.index('Right Hand')
elbow_index_robots = body_segment_names_robots.index('Right Forearm')
shoulder_index_robots = body_segment_names_robots.index('Right Upper Arm')

# Create feature matrices for each trial, labeled as human or robot.
# The current matrices will concatenate shoulder, elbow, and hand positions and hand orientation.
feature_matrices = []
labels = []

# Helper to create a feature matrix from the processed trajectory data.
def add_training_segment(time_s, body_segment_position_cm, body_segment_quaternion, is_human):
  
  # Concatenate position and orientation features.
  # Result will be Tx13, where T is timesteps and 13 is (xyz axes) x (hand, elbow, shoulder) + (xyzw quaternion) x (hand)
  num_timesteps = body_segment_position_cm.shape[0]
  if is_human:
    position_features = np.reshape(body_segment_position_cm[:,[hand_index_humans, elbow_index_humans, shoulder_index_humans],:], (num_timesteps, -1)) # unwrap xyz from each segment
    orientation_features =  np.reshape(body_segment_quaternion[:,hand_index_humans,:], (num_timesteps, -1))
  else:
    position_features = np.reshape(body_segment_position_cm[:,[hand_index_robots, elbow_index_robots, shoulder_index_robots],:], (num_timesteps, -1)) # unwrap xyz from each segment
    orientation_features =  np.reshape(body_segment_quaternion[:,hand_index_robots,:], (num_timesteps, -1))
  feature_matrix = np.concatenate([position_features, orientation_features], axis=1)

  # Normalize time and resample.
  time_s = time_s - min(time_s)
  time_s = time_s / max(time_s)
  fn_interpolate_data = interpolate.interp1d(
      time_s, # x values
      feature_matrix,   # y values
      axis=0,        # axis of the data along which to interpolate
      kind='linear', # interpolation method, such as 'linear', 'zero', 'nearest', 'quadratic', 'cubic', etc.
      fill_value='extrapolate' # how to handle x values outside the original range
  )
  feature_matrix_resampled = fn_interpolate_data(normalized_timestamps)

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
    hand_path = np.squeeze(feature_matrix_resampled[:,0:3])
    print(hand_path.shape)
    ax.plot3D(hand_path[:, 0], hand_path[:, 1], hand_path[:, 2], alpha=1)
    ax.set_box_aspect([ub - lb for lb, ub in (ax.get_xlim(), ax.get_ylim(), ax.get_zlim())])
    plt.show()
  
  feature_matrices.append(feature_matrix_resampled)
  labels.append('human' if is_human else 'robot')
  

# Loop through each subject and trial.
print()
referenceObject_positions_cm = []
for group_name in data_file_humans.keys():
  # Check if this group is for a subject's data (or something else such as metadata)
  try:
    subject_id = int(group_name.split('subject_')[-1])
  except:
    continue
  print('See data for subject %02d' % subject_id)
  subject_group_human = data_file_humans[group_name]
  subject_group_robot = data_file_robots[group_name]
  
  # Loop through each trial for this subject.
  for (trial_name, trial_group_human) in subject_group_human.items():
    trial_id = int(trial_name.split('trial_')[-1])
    trial_group_robot = subject_group_robot[trial_name]
    
    # Get the timestamp for each sample, as seconds since trial start.
    # NOTE: Not currently used in the features, but can be used to ensure
    #  that the human and robot paths used the same timesteps.
    time_s = np.squeeze(np.array(trial_group_human['time_s']))
    assert(np.array_equal(np.squeeze(np.array(trial_group_robot['time_s'])), time_s))
    
    # Get the global xyz position of each body segment for the human demonstration.
    # It is a Tx23x3 matrix, indexed as [timestep][body_segment_index][xyz].
    body_segment_position_cm = np.squeeze(np.array(trial_group_human['body_segment_position_cm']))
    # Get the orientation of each body segment as a quaternion.
    # It is a Tx23x4 matrix, indexed as [timestep][body_segment_index][xyzw].
    body_segment_quaternion = np.squeeze(np.array(trial_group_human['body_segment_quaternion']))
    # Get the global xyz positions, orientations, and timestamp
    #  of the inferred stationary position (when the person holds the pitcher or scooper relatively still).
    # NOTE: Not currently used in the features, but shown here in case it becomes useful.
    stationary_position_cm = np.squeeze(np.array(trial_group_human['stationary']['body_segment_position_cm']))
    stationary_quaternion = np.squeeze(np.array(trial_group_human['stationary']['body_segment_quaternion']))
    stationary_time_s = np.squeeze(np.array(trial_group_human['stationary']['time_s']))
    stationary_index = np.where(abs(stationary_time_s - time_s) == np.min(abs(stationary_time_s - time_s)))[0][-1]
    
    # Add a labeled feature matrix for this trial.
    add_training_segment(time_s, body_segment_position_cm, body_segment_quaternion, is_human=True)
    
    # Do the same for the robot path based on this trial.
    body_segment_position_cm = np.squeeze(np.array(trial_group_robot['body_segment_position_cm']))
    body_segment_quaternion = np.squeeze(np.array(trial_group_robot['body_segment_quaternion']))
    add_training_segment(time_s, body_segment_position_cm, body_segment_quaternion, is_human=False)

    # Get the reference object position.
    # Add it twice since it will be the same for human and robot examples that were just added.
    referenceObject_positions_cm.append(np.array(trial_group_human['reference_object']['position_cm']))
    referenceObject_positions_cm.append(np.array(trial_group_human['reference_object']['position_cm']))

print()

# Make a single matrix from the reference object positions.
referenceObject_positions_cm = np.stack(referenceObject_positions_cm)

# Clean up.
data_file_humans.close()
data_file_robots.close()

# Save the data if desired
if training_data_filepath is not None:
  training_data_file = h5py.File(training_data_filepath, 'w')
  training_data_file.create_dataset('feature_matrices', data=np.array(feature_matrices))
  training_data_file.create_dataset('labels', data=labels)
  training_data_file.close()
  # Write a file with the reference object positions.
  referenceObject_positions_file = h5py.File(referenceOjbect_positions_filepath, 'w')
  referenceObject_positions_file.create_dataset('position_cm', data=np.array(referenceObject_positions_cm))
  referenceObject_positions_file.close()












