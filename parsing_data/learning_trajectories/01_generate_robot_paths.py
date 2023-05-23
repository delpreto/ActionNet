
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

# Specify the folder of experiments to parse.
data_dir = os.path.realpath(os.path.join(script_dir, '..', '..', '..', 'results', 'learning_trajectories'))
# Specify the input file of extracted trajectory data.
trajectory_data_filepath_humans = os.path.join(data_dir, 'pouring_paths_humans.hdf5')
# Specify the output file for robot trajectory data.
trajectory_data_filepath_robots = os.path.join(data_dir, 'pouring_paths_robots.hdf5')

plot_human_and_robot_hand_paths = False

###################################################################
###################################################################
###################################################################

# Open the files of extracted trajectory data.
data_file_humans = h5py.File(trajectory_data_filepath_humans, 'r')
data_file_robots = h5py.File(trajectory_data_filepath_robots, 'w')

# Get the list of body segments, which can be used to index data matrices below.
body_segment_names_humans = data_file_humans['body_segment_names']
body_segment_names_humans = [name.decode('utf-8') for name in body_segment_names_humans]
print()
print('See the following list of body segments.')
print('The starred segments should be used as the hand/elbow/shoulder chain.')
for (body_segment_index, body_segment_name) in enumerate(body_segment_names_humans):
  if body_segment_name in ['Right Upper Arm', 'Right Forearm', 'Right Hand']:
    print('*', end='')
  else:
    print(' ', end='')
  print(' %02d: %s' % (body_segment_index, body_segment_name))
# Highlight segment indexes useful for the pouring trajectories:
hand_index_humans = body_segment_names_humans.index('Right Hand')
elbow_index_humans = body_segment_names_humans.index('Right Forearm')
shoulder_index_humans = body_segment_names_humans.index('Right Upper Arm')

# Save a sample list of body segments for the robot, which will correspond to the data matrices below.
# TODO Update if needed
body_segment_names_robots = ['Right Hand', 'Right Forearm', 'Right Upper Arm']
hand_index_robots = body_segment_names_robots.index('Right Hand')
elbow_index_robots = body_segment_names_robots.index('Right Forearm')
shoulder_index_robots = body_segment_names_robots.index('Right Upper Arm')
data_file_robots.create_dataset('body_segment_names', data=body_segment_names_robots)

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
  
  # Make an analogous group for the robot trajectories.
  subject_group_robot = data_file_robots.create_group(group_name)
  
  # Loop through each trial for this subject.
  for (trial_name, trial_group_human) in subject_group_human.items():
    trial_id = int(trial_name.split('trial_')[-1])
    
    # Get the timestamp for each sample, as seconds since trial start.
    time_s = np.squeeze(np.array(trial_group_human['time_s']))
    # Get the global xyz position of each body segment.
    # It is a Tx23x3 matrix, indexed as [timestep][body_segment_index][xyz].
    body_segment_position_cm = np.squeeze(np.array(trial_group_human['body_segment_position_cm']))
    # Get the orientation of each body segment as a quaternion.
    # It is a Tx23x4 matrix, indexed as [timestep][body_segment_index][xyzw].
    body_segment_quaternion = np.squeeze(np.array(trial_group_human['body_segment_quaternion']))
    # Get the global xyz positions, orientations, and timestamp
    #  of the inferred pour position (when the person holds the pitcher relatively still).
    pouring_position_cm = np.squeeze(np.array(trial_group_human['pouring']['body_segment_position_cm']))
    pouring_quaternion = np.squeeze(np.array(trial_group_human['pouring']['body_segment_quaternion']))
    pouring_time_s = np.squeeze(np.array(trial_group_human['pouring']['time_s']))
    pouring_index = np.where(abs(pouring_time_s - time_s) == np.min(abs(pouring_time_s - time_s)))[0][-1]
    
    # Highlight data that should be used to generate a robot path.
    # The path should have the following characteristics:
    #   A sample at each timestamp in sampling_times_s
    #   At start_time_s (probably 0):
    #     The hand/elbow/shoulder should be at start_positions_cm[hand_index/elbow_index/shoulder_index]
    #     The hand orientation should be given by start_quaternions[hand_index]
    #   At pour_time_s:
    #     The hand/elbow/shoulder should be at pour_positions_cm[hand_index/elbow_index/shoulder_index]
    #     The hand orientation should be given by pour_quaternions[hand_index]
    #   At end_time_s (the last sample):
    #     The hand/elbow/shoulder should be at end_positions_cm[hand_index/elbow_index/shoulder_index]
    #     The hand orientation should be given by end_quaternions[hand_index]
    start_positions_cm = np.squeeze(body_segment_position_cm[0,:,:])
    pour_positions_cm = np.squeeze(body_segment_position_cm[pouring_index,:,:])
    end_positions_cm = np.squeeze(body_segment_position_cm[-1,:,:])
    start_quaternions = np.squeeze(body_segment_quaternion[0,:,:])
    pour_quaternions = np.squeeze(body_segment_quaternion[pouring_index,:,:])
    end_quaternions = np.squeeze(body_segment_quaternion[-1,:,:])
    start_time_s = time_s[0]
    pour_time_s = time_s[pouring_index]
    end_time_s = time_s[-1]
    sampling_times_s = time_s
    
    # Use linear interpolation for now.
    # TODO Use another planning method?
    pour_duration_s = 2 # might want to match pour_position_variance_buffer_duration_s used when extracting human demonstrations?
    time_s_robot = np.array([
        start_time_s,
        pour_time_s-pour_duration_s/2,
        pour_time_s+pour_duration_s/2,
        end_time_s
      ])
    body_segment_position_cm_robot = np.array([
        start_positions_cm,
        pour_positions_cm,
        pour_positions_cm,
        end_positions_cm
      ])
    body_segment_quaternion_robot = np.array([
        start_quaternions,
        pour_quaternions,
        pour_quaternions,
        end_quaternions
      ])
    fn_interpolate_data = interpolate.interp1d(
        time_s_robot, # x values
        body_segment_position_cm_robot,   # y values
        axis=0,        # axis of the data along which to interpolate
        kind='linear', # interpolation method, such as 'linear', 'zero', 'nearest', 'quadratic', 'cubic', etc.
        fill_value='extrapolate' # how to handle x values outside the original range
    )
    body_segment_position_cm_robot = fn_interpolate_data(sampling_times_s)
    fn_interpolate_data = interpolate.interp1d(
        time_s_robot, # x values
        body_segment_quaternion_robot,   # y values
        axis=0,        # axis of the data along which to interpolate
        kind='linear', # interpolation method, such as 'linear', 'zero', 'nearest', 'quadratic', 'cubic', etc.
        fill_value='extrapolate' # how to handle x values outside the original range
    )
    body_segment_quaternion_robot = fn_interpolate_data(sampling_times_s)
    
    time_s_robot = sampling_times_s

    # Only use select body segments for the interpolated robot paths.
    # Should match body_segment_names_robots defined above.
    body_segment_position_cm_robot = body_segment_position_cm_robot[:,[hand_index_humans, elbow_index_humans, shoulder_index_humans],:]
    body_segment_quaternion_robot = body_segment_quaternion_robot[:,[hand_index_humans, elbow_index_humans, shoulder_index_humans],:]
    
    if plot_human_and_robot_hand_paths:
      fig = plt.figure()
      figManager = plt.get_current_fig_manager()
      figManager.window.showMaximized()
      fig.add_subplot(1, 1, 1, projection='3d')
      ax = fig.get_axes()[0]
      ax.view_init(16, 44)
      ax.set_xlabel('X [cm]')
      ax.set_ylabel('Y [cm]')
      ax.set_zlabel('Z [cm]')
      human_hand_path = np.squeeze(body_segment_position_cm[:,hand_index_humans,:])
      print(human_hand_path.shape)
      robot_hand_path = np.squeeze(body_segment_position_cm_robot[:,hand_index_robots,:])
      ax.plot3D(human_hand_path[:, 0], human_hand_path[:, 1], human_hand_path[:, 2], alpha=1)
      ax.plot3D(robot_hand_path[:, 0], robot_hand_path[:, 1], robot_hand_path[:, 2], alpha=1)
      ax.set_box_aspect([ub - lb for lb, ub in (ax.get_xlim(), ax.get_ylim(), ax.get_zlim())])
      ax.set_title('Subject %02d Trial %02d' % (subject_id, trial_id))
      plt.show()
    
    
    # Save robot data.
    # Make an analogous group for the robot trajectories.
    trial_group_robot = subject_group_robot.create_group(trial_name)
    trial_group_robot.create_dataset('body_segment_position_cm', data=body_segment_position_cm_robot)
    trial_group_robot.create_dataset('body_segment_quaternion', data=body_segment_quaternion_robot)
    trial_group_robot.create_dataset('time_s', data=time_s_robot)
    
print()

# Clean up.
data_file_humans.close()
data_file_robots.close()









