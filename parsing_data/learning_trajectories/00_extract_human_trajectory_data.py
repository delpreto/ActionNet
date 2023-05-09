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
from scipy.spatial.transform import Rotation
from collections import OrderedDict
import os
script_dir = os.path.dirname(os.path.realpath(__file__))
import sys
import cv2

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d

from utils.numpy_scipy_utils import *
from utils.print_utils import *

# Specify the subjects to consider.
subject_ids_filter = None # None to consider all subjects

# Specify the folder of experiments to parse.
data_dir = os.path.realpath(os.path.join(script_dir, '..', '..', '..', 'data'))
experiments_dir = os.path.join(data_dir, 'experiments')
output_dir = os.path.realpath(os.path.join(script_dir, '..', '..', '..', 'results', 'learning_trajectories'))

animate_pouring_plots = False # show an animated plot of the skeleton for each trial
plot_all_pouring_paths = False # make a subplot for each subject, which shows all paths from that subject
save_eye_videos = False # save the eye-tracking video for each trial
save_composite_videos = False # save the eye-tracking video and animated plot for each trial
save_results_data = True

resampled_fs_hz = 50

use_manual_startEnd_times = True
manual_pouring_start_times_s = [
  '2022-06-07 18:44:42.541198',
  '2022-06-07 18:45:16.441165',
  '2022-06-07 18:45:30.841151',
  '2022-06-07 18:45:42.141141',
  '2022-06-07 18:45:55.741128',
  '2022-06-13 22:39:48.417483',
  '2022-06-13 22:39:59.017473',
  '2022-06-13 22:40:07.017465',
  '2022-06-13 22:40:15.917457',
  '2022-06-13 22:40:26.017447',
  '2022-06-14 14:03:57.492392',
  '2022-06-14 14:04:09.692380',
  '2022-06-14 14:04:18.992372',
  '2022-06-14 14:04:28.492362',
  '2022-06-14 14:04:36.792355',
  '2022-06-14 17:18:49.326952',
  '2022-06-14 17:18:58.226943',
  '2022-06-14 17:19:06.226936',
  '2022-06-14 17:19:14.026928',
  '2022-06-14 17:19:23.026920',
  '2022-06-14 21:19:39.860983',
  '2022-06-14 21:19:48.660974',
  '2022-06-14 21:19:56.460967',
  '2022-06-14 21:20:04.160959',
  '2022-06-14 21:20:11.260953',
  '2022-07-12 15:27:07.005319',
  '2022-07-12 15:27:36.105291',
  '2022-07-12 15:27:47.505280',
  '2022-07-12 15:27:58.005270',
  '2022-07-12 15:28:08.005260',
  '2022-07-13 11:50:11.389303',
  '2022-07-13 11:50:18.789296',
  '2022-07-13 11:50:25.389290',
  '2022-07-13 11:50:30.389285',
  '2022-07-13 11:50:36.889279',
  '2022-07-13 15:04:19.351326',
  '2022-07-13 15:04:30.151315',
  '2022-07-13 15:04:42.751303',
  '2022-07-13 15:04:54.551292',
  '2022-07-13 15:05:04.451283',
  '2022-07-14 10:39:23.020857',
  '2022-07-14 10:39:33.020847',
  '2022-07-14 10:39:43.020838',
  '2022-07-14 10:39:51.620830',
  '2022-07-14 10:40:02.020820',
]
manual_pouring_end_times_s = [
  '2022-06-07 18:44:52.841188',
  '2022-06-07 18:45:26.041156',
  '2022-06-07 18:45:39.141144',
  '2022-06-07 18:45:50.541133',
  '2022-06-07 18:46:04.141120',
  '2022-06-13 22:39:55.617476',
  '2022-06-13 22:40:05.817466',
  '2022-06-13 22:40:14.117459',
  '2022-06-13 22:40:22.417450',
  '2022-06-13 22:40:32.117441',
  '2022-06-14 14:04:09.492381',
  '2022-06-14 14:04:16.992373',
  '2022-06-14 14:04:26.792364',
  '2022-06-14 14:04:34.692357',
  '2022-06-14 14:04:44.492347',
  '2022-06-14 17:18:54.826946',
  '2022-06-14 17:19:02.326939',
  '2022-06-14 17:19:10.826931',
  '2022-06-14 17:19:18.526924',
  '2022-06-14 17:19:27.526915',
  '2022-06-14 21:19:46.160977',
  '2022-06-14 21:19:54.260969',
  '2022-06-14 21:20:01.760962',
  '2022-06-14 21:20:09.660954',
  '2022-06-14 21:20:16.360948',
  '2022-07-12 15:27:14.305312',
  '2022-07-12 15:27:43.805284',
  '2022-07-12 15:27:54.405273',
  '2022-07-12 15:28:05.005263',
  '2022-07-12 15:28:15.305254',
  '2022-07-13 11:50:18.289297',
  '2022-07-13 11:50:24.889290',
  '2022-07-13 11:50:30.189285',
  '2022-07-13 11:50:36.089280',
  '2022-07-13 11:50:42.989273',
  '2022-07-13 15:04:25.751319',
  '2022-07-13 15:04:38.451307',
  '2022-07-13 15:04:50.551296',
  '2022-07-13 15:05:01.251286',
  '2022-07-13 15:05:12.051275',
  '2022-07-14 10:39:31.120849',
  '2022-07-14 10:39:39.820841',
  '2022-07-14 10:39:49.320832',
  '2022-07-14 10:39:58.720823',
  '2022-07-14 10:40:07.320815',
]

bodySegment_labels = [
  'Pelvis',
  'L5',
  'L3',
  'T12',
  'T8',
  'Neck',
  'Head',
  'Right Shoulder',
  'Right Upper Arm',
  'Right Forearm',
  'Right Hand',
  'Left Shoulder',
  'Left Upper Arm',
  'Left Forearm',
  'Left Hand',
  'Right Upper Leg',
  'Right Lower Leg',
  'Right Foot',
  'Right Toe',
  'Left Upper Leg',
  'Left Lower Leg',
  'Left Foot',
  'Left Toe',
]

bodySegment_chains_labels_toPlot = {
  # 'Left Leg':  ['Left Upper Leg', 'Left Lower Leg', 'Left Foot', 'Left Toe'],
  # 'Right Leg': ['Right Upper Leg', 'Right Lower Leg', 'Right Foot', 'Right Toe'],
  'Spine':     ['Head', 'Neck', 'T8', 'T12', 'L3', 'L5', 'Pelvis'], # top down
  # 'Hip':       ['Left Upper Leg', 'Pelvis', 'Right Upper Leg'],
  'Shoulders': ['Left Upper Arm', 'Left Shoulder', 'Right Shoulder', 'Right Upper Arm'],
  'Left Arm':  ['Left Upper Arm', 'Left Forearm', 'Left Hand'],
  'Right Arm': ['Right Upper Arm', 'Right Forearm', 'Right Hand'],
}

# hand_box_dimensions_cm = np.array([2, 9, 18]) # open hand
hand_box_dimensions_cm = np.array([5, 9, 8]) # fist

# rolling buffer duration for looking at hand position variance when inferring pour position
pour_position_variance_buffer_duration_s = 2

###################################################################
###################################################################
###################################################################

# Define a helper to extract hand path data from an HDF5 file.
def get_bodyPath_data(h5_file):
  # Extract hand position.
  # Specify the device and stream.
  device_name = 'xsens-segments'
  stream_name = 'position_cm'

  # Get the timestamps for each entry as seconds since epoch.
  time_s = h5_file[device_name][stream_name]['time_s']
  time_s = np.squeeze(np.array(time_s)) # squeeze (optional) converts from a list of single-element lists to a 1D list
  # Get the timestamps for each entry as human-readable strings.
  time_str = h5_file[device_name][stream_name]['time_str']
  time_str = np.squeeze(np.array(time_str)) # squeeze (optional) converts from a list of single-element lists to a 1D list
  
  # Get segment position data as an NxJx3 matrix as [timestep][segment][xyz]
  segment_data = h5_file[device_name][stream_name]['data']
  segment_data = np.array(segment_data)
  bodySegment_xyz_data = OrderedDict()
  for (segment_index, segment_name) in enumerate(bodySegment_labels):
    bodySegment_xyz_data[segment_name] = np.squeeze(segment_data[:, segment_index, :])
  # # Find the index of the hand segment in the metadata, then use that to slice the matrix.
  # metadata = dict(h5_file[device_name][stream_name].attrs.items())
  # segment_names = eval(metadata['Data headings'])
  # segmentNames_xCoords = [segment_name for segment_name in segment_names if '(x)' in segment_name]
  # rightHand_segment_index = segmentNames_xCoords.index('Right Hand (x)')

  # Get segment orientation data as an NxJx3 matrix as [timestep][segment][xyz]
  device_name = 'xsens-segments'
  stream_name = 'orientation_quaternion'
  segment_data = h5_file[device_name][stream_name]['data']
  segment_data = np.array(segment_data)
  bodySegment_quaternion_data = OrderedDict()
  for (segment_index, segment_name) in enumerate(bodySegment_labels):
    bodySegment_quaternion_data[segment_name] = np.squeeze(segment_data[:, segment_index, :])
  
  # Combine the position and orientation data.
  # NOTE: Assumes the timestamps are the same.
  bodySegment_data = {
    'position_cm': bodySegment_xyz_data,
    'quaternion': bodySegment_quaternion_data
  }
  return (time_s, bodySegment_data)

# Helper to extract path data for specified activities.
def get_pouring_bodyPath_data(h5_file, start_times_s, end_times_s):
  # Get path data throughout the whole experiment.
  (time_s, bodySegment_data) = get_bodyPath_data(h5_file)
  
  times_s = []
  bodySegment_datas = []
  # Extract each desired activity.
  for time_index in range(len(start_times_s)):
    start_time_s = start_times_s[time_index]
    end_time_s = end_times_s[time_index]
    indexes_forLabel = np.where((time_s >= start_time_s) & (time_s <= end_time_s))[0]
    if indexes_forLabel.size > 0:
      times_s.append(time_s[indexes_forLabel])
      bodySegment_datas.append(OrderedDict())
      for data_type in bodySegment_data.keys():
        bodySegment_datas[-1].setdefault(data_type, OrderedDict())
        for body_segment in bodySegment_data[data_type].keys():
          bodySegment_datas[-1][data_type].setdefault(body_segment, OrderedDict())
          bodySegment_datas[-1][data_type][body_segment] = bodySegment_data[data_type][body_segment][indexes_forLabel, :]
          # if data_type == 'position_cm':
          #   print(data_type, body_segment)
          #   print('start time index', time_index, 'start time', start_time_s, get_time_str(start_time_s))
          #   print('first entry index', indexes_forLabel[0])
          #   print(bodySegment_datas[-1][data_type][body_segment])
          
  # Rotate each segment to a coordinate frame based on the body, such that
  #  the y axis is aligned with the shoudlers/hips.
  # print()
  for trial_index in range(len(bodySegment_datas)):
    # Set the origin between the hips (very close to the pelvis but not exactly).
    # pelvis_position_cm = bodySegment_datas[trial_index]['position_cm']['Pelvis']
    # origin_cm = pelvis_position_cm[0,:]
    origin_cm = np.mean(np.array(
          [bodySegment_datas[trial_index]['position_cm']['Right Upper Leg'][0, :],
          bodySegment_datas[trial_index]['position_cm']['Left Upper Leg'][0, :]]), axis=0)
    # print(origin_cm)
    for body_segment in bodySegment_data['position_cm'].keys():
      position_cm = bodySegment_datas[trial_index]['position_cm'][body_segment]
      position_cm = position_cm - origin_cm
      bodySegment_datas[trial_index]['position_cm'][body_segment] = position_cm
    
    # Use the hip orientation to create the y axis.
    y_axis_right = np.append(bodySegment_datas[trial_index]['position_cm']['Right Upper Leg'][0, 0:2], 0)
    y_axis_left = np.append(bodySegment_datas[trial_index]['position_cm']['Left Upper Leg'][0, 0:2], 0)
    # # Average the shoulders and hips (projected onto xy plane) to create a y axis.
    # y_axis_right = np.mean(np.array(
    #   [bodySegment_datas[trial_index]['position_cm']['Right Upper Arm'][0, 0:2],
    #   bodySegment_datas[trial_index]['position_cm']['Right Upper Leg'][0, 0:2]]), axis=0)
    # y_axis_left = np.mean(np.array(
    #   [bodySegment_datas[trial_index]['position_cm']['Left Upper Arm'][0, 0:2],
    #   bodySegment_datas[trial_index]['position_cm']['Left Upper Leg'][0, 0:2]]), axis=0)
    y_axis_center = np.mean(np.array([y_axis_right, y_axis_left]), axis=0)
    rotation_matrix = rotation_matrix_from_vectors(y_axis_right, [0, 1, 0])
    for body_segment in bodySegment_data['position_cm'].keys():
      position_cm = bodySegment_datas[trial_index]['position_cm'][body_segment]
      for time_index in range(position_cm.shape[0]):
        position_cm[time_index,:] = rotation_matrix.dot(position_cm[time_index,:])
    # NOTE: The segment orientations probably don't need to be adjusted,
    #   since they represent intrinsic rotations that shouldn't be affected by global frame drift.
    #  This might mean that a segment orientation isn't quite aligned with the body path,
    #   but if the above hip frame is close to the true global frame then it should be close to correct.
    #  For example, rotating the hand according to its quaternion should create the correct
    #   hand pose in global space, but then translating it to the hand position at the end
    #   of the arm may make it at the wrong relative angle compared to the rest of the arm
    #   since the arm joints are positioned based on global position estimates.
    #   But since we correct the frame above to remove drift (hopefully), attaching
    #   the hand at the end should work.
    #  This was visually confirmed with the hand orientation relative to the arm
    #   (it looks like the pose in the eye-tracking video after imposing the hip frame
    #    but not changing the quaternion), but it wasn't quantified.
    
  return (times_s, bodySegment_datas)

# Resample to a consistent rate
def resample_data(times_s, bodySegment_datas):
  for trial_index in range(len(times_s)):
    time_s = times_s[trial_index]
    time_s_resampled = np.arange(min(time_s), max(time_s), 1/resampled_fs_hz)
    for data_type in bodySegment_datas[trial_index].keys():
      for (body_segment, data) in bodySegment_datas[trial_index][data_type].items():
        fn_interpolate_data = interpolate.interp1d(
            time_s, # x values
            data,   # y values
            axis=0,        # axis of the data along which to interpolate
            kind='linear', # interpolation method, such as 'linear', 'zero', 'nearest', 'quadratic', 'cubic', etc.
            fill_value='extrapolate' # how to handle x values outside the original range
        )
        data_resampled = fn_interpolate_data(time_s_resampled)
        bodySegment_datas[trial_index][data_type][body_segment] = data_resampled
    times_s[trial_index] = time_s_resampled
  return (times_s, bodySegment_datas)

# Infer the hand position when the water is being poured.
def infer_pouring_pose(times_s, bodySegment_datas):
  body_positions_pouring_cm = []
  pouring_times_s = []
  for trial_index, bodySegment_data in enumerate(bodySegment_datas):
    body_position_cm = bodySegment_datas[trial_index]['position_cm']
    body_quaternion = bodySegment_datas[trial_index]['quaternion']
    time_s = times_s[trial_index]
    num_timesteps = time_s.shape[0]
    min_average_distance_cm = None
    min_average_distance_buffer_start_index = None
    min_average_distance_buffer_end_index = None
    body_position_pouring_buffer_cm = None
    body_quaternion_pouring_buffer = None
    average_distances_cm = []
    for buffer_start_index in range(num_timesteps):
      buffer_start_time_s = time_s[buffer_start_index]
      buffer_end_time_s = buffer_start_time_s + pour_position_variance_buffer_duration_s
      if buffer_end_time_s > time_s[-1]:
        break
      buffer_end_index = np.where(time_s <= buffer_end_time_s)[0][-1]
      body_position_buffers_cm = dict([(name, position_cm[buffer_start_index:buffer_end_index, :]) for (name, position_cm) in body_position_cm.items()])
      body_quaternion_buffers = dict([(name, quaternion[buffer_start_index:buffer_end_index, :]) for (name, quaternion) in body_quaternion.items()])
      median_hand_position_cm = np.median(body_position_buffers_cm['Right Hand'], axis=0)
      distances_cm = np.linalg.norm(body_position_buffers_cm['Right Hand'] - median_hand_position_cm, axis=1)
      average_distance_cm = np.mean(distances_cm, axis=0)
      average_distances_cm.append(average_distance_cm)
      if min_average_distance_cm is None or average_distance_cm < min_average_distance_cm:
        min_average_distance_cm = average_distance_cm
        min_average_distance_buffer_start_index = buffer_start_index
        min_average_distance_buffer_end_index = buffer_end_index
        body_position_pouring_buffer_cm = body_position_buffers_cm
        body_quaternion_pouring_buffer = body_quaternion_buffers
    # print(time_s[min_average_distance_buffer_start_index] - min(time_s),
    #       time_s[min_average_distance_buffer_end_index] - min(time_s))
    body_positions_pouring_cm.append({
      'position_cm':
        dict([(name, np.median(position_cm, axis=0)) for (name, position_cm) in body_position_pouring_buffer_cm.items()]),
      'quaternion':
        dict([(name, quaternion[int(quaternion.shape[0]/2),:]) for (name, quaternion) in body_quaternion_pouring_buffer.items()]),
      })
    pouring_times_s.append(time_s[int(np.mean([min_average_distance_buffer_start_index,
                                               min_average_distance_buffer_end_index]))]-min(time_s))
  # print(pouring_times_s)
  return (pouring_times_s, body_positions_pouring_cm)
  
# Plot hand paths.
def plt_wait_for_keyboard_press(timeout_s=-1.0):
  keyboardClick=False
  while keyboardClick == False:
    keyboardClick = plt.waitforbuttonpress(timeout=timeout_s)

def plot_3d_box(ax, center, quaternion): # function created using ChatGPT
  # Define vertices of a unit box
  corners = np.array([
    [-1, -1, -1],
    [-1, -1, 1],
    [-1, 1, -1],
    [-1, 1, 1],
    [1, -1, -1],
    [1, -1, 1],
    [1, 1, -1],
    [1, 1, 1]
  ]) * 0.5

  # Define faces of the box, using corner indexes
  faces = np.array([
    [0, 1, 3, 2],
    [0, 2, 6, 4],
    [0, 1, 5, 4],
    [4, 5, 7, 6],
    [1, 3, 7, 5],
    [2, 3, 7, 6]
  ])
  
  # Scale the box
  corners = corners * hand_box_dimensions_cm
  
  # Rotate the box using the quaternion
  rot = Rotation.from_quat(quaternion).as_matrix()
  corners = np.dot(corners, rot)
  
  # Translate the box
  corners = corners + center
  
  # Plot the box
  # ax.set_box_aspect([np.ptp(corners[:,dim]) for dim in range(3)])
  # ax.set_xlim3d(corners[:,0].min(), corners[:,0].max())
  # ax.set_ylim3d(corners[:,1].min(), corners[:,1].max())
  # ax.set_zlim3d(corners[:,2].min(), corners[:,2].max())
  box = art3d.Poly3DCollection([corners[face] for face in faces],
                               alpha=0.8,
                               facecolor=0.5*np.array([1,1,1]),
                               edgecolor=0.3*np.array([1,1,1]))
  ax.add_collection3d(box)
  return box

def plot_handPath_data(fig, subplot_index,
                       times_s, bodySegment_datas, pouring_pose,
                       subject_id, num_subjects,
                       spf=0.1,  # None to view all frames
                       include_skeleton=True,
                       trial_indexes_filter=None,
                       target_times_s=None, start_times_s=None, end_times_s=None,
                       pause_between_trials=False, pause_between_frames=True,
                       clear_axes_each_trial=True, hide_figure_window=False):
  if fig is None:
    if hide_figure_window:
      matplotlib.use("Agg")
    fig = plt.figure()
    if not hide_figure_window:
      figManager = plt.get_current_fig_manager()
      figManager.window.showMaximized()
    plt.ion()
    num_rows = int(np.sqrt(num_subjects))
    num_cols = int(np.ceil(num_subjects/num_rows))
    for s in range(num_subjects):
      fig.add_subplot(num_rows, num_cols, s+1, projection='3d')
  ax = fig.get_axes()[subplot_index]
  for trial_index, bodySegment_data in enumerate(bodySegment_datas):
    if trial_indexes_filter is not None and trial_index not in trial_indexes_filter:
      continue
    if clear_axes_each_trial:
      ax.clear()
    time_s = times_s[trial_index]
    
    # Add labels and titles.
    ax.set_xlabel('X [cm]')
    ax.set_ylabel('Y [cm]')
    ax.set_zlabel('Z [cm]')
    
    # Set the view angle
    ax.view_init(16, 44)
    # ax.view_init(90, 0)
    
    # Plot trajectories of the right arm and pelvis.
    hand_position_cm = bodySegment_data['position_cm']['Right Hand']
    forearm_position_cm = bodySegment_data['position_cm']['Right Forearm']
    upperArm_position_cm = bodySegment_data['position_cm']['Right Upper Arm']
    pelvis_position_cm = bodySegment_data['position_cm']['Pelvis']
    h_hand_path = ax.plot3D(hand_position_cm[:, 0], hand_position_cm[:, 1], hand_position_cm[:, 2], alpha=1)
    if include_skeleton:
      ax.plot3D(forearm_position_cm[:, 0], forearm_position_cm[:, 1], forearm_position_cm[:, 2], alpha=0.3)
      ax.plot3D(upperArm_position_cm[:, 0], upperArm_position_cm[:, 1], upperArm_position_cm[:, 2], alpha=0.3)
      ax.plot3D(pelvis_position_cm[:, 0], pelvis_position_cm[:, 1], pelvis_position_cm[:, 2], alpha=0.3)
    
    # Plot origin and start/end/pour hand positions.
    ax.scatter(0, 0, 0, s=25, color=[0, 0, 0])
    ax.scatter(hand_position_cm[0, 0], hand_position_cm[0, 1], hand_position_cm[0, 2], s=25, color='g')
    ax.scatter(hand_position_cm[-1, 0], hand_position_cm[-1, 1], hand_position_cm[-1, 2], s=25, color='r')
    ax.scatter(pouring_pose[trial_index]['position_cm']['Right Hand'][0],
               pouring_pose[trial_index]['position_cm']['Right Hand'][1],
               pouring_pose[trial_index]['position_cm']['Right Hand'][2],
               s=25, color=h_hand_path[0].get_color(), edgecolor='k')
    
    if include_skeleton:
      # Animate the whole skeleton.
      h_chains = []
      h_scatters = []
      h_hand = None
      sampling_rate_hz = (time_s.shape[0]-1)/(time_s[-1] - time_s[0])
      spf = spf or 1/sampling_rate_hz
      timestep_interval = max([1, int(sampling_rate_hz*spf)])
      for time_index in range(0, hand_position_cm.shape[0], timestep_interval):
        if start_times_s is not None and time_s[time_index] < start_times_s[trial_index]:
          continue
        if end_times_s is not None and time_s[time_index] > end_times_s[trial_index]:
          continue
        if target_times_s is not None and abs(time_s[time_index] - target_times_s[trial_index]) > min(abs(time_s - target_times_s[trial_index])):
          continue
        for i in range(len(h_chains)):
          h_chains[i][0].remove()
          h_scatters[i].remove()
        if h_hand is not None:
          h_hand.remove()
        h_chains = []
        h_scatters = []
        h_hand = None
        
        # Draw the skeleton chains
        for chain_name, segment_names in bodySegment_chains_labels_toPlot.items():
          x = []
          y = []
          z = []
          for segment_name in segment_names:
            position_cm = bodySegment_data['position_cm'][segment_name]
            x.append(position_cm[time_index, 0])
            y.append(position_cm[time_index, 1])
            z.append(position_cm[time_index, 2])
          h_chains.append(ax.plot3D(x, y, z, color='k'))
          h_scatters.append(ax.scatter(x, y, z, s=25, color=[0.5, 0.5, 0.5]))
        
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        z_lim = ax.get_zlim()
        
        # Visualize a box as the hand.
        # hand_dimensions_cm = [1, 3, 5]
        # hand_rotation_matrix = Rotation.from_quat(bodySegment_data['quaternion']['Right Hand'])
        # print(hand_rotation_matrix.apply(hand_dimensions_cm))
        # hand_box_data = np.ones(hand_dimensions_cm, dtype=bool)
        # hand_colors = np.empty(hand_dimensions_cm + [4], dtype=np.float32)
        # hand_colors[:] = [1, 0, 0, 0.8]
        # h_hand = ax.voxels(hand_box_data, facecolors=hand_colors)
        h_hand = plot_3d_box(ax, center=bodySegment_data['position_cm']['Right Hand'][time_index, :],
                             quaternion=bodySegment_data['quaternion']['Right Hand'][time_index, :])
        
        # Set the aspect ratio
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_zlim(z_lim)
        ax.set_box_aspect([ub - lb for lb, ub in (x_lim, y_lim, z_lim)])
        
        # Set the title.
        ax.set_title('Subject %02d Trial %02d\nt=%0.2fs' % (subject_id, trial_index, times_s[trial_index][time_index]-times_s[trial_index][0]))
        
        # Show the plot and wait for the next timestep.
        plt.draw()
        if pause_between_frames:
          plt_wait_for_keyboard_press(timeout_s=-1)#spf)
    else:
      # Set the aspect ratio
      ax.set_box_aspect([ub - lb for lb, ub in (ax.get_xlim(), ax.get_ylim(), ax.get_zlim())])
      # Set the title.
      ax.set_title('Subject %02d' % (subject_id))
      # Show the plot.
      plt.draw()
      
    if pause_between_trials:
      plt_wait_for_keyboard_press()
      print('view elev/azim:', ax.elev, ax.azim)
  return fig


# Save the first-person videos during each hand path.
def save_pouring_eyeVideos(h5_file, eyeVideo_filepath, times_s, subject_id, trial_indexes_filter=None):
  device_name = 'eye-tracking-video-worldGaze'
  stream_name = 'frame_timestamp'
  frames_time_s = h5_file[device_name][stream_name]['data']
  
  for trial_index, time_s in enumerate(times_s):
    print('Saving eye video for Subject S%02d trial %02d' % (subject_id, trial_index))
    if trial_indexes_filter is not None and trial_index not in trial_indexes_filter:
      continue
    start_time_s = min(time_s)
    end_time_s = max(time_s)
    frame_indexes = np.where((frames_time_s >= start_time_s) & (frames_time_s <= end_time_s))[0]
    start_frame_index = min(frame_indexes)
    num_frames = len(frame_indexes)
    video_reader = cv2.VideoCapture(eyeVideo_filepath)
    video_writer = cv2.VideoWriter(os.path.join(output_dir, 'pouring_eyeVideo_S%02d_%02d.mp4' % (subject_id, trial_index)),
                                   cv2.VideoWriter_fourcc(*'MP4V'), # for AVI: cv2.VideoWriter_fourcc(*'MJPG'),
                                   video_reader.get(cv2.CAP_PROP_FPS),
                                   (int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                                   )
    video_reader.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index-1)
    for i in range(num_frames):
      res, frame = video_reader.read()
      video_writer.write(frame)
    video_reader.release()
    video_writer.release()

def resize_image(img, target_width=None, target_height=None):
  img_width = img.shape[1]
  img_height = img.shape[0]
  if target_width is not None and target_height is not None:
    # Check if the width or height will be the controlling dimension.
    scale_factor_fromWidth = target_width/img_width
    scale_factor_fromHeight = target_height/img_height
    if img_height*scale_factor_fromWidth > target_height:
      scale_factor = scale_factor_fromHeight
    else:
      scale_factor = scale_factor_fromWidth
  elif target_width is not None:
    scale_factor = target_width/img_width
  elif target_height is not None:
    scale_factor = target_height/img_height
  else:
    raise AssertionError('No target dimension provided when resizing the image')
  # Resize the image.
  return cv2.resize(img, (0,0), None, scale_factor, scale_factor)
  
def save_pouring_composite_videos(h5_file, eyeVideo_filepath,
                                  fig, subplot_index,
                                  times_s, bodySegment_datas,
                                  subject_id, num_subjects, trial_indexes_filter=None):
  device_name = 'eye-tracking-video-worldGaze'
  stream_name = 'frame_timestamp'
  frames_time_s = h5_file[device_name][stream_name]['data']
  
  for trial_index, time_s in enumerate(times_s):
    print('Saving composite video for Subject S%02d trial %02d' % (subject_id, trial_index))
    if trial_indexes_filter is not None and trial_index not in trial_indexes_filter:
      continue
    start_time_s = min(time_s)
    end_time_s = max(time_s)
    frame_indexes = np.where((frames_time_s >= start_time_s) & (frames_time_s <= end_time_s))[0]
    start_frame_index = min(frame_indexes)
    num_frames = len(frame_indexes)
    video_reader = cv2.VideoCapture(eyeVideo_filepath)
    video_reader.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index-1)
    video_writer = None
    for i in range(num_frames):
      if i % (np.ceil(num_frames/5)) == 0 and i > 0:
        print('  Completed %d/%d frames (%0.1f%%)' % (i, num_frames, 100*i/num_frames))
      # Get the frame of the eye video and its timestamp
      _, eye_frame = video_reader.read()
      target_time_s = frames_time_s[start_frame_index + i]
      # Plot the skeleton at the closest time to the eye-video frame timestamp
      target_times_s = [None]*len(times_s) # an entry for each trial
      target_times_s[trial_index] = target_time_s
      fig = plot_handPath_data(fig, subplot_index,
                               times_s, bodySegment_datas, pouring_pose,
                               subject_id, 1,
                               spf=None,  # scan all frames for the closest time match
                               include_skeleton=True,
                               trial_indexes_filter=[trial_index],
                               target_times_s=target_times_s,
                               pause_between_trials=False, pause_between_frames=False,
                               hide_figure_window=True,
                               clear_axes_each_trial=True)
      plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
      plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
      plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)
      # Concatenate the eye-video frame and the plot
      eye_frame = resize_image(eye_frame, target_height=plot_img.shape[0])
      composite_frame = cv2.hconcat([plot_img, eye_frame])
      # cv2.imshow('my frame', composite_frame)
      # cv2.waitKey(10)
      
      # Write to the output video
      if video_writer is None:
        video_writer = cv2.VideoWriter(os.path.join(output_dir, 'pouring_composite_S%02d_%02d.mp4' % (subject_id, trial_index)),
                                       cv2.VideoWriter_fourcc(*'MP4V'), # for AVI: cv2.VideoWriter_fourcc(*'MJPG'),
                                       video_reader.get(cv2.CAP_PROP_FPS),
                                       (composite_frame.shape[1], composite_frame.shape[0])
                                       )
      video_writer.write(composite_frame)
    video_writer.release()
    video_reader.release()

def export_path_data(times_s_allSubjects, bodySegment_datas_allSubjects,
                     pouring_times_s_allSubjects, pouring_pose_allSubjects):
  # Open the output HDF5 file
  hdf5_output_filepath = os.path.join(output_dir, 'pouring_paths_humans.hdf5')
  if os.path.exists(hdf5_output_filepath):
    print()
    print('Output file exists at [%s]' % hdf5_output_filepath)
    print('  Overwrite the file? [y/N] ', end='')
    overwrite_file = input()
    if overwrite_file.lower().strip() != 'y':
      print('  Aborting')
      return
  hdf5_file = h5py.File(hdf5_output_filepath, 'w')
  
  for subject_id in bodySegment_datas_allSubjects:
    num_trials = len(bodySegment_datas_allSubjects[subject_id])
    subject_group = hdf5_file.create_group('subject_%02d' % subject_id)
    for trial_index in range(num_trials):
      trial_group = subject_group.create_group('trial_%02d' % trial_index)
      # Add timestamps
      time_s = times_s_allSubjects[subject_id][trial_index]
      time_s = time_s - min(time_s)
      trial_group.create_dataset('time_s', data=time_s)
      # time_str = [get_time_str(t, '%Y-%m-%d %H:%M:%S.%f') for t in time_s]
      # trial_group.create_dataset('time_str', data=time_str, dtype='S26')
      # Add body segment position data
      data_segmentDict = bodySegment_datas_allSubjects[subject_id][trial_index]['position_cm']
      data = list(data_segmentDict.values())
      data = np.stack(data, axis=0)
      data = np.moveaxis(data, 0, 1) # convert from [segment][time][xyz] to [time][segment][xyz]
      trial_group.create_dataset('body_segment_position_cm', data=data)
      # Add body segment orientation data
      data_segmentDict = bodySegment_datas_allSubjects[subject_id][trial_index]['quaternion']
      data = list(data_segmentDict.values())
      data = np.stack(data, axis=0)
      data = np.moveaxis(data, 0, 1) # convert from [segment][time][xyzw] to [time][segment][xyzw]
      trial_group.create_dataset('body_segment_quaternion', data=data)
      # Add estimated pouring position
      pouring_group = trial_group.create_group('pouring')
      data_segmentDict = pouring_pose_allSubjects[subject_id][trial_index]['position_cm']
      data = list(data_segmentDict.values())
      data = np.stack(data, axis=0)
      pouring_group.create_dataset('body_segment_position_cm', data=data)
      # Add estimated pouring quaternion
      data_segmentDict = pouring_pose_allSubjects[subject_id][trial_index]['quaternion']
      data = list(data_segmentDict.values())
      data = np.stack(data, axis=0)
      pouring_group.create_dataset('body_segment_quaternion', data=data)
      # Add pouring time
      pouring_group.create_dataset('time_s',
                                   data=pouring_times_s_allSubjects[subject_id][trial_index] - min(time_s))
  
  # Add segment names
  hdf5_file.create_dataset('body_segment_names',
                           data=list(bodySegment_datas_allSubjects[0][0]['position_cm'].keys()))
  
  # Close the output file
  hdf5_file.close()

# Helper to get start and end times of each activity.
# Optionally exclude activities marked as bad.
#   Some activities may have been marked as 'Bad' or 'Maybe' by the experimenter.
#   Submitted notes with the activity typically give more information.
def get_activity_startEnd_times_s(h5_file, exclude_bad_labels=True):
  device_name = 'experiment-activities'
  stream_name = 'activities'
  
  # Get the timestamped label data.
  # As described in the HDF5 metadata, each row has entries for ['Activity', 'Start/Stop', 'Valid', 'Notes'].
  activity_datas = h5_file[device_name][stream_name]['data']
  activity_times_s = h5_file[device_name][stream_name]['time_s']
  activity_times_s = np.squeeze(np.array(activity_times_s))  # squeeze (optional) converts from a list of single-element lists to a 1D list
  # Convert to strings for convenience.
  activity_datas = [[x.decode('utf-8') for x in datas] for datas in activity_datas]
  
  # Combine start/stop rows to single activity entries with start/stop times.
  #   Each row is either the start or stop of the label.
  #   The notes and ratings fields are the same for the start/stop rows of the label, so only need to check one.
  activities_labels = []
  activities_start_times_s = []
  activities_end_times_s = []
  activities_ratings = []
  activities_notes = []
  for (row_index, time_s) in enumerate(activity_times_s):
    label    = activity_datas[row_index][0]
    is_start = activity_datas[row_index][1] == 'Start'
    is_stop  = activity_datas[row_index][1] == 'Stop'
    rating   = activity_datas[row_index][2]
    notes    = activity_datas[row_index][3]
    if exclude_bad_labels and rating in ['Bad', 'Maybe']:
      continue
    # Record the start of a new activity.
    if is_start:
      activities_labels.append(label)
      activities_start_times_s.append(time_s)
      activities_ratings.append(rating)
      activities_notes.append(notes)
    # Record the end of the previous activity.
    if is_stop:
      activities_end_times_s.append(time_s)

  activities_labels = np.array(activities_labels)
  activities_start_times_s = np.array(activities_start_times_s)
  activities_end_times_s = np.array(activities_end_times_s)
  return (activities_labels, activities_start_times_s, activities_end_times_s)

# Helper to get start and end times of the pouring water activity.
def get_pouring_startEnd_times_s(h5_file, exclude_bad_labels=True):
  # Get start/end times of every activity.
  (activities_labels, activities_start_times_s, activities_end_times_s) = get_activity_startEnd_times_s(h5_file, exclude_bad_labels=exclude_bad_labels)
  
  # Filter by pouring water.
  pouring_indexes = [i for (i, label) in enumerate(activities_labels) if 'Pour' in label]
  if len(pouring_indexes) == 0:
    return (None, None, None)
  activities_labels = activities_labels[pouring_indexes]
  activities_start_times_s = activities_start_times_s[pouring_indexes]
  activities_end_times_s = activities_end_times_s[pouring_indexes]
  
  return (activities_labels, activities_start_times_s, activities_end_times_s)

# Helper to get manual start and end times of the pouring water activity.
def get_manual_pouring_startEnd_times_s(subject_id):
  num_trials = 5
  subject_index = subject_id
  if subject_id == 0:
    subject_start_index = 0
  else:
    subject_start_index = (subject_index-1)*num_trials # no subject id 1
    
  activities_start_times_str = manual_pouring_start_times_s[subject_start_index:(subject_start_index+num_trials)]
  activities_end_times_str = manual_pouring_end_times_s[subject_start_index:(subject_start_index+num_trials)]

  activities_start_times_s = []
  activities_end_times_s = []
  for i in range(num_trials):
    activities_start_times_s.append(
        get_time_s_from_local_str(activities_start_times_str[i].split(' ')[1], input_time_format='%H:%M:%S.%f',
                                  date_local_str=activities_start_times_str[i].split(' ')[0], input_date_format='%Y-%m-%d'))
    activities_end_times_s.append(
        get_time_s_from_local_str(activities_end_times_str[i].split(' ')[1], input_time_format='%H:%M:%S.%f',
                                  date_local_str=activities_end_times_str[i].split(' ')[0], input_date_format='%Y-%m-%d'))
  
  activities_labels = ['Pour water from a pitcher into a glass']*num_trials
  
  return (activities_labels, activities_start_times_s, activities_end_times_s)


###################################################################
###################################################################
###################################################################



# Find folders of log data, and record filepaths for the HDF5s and first-person videos.
hdf5_filepaths = OrderedDict() # map subject IDs to list of filepaths
eyeVideo_filepaths = OrderedDict() # map subject IDs to list of filepaths
for subdir, dirs, filenames in os.walk(experiments_dir):
  hdf5_filepath = [filename for filename in filenames if '.hdf5' in filename]
  eyeVideo_filepath = [filename for filename in filenames if 'eye-tracking-video-worldGaze_frame.avi' in filename]
  log_filepath = [filename for filename in filenames if 'log_history.txt' in filename]
  try:
    subject_id = int(subdir.split('_')[-1][1:])
  except:
    subject_id = None
  is_a_root_log_folder = len(hdf5_filepath) == 1 and len(eyeVideo_filepath) == 1 \
                         and len(log_filepath) == 1 and subject_id is not None
  if is_a_root_log_folder and (subject_ids_filter is None or subject_id in subject_ids_filter):
    print(subdir, subject_id)
    hdf5_filepaths.setdefault(subject_id, [])
    eyeVideo_filepaths.setdefault(subject_id, [])
    hdf5_filepaths[subject_id].append(os.path.join(subdir, hdf5_filepath[0]))
    eyeVideo_filepaths[subject_id].append(os.path.join(subdir, eyeVideo_filepath[0]))

# Loop through experiment files to extract trajectories for each action instance.
print()
num_subjects = len(hdf5_filepaths)
fig = None
subplot_index = 0
times_s_allSubjects = OrderedDict()
bodySegment_datas_allSubjects = OrderedDict()
pouring_times_s_allSubjects = OrderedDict()
pouring_pose_allSubjects = OrderedDict()
for subject_id, subject_hdf5_filepaths in hdf5_filepaths.items():
  print('Processing subject %02d' % subject_id)
  times_s_allSubjects.setdefault(subject_id, [])
  bodySegment_datas_allSubjects.setdefault(subject_id, [])
  pouring_times_s_allSubjects.setdefault(subject_id, [])
  pouring_pose_allSubjects.setdefault(subject_id, [])
  pouring_trial_index_start = 0
  for (filepath_index, hdf5_filepath) in enumerate(subject_hdf5_filepaths):
    print(' ', hdf5_filepath)
    eyeVideo_filepath = eyeVideo_filepaths[subject_id][filepath_index]
    # Open the HDF5 file.
    h5_file = h5py.File(hdf5_filepath, 'r')
    
    # Determine the start/end times of each hand path.
    (activities_labels, activities_start_times_s, activities_end_times_s) = \
      get_pouring_startEnd_times_s(h5_file, exclude_bad_labels=True)
    if activities_labels is None:
      continue
    num_trials_inFile = len(activities_labels)
    if use_manual_startEnd_times:
      (activities_labels, activities_start_times_s, activities_end_times_s) = \
        get_manual_pouring_startEnd_times_s(subject_id)
      activities_labels = activities_labels[pouring_trial_index_start:(pouring_trial_index_start+num_trials_inFile)]
      activities_start_times_s = activities_start_times_s[pouring_trial_index_start:(pouring_trial_index_start+num_trials_inFile)]
      activities_end_times_s = activities_end_times_s[pouring_trial_index_start:(pouring_trial_index_start+num_trials_inFile)]
      # print([activities_start_times_s[i] - activities_start_times_s[i] for i in range(5)])
      # print([activities_end_times_s[i] - activities_end_times_s[i] for i in range(5)])
      # print([get_time_str(t) for t in activities_start_times_s])
      # print([get_time_str(t) for t in activities_start_times_s])
    pouring_trial_index_start += num_trials_inFile
    
    # Get the hand paths.
    (times_s, bodySegment_datas) = get_pouring_bodyPath_data(h5_file, activities_start_times_s, activities_end_times_s)
    (times_s, bodySegment_datas) = resample_data(times_s, bodySegment_datas)
    # Infer the hand position while pouring
    (pour_times_s, pouring_pose) = infer_pouring_pose(times_s, bodySegment_datas)
    
    # Store the results
    times_s_allSubjects[subject_id].extend(times_s)
    bodySegment_datas_allSubjects[subject_id].extend(bodySegment_datas)
    pouring_times_s_allSubjects[subject_id].extend(pour_times_s)
    pouring_pose_allSubjects[subject_id].extend(pouring_pose)

    # Plot the paths.
    if animate_pouring_plots:
      fig = plot_handPath_data(fig, subplot_index,
                               times_s, bodySegment_datas,
                               pouring_pose,
                               subject_id, 1, trial_indexes_filter=None,
                               spf=0.25,
                               pause_between_trials=False, pause_between_frames=True)
    if plot_all_pouring_paths:
      fig = plot_handPath_data(fig, subplot_index,
                               times_s, bodySegment_datas,
                               pouring_pose,
                               subject_id, num_subjects, trial_indexes_filter=None,
                               target_times_s=pour_times_s,
                               include_skeleton=False,
                               clear_axes_each_trial=False)
      
    if save_eye_videos:
      save_pouring_eyeVideos(h5_file, eyeVideo_filepath, times_s, subject_id, trial_indexes_filter=None)
    if save_composite_videos:
      save_pouring_composite_videos(h5_file, eyeVideo_filepath,
                                    fig, subplot_index,
                                    times_s, bodySegment_datas,
                                    subject_id, num_subjects, trial_indexes_filter=None)
    # Close the HDF5 file.
    h5_file.close()
    
  # Advance subplot index if putting each subject in a new subplot
  if plot_all_pouring_paths:
    subplot_index += 1
  
# Export the results if desired
if save_results_data:
  export_path_data(times_s_allSubjects, bodySegment_datas_allSubjects,
                   pouring_times_s_allSubjects, pouring_pose_allSubjects)
  
# Show the final plot
if animate_pouring_plots or plot_all_pouring_paths:
  plt.show(block=True)




















