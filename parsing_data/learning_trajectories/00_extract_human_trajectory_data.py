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
# experiments_dir = os.path.join(data_dir, 'experiments', '2023-08-18_experiment_S10')
experiments_dir = os.path.join(data_dir, 'experiments', '2023-09-10_experiment_S00')
# experiments_dir = os.path.join(data_dir, 'experiments', '2023-09-10_experiment_S11')
output_dir = os.path.realpath(os.path.join(script_dir, '..', '..', '..', 'results', 'learning_trajectories', 'S00'))
# output_dir = os.path.realpath(os.path.join(script_dir, '..', '..', '..', 'results', 'learning_trajectories', 'S11'))
os.makedirs(output_dir, exist_ok=True)

animate_trajectory_plots = False # show an animated plot of the skeleton for each trial
plot_all_trajectories = False # make a subplot for each subject, which shows all paths from that subject
save_plot_all_trajectories = False # make a subplot for each subject, which shows all paths from that subject
save_eye_videos = False # save the eye-tracking video for each trial
save_composite_videos = False # save the eye-tracking video and animated plot for each trial
save_results_data = True

resampled_fs_hz = 50

# CHOOSE THE ACTIVITY TO PROCESS
configure_for_pouring = True # if False, will be scooping

if configure_for_pouring:
  target_activity_label = 'Pour water from a pitcher into a glass'
  target_activity_keyword = 'pouring'
else:
  target_activity_label = 'Scoop from a pan to a plate'
  target_activity_keyword = 'scooping'

# Decide how to extract the stationary waypoint (e.g. pouring the water or scooping from the plate).
# Can use a rolling buffer and look at hand position variance when inferring hold position,
#  or can use the pose at a hard-coded fraction into the trial.
stationary_position_use_variance = True if configure_for_pouring else False
stationary_position_hardcoded_time_fraction = 0.3 # will use the hand position/pose at this ratio of time into the trial
# Define the buffer length to use when computing variance OR when averaging around the hard-coded time.
stationary_position_buffer_duration_s = 2 if configure_for_pouring else 1

# Specify the segment that is a proxy for the reference object.
referenceObject_bodySegment_name = 'LeftHand'
referenceObject_offset_cm = np.array([0, 9, -(6-4)]) # [up along thumb, out along forearm axis, out from back of hand]
referenceObject_diameter_cm = 7.3 # glass top 7.3 bottom 6.3

# hand_box_dimensions_cm = np.array([2, 9, 18]) # open hand
hand_box_dimensions_cm = np.array([4.8, 3, 1.3])
hand_box_color = 0.8*np.array([1, 0.6, 0])
pitcher_box_dimensions_cm = np.array([23, 23, 10.8]) # [height, top length, width]
pitcher_box_color = 0.8*np.array([1, 1, 1])
hand_to_pitcher_rotation = Rotation.from_rotvec(np.pi/2 * np.array([1, 0, 0]))
hand_to_pitcher_offset_cm = np.array([0, -13, 0])

use_manual_startEnd_times = False
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

bodySegment_chains_labels_toPlot = {
  # 'Left Leg':  ['Left Upper Leg', 'Left Lower Leg', 'Left Foot', 'Left Toe'],
  # 'Right Leg': ['RightUpperLeg', 'Right Lower Leg', 'Right Foot', 'Right Toe'],
  'Spine':     ['Head', 'Neck', 'T8', 'T12', 'L3', 'L5', 'Pelvis'], # top down
  # 'Hip':       ['Left Upper Leg', 'Pelvis', 'RightUpperLeg'],
  'Shoulders': ['LeftUpperArm', 'LeftShoulder', 'RightShoulder', 'RightUpperArm'],
  'Left Arm':  ['LeftUpperArm', 'LeftForeArm', 'LeftHand'],
  'Right Arm': ['RightUpperArm', 'RightForeArm', 'RightHand'],
}



###################################################################
###################################################################
###################################################################

# Define a helper to extract hand path data from an HDF5 file.
def get_bodyPath_data(h5_file):
  # Extract hand position.
  # Specify the device and stream.
  device_name = 'xsens-segments'
  stream_name = 'body_position_xyz_m'
  metadata = dict(h5_file[device_name][stream_name].attrs.items())
  bodySegment_labels = eval(metadata['segment_names_body'])

  # Get the timestamps for each entry as seconds since epoch.
  time_s = h5_file[device_name][stream_name]['time_s']
  time_s = np.squeeze(np.array(time_s)) # squeeze (optional) converts from a list of single-element lists to a 1D list
  # Get the timestamps for each entry as human-readable strings.
  time_str = h5_file[device_name][stream_name]['time_str']
  time_str = np.squeeze(np.array(time_str)) # squeeze (optional) converts from a list of single-element lists to a 1D list
  
  # Get segment position data as an NxJx3 matrix as [timestep][segment][xyz]
  segment_data = h5_file[device_name][stream_name]['data']
  segment_data = np.array(segment_data)
  bodySegment_position_xyz_m = OrderedDict()
  for (segment_index, segment_name) in enumerate(bodySegment_labels):
    bodySegment_position_xyz_m[segment_name] = np.squeeze(segment_data[:, segment_index, :])

  # Get segment orientation data as an NxJx4 matrix as [timestep][segment][wijk]
  device_name = 'xsens-segments'
  stream_name = 'body_orientation_quaternion_wijk'
  segment_data = h5_file[device_name][stream_name]['data']
  segment_data = np.array(segment_data)
  bodySegment_quaternion_wijk = OrderedDict()
  for (segment_index, segment_name) in enumerate(bodySegment_labels):
    bodySegment_quaternion_wijk[segment_name] = np.squeeze(segment_data[:, segment_index, :])

  # Get joint angle data as an NxJx3 matrix as [timestep][joint][xyz]
  device_name = 'xsens-joints'
  stream_name = 'body_joint_angles_eulerZXY_xyz_rad'
  metadata = dict(h5_file[device_name][stream_name].attrs.items())
  bodyJoint_labels = eval(metadata['joint_names_body'])
  joint_data = h5_file[device_name][stream_name]['data']
  joint_data = np.array(joint_data)
  bodyJoint_angles_eulerZXY_xyz_rad = OrderedDict()
  for (joint_index, joint_name) in enumerate(bodyJoint_labels):
    bodyJoint_angles_eulerZXY_xyz_rad[joint_name] = np.squeeze(joint_data[:, joint_index, :])
  # Get joint angle data as an NxJx3 matrix as [timestep][joint][xyz]
  device_name = 'xsens-joints'
  stream_name = 'body_joint_angles_eulerXZY_xyz_rad'
  metadata = dict(h5_file[device_name][stream_name].attrs.items())
  bodyJoint_labels = eval(metadata['joint_names_body'])
  joint_data = h5_file[device_name][stream_name]['data']
  joint_data = np.array(joint_data)
  bodyJoint_angles_eulerXZY_xyz_rad = OrderedDict()
  for (joint_index, joint_name) in enumerate(bodyJoint_labels):
    bodyJoint_angles_eulerXZY_xyz_rad[joint_name] = np.squeeze(joint_data[:, joint_index, :])

  # Combine the position and orientation data.
  # NOTE: Assumes the timestamps are the same.
  bodyPath_data = {
    'position_m': bodySegment_position_xyz_m,
    'joint_angle_eulerZXY_xyz_rad': bodyJoint_angles_eulerZXY_xyz_rad,
    'joint_angle_eulerXZY_xyz_rad': bodyJoint_angles_eulerXZY_xyz_rad,
    'quaternion_wijk': bodySegment_quaternion_wijk
  }
  return (time_s, bodyPath_data)

# Helper to extract path data for specified activities.
def get_activity_bodyPath_data(h5_file, start_times_s, end_times_s):
  # Get path data throughout the whole experiment.
  (time_s, bodyPath_data) = get_bodyPath_data(h5_file)
  
  times_s = []
  bodyPath_datas = []
  # Extract each desired activity.
  for time_index in range(len(start_times_s)):
    start_time_s = start_times_s[time_index]
    end_time_s = end_times_s[time_index]
    indexes_forLabel = np.where((time_s >= start_time_s) & (time_s <= end_time_s))[0]
    if indexes_forLabel.size > 0:
      times_s.append(time_s[indexes_forLabel])
      bodyPath_datas.append(OrderedDict())
      for data_type in bodyPath_data.keys():
        bodyPath_datas[-1].setdefault(data_type, OrderedDict())
        for body_segment in bodyPath_data[data_type].keys():
          bodyPath_datas[-1][data_type].setdefault(body_segment, OrderedDict())
          bodyPath_datas[-1][data_type][body_segment] = bodyPath_data[data_type][body_segment][indexes_forLabel, :]
          
  # Rotate each segment to a coordinate frame based on the body, such that
  #  the y axis is aligned with the shoulders/hips.
  # print()
  for trial_index in range(len(bodyPath_datas)):
    # Set the origin between the hips (very close to the pelvis but not exactly).
    # pelvis_position_m = bodySegment_datas[trial_index]['position_m']['Pelvis']
    # origin_cm = pelvis_position_m[0,:]
    origin_cm = np.mean(np.array(
          [bodyPath_datas[trial_index]['position_m']['RightUpperLeg'][0, :],
           bodyPath_datas[trial_index]['position_m']['LeftUpperLeg'][0, :]]), axis=0)
    # print(origin_cm)
    for body_segment in bodyPath_data['position_m'].keys():
      position_m = bodyPath_datas[trial_index]['position_m'][body_segment]
      position_m = position_m - origin_cm
      bodyPath_datas[trial_index]['position_m'][body_segment] = position_m

    # Use the hip orientation to create the y axis.
    y_axis_right = np.append(bodyPath_datas[trial_index]['position_m']['RightUpperLeg'][0, 0:2], 0)
    # y_axis_left = np.append(bodyPath_datas[trial_index]['position_m']['Left Upper Leg'][0, 0:2], 0)
    # # Average the shoulders and hips (projected onto xy plane) to create a y axis.
    # y_axis_right = np.mean(np.array(
    #   [bodySegment_datas[trial_index]['position_m']['RightUpperArm'][0, 0:2],
    #   bodySegment_datas[trial_index]['position_m']['RightUpperLeg'][0, 0:2]]), axis=0)
    # y_axis_left = np.mean(np.array(
    #   [bodySegment_datas[trial_index]['position_m']['LeftUpperArm'][0, 0:2],
    #   bodySegment_datas[trial_index]['position_m']['Left Upper Leg'][0, 0:2]]), axis=0)
    # y_axis_center = np.mean(np.array([y_axis_right, y_axis_left]), axis=0)

    # Rotate each position.
    alignment_rotation_matrix = rotation_matrix_from_vectors(y_axis_right, [0, 1, 0])
    for body_segment in bodyPath_data['position_m'].keys():
      position_m = bodyPath_datas[trial_index]['position_m'][body_segment]
      for time_index in range(position_m.shape[0]):
        position_m[time_index,:] = alignment_rotation_matrix.dot(position_m[time_index,:])

    # Compose the rotation with quaternion orientations.
    alignment_rotation = Rotation.from_matrix(alignment_rotation_matrix)
    for body_segment in bodyPath_data['quaternion_wijk'].keys():
      quaternion_wijk = bodyPath_datas[trial_index]['quaternion_wijk'][body_segment]
      for time_index in range(quaternion_wijk.shape[0]):
        quaternion_forTime_ijkw = quaternion_wijk[time_index, [1,2,3,0]]
        quaternion_rotation = Rotation.from_quat(quaternion_forTime_ijkw)
        aligned_quaternion_rotation = alignment_rotation * quaternion_rotation # note that multiplication is overloaded for scipy Rotation objects
        aligned_quaternion_ijkw = aligned_quaternion_rotation.as_quat()
        quaternion_wijk[time_index,:] = aligned_quaternion_ijkw[[3,0,1,2]]
    
  return (times_s, bodyPath_datas)

# Resample to a consistent rate
def resample_data(times_s, bodySegment_datas):
  for trial_index in range(len(times_s)):
    time_s = times_s[trial_index]
    time_s_resampled = np.arange(min(time_s), max(time_s)+1/resampled_fs_hz/2, 1/resampled_fs_hz)
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

# Infer the hand position at a stationary point (such as when water is being poured)
def infer_stationary_poses(times_s, bodySegment_datas):
  body_positions_stationary_cm = []
  stationary_times_s = []
  for trial_index, bodySegment_data in enumerate(bodySegment_datas):
    body_position_m = bodySegment_datas[trial_index]['position_m']
    body_quaternion_wijk = bodySegment_datas[trial_index]['quaternion_wijk']
    time_s = times_s[trial_index]
    num_timesteps = time_s.shape[0]
    fs = (num_timesteps-1)/(time_s[-1] - time_s[0])
    if stationary_position_use_variance:
      if time_s[-1] - time_s[0] < 2*stationary_position_buffer_duration_s:
        # Add a dummy stationary position for now
        # TODO do something better?
        body_positions_stationary_cm.append({
        'position_m':
          dict([(name, 0) for (name, position_m) in body_position_m.items()]),
        'quaternion_wijk':
          dict([(name, 0) for (name, quaternion) in body_position_m.items()]),
        })
        stationary_times_s.append(0)
        continue
      min_average_distance_cm = None
      min_average_distance_buffer_start_index = None
      min_average_distance_buffer_end_index = None
      body_position_stationary_buffer_m = None
      body_quaternion_wijk_stationary_buffer = None
      average_distances_cm = []
      for buffer_start_index in range(num_timesteps):
        buffer_start_time_s = time_s[buffer_start_index]
        buffer_end_time_s = buffer_start_time_s + stationary_position_buffer_duration_s
        if buffer_end_time_s > time_s[-1]:
          break
        buffer_end_index = np.where(time_s <= buffer_end_time_s)[0][-1]
        body_position_buffers_m = dict([(name, position_m[buffer_start_index:buffer_end_index, :]) for (name, position_m) in body_position_m.items()])
        body_quaternion_wijk_buffers = dict([(name, quaternion_wijk[buffer_start_index:buffer_end_index, :]) for (name, quaternion_wijk) in body_quaternion_wijk.items()])
        median_hand_position_m = np.median(body_position_buffers_m['RightHand'], axis=0)
        distances_cm = np.linalg.norm(body_position_buffers_m['RightHand'] - median_hand_position_m, axis=1)
        average_distance_m = np.mean(distances_cm, axis=0)
        average_distances_cm.append(average_distance_m)
        if min_average_distance_cm is None or average_distance_m < min_average_distance_cm:
          min_average_distance_cm = average_distance_m
          min_average_distance_buffer_start_index = buffer_start_index
          min_average_distance_buffer_end_index = buffer_end_index
          body_position_stationary_buffer_m = body_position_buffers_m
          body_quaternion_wijk_stationary_buffer = body_quaternion_wijk_buffers
      # print(time_s[min_average_distance_buffer_start_index] - min(time_s),
      #       time_s[min_average_distance_buffer_end_index] - min(time_s))
      body_positions_stationary_cm.append({
        'position_m':
          dict([(name, np.median(position_m, axis=0)) for (name, position_m) in body_position_stationary_buffer_m.items()]),
        'quaternion_wijk':
          dict([(name, quaternion_wijk[int(quaternion_wijk.shape[0]/2),:]) for (name, quaternion_wijk) in body_quaternion_wijk_stationary_buffer.items()]),
        })
      stationary_times_s.append(time_s[int(np.mean([min_average_distance_buffer_start_index,
                                                    min_average_distance_buffer_end_index]))]-min(time_s))
    else: # use hard-coded time fraction instead of computing the variance
      stationary_position_time_s = time_s[0] + (time_s[-1] - time_s[0])*stationary_position_hardcoded_time_fraction
      stationary_position_index = time_s.searchsorted(stationary_position_time_s)
      buffer_length = fs * stationary_position_buffer_duration_s
      buffer_start_index = round(stationary_position_index - buffer_length/2)
      buffer_end_index = round(stationary_position_index + buffer_length/2)
      body_position_stationary_buffer_m = dict([(name, position_cm[buffer_start_index:buffer_end_index, :]) for (name, position_cm) in body_position_m.items()])
      body_quaternion_wijk_stationary_buffer = dict([(name, quaternion[buffer_start_index:buffer_end_index, :]) for (name, quaternion) in body_quaternion_wijk.items()])
      body_positions_stationary_cm.append({
        'position_m':
          dict([(name, np.median(position_m, axis=0)) for (name, position_m) in body_position_stationary_buffer_m.items()]),
        'quaternion_wijk':
          dict([(name, quaternion_wijk[int(quaternion_wijk.shape[0]/2),:]) for (name, quaternion_wijk) in body_quaternion_wijk_stationary_buffer.items()]),
        })
      stationary_times_s.append(time_s[int(np.mean([buffer_start_index,
                                                    buffer_end_index]))]-min(time_s))

  # print(stationary_times_s)
  return (stationary_times_s, body_positions_stationary_cm)

# Infer the hand position at a stationary point (such as when water is being poured)
def infer_referenceObject_positions(bodySegment_datas, times_s, referenceObject_times_s):
  referenceObject_positions_m = []
  for trial_index, bodySegment_data in enumerate(bodySegment_datas):
    body_position_m = bodySegment_datas[trial_index]['position_m']
    body_quaternion_wijk = bodySegment_datas[trial_index]['quaternion_wijk']
    referenceObject_index = np.where(np.abs(times_s[trial_index] - referenceObject_times_s[trial_index]) == np.min(np.abs(times_s[trial_index] - referenceObject_times_s[trial_index])))[0]
    referenceObject_segment_position_m = np.squeeze(
                                          body_position_m[referenceObject_bodySegment_name][referenceObject_index,:],
                                          axis=0)
    referenceObject_segment_quaternion_wijk = np.squeeze(body_quaternion_wijk[referenceObject_bodySegment_name][referenceObject_index, :])
    quat_ijkw = referenceObject_segment_quaternion_wijk[[1,2,3,0]]
    quat_ijkw = [-quat_ijkw[0], -quat_ijkw[1], -quat_ijkw[2], quat_ijkw[3]]
    referenceObject_segment_rot = Rotation.from_quat(quat_ijkw).as_matrix()
    referenceObject_offset_rotated_cm = np.dot(referenceObject_offset_cm, referenceObject_segment_rot)
    referenceObject_position_m = referenceObject_segment_position_m + referenceObject_offset_rotated_cm/100
    referenceObject_positions_m.append(referenceObject_position_m)
  return referenceObject_positions_m

# Plot hand paths.
def plt_wait_for_keyboard_press(timeout_s=-1.0):
  keyboardClick=False
  while keyboardClick == False:
    keyboardClick = plt.waitforbuttonpress(timeout=timeout_s)

def plot_hand_box(ax, hand_quaternion_localToGlobal_wijk, hand_center):
  return plot_3d_box(ax, hand_quaternion_localToGlobal_wijk, hand_center, np.array([0,0,0]),
                     hand_box_dimensions_cm, hand_box_color)

def plot_pitcher_box(ax, hand_quaternion_localToGlobal_wijk, hand_center):
  hand_rotation = Rotation.from_quat(hand_quaternion_localToGlobal_wijk[[1,2,3,0]])
  pitcher_rotation = hand_rotation * hand_to_pitcher_rotation
  pitcher_quaternion_localToGlobal_ijkw = pitcher_rotation.as_quat()
  return plot_3d_box(ax, pitcher_quaternion_localToGlobal_ijkw[[3,0,1,2]],
                     hand_center, hand_to_pitcher_offset_cm,
                     pitcher_box_dimensions_cm, pitcher_box_color)
  
def plot_3d_box(ax, quaternion_localToGlobal_wijk, center_cm, center_preRotation_cm, box_dimensions_cm, color): # function created using ChatGPT
  # Define vertices of a unit box in the global frame
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
  # Define faces of the box in the global frame, using corner indexes
  faces = np.array([
    [0, 1, 3, 2], # bottom face
    [0, 2, 6, 4],
    [0, 1, 5, 4],
    [4, 5, 7, 6], # top face
    [1, 3, 7, 5],
    [2, 3, 7, 6], # hand-side face
  ])
  # Scale the box
  corners = corners * box_dimensions_cm
  
  # Translate the box
  corners = corners + center_preRotation_cm
  
  # Invert the quaternion.
  quaternion_globalToLocal_ijkw = [
    -quaternion_localToGlobal_wijk[1],
    -quaternion_localToGlobal_wijk[2],
    -quaternion_localToGlobal_wijk[3],
    quaternion_localToGlobal_wijk[0],
    ]
  # Rotate the box using the quaternion,
  rot = Rotation.from_quat(quaternion_globalToLocal_ijkw).as_matrix()
  corners = np.dot(corners, rot)
  
  # Translate the box
  corners = corners + center_cm
  
  # Plot the box
  # ax.set_box_aspect([np.ptp(corners[:,dim]) for dim in range(3)])
  # ax.set_xlim3d(corners[:,0].min(), corners[:,0].max())
  # ax.set_ylim3d(corners[:,1].min(), corners[:,1].max())
  # ax.set_zlim3d(corners[:,2].min(), corners[:,2].max())
  box = art3d.Poly3DCollection([corners[face] for face in faces],
                               alpha=0.8,
                               facecolor=color,
                               edgecolor=0.4*color)
  ax.add_collection3d(box)
  return box

def plot_handPath_data(fig, subplot_index,
                       times_s, bodySegment_datas,
                       stationary_poses, referenceObject_positions_m,
                       subject_id, num_subjects,
                       spf=0.1,  # None to view all frames
                       include_skeleton=True,
                       trial_indexes_filter=None,
                       trial_start_index_offset=0,
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
    trial_index_withOffset = trial_index + trial_start_index_offset
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
    hand_position_cm = 100*bodySegment_data['position_m']['RightHand']
    forearm_position_cm = 100*bodySegment_data['position_m']['RightForeArm']
    upperArm_position_cm = 100*bodySegment_data['position_m']['RightUpperArm']
    pelvis_position_cm = 100*bodySegment_data['position_m']['Pelvis']
    h_hand_path = ax.plot3D(hand_position_cm[:, 0], hand_position_cm[:, 1], hand_position_cm[:, 2], alpha=1)
    if include_skeleton:
      ax.plot3D(forearm_position_cm[:, 0], forearm_position_cm[:, 1], forearm_position_cm[:, 2], alpha=0.3)
      ax.plot3D(upperArm_position_cm[:, 0], upperArm_position_cm[:, 1], upperArm_position_cm[:, 2], alpha=0.3)
      ax.plot3D(pelvis_position_cm[:, 0], pelvis_position_cm[:, 1], pelvis_position_cm[:, 2], alpha=0.3)
    
    # Plot origin and start/end/stationary hand positions.
    ax.scatter(0, 0, 0, s=25, color=[0, 0, 0])
    ax.scatter(hand_position_cm[0, 0], hand_position_cm[0, 1], hand_position_cm[0, 2], s=25, color='g')
    ax.scatter(hand_position_cm[-1, 0], hand_position_cm[-1, 1], hand_position_cm[-1, 2], s=25, color='r')
    ax.scatter(100*stationary_poses[trial_index]['position_m']['RightHand'][0],
               100*stationary_poses[trial_index]['position_m']['RightHand'][1],
               100*stationary_poses[trial_index]['position_m']['RightHand'][2],
               s=25, color=h_hand_path[0].get_color(), edgecolor='k')
    ax.scatter(100*referenceObject_positions_m[trial_index][0],
               100*referenceObject_positions_m[trial_index][1],
               100*referenceObject_positions_m[trial_index][2],
               s=25, color=h_hand_path[0].get_color(), edgecolor='c')

    if include_skeleton:
      # Animate the whole skeleton.
      h_chains = []
      h_scatters = []
      h_hand = None
      h_pitcher = None
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
        if h_pitcher is not None:
          h_pitcher.remove()
        h_chains = []
        h_scatters = []
        h_hand = None
        h_pitcher = None
        
        # Draw the skeleton chains
        for chain_name, segment_names in bodySegment_chains_labels_toPlot.items():
          x = []
          y = []
          z = []
          for segment_name in segment_names:
            position_cm = 100*bodySegment_data['position_m'][segment_name]
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
        # hand_rotation_matrix = Rotation.from_quat(bodySegment_data['quaternion']['RightHand'])
        # print(hand_rotation_matrix.apply(hand_dimensions_cm))
        # hand_box_data = np.ones(hand_dimensions_cm, dtype=bool)
        # hand_colors = np.empty(hand_dimensions_cm + [4], dtype=np.float32)
        # hand_colors[:] = [1, 0, 0, 0.8]
        # h_hand = ax.voxels(hand_box_data, facecolors=hand_colors)
        h_hand = plot_hand_box(ax, hand_center=100*bodySegment_data['position_m']['RightHand'][time_index, :],
                               hand_quaternion_localToGlobal_wijk=bodySegment_data['quaternion_wijk']['RightHand'][time_index, :])
        h_pitcher = plot_pitcher_box(ax, hand_center=100*bodySegment_data['position_m']['RightHand'][time_index, :],
                                  hand_quaternion_localToGlobal_wijk=bodySegment_data['quaternion_wijk']['RightHand'][time_index, :])
        
        # Set the aspect ratio
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_zlim(z_lim)
        ax.set_box_aspect([ub - lb for lb, ub in (x_lim, y_lim, z_lim)])
        
        # Set the title.
        ax.set_title('Subject %02d Trial %02d\nt=%0.2fs' % (subject_id, trial_index_withOffset, times_s[trial_index][time_index]-times_s[trial_index][0]))
        
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
def save_trial_eyeVideos(h5_file, eyeVideo_filepath, times_s, subject_id, trial_indexes_filter=None, trial_start_index_offset=0):
  device_name = 'eye-tracking-video-worldGaze'
  stream_name = 'frame_timestamp'
  frames_time_s = h5_file[device_name][stream_name]['data']
  
  for trial_index, time_s in enumerate(times_s):
    trial_index_withOffset = trial_index+trial_start_index_offset
    print('Saving eye video for Subject S%02d trial %02d' % (subject_id, trial_index_withOffset))
    if trial_indexes_filter is not None and trial_index not in trial_indexes_filter:
      continue
    start_time_s = min(time_s)
    end_time_s = max(time_s)
    frame_indexes = np.where((frames_time_s >= start_time_s) & (frames_time_s <= end_time_s))[0]
    start_frame_index = min(frame_indexes)
    num_frames = len(frame_indexes)
    video_reader = cv2.VideoCapture(eyeVideo_filepath)
    video_writer = cv2.VideoWriter(os.path.join(output_dir, '%s_eyeVideo_S%02d_%02d.mp4' % (target_activity_keyword, subject_id, trial_index_withOffset)),
                                   cv2.VideoWriter_fourcc(*'MP4V'),  # for AVI: cv2.VideoWriter_fourcc(*'MJPG'),
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
  
def save_activity_composite_videos(h5_file, eyeVideo_filepath,
                                   times_s, bodySegment_datas,
                                   stationary_poses, referenceObject_positions_cm,
                                   subject_id, num_subjects, trial_indexes_filter=None,
                                   trial_start_index_offset=0):
  device_name = 'eye-tracking-video-worldGaze'
  stream_name = 'frame_timestamp'
  frames_time_s = h5_file[device_name][stream_name]['data']
  fig = None
  
  for trial_index, time_s in enumerate(times_s):
    trial_index_withOffset = trial_index + trial_start_index_offset
    print('Saving composite video for Subject S%02d trial %02d' % (subject_id, trial_index_withOffset))
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
      fig = plot_handPath_data(fig, 0,
                               times_s, bodySegment_datas,
                               stationary_poses, referenceObject_positions_cm,
                               subject_id, 1,
                               spf=None,  # scan all frames for the closest time match
                               include_skeleton=True,
                               trial_indexes_filter=[trial_index],
                               trial_start_index_offset=trial_start_index_offset,
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
        video_writer = cv2.VideoWriter(os.path.join(output_dir, '%s_composite_S%02d_%02d.mp4' % (target_activity_keyword, subject_id, trial_index_withOffset)),
                                       cv2.VideoWriter_fourcc(*'MP4V'), # for AVI: cv2.VideoWriter_fourcc(*'MJPG'),
                                       video_reader.get(cv2.CAP_PROP_FPS),
                                       (composite_frame.shape[1], composite_frame.shape[0])
                                       )
      video_writer.write(composite_frame)
    video_reader.release()

def export_path_data(times_s_allSubjects, bodyPath_datas_allSubjects,
                     stationary_times_s_allSubjects, stationary_pose_allSubjects,
                     referenceObject_positions_m_allSubjects):
  # Open the output HDF5 file
  hdf5_output_filepath = os.path.join(output_dir, '%s_paths_humans.hdf5' % target_activity_keyword)
  if os.path.exists(hdf5_output_filepath):
    print()
    print('Output file exists at [%s]' % hdf5_output_filepath)
    print('  Overwrite the file? [y/N] ', end='')
    overwrite_file = 'y'#input()
    if overwrite_file.lower().strip() != 'y':
      print('  Aborting')
      return
  hdf5_file = h5py.File(hdf5_output_filepath, 'w')
  
  body_segment_names = None
  body_joint_names = None
  
  for subject_id in bodyPath_datas_allSubjects:
    num_trials = len(bodyPath_datas_allSubjects[subject_id])
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
      data_segmentDict = bodyPath_datas_allSubjects[subject_id][trial_index]['position_m']
      data = list(data_segmentDict.values())
      data = np.stack(data, axis=0)
      data = np.moveaxis(data, 0, 1) # convert from [segment][time][xyz] to [time][segment][xyz]
      trial_group.create_dataset('body_segment_position_m', data=data)
      # Add body segment orientation data
      data_segmentDict = bodyPath_datas_allSubjects[subject_id][trial_index]['quaternion_wijk']
      data = list(data_segmentDict.values())
      data = np.stack(data, axis=0)
      data = np.moveaxis(data, 0, 1) # convert from [segment][time][wxyz] to [time][segment][wxyz]
      trial_group.create_dataset('body_segment_quaternion_wijk', data=data)
      # Add body joint angle data
      data_jointDict = bodyPath_datas_allSubjects[subject_id][trial_index]['joint_angle_eulerZXY_xyz_rad']
      data = list(data_jointDict.values())
      data = np.stack(data, axis=0)
      data = np.moveaxis(data, 0, 1) # convert from [joint][time][xyz] to [time][joint][xzy]
      trial_group.create_dataset('joint_angle_eulerZXY_xyz_rad', data=data)
      data_jointDict = bodyPath_datas_allSubjects[subject_id][trial_index]['joint_angle_eulerXZY_xyz_rad']
      data = list(data_jointDict.values())
      data = np.stack(data, axis=0)
      data = np.moveaxis(data, 0, 1) # convert from [joint][time][xyz] to [time][joint][xzy]
      trial_group.create_dataset('joint_angle_eulerXZY_xyz_rad', data=data)
      # Add estimated stationary position
      stationary_group = trial_group.create_group('stationary')
      data_segmentDict = stationary_pose_allSubjects[subject_id][trial_index]['position_m']
      data = list(data_segmentDict.values())
      data = np.stack(data, axis=0)
      stationary_group.create_dataset('body_segment_position_m', data=data)
      # Add estimated stationary quaternion
      data_segmentDict = stationary_pose_allSubjects[subject_id][trial_index]['quaternion_wijk']
      data = list(data_segmentDict.values())
      data = np.stack(data, axis=0)
      stationary_group.create_dataset('body_segment_quaternion_wijk', data=data)
      # Add stationary time
      stationary_group.create_dataset('time_s',
                                      data=stationary_times_s_allSubjects[subject_id][trial_index] - min(time_s))
      # Add estimated reference object position
      referenceObject_group = trial_group.create_group('reference_object')
      data = referenceObject_positions_m_allSubjects[subject_id][trial_index]
      data = np.stack(data, axis=0)
      referenceObject_group.create_dataset('position_m', data=data)
      
      # Store body segment and joint names.
      # This should be the same for every subject/trial.
      body_segment_names_forTrial = list(bodyPath_datas_allSubjects[subject_id][0]['position_m'].keys())
      body_joint_names_forTrial = list(bodyPath_datas_allSubjects[subject_id][0]['joint_angle_eulerZXY_xyz_rad'].keys())
      if body_segment_names is None:
        body_segment_names = body_segment_names_forTrial
        body_joint_names = body_joint_names_forTrial
      else:
        assert body_segment_names_forTrial == body_segment_names
        assert body_joint_names_forTrial == body_joint_names
  
  # Add segment names
  hdf5_file.create_dataset('body_segment_names', data=body_segment_names)
  # Add joint names
  hdf5_file.create_dataset('body_joint_names', data=body_joint_names)

  # Close the output file
  hdf5_file.close()

  # # Make a file dedicated to the reference object positions.
  # hdf5_output_filepath = os.path.join(output_dir, '%s_referenceOjbects.hdf5' % target_activity_keyword)
  # if os.path.exists(hdf5_output_filepath):
  #   print()
  #   print('Output file exists at [%s]' % hdf5_output_filepath)
  #   print('  Overwrite the file? [y/N] ', end='')
  #   overwrite_file = input()
  #   if overwrite_file.lower().strip() != 'y':
  #     print('  Aborting')
  #     return
  # hdf5_file = h5py.File(hdf5_output_filepath, 'w')
  # for subject_id in bodySegment_datas_allSubjects:
  #   subject_group = hdf5_file.create_group('subject_%02d' % subject_id)
  #   data = referenceObject_positions_m_allSubjects[subject_id]
  #   data = np.stack(data, axis=0)
  #   subject_group.create_dataset('position_m', data=data)
  # hdf5_file.close()

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

# Helper to get start and end times of the target activity.
def get_targetActivity_startEnd_times_s(h5_file, exclude_bad_labels=True):
  # Get start/end times of every activity.
  (activities_labels, activities_start_times_s, activities_end_times_s) = get_activity_startEnd_times_s(h5_file, exclude_bad_labels=exclude_bad_labels)
  
  # Filter by the target activity label.
  targetActivity_indexes = [i for (i, label) in enumerate(activities_labels) if target_activity_label in label]
  if len(targetActivity_indexes) == 0:
    return (None, None, None)
  activities_labels = activities_labels[targetActivity_indexes]
  activities_start_times_s = activities_start_times_s[targetActivity_indexes]
  activities_end_times_s = activities_end_times_s[targetActivity_indexes]
  
  return (activities_labels, activities_start_times_s, activities_end_times_s)

# Helper to get manual start and end times of the target activity.
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
  
  activities_labels = [target_activity_label]*num_trials
  
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
fig_animatePath = None
fig_allPaths = None
allPaths_subplot_index = 0
times_s_allSubjects = OrderedDict()
bodyPath_datas_allSubjects = OrderedDict()
stationary_times_s_allSubjects = OrderedDict()
stationary_poses_allSubjects = OrderedDict()
referenceObject_positions_m_allSubjects = OrderedDict()
for subject_id, subject_hdf5_filepaths in hdf5_filepaths.items():
  print('Processing subject %02d' % subject_id)
  times_s_allSubjects.setdefault(subject_id, [])
  bodyPath_datas_allSubjects.setdefault(subject_id, [])
  stationary_times_s_allSubjects.setdefault(subject_id, [])
  stationary_poses_allSubjects.setdefault(subject_id, [])
  referenceObject_positions_m_allSubjects.setdefault(subject_id, [])
  targetActivity_trial_index_start = 0
  for (filepath_index, hdf5_filepath) in enumerate(subject_hdf5_filepaths):
    print(' ', hdf5_filepath)
    eyeVideo_filepath = eyeVideo_filepaths[subject_id][filepath_index]
    # Open the HDF5 file.
    h5_file = h5py.File(hdf5_filepath, 'r')
    
    # Determine the start/end times of each hand path.
    (activities_labels, activities_start_times_s, activities_end_times_s) = \
      get_targetActivity_startEnd_times_s(h5_file, exclude_bad_labels=True)
    if activities_labels is None:
      continue
    num_trials_inFile = len(activities_labels)
    if use_manual_startEnd_times:
      (activities_labels, activities_start_times_s, activities_end_times_s) = \
        get_manual_pouring_startEnd_times_s(subject_id)
      activities_labels = activities_labels[targetActivity_trial_index_start:(targetActivity_trial_index_start + num_trials_inFile)]
      activities_start_times_s = activities_start_times_s[targetActivity_trial_index_start:(targetActivity_trial_index_start + num_trials_inFile)]
      activities_end_times_s = activities_end_times_s[targetActivity_trial_index_start:(targetActivity_trial_index_start + num_trials_inFile)]
      # print([activities_start_times_s[i] - activities_start_times_s[i] for i in range(5)])
      # print([activities_end_times_s[i] - activities_end_times_s[i] for i in range(5)])
      # print([get_time_str(t) for t in activities_start_times_s])
      # print([get_time_str(t) for t in activities_start_times_s])

    # Get the hand paths.
    (times_s, bodyPath_datas) = get_activity_bodyPath_data(h5_file, activities_start_times_s, activities_end_times_s)
    (times_s, bodyPath_datas) = resample_data(times_s, bodyPath_datas)
    # Infer the hand position while being relatively stationary
    (stationary_times_s, stationary_poses) = infer_stationary_poses(times_s, bodyPath_datas)
    # Infer the reference object position
    referenceObject_positions_m = infer_referenceObject_positions(bodyPath_datas, times_s, stationary_times_s)

    # Store the results
    times_s_allSubjects[subject_id].extend(times_s)
    bodyPath_datas_allSubjects[subject_id].extend(bodyPath_datas)
    stationary_times_s_allSubjects[subject_id].extend(stationary_times_s)
    stationary_poses_allSubjects[subject_id].extend(stationary_poses)
    referenceObject_positions_m_allSubjects[subject_id].extend(referenceObject_positions_m)

    # Plot the paths.
    if animate_trajectory_plots:
      fig_animatePath = plot_handPath_data(fig_animatePath, 0,
                                           times_s, bodyPath_datas,
                                           stationary_poses, referenceObject_positions_m,
                                           subject_id, 1, trial_indexes_filter=None,
                                           trial_start_index_offset=targetActivity_trial_index_start,
                                           spf=0.25,
                                           pause_between_trials=False, pause_between_frames=True)
    if plot_all_trajectories or save_plot_all_trajectories:
      fig_allPaths = plot_handPath_data(fig_allPaths, allPaths_subplot_index,
                                        times_s, bodyPath_datas,
                                        stationary_poses, referenceObject_positions_m,
                                        subject_id, num_subjects, trial_indexes_filter=None,
                                        trial_start_index_offset=targetActivity_trial_index_start,
                                        target_times_s=stationary_times_s,
                                        include_skeleton=False,
                                        clear_axes_each_trial=False)
      
    if save_eye_videos:
      save_trial_eyeVideos(h5_file, eyeVideo_filepath, times_s, subject_id,
                           trial_start_index_offset=targetActivity_trial_index_start,
                           trial_indexes_filter=None)
    if save_composite_videos:
      save_activity_composite_videos(h5_file, eyeVideo_filepath,
                                     times_s, bodyPath_datas,
                                     stationary_poses, referenceObject_positions_m,
                                     subject_id, num_subjects, trial_indexes_filter=None,
                                     trial_start_index_offset=targetActivity_trial_index_start, )
    # Close the HDF5 file.
    h5_file.close()

    # Increment the trial index offset counter for this subject.
    targetActivity_trial_index_start += num_trials_inFile
    
  # Advance subplot index if putting each subject in a new subplot
  if plot_all_trajectories or save_plot_all_trajectories:
    allPaths_subplot_index += 1

# Save the all-trajectories plot if desired.
if save_plot_all_trajectories:
  fig_allPaths.savefig(os.path.join(output_dir, '%s_allHumanTrajectories.jpg' % (target_activity_keyword)),
                       dpi=300)
  
# Export the results if desired
if save_results_data:
  export_path_data(times_s_allSubjects, bodyPath_datas_allSubjects,
                   stationary_times_s_allSubjects, stationary_poses_allSubjects,
                   referenceObject_positions_m_allSubjects)
  
# Show the final plot
if animate_trajectory_plots or plot_all_trajectories:
  plt.show(block=True)




















