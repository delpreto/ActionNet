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
# See https://action-net.csail.mit.edu for more usage information.
# Created 2021-2024 for the MIT ActionSense project by Joseph DelPreto [https://josephdelpreto.com].
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

from learning_trajectories.helpers.configuration import *
from learning_trajectories.helpers.parse_process_feature_data import *
from learning_trajectories.helpers.numpy_scipy_utils import *
from learning_trajectories.helpers.printing import *

import matplotlib.pyplot as plt

#===========================================================
# Extract hand path data from an HDF5 file.
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
  # # Get the timestamps for each entry as human-readable strings.
  # time_str = h5_file[device_name][stream_name]['time_str']
  # time_str = np.squeeze(np.array(time_str)) # squeeze (optional) converts from a list of single-element lists to a 1D list
  
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

#===========================================================
# Extract path data for specified activities.
def get_bodyPath_data_byTrial(h5_file, start_times_s, end_times_s):
  # Get path data throughout the whole experiment.
  (time_s, bodyPath_data) = get_bodyPath_data(h5_file)
  
  time_s_byTrial = []
  bodyPath_data_byTrial = []
  # Extract each desired activity.
  for time_index in range(len(start_times_s)):
    start_time_s = start_times_s[time_index]
    end_time_s = end_times_s[time_index]
    indexes_forTrial = np.where((time_s >= start_time_s) & (time_s <= end_time_s))[0]
    if indexes_forTrial.size > 0:
      time_s_byTrial.append(time_s[indexes_forTrial])
      bodyPath_data_byTrial.append(OrderedDict())
      for data_type in bodyPath_data.keys():
        bodyPath_data_byTrial[-1].setdefault(data_type, OrderedDict())
        for body_segment in bodyPath_data[data_type].keys():
          bodyPath_data_byTrial[-1][data_type].setdefault(body_segment, OrderedDict())
          bodyPath_data_byTrial[-1][data_type][body_segment] = bodyPath_data[data_type][body_segment][indexes_forTrial, :]
  
  return (time_s_byTrial, bodyPath_data_byTrial)

#===========================================================
# Shift and rotate body path data to a person-centric coordinate frame.
def transform_bodyPath_data_personFrame(time_s_byTrial, bodyPath_data_byTrial, activity_type=None):
  if not isinstance(bodyPath_data_byTrial, (list, tuple)):
    bodyPath_data_byTrial = [bodyPath_data_byTrial]
    time_s_byTrial = [time_s_byTrial]
    return_single_trial = True
  else:
    return_single_trial = False
  
  bodyPath_origin_xyz_m_byTrial = []
  
  # Shift to a coordinate frame with the origin between the hips and at the table height.
  starting_positions_m = {}
  for trial_index in range(len(bodyPath_data_byTrial)):
    # Set the origin between the hips (very close to the pelvis but not exactly).
    # pelvis_position_m = bodySegment_datas[trial_index]['position_m']['Pelvis']
    # origin_m = pelvis_position_m[0,:]
    origin_m = np.mean(np.array(
          [bodyPath_data_byTrial[trial_index]['position_m']['RightUpperLeg'][0, :],
           bodyPath_data_byTrial[trial_index]['position_m']['LeftUpperLeg'][0, :]]), axis=0)
    bodyPath_origin_xyz_m_byTrial.append(origin_m)
    # Use the table as the z origin.
    origin_m[2] = table_height_cm/100
    # Adjust all coordinates for the new origin.
    for body_segment in bodyPath_data_byTrial[trial_index]['position_m'].keys():
      position_m = bodyPath_data_byTrial[trial_index]['position_m'][body_segment]
      position_m = position_m - origin_m
      bodyPath_data_byTrial[trial_index]['position_m'][body_segment] = position_m
      starting_positions_m.setdefault(body_segment, [])
      starting_positions_m[body_segment].append(bodyPath_data_byTrial[trial_index]['position_m'][body_segment][0,:])
  print('      Right hand starting positions [cm]:')
  print('        medn', 100*np.median(starting_positions_m['RightHand'], axis=0))
  print('        std ', 100*np.std(starting_positions_m['RightHand'], axis=0))
  print('        min ', 100*np.min(starting_positions_m['RightHand'], axis=0))
  print('        max ', 100*np.max(starting_positions_m['RightHand'], axis=0))
  print('      Left hand starting positions [cm]:')
  print('        medn', 100*np.median(starting_positions_m['LeftHand'], axis=0))
  print('        std ', 100*np.std(starting_positions_m['LeftHand'], axis=0))
  print('        min ', 100*np.min(starting_positions_m['LeftHand'], axis=0))
  print('        max ', 100*np.max(starting_positions_m['LeftHand'], axis=0))
  # Correct for what seems to be sensor error for this subject,
  #  given that we expect the motion hand to be at a known height at the start.
  z_offsets_m = []
  if activity_type in target_starting_height_cm and isinstance(target_starting_height_cm[activity_type], dict):
    for (body_segment, target_height_cm) in target_starting_height_cm[activity_type].items():
      starting_height_m = np.median(starting_positions_m[body_segment], axis=0)[2]
      z_offsets_m.append(starting_height_m - target_height_cm/100)
  if len(z_offsets_m) > 0:
    z_offset_m = np.median(z_offsets_m)
    for trial_index in range(len(bodyPath_data_byTrial)):
      for body_segment in bodyPath_data_byTrial[trial_index]['position_m'].keys():
        position_m = bodyPath_data_byTrial[trial_index]['position_m'][body_segment]
        position_m[:, 2] = position_m[:, 2] - z_offset_m
        bodyPath_data_byTrial[trial_index]['position_m'][body_segment] = position_m
  else:
    z_offset_m = 0 # for printing purposes
  
  # Rotate to a coordinate frame based on the body, such that
  #  the y axis is aligned with the shoulders/hips.
  starting_positions_m = {}
  for trial_index in range(len(bodyPath_data_byTrial)):
    # Use the hip orientation to create the y axis.
    y_axis_right = np.append(bodyPath_data_byTrial[trial_index]['position_m']['RightUpperLeg'][0, 0:2], 0)
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
    for body_segment in bodyPath_data_byTrial[trial_index]['position_m'].keys():
      position_m = bodyPath_data_byTrial[trial_index]['position_m'][body_segment]
      for time_index in range(position_m.shape[0]):
        position_m[time_index,:] = alignment_rotation_matrix.dot(position_m[time_index,:])

    # Compose the rotation with quaternion orientations.
    alignment_rotation = Rotation.from_matrix(alignment_rotation_matrix)
    for body_segment in bodyPath_data_byTrial[trial_index]['quaternion_wijk'].keys():
      quaternion_wijk = bodyPath_data_byTrial[trial_index]['quaternion_wijk'][body_segment]
      for time_index in range(quaternion_wijk.shape[0]):
        quaternion_forTime_ijkw = quaternion_wijk[time_index, [1,2,3,0]]
        quaternion_rotation = Rotation.from_quat(quaternion_forTime_ijkw)
        aligned_quaternion_rotation = alignment_rotation * quaternion_rotation # note that multiplication is overloaded for scipy Rotation objects
        aligned_quaternion_ijkw = aligned_quaternion_rotation.as_quat()
        quaternion_wijk[time_index,:] = aligned_quaternion_ijkw[[3,0,1,2]]
      starting_positions_m.setdefault(body_segment, [])
      starting_positions_m[body_segment].append(bodyPath_data_byTrial[trial_index]['position_m'][body_segment][0,:])
    
  print('      %s hand starting positions after correcting for a z offset of %0.3f cm and rotating for the hip axis:' % (motionObject_rightOrLeftArm[activity_to_process], 100*z_offset_m))
  print('        medn', 100*np.median(starting_positions_m['%sHand' % motionObject_rightOrLeftArm[activity_to_process]], axis=0))
  print('        std ', 100*np.std(starting_positions_m['%sHand' % motionObject_rightOrLeftArm[activity_to_process]], axis=0))
  print('        min ', 100*np.min(starting_positions_m['%sHand' % motionObject_rightOrLeftArm[activity_to_process]], axis=0))
  print('        max ', 100*np.max(starting_positions_m['%sHand' % motionObject_rightOrLeftArm[activity_to_process]], axis=0))
  print('      %s hand starting positions after correcting for a z offset of %0.3f cm and rotating for the hip axis:' % (motionObject_rightOrLeftArm[activity_to_process], 100*z_offset_m))
  print('        medn', 100*np.median(starting_positions_m['%sHand' % motionObject_rightOrLeftArm[activity_to_process]], axis=0))
  print('        std ', 100*np.std(starting_positions_m['%sHand' % motionObject_rightOrLeftArm[activity_to_process]], axis=0))
  print('        min ', 100*np.min(starting_positions_m['%sHand' % motionObject_rightOrLeftArm[activity_to_process]], axis=0))
  print('        max ', 100*np.max(starting_positions_m['%sHand' % motionObject_rightOrLeftArm[activity_to_process]], axis=0))
  
  # Return the transformed body path data.
  if return_single_trial:
    return (time_s_byTrial[0], bodyPath_data_byTrial[0], bodyPath_origin_xyz_m_byTrial[0])
  else:
    return (time_s_byTrial, bodyPath_data_byTrial, bodyPath_origin_xyz_m_byTrial)

#===========================================================
# Shift the trials upwards if they are too low for the reference object during the stationary window.
def transform_bodyPath_data_referenceObjectHeight(
        bodyPath_data_byTrial, stationary_pose_byTrial, referenceObject_position_m_byTrial,
        activity_type):
  if not isinstance(bodyPath_data_byTrial, (list, tuple)):
    bodyPath_data_byTrial = [bodyPath_data_byTrial]
    return_single_trial = True
  else:
    return_single_trial = False
  
  # Translate vertically to have the object keypoint above the reference object.
  offsets_to_referenceObject_m = []
  offsets_to_referenceObject_adjusted_m = []
  for trial_index in range(len(bodyPath_data_byTrial)):
    # Infer the average stationary position.
    stationary_start_time_index = stationary_pose_byTrial[trial_index]['start_time_index']
    stationary_end_time_index = stationary_pose_byTrial[trial_index]['end_time_index']
    motionObjectKeypoint_positions_m = []
    for time_index in range(stationary_start_time_index, stationary_end_time_index+1):
      motionObjectKeypoint_positions_m.append(eval(infer_motionObjectKeypoint_position_m_fn[activity_type])(
        bodyPath_data_to_parsed_feature_data(bodyPath_data_byTrial[trial_index]), activity_type, time_index))
    motionObjectKeypoint_position_m = np.median(motionObjectKeypoint_positions_m, axis=0)
    offset_m = motionObjectKeypoint_position_m[2] - referenceObject_position_m_byTrial[trial_index][2]
    offsets_to_referenceObject_m.append(offset_m)
    # Raise the trajectory if needed.
    if offset_m < 0/100:
      amount_to_raise_m = abs(offset_m)-0/100
      print('      Raising the body data for trial %d by %0.2fcm' % (trial_index, 100*amount_to_raise_m))
      for body_segment in bodyPath_data_byTrial[trial_index]['position_m'].keys():
        position_m = bodyPath_data_byTrial[trial_index]['position_m'][body_segment]
        position_m[:, 2] = position_m[:, 2] + amount_to_raise_m
        bodyPath_data_byTrial[trial_index]['position_m'][body_segment] = position_m
    # Infer the average stationary position again to check the result.
    stationary_start_time_index = stationary_pose_byTrial[trial_index]['start_time_index']
    stationary_end_time_index = stationary_pose_byTrial[trial_index]['end_time_index']
    motionObjectKeypoint_positions_m = []
    for time_index in range(stationary_start_time_index, stationary_end_time_index+1):
      motionObjectKeypoint_positions_m.append(eval(infer_motionObjectKeypoint_position_m_fn[activity_type])(
        bodyPath_data_to_parsed_feature_data(bodyPath_data_byTrial[trial_index]), activity_type, time_index))
    motionObjectKeypoint_position_m = np.median(motionObjectKeypoint_positions_m, axis=0)
    offset_m = motionObjectKeypoint_position_m[2] - referenceObject_position_m_byTrial[trial_index][2]
    offsets_to_referenceObject_adjusted_m.append(offset_m)
    
  print('      Motion object keypoint height above reference object height')
  print('        mean', 100*np.mean(offsets_to_referenceObject_m, axis=0))
  print('        std ', 100*np.std(offsets_to_referenceObject_m, axis=0))
  print('        min ', 100*np.min(offsets_to_referenceObject_m, axis=0))
  print('        max ', 100*np.max(offsets_to_referenceObject_m, axis=0))
  print('      Motion object keypoint height above reference object height after raising trajectories when needed')
  print('        mean', 100*np.mean(offsets_to_referenceObject_adjusted_m, axis=0))
  print('        std ', 100*np.std(offsets_to_referenceObject_adjusted_m, axis=0))
  print('        min ', 100*np.min(offsets_to_referenceObject_adjusted_m, axis=0))
  print('        max ', 100*np.max(offsets_to_referenceObject_adjusted_m, axis=0))
  
  # Return the transformed body path data.
  if return_single_trial:
    return bodyPath_data_byTrial[0]
  else:
    return bodyPath_data_byTrial
  
#===========================================================
# Infer the left hand position, which is holding the reference object.
def infer_referenceObject_position_m_byTrial(bodyPath_data_byTrial, time_s_byTrial,
                                             referenceObject_start_time_s_byTrial,
                                             referenceObject_end_time_s_byTrial,
                                             activity_type,
                                             use_motionObjectKeypoint_position_xy=False,
                                             stationary_pose_byTrial=None):
  referenceObject_position_m_byTrial = []
  handsegment_to_motionObjectKeypoint_diffs_m = []
  for (trial_index, bodyPath_data) in enumerate(bodyPath_data_byTrial):
    # Infer the location using the specified body segment.
    time_s = time_s_byTrial[trial_index]
    body_position_m = bodyPath_data['position_m']
    body_quaternion_wijk = bodyPath_data['quaternion_wijk']
    referenceObject_start_time_s = referenceObject_start_time_s_byTrial[trial_index]
    referenceObject_end_time_s = referenceObject_end_time_s_byTrial[trial_index]
    referenceObject_start_index = time_s.searchsorted(referenceObject_start_time_s)
    referenceObject_end_index = time_s.searchsorted(referenceObject_end_time_s)
    referenceObject_segment_position_m = np.median(
      body_position_m[referenceObject_bodySegment_name[activity_type]][referenceObject_start_index:referenceObject_end_index,:],
      axis=0)
    referenceObject_segment_quaternion_wijk = body_quaternion_wijk[referenceObject_bodySegment_name[activity_type]][int(np.mean([referenceObject_start_index, referenceObject_end_index])), :]
    quat_ijkw = referenceObject_segment_quaternion_wijk[[1,2,3,0]]
    quat_ijkw = [-quat_ijkw[0], -quat_ijkw[1], -quat_ijkw[2], quat_ijkw[3]]
    referenceObject_segment_rot = Rotation.from_quat(quat_ijkw).as_matrix()
    referenceObject_offset_rotated_cm = np.dot(referenceObject_offset_cm[activity_type], referenceObject_segment_rot)
    referenceObject_position_m_fromSegment = referenceObject_segment_position_m + referenceObject_offset_rotated_cm/100.0
    referenceObject_position_m = referenceObject_position_m_fromSegment.copy()
    
    # Use a known height for the reference object.
    # Assume z=0 is at the table height.
    referenceObject_position_m[2] = referenceObject_height_cm[activity_type]/100
    
    # Move the xy position to the stationary position if desired.
    if use_motionObjectKeypoint_position_xy:
      stationary_start_time_index = stationary_pose_byTrial[trial_index]['start_time_index']
      stationary_end_time_index = stationary_pose_byTrial[trial_index]['end_time_index']
      motionObjectKeypoint_positions_m = []
      for time_index in range(stationary_start_time_index, stationary_end_time_index+1):
        motionObjectKeypoint_positions_m.append(eval(infer_motionObjectKeypoint_position_m_fn[activity_type])(
          bodyPath_data_to_parsed_feature_data(bodyPath_data), activity_type, time_index))
      motionObjectKeypoint_position_m = np.median(motionObjectKeypoint_positions_m, axis=0)
      referenceObject_position_m[0:2] = motionObjectKeypoint_position_m[0:2]
    
    # Store the result.
    referenceObject_position_m_byTrial.append(referenceObject_position_m)
    handsegment_to_motionObjectKeypoint_diffs_m.append(referenceObject_position_m - referenceObject_position_m_fromSegment)
  
  if use_motionObjectKeypoint_position_xy:
    print('      Amount reference object was moved from hand-based inference to motion object keypoint position [cm]:')
    print('       mean', 100*np.mean(handsegment_to_motionObjectKeypoint_diffs_m, axis=0))
    print('       std ', 100*np.std(handsegment_to_motionObjectKeypoint_diffs_m, axis=0))
    print('       min ', 100*np.min(handsegment_to_motionObjectKeypoint_diffs_m, axis=0))
    print('       max ', 100*np.max(handsegment_to_motionObjectKeypoint_diffs_m, axis=0))
  return referenceObject_position_m_byTrial


#===========================================================
# Infer the pitcher angle in the hand, based on distance to pouring success.
def infer_pitcher_holding_angle_rad_byTrial(bodyPath_data_byTrial,
                                            stationary_pose_byTrial,
                                            referenceObject_position_m_byTrial,
                                            plot_distance_metrics_eachTrial=False,
                                            subject_id=None):
  # Specify holding angles to test.
  hand_to_pitcher_angles_rad_toTest = []
  hand_to_pitcher_rotations_toTest = []
  
  # tilt left/right (positive/negative)
  x_degs = np.arange(60, 130+0.1, 2)
  # x_degs = np.arange(90, 90+0.1, 4)
  
  # tilt down/up (positive/negative)
  y_degs = np.arange(-20, 20+0.1, 5)
  # y_degs = np.arange(0, 0+0.1, 1)
  
  # tilt inward/outward (positive/negative)
  z_degs = np.arange(-20, 20+0.1, 5)
  # z_degs = np.arange(-5, -5+0.1, 1)
  
  for x_deg in x_degs:
    for y_deg in y_degs:
      for z_deg in z_degs:
        hand_to_pitcher_angles_rad_toTest.append(np.radians(np.array([x_deg, y_deg, z_deg])))
        hand_to_pitcher_rotations_toTest.append(
          Rotation.from_rotvec(hand_to_pitcher_angles_rad_toTest[-1])
        )
  hand_to_pitcher_angles_rad_toTest = np.array(hand_to_pitcher_angles_rad_toTest)
  best_hand_to_pitcher_angles_rad_byTrial = []
  if plot_distance_metrics_eachTrial:
    fig = plt.figure()
  else:
    fig = None
  # Test all angles for each trial.
  num_trials = len(bodyPath_data_byTrial)
  for trial_index in range(num_trials):
    stationary_pose = stationary_pose_byTrial[trial_index]
    stationary_time_index = stationary_pose['time_index']
    referenceObject_position_m = referenceObject_position_m_byTrial[trial_index]
    bodyPath_data = bodyPath_data_byTrial[trial_index]
    parsed_feature_data = bodyPath_data_to_parsed_feature_data(bodyPath_data, time_s=None,
                                                               referenceObject_position_m=referenceObject_position_m,
                                                               hand_to_pitcher_angles_rad=None)
    distances_to_success = []
    distances_to_center = []
    is_successful = []
    for (test_index, hand_to_pitcher_rotation_toTest) in enumerate(hand_to_pitcher_rotations_toTest):
      # Infer the spout pouring relative offset.
      spout_relativeOffset_cm = infer_motionObject_referenceObject_relativeOffset_cm(
        parsed_feature_data, referenceObject_position_m, time_index=stationary_time_index,
        hand_to_motionObject_rotation_toUse=hand_to_pitcher_rotation_toTest)
      # Compute a metric about pouring success.
      glass_rim_buffer_cm = 0.5
      behind_glass_threshold_cm = 1.5
      distance_to_glass_center_threshold_cm = (referenceObject_diameter_cm/2 - glass_rim_buffer_cm)
      
      # horizontal_success = abs(spout_relativeOffset_cm[0]) < distance_to_glass_center_threshold_cm
      # vertical_success = False
      # if spout_relativeOffset_cm[1] > 0 and spout_relativeOffset_cm[1] < distance_to_glass_center_threshold_cm:
      #   vertical_success = True
      # if spout_relativeOffset_cm[1] < 0 and abs(spout_relativeOffset_cm[1]) < behind_glass_threshold_cm:
      #   vertical_success = True
      # pouring_success = horizontal_success and vertical_success
      
      horizontal_distance_to_success = abs(spout_relativeOffset_cm[0]) - distance_to_glass_center_threshold_cm
      if spout_relativeOffset_cm[1] > 0:
        if horizontal_distance_to_success <= 0:
          threshold_y_cm = np.sqrt(distance_to_glass_center_threshold_cm**2 - spout_relativeOffset_cm[0]**2)
          vertical_distance_to_success = spout_relativeOffset_cm[1] - threshold_y_cm
        else:
          vertical_distance_to_success = spout_relativeOffset_cm[1] - distance_to_glass_center_threshold_cm
      else:
        if horizontal_distance_to_success <= 0:
          threshold_y_cm = -(np.sqrt((referenceObject_diameter_cm/2)**2 - spout_relativeOffset_cm[0]**2))
          threshold_y_cm -= behind_glass_threshold_cm
          vertical_distance_to_success = abs(spout_relativeOffset_cm[1]) - abs(threshold_y_cm)
        else:
          vertical_distance_to_success = abs(spout_relativeOffset_cm[1]) - (referenceObject_diameter_cm/2+behind_glass_threshold_cm)
      pouring_success = (horizontal_distance_to_success < 0) and (vertical_distance_to_success < 0)
      is_successful.append(pouring_success)
      distances_to_success.append(np.linalg.norm([max(0, horizontal_distance_to_success),
                                                  max(0, vertical_distance_to_success)]))
      distances_to_center.append(np.linalg.norm(spout_relativeOffset_cm))
    
    distances_to_success = np.array(distances_to_success)
    distances_to_center = np.array(distances_to_center)
    is_successful = np.array(is_successful)
    # If at least one was successful, find the one closest to the center.
    # Otherwise, find the one closest to any successful point.
    if np.any(is_successful):
      best_index = np.argmin(distances_to_center)
    else:
      best_index = np.argmin(distances_to_success)
    best_hand_to_pitcher_angles_rad_byTrial.append(hand_to_pitcher_angles_rad_toTest[best_index, :])
  
    # Plot if desired.
    if plot_distance_metrics_eachTrial:
      figManager = plt.get_current_fig_manager()
      figManager.window.showMaximized()
      x_degs = np.degrees(hand_to_pitcher_angles_rad_toTest[:, 0])
      plt.plot(x_degs[successful_indexes], distances_to_success[successful_indexes], 'k*')
      plt.plot(x_degs, distances_to_success, '.-')
      plt.grid(True, color='lightgray')
      plt.title('Subject %s: Pouring Distance Metric by Pitcher Holding Angle' % subject_id)
      plt.xlabel('X Holding Angle [deg]')
      plt.ylabel('Pouring Distance Metric')
  return (best_hand_to_pitcher_angles_rad_byTrial, fig)



#===========================================================
# Get start and end times of each activity.
# Optionally exclude activities marked as bad.
#   Some activities may have been marked as 'Bad' or 'Maybe' by the experimenter.
#   Submitted notes with the activity typically give more information.
def get_activity_startEnd_times_s(h5_file, start_offset_s=0, end_offset_s=0, exclude_bad_labels=True):
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
  # Offset the times if desired.
  activities_start_times_s = activities_start_times_s + start_offset_s
  activities_end_times_s = activities_end_times_s + end_offset_s
  return (activities_labels, activities_start_times_s, activities_end_times_s)

#===========================================================
# Get start and end times of the target activity.
def get_targetActivity_startEnd_times_s(h5_file, target_activity_label,
                                        start_offset_s=0, end_offset_s=0,
                                        exclude_bad_labels=True):
  # Get start/end times of every activity.
  (activities_labels, activities_start_times_s, activities_end_times_s) = \
    get_activity_startEnd_times_s(h5_file, start_offset_s=start_offset_s, end_offset_s=end_offset_s,
                                  exclude_bad_labels=exclude_bad_labels)
  
  # Filter by the target activity label.
  targetActivity_indexes = [i for (i, label) in enumerate(activities_labels) if target_activity_label in label]
  if len(targetActivity_indexes) == 0:
    return (None, None, None)
  activities_labels = activities_labels[targetActivity_indexes]
  activities_start_times_s = activities_start_times_s[targetActivity_indexes]
  activities_end_times_s = activities_end_times_s[targetActivity_indexes]
  
  return (activities_labels, activities_start_times_s, activities_end_times_s)

# #===========================================================
# # Get manual start and end times of the target activity.
# def get_manual_pouring_startEnd_times_s(subject_id):
#   num_trials = 5
#   subject_index = subject_id
#   if subject_id == 0:
#     subject_start_index = 0
#   else:
#     subject_start_index = (subject_index-1)*num_trials # no subject id 1
#
#   activities_start_times_str = manual_pouring_start_times_s[subject_start_index:(subject_start_index+num_trials)]
#   activities_end_times_str = manual_pouring_end_times_s[subject_start_index:(subject_start_index+num_trials)]
#
#   activities_start_times_s = []
#   activities_end_times_s = []
#   for i in range(num_trials):
#     activities_start_times_s.append(
#         get_time_s_from_local_str(activities_start_times_str[i].split(' ')[1], input_time_format='%H:%M:%S.%f',
#                                   date_local_str=activities_start_times_str[i].split(' ')[0], input_date_format='%Y-%m-%d'))
#     activities_end_times_s.append(
#         get_time_s_from_local_str(activities_end_times_str[i].split(' ')[1], input_time_format='%H:%M:%S.%f',
#                                   date_local_str=activities_end_times_str[i].split(' ')[0], input_date_format='%Y-%m-%d'))
#
#   activities_labels = [target_activity_label]*num_trials
#
#   return (activities_labels, activities_start_times_s, activities_end_times_s)
#
# use_manual_startEnd_times = False
# manual_pouring_start_times_s = [
#   '2022-06-07 18:44:42.541198',
#   '2022-06-07 18:45:16.441165',
#   '2022-06-07 18:45:30.841151',
#   '2022-06-07 18:45:42.141141',
#   '2022-06-07 18:45:55.741128',
#   '2022-06-13 22:39:48.417483',
#   '2022-06-13 22:39:59.017473',
#   '2022-06-13 22:40:07.017465',
#   '2022-06-13 22:40:15.917457',
#   '2022-06-13 22:40:26.017447',
#   '2022-06-14 14:03:57.492392',
#   '2022-06-14 14:04:09.692380',
#   '2022-06-14 14:04:18.992372',
#   '2022-06-14 14:04:28.492362',
#   '2022-06-14 14:04:36.792355',
#   '2022-06-14 17:18:49.326952',
#   '2022-06-14 17:18:58.226943',
#   '2022-06-14 17:19:06.226936',
#   '2022-06-14 17:19:14.026928',
#   '2022-06-14 17:19:23.026920',
#   '2022-06-14 21:19:39.860983',
#   '2022-06-14 21:19:48.660974',
#   '2022-06-14 21:19:56.460967',
#   '2022-06-14 21:20:04.160959',
#   '2022-06-14 21:20:11.260953',
#   '2022-07-12 15:27:07.005319',
#   '2022-07-12 15:27:36.105291',
#   '2022-07-12 15:27:47.505280',
#   '2022-07-12 15:27:58.005270',
#   '2022-07-12 15:28:08.005260',
#   '2022-07-13 11:50:11.389303',
#   '2022-07-13 11:50:18.789296',
#   '2022-07-13 11:50:25.389290',
#   '2022-07-13 11:50:30.389285',
#   '2022-07-13 11:50:36.889279',
#   '2022-07-13 15:04:19.351326',
#   '2022-07-13 15:04:30.151315',
#   '2022-07-13 15:04:42.751303',
#   '2022-07-13 15:04:54.551292',
#   '2022-07-13 15:05:04.451283',
#   '2022-07-14 10:39:23.020857',
#   '2022-07-14 10:39:33.020847',
#   '2022-07-14 10:39:43.020838',
#   '2022-07-14 10:39:51.620830',
#   '2022-07-14 10:40:02.020820',
# ]
# manual_pouring_end_times_s = [
#   '2022-06-07 18:44:52.841188',
#   '2022-06-07 18:45:26.041156',
#   '2022-06-07 18:45:39.141144',
#   '2022-06-07 18:45:50.541133',
#   '2022-06-07 18:46:04.141120',
#   '2022-06-13 22:39:55.617476',
#   '2022-06-13 22:40:05.817466',
#   '2022-06-13 22:40:14.117459',
#   '2022-06-13 22:40:22.417450',
#   '2022-06-13 22:40:32.117441',
#   '2022-06-14 14:04:09.492381',
#   '2022-06-14 14:04:16.992373',
#   '2022-06-14 14:04:26.792364',
#   '2022-06-14 14:04:34.692357',
#   '2022-06-14 14:04:44.492347',
#   '2022-06-14 17:18:54.826946',
#   '2022-06-14 17:19:02.326939',
#   '2022-06-14 17:19:10.826931',
#   '2022-06-14 17:19:18.526924',
#   '2022-06-14 17:19:27.526915',
#   '2022-06-14 21:19:46.160977',
#   '2022-06-14 21:19:54.260969',
#   '2022-06-14 21:20:01.760962',
#   '2022-06-14 21:20:09.660954',
#   '2022-06-14 21:20:16.360948',
#   '2022-07-12 15:27:14.305312',
#   '2022-07-12 15:27:43.805284',
#   '2022-07-12 15:27:54.405273',
#   '2022-07-12 15:28:05.005263',
#   '2022-07-12 15:28:15.305254',
#   '2022-07-13 11:50:18.289297',
#   '2022-07-13 11:50:24.889290',
#   '2022-07-13 11:50:30.189285',
#   '2022-07-13 11:50:36.089280',
#   '2022-07-13 11:50:42.989273',
#   '2022-07-13 15:04:25.751319',
#   '2022-07-13 15:04:38.451307',
#   '2022-07-13 15:04:50.551296',
#   '2022-07-13 15:05:01.251286',
#   '2022-07-13 15:05:12.051275',
#   '2022-07-14 10:39:31.120849',
#   '2022-07-14 10:39:39.820841',
#   '2022-07-14 10:39:49.320832',
#   '2022-07-14 10:39:58.720823',
#   '2022-07-14 10:40:07.320815',
# ]



















