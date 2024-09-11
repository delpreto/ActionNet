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

import copy

from learning_trajectories.helpers.configuration import *
from learning_trajectories.helpers.transformation import *
from learning_trajectories.helpers.timeseries_processing import *
from learning_trajectories.helpers.numpy_scipy_utils import *

# ================================================================
# Parse feature data
# ================================================================

# ================================================================
def parse_feature_data(feature_data):
  # Check if the provided data is already parsed.
  if 'position_m' in feature_data:
    return copy.deepcopy(feature_data)
  # Parse the latest feature data format.
  try:
    return {
      'position_m' : {
        'hand': feature_data['hand_position_m'],
        'elbow': feature_data['elbow_position_m'],
        'shoulder': feature_data['shoulder_position_m'],
      },
      'quaternion_wijk': {
        'hand': feature_data['hand_quaternion_wijk'],
        'elbow': feature_data['elbow_quaternion_wijk'],
        'shoulder': feature_data['shoulder_quaternion_wijk'],
      },
      'joint_angle_rad': {
        'hand': feature_data['wrist_joint_angle_xyz_rad'],
        'elbow': feature_data['elbow_joint_angle_xyz_rad'],
        'shoulder': feature_data['shoulder_joint_angle_xyz_rad'],
      },
      'time_s': feature_data['time_s'],
      'referenceObject_position_m': np.squeeze(feature_data['referenceObject_position_m']),
      # 'hand_to_pitcher_angles_rad': np.squeeze(feature_data['hand_to_pitcher_angles_rad']),
    }
  except:
    pass
  # Parse a model output.
  try:
    return {
      'position_m' : {
        'hand': feature_data['hand_position_m'],
        # 'elbow':    None,
        # 'shoulder': None,
      },
      'quaternion_wijk': {
        'hand': feature_data['hand_quaternion_wijk'],
        # 'elbow':    None,
        # 'shoulder': None,
      },
      # 'joint_angle_rad': {
      #   'hand': feature_data[:, 7:10],
      #   'elbow': feature_data[:, 10:13],
      #   'shoulder': feature_data[:, 13:16],
      # },
      # 'time_s': np.linspace(0, 10, feature_data['hand_position_m'].shape[1]),
      'time_s': feature_data['time_s'],
      'referenceObject_position_m': np.squeeze(feature_data['referenceObject_position_m']),
    }
  except:
    pass
  # Parse a legacy feature matrix of human demonstrations.
  if isinstance(feature_data, np.ndarray) and feature_data.shape[-1] == 31:
    return {
      'position_m' : {
        'hand': feature_data[:, 0:3],
        'elbow': feature_data[:, 3:6],
        'shoulder': feature_data[:, 6:9],
      },
      'quaternion_wijk': {
        'hand': feature_data[:, 9:13],
        'elbow': feature_data[:, 13:17],
        'shoulder': feature_data[:, 17:21],
      },
      'joint_angle_rad': {
        'hand': feature_data[:, 21:24],
        'elbow': feature_data[:, 24:27],
        'shoulder': feature_data[:, 27:30],
      },
      'time_s': feature_data[:, 30]
    }
  
  # Parse a legacy model output.
  if isinstance(feature_data, np.ndarray) and feature_data.shape[-1] == 16:
    return {
      'position_m' : {
        'hand': feature_data[:, 0:3],
        # 'elbow':    None,
        # 'shoulder': None,
      },
      'quaternion_wijk': {
        'hand': feature_data[:, 3:7],
        # 'elbow':    None,
        # 'shoulder': None,
      },
      'joint_angle_rad': {
        'hand': feature_data[:, 7:10],
        'elbow': feature_data[:, 10:13],
        'shoulder': feature_data[:, 13:16],
      },
      'time_s': np.linspace(0, 10, feature_data.shape[0]),
    }

def bodyPath_data_to_parsed_feature_data(bodyPath_data, time_s=None,
                                         referenceObject_position_m=None,
                                         hand_to_pitcher_angles_rad=None):
  feature_data = {
    'position_m' : {
      'hand': bodyPath_data['position_m']['RightHand'],
      'elbow': bodyPath_data['position_m']['RightForeArm'],
      'shoulder': bodyPath_data['position_m']['RightUpperArm'],
    },
    'quaternion_wijk': {
      'hand': bodyPath_data['quaternion_wijk']['RightHand'],
      'elbow': bodyPath_data['quaternion_wijk']['RightForeArm'],
      'shoulder': bodyPath_data['quaternion_wijk']['RightUpperArm'],
    },
    'joint_angle_rad': {
      'hand': bodyPath_data['joint_angle_eulerZXY_xyz_rad']['RightWrist'],
      'elbow': bodyPath_data['joint_angle_eulerZXY_xyz_rad']['RightElbow'],
      'shoulder': bodyPath_data['joint_angle_eulerXZY_xyz_rad']['RightShoulder'],
    }
  }
  if time_s is not None:
    feature_data['time_s'] = time_s
  if referenceObject_position_m is not None:
    feature_data['referenceObject_position_m'] = referenceObject_position_m
  if hand_to_pitcher_angles_rad is not None:
    feature_data['hand_to_pitcher_angles_rad'] = hand_to_pitcher_angles_rad
  return feature_data
  
# ================================================================
# Get feature data for specified time indexes.
def get_feature_data_for_trial(parsed_feature_data, trial_indexes):
  def extract_trials(x):
    if isinstance(x, np.ndarray):
      return x[trial_indexes]
    if isinstance(x, dict):
      for (key, data) in x.items():
        x[key] = extract_trials(x[key])
      return x
  return extract_trials(copy.deepcopy(parsed_feature_data))
  
# ================================================================
# Get the 3D positions of body segments.
# Will return a dict with 'hand', 'elbow', and 'shoulder' each of which has xyz positions.
def get_body_position_m(feature_data):
  parsed_data = parse_feature_data(feature_data)
  return parsed_data['position_m']

# ================================================================
# Get the 3D rotation angles of each joint.
# Will return a dict with 'hand', 'elbow', and 'shoulder' each of which has xyz rotations.
def get_body_joint_angles_rad(feature_data):
  parsed_data = parse_feature_data(feature_data)
  return parsed_data['joint_angle_rad']

# ================================================================
# Get the 3D speeds of body segments.
# Will return a dict with 'hand', 'elbow', and 'shoulder' each of which has a speed vector.
def get_body_speed_m_s(feature_data):
  positions_m = get_body_position_m(feature_data)
  feature_data = parse_feature_data(feature_data)
  times_s = feature_data['time_s']
  dt = np.reshape(np.diff(times_s, axis=0), (-1, 1))
  speeds_m_s = dict.fromkeys(positions_m.keys())
  for (body_key, position_m) in positions_m.items():
    if position_m is None:
      continue
    # Infer the speed.
    dxdydz = np.diff(position_m, axis=0)
    speed_m_s = np.hstack([np.squeeze([0]), np.linalg.norm(dxdydz, axis=1)/np.squeeze(dt)])
    speeds_m_s[body_key] = speed_m_s
  return speeds_m_s
  
# ================================================================
# Get the 3D accelerations of body segments.
# Will return a dict with 'hand', 'elbow', and 'shoulder' each of which has an acceleration vector.
def get_body_acceleration_m_s_s(feature_data):
  speeds_m_s = get_body_speed_m_s(feature_data)
  feature_data = parse_feature_data(feature_data)
  times_s = feature_data['time_s']
  dt = np.reshape(np.diff(times_s, axis=0), (-1, 1))
  accelerations_m_s_s = dict.fromkeys(speeds_m_s.keys())
  for (body_key, speed_m_s) in speeds_m_s.items():
    if speed_m_s is None:
      continue
    # Infer the acceleration.
    dv = np.diff(speed_m_s, axis=0)
    dt = np.reshape(np.diff(times_s, axis=0), (-1, 1))
    acceleration_m_s_s = np.hstack([np.squeeze([0]), dv/np.squeeze(dt)])
    accelerations_m_s_s[body_key] = acceleration_m_s_s
  return accelerations_m_s_s

# ================================================================
# Get the 3D jerks of body segments.
# Will return a dict with 'hand', 'elbow', and 'shoulder' each of which has a jerk vector.
def get_body_jerk_m_s_s_s(feature_data):
  accelerations_m_s_s = get_body_acceleration_m_s_s(feature_data)
  feature_data = parse_feature_data(feature_data)
  times_s = feature_data['time_s']
  dt = np.reshape(np.diff(times_s, axis=0), (-1, 1))
  jerks_m_s_s_s = dict.fromkeys(accelerations_m_s_s.keys())
  for (body_key, acceleration_m_s_s) in accelerations_m_s_s.items():
    if acceleration_m_s_s is None:
      continue
    # Infer the jerk.
    da = np.diff(acceleration_m_s_s, axis=0)
    dt = np.reshape(np.diff(times_s, axis=0), (-1, 1))
    jerk_m_s_s_s = np.hstack([np.squeeze([0]), da/np.squeeze(dt)])
    jerks_m_s_s_s[body_key] = jerk_m_s_s_s
  return jerks_m_s_s_s

# ================================================================
# Get the timestamps of the demonstration.
# Will return a list of timestamps.
def get_trajectory_time_s(feature_data):
  parsed_data = parse_feature_data(feature_data)
  return parsed_data['time_s']

# ================================================================
# Get the demonstration duration.
def get_trajectory_duration_s(feature_data):
  time_s = get_trajectory_time_s(feature_data)
  return time_s[-1]

# ================================================================
# Get the demonstration sampling rate.
def get_trajectory_Fs_hz(feature_data):
  parsed_data = parse_feature_data(feature_data)
  time_s = parsed_data['time_s']
  Fs_hz = (len(time_s) - 1)/(time_s[-1] - time_s[0])
  return Fs_hz


##################################################################
# Infer metrics or other quantities about trajectories.
##################################################################

# ================================================================
# Get the body position and orientation during an inferred pouring window.
# Will infer the pouring window by finding a region that is the most stationary.
def infer_pour_pose(feature_data):
  # Parse the feature data if needed.
  parsed_data = parse_feature_data(feature_data)
  # Infer the stationary segment and pose.
  (stationary_time_s, stationary_pose) = infer_stationary_poses(
    parsed_data['time_s'], parsed_data,
    use_variance=True, hand_segment_key='hand')
  # Return the pose.
  return stationary_pose

# ================================================================
# Get the tilt angle of the spout at a specific time index or during the entire trial.
def infer_spout_tilting(feature_data, time_index=None, hand_to_pitcher_rotation_toUse=None):
  # Parse the feature data if needed.
  parsed_data = parse_feature_data(feature_data)
  if hand_to_pitcher_rotation_toUse is None and 'hand_to_pitcher_angles_rad' in parsed_data:
    hand_to_pitcher_angles_rad = np.squeeze(parsed_data['hand_to_pitcher_angles_rad'])
    hand_to_pitcher_rotation_toUse = Rotation.from_rotvec(hand_to_pitcher_angles_rad)
  if hand_to_pitcher_rotation_toUse is None:
    hand_to_pitcher_rotation_toUse = hand_to_pitcher_rotation
    
  # Get tilt for all time if desired
  if time_index is None:
    spout_tilts = []
    for time_index in range(parsed_data['time_s'].shape[0]):
      spout_tilts.append(infer_spout_tilting(feature_data, time_index=time_index))
    return np.array(spout_tilts)
  
  # Rotate a box for the pitcher according to the hand quaternion.
  hand_center_cm = 100*parsed_data['position_m']['hand'][time_index, :]
  hand_quaternion_localToGlobal_wijk = parsed_data['quaternion_wijk']['hand'][time_index, :]
  hand_rotation = Rotation.from_quat(hand_quaternion_localToGlobal_wijk[[1,2,3,0]])
  pitcher_rotation = hand_rotation * hand_to_pitcher_rotation_toUse
  pitcher_quaternion_localToGlobal_ijkw = pitcher_rotation.as_quat()
  (corners, faces) = rotate_3d_box(pitcher_quaternion_localToGlobal_ijkw[[3,0,1,2]], hand_to_pitcher_offset_cm, pitcher_box_dimensions_cm)
  # Get a line segment along the long axis of the top of the pitcher, and move it to the origin.
  pitcher_topFace_line = corners[4,:] - corners[6,:]
  # Compute the angle between the pitcher top and the xy plane.
  angle_toZ_rad = np.arccos(np.dot(pitcher_topFace_line, [0, 0, 1]) / (np.linalg.norm(pitcher_topFace_line)*1))
  angle_toXY_rad = (np.pi/2) - angle_toZ_rad
  return angle_toXY_rad

# ================================================================
# Get the 3D spout position at a specific time index or during the entire trial.
def infer_spout_position_m(feature_data, time_index=None, hand_to_pitcher_rotation_toUse=None):
  # Parse the feature data if needed.
  feature_data = parse_feature_data(feature_data)
  if hand_to_pitcher_rotation_toUse is None and 'hand_to_pitcher_angles_rad' in feature_data:
    hand_to_pitcher_angles_rad = np.squeeze(feature_data['hand_to_pitcher_angles_rad'])
    hand_to_pitcher_rotation_toUse = Rotation.from_rotvec(hand_to_pitcher_angles_rad)
  if hand_to_pitcher_rotation_toUse is None:
    hand_to_pitcher_rotation_toUse = hand_to_pitcher_rotation
  
  # Get position for all time if desired
  if time_index is None:
    spout_position_m = []
    for time_index in range(feature_data['time_s'].shape[0]):
      spout_position_m.append(infer_spout_position_m(feature_data, time_index=time_index))
    return np.array(spout_position_m)
  
  # Rotate a box for the pitcher according to the hand quaternion.
  hand_center_cm = 100*feature_data['position_m']['hand'][time_index, :]
  hand_quaternion_localToGlobal_wijk = feature_data['quaternion_wijk']['hand'][time_index, :]
  hand_rotation = Rotation.from_quat(hand_quaternion_localToGlobal_wijk[[1,2,3,0]])
  pitcher_rotation = hand_rotation * hand_to_pitcher_rotation_toUse
  pitcher_quaternion_localToGlobal_ijkw = pitcher_rotation.as_quat()
  (corners, faces) = rotate_3d_box(pitcher_quaternion_localToGlobal_ijkw[[3,0,1,2]],
                                   hand_to_pitcher_offset_cm, pitcher_box_dimensions_cm)
  corners = corners + hand_center_cm
  corners = corners/100
  faces = faces/100
  
  # Average two points at the front of the pitcher to get the spout position.
  return np.mean(corners[[4,5],:], axis=0)

# ================================================================
# Get the spout speed at a specific time index or during the entire trial.
def infer_spout_speed_m_s(feature_data, time_index=None):
  # Get the spout position for all time indexes.
  spout_position_m = infer_spout_position_m(feature_data, time_index=None)
  times_s = get_trajectory_time_s(feature_data)
  # Infer the speed.
  dxdydz = np.diff(spout_position_m, axis=0)
  dt_s = np.reshape(np.diff(times_s, axis=0), (-1, 1))
  spout_speed_m_s = np.hstack([np.squeeze([0]), np.linalg.norm(dxdydz, axis=1)/np.squeeze(dt_s)])
  if time_index is None:
    return spout_speed_m_s
  else:
    return spout_speed_m_s[time_index]

# ================================================================
# Get the spout acceleration at a specific time index or during the entire trial.
def infer_spout_acceleration_m_s_s(feature_data, time_index=None):
  # Get the spout speed for all time indexes.
  spout_speed_m_s = infer_spout_speed_m_s(feature_data, time_index=None)
  times_s = get_trajectory_time_s(feature_data)
  # Infer the acceleration.
  dv = np.diff(spout_speed_m_s, axis=0)
  dt_s = np.reshape(np.diff(times_s, axis=0), (-1, 1))
  spout_acceleration_m_s_s = np.hstack([np.squeeze([0]), dv/np.squeeze(dt_s)])
  if time_index is None:
    return spout_acceleration_m_s_s
  else:
    return spout_acceleration_m_s_s[time_index]

# ================================================================
# Get the spout jerk at a specific time index or during the entire trial.
def infer_spout_jerk_m_s_s_s(feature_data, time_index=None):
  # Get the spout speed.
  spout_acceleration_m_s_s = infer_spout_acceleration_m_s_s(feature_data, time_index=None)
  times_s = get_trajectory_time_s(feature_data)
  # Infer the jerk.
  da = np.diff(spout_acceleration_m_s_s, axis=0)
  dt_s = np.reshape(np.diff(times_s, axis=0), (-1, 1))
  spout_jerk_m_s_s_s = np.hstack([np.squeeze([0]), da/np.squeeze(dt_s)])
  if time_index is None:
    return spout_jerk_m_s_s_s
  else:
    return spout_jerk_m_s_s_s[time_index]

# ================================================================
# Get the spout yaw vector at a specific time index or during the entire trial.
def infer_spout_yawvector(feature_data, time_index=None, hand_to_pitcher_rotation_toUse=None):
  # Parse the feature data if needed.
  feature_data = parse_feature_data(feature_data)
  if hand_to_pitcher_rotation_toUse is None and 'hand_to_pitcher_angles_rad' in feature_data:
    hand_to_pitcher_angles_rad = np.squeeze(feature_data['hand_to_pitcher_angles_rad'])
    hand_to_pitcher_rotation_toUse = Rotation.from_rotvec(hand_to_pitcher_angles_rad)
  if hand_to_pitcher_rotation_toUse is None:
    hand_to_pitcher_rotation_toUse = hand_to_pitcher_rotation
  
  # Get vector for all time indexes if desired.
  if time_index is None:
    spout_yawvectors = []
    for time_index in range(feature_data['time_s'].shape[0]):
      spout_yawvectors.append(infer_spout_yawvector(feature_data, time_index=time_index))
    return np.array(spout_yawvectors)
  
  # Rotate a box for the pitcher according to the hand quaternion.
  hand_center_cm = 100*feature_data['position_m']['hand'][time_index, :]
  hand_quaternion_localToGlobal_wijk = feature_data['quaternion_wijk']['hand'][time_index, :]
  hand_rotation = Rotation.from_quat(hand_quaternion_localToGlobal_wijk[[1,2,3,0]])
  pitcher_rotation = hand_rotation * hand_to_pitcher_rotation_toUse
  pitcher_quaternion_localToGlobal_ijkw = pitcher_rotation.as_quat()
  (corners, faces) = rotate_3d_box(pitcher_quaternion_localToGlobal_ijkw[[3,0,1,2]],
                                   hand_to_pitcher_offset_cm, pitcher_box_dimensions_cm)
  corners = corners + hand_center_cm
  corners = corners/100
  faces = faces/100
  
  # Get a line segment along the long axis of the top of the pitcher.
  handside_point = corners[6,:]
  spoutside_point = corners[4,:]
  # Project it, move it to the origin, and normalize.
  handside_point[2] = 0
  spoutside_point[2] = 0
  yawvector = spoutside_point - handside_point
  return yawvector/np.linalg.norm(yawvector)

# ================================================================
# Get the pouring relative offset at a specific time index or during the entire trial.
# Will rotate such that positive y is the direction of water pouring.
def infer_spout_relativeOffset_cm(feature_data, referenceObject_position_m=None,
                                  time_index=None, hand_to_pitcher_rotation_toUse=None):
  # Parse the feature data if needed.
  feature_data = parse_feature_data(feature_data)
  if referenceObject_position_m is None and 'referenceObject_position_m' in feature_data:
    referenceObject_position_m = np.squeeze(feature_data['referenceObject_position_m'])
  if hand_to_pitcher_rotation_toUse is None and 'hand_to_pitcher_angles_rad' in feature_data:
    hand_to_pitcher_angles_rad = np.squeeze(feature_data['hand_to_pitcher_angles_rad'])
    hand_to_pitcher_rotation_toUse = Rotation.from_rotvec(hand_to_pitcher_angles_rad)
  if hand_to_pitcher_rotation_toUse is None:
    hand_to_pitcher_rotation_toUse = hand_to_pitcher_rotation
    
  # Get vector for all time indexes if desired.
  if time_index is None:
    spout_relativeOffsets_cm = []
    for time_index in range(feature_data['time_s'].shape[0]):
      spout_relativeOffsets_cm.append(infer_spout_relativeOffset_cm(
        feature_data, referenceObject_position_m, time_index=time_index,
        hand_to_pitcher_rotation_toUse=hand_to_pitcher_rotation_toUse))
    return np.array(spout_relativeOffsets_cm)
  
  # Infer the spout position and yaw.
  spout_position_cm = 100*infer_spout_position_m(
      feature_data=feature_data,
      time_index=time_index,
      hand_to_pitcher_rotation_toUse=hand_to_pitcher_rotation_toUse)
  spout_yawvector = infer_spout_yawvector(feature_data=feature_data,
                                          time_index=time_index,
                                          hand_to_pitcher_rotation_toUse=hand_to_pitcher_rotation_toUse)
  # Project everything to the XY plane.
  spout_position_cm = spout_position_cm[0:2]
  spout_yawvector = spout_yawvector[0:2]
  referenceObject_position_cm = 100*referenceObject_position_m[0:2]
  # Use the spout projection as the origin.
  referenceObject_position_cm = referenceObject_position_cm - spout_position_cm
  # Rotate so the yaw vector is the new y-axis.
  yaw_rotation_matrix = rotation_matrix_from_vectors([spout_yawvector[0], spout_yawvector[1], 0],
                                                     [0, 1, 0])
  referenceObject_position_cm = yaw_rotation_matrix.dot(np.array([referenceObject_position_cm[0], referenceObject_position_cm[1], 0]))
  referenceObject_position_cm = referenceObject_position_cm[0:2]
  # Move the origin to the reference object.
  spout_relativeOffset_cm = -referenceObject_position_cm
  # Return the result.
  return spout_relativeOffset_cm


  

















