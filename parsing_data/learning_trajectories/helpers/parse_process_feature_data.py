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

from learning_trajectories.helpers.configuration import *
from learning_trajectories.helpers.transformation import *
from learning_trajectories.helpers.timeseries_processing import *

# ================================================================
# Parse feature matrices
# ================================================================

# ================================================================
# Feature_matrices should be Tx31, where
#   T is the number of timesteps in each trial
#   31 is the concatenation of:
#     xyz position for hand > elbow > shoulder
#     wijk quaternion for hand > lower arm > upper arm
#     xzy joint angle for wrist > elbow > shoulder
#     time_s
def parse_feature_matrix(feature_matrix):
  if feature_matrix.shape[-1] == 31: # human demonstrations
    return {
      'position_m' : {
        'hand':     feature_matrix[:, 0:3],
        'elbow':    feature_matrix[:, 3:6],
        'shoulder': feature_matrix[:, 6:9],
      },
      'quaternion_wijk': {
        'hand':     feature_matrix[:, 9:13],
        'elbow':    feature_matrix[:, 13:17],
        'shoulder': feature_matrix[:, 17:21],
      },
      'joint_angle_rad': {
        'hand':     feature_matrix[:, 21:24],
        'elbow':    feature_matrix[:, 24:27],
        'shoulder': feature_matrix[:, 27:30],
      },
      'time_s': feature_matrix[:, 30]
    }
  elif feature_matrix.shape[-1] == 16: # model outputs
    return {
      'position_m' : {
        'hand':     feature_matrix[:, 0:3],
        # 'elbow':    None,
        # 'shoulder': None,
      },
      'quaternion_wijk': {
        'hand':     feature_matrix[:, 3:7],
        # 'elbow':    None,
        # 'shoulder': None,
      },
      'joint_angle_rad': {
        'hand':     feature_matrix[:, 7:10],
        'elbow':    feature_matrix[:, 10:13],
        'shoulder': feature_matrix[:, 13:16],
      },
      'time_s': np.linspace(0, 10, feature_matrix.shape[0]),
    }
    

# ================================================================
# Get the 3D positions of body segments.
# Will return a dict with 'hand', 'elbow', and 'shoulder' each of which has xyz positions.
def get_body_position_m(feature_matrix):
  parsed_data = parse_feature_matrix(feature_matrix)
  return parsed_data['position_m']

# ================================================================
# Get the 3D rotation angles of each joint.
# Will return a dict with 'hand', 'elbow', and 'shoulder' each of which has xyz rotations.
def get_body_joint_angles_rad(feature_matrix):
  parsed_data = parse_feature_matrix(feature_matrix)
  return parsed_data['joint_angle_rad']

# ================================================================
# Get the 3D speeds of body segments.
# Will return a dict with 'hand', 'elbow', and 'shoulder' each of which has a speed vector.
def get_body_speed_m_s(feature_matrix):
  positions_m = get_body_position_m(feature_matrix)
  times_s = feature_matrix[:,-1]
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
def get_body_acceleration_m_s_s(feature_matrix):
  speeds_m_s = get_body_speed_m_s(feature_matrix)
  times_s = feature_matrix[:,-1]
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
def get_body_jerk_m_s_s_s(feature_matrix):
  accelerations_m_s_s = get_body_acceleration_m_s_s(feature_matrix)
  times_s = feature_matrix[:,-1]
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
def get_trajectory_time_s(feature_matrix):
  parsed_data = parse_feature_matrix(feature_matrix)
  return parsed_data['time_s']

# ================================================================
# Get the demonstration duration.
def get_trajectory_duration_s(feature_matrix):
  time_s = get_trajectory_time_s(feature_matrix)
  return time_s[-1]

# ================================================================
# Get the demonstration sampling rate.
def get_trajectory_Fs_hz(feature_matrix):
  parsed_data = parse_feature_matrix(feature_matrix)
  time_s = parsed_data['time_s']
  Fs_hz = (len(time_s) - 1)/(time_s[-1] - time_s[0])
  return Fs_hz


##################################################################
# Infer metrics or other quantities about trajectories.
##################################################################

# ================================================================
# Get the body position and orientation during an inferred pouring window.
# Will infer the pouring window by finding a region that is the most stationary.
def infer_pour_pose(feature_matrix):
  # Parse the feature matrix.
  parsed_data = parse_feature_matrix(feature_matrix)
  # Infer the stationary segment and pose.
  (stationary_time_s, stationary_pose) = infer_stationary_poses(
    parsed_data['time_s'], parsed_data,
    use_variance=True, hand_segment_key='hand')
  # Return the pose.
  return stationary_pose

# ================================================================
# Get the tilt angle of the spout at a specific time index or during the entire trial.
def infer_spout_tilting(feature_matrix, time_index=None):
  # Get tilt for all time if desired
  if time_index is None:
    spout_tilts = []
    for time_index in range(feature_matrix.shape[0]):
      spout_tilts.append(infer_spout_tilting(feature_matrix, time_index=time_index))
    return np.array(spout_tilts)
  
  # Parse the feature matrix.
  parsed_data = parse_feature_matrix(feature_matrix)
  # Rotate a box for the pitcher according to the hand quaternion.
  hand_center_cm = 100*parsed_data['position_m']['hand'][time_index, :]
  hand_quaternion_localToGlobal_wijk = parsed_data['quaternion_wijk']['hand'][time_index, :]
  hand_rotation = Rotation.from_quat(hand_quaternion_localToGlobal_wijk[[1,2,3,0]])
  pitcher_rotation = hand_rotation * hand_to_pitcher_rotation
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
def infer_spout_position_m(feature_matrix=None, parsed_data=None, time_index=None):
  # Parse the feature matrix if needed.
  if feature_matrix is not None:
    parsed_data = parse_feature_matrix(feature_matrix)
  
  # Get position for all time if desired
  if time_index is None:
    spout_position_m = []
    for time_index in range(parsed_data['time_s'].shape[0]):
      spout_position_m.append(infer_spout_position_m(parsed_data=parsed_data, time_index=time_index))
    return np.array(spout_position_m)
  
  # Rotate a box for the pitcher according to the hand quaternion.
  hand_center_cm = 100*parsed_data['position_m']['hand'][time_index, :]
  hand_quaternion_localToGlobal_wijk = parsed_data['quaternion_wijk']['hand'][time_index, :]
  hand_rotation = Rotation.from_quat(hand_quaternion_localToGlobal_wijk[[1,2,3,0]])
  pitcher_rotation = hand_rotation * hand_to_pitcher_rotation
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
def infer_spout_speed_m_s(feature_matrix, time_index=None):
  # Get the spout position for all time indexes.
  spout_position_m = infer_spout_position_m(feature_matrix, time_index=None)
  times_s = get_trajectory_time_s(feature_matrix)
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
def infer_spout_acceleration_m_s_s(feature_matrix, time_index=None):
  # Get the spout speed for all time indexes.
  spout_speed_m_s = infer_spout_speed_m_s(feature_matrix, time_index=None)
  times_s = get_trajectory_time_s(feature_matrix)
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
def infer_spout_jerk_m_s_s_s(feature_matrix, time_index=None):
  # Get the spout speed.
  spout_acceleration_m_s_s = infer_spout_acceleration_m_s_s(feature_matrix, time_index=None)
  times_s = get_trajectory_time_s(feature_matrix)
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
def infer_spout_yawvector(feature_matrix=None, parsed_data=None, time_index=None):
  # Parse the feature matrix if needed.
  if feature_matrix is not None:
    parsed_data = parse_feature_matrix(feature_matrix)
  
  # Get vector for all time indexes if desired.
  if time_index is None:
    spout_yawvectors = []
    for time_index in range(parsed_data['time_s'].shape[0]):
      spout_yawvectors.append(infer_spout_yawvector(parsed_data=parsed_data, time_index=time_index))
    return np.array(spout_yawvectors)
  
  # Rotate a box for the pitcher according to the hand quaternion.
  hand_center_cm = 100*parsed_data['position_m']['hand'][time_index, :]
  hand_quaternion_localToGlobal_wijk = parsed_data['quaternion_wijk']['hand'][time_index, :]
  hand_rotation = Rotation.from_quat(hand_quaternion_localToGlobal_wijk[[1,2,3,0]])
  pitcher_rotation = hand_rotation * hand_to_pitcher_rotation
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



















