############
#
# Copyright (c) 2026 MIT CSAIL and Joseph DelPreto
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
# Created 2026 for the MIT ActionSense project by Joseph DelPreto [https://josephdelpreto.com].
# [Add additional updates and authors here]
#
############

import h5py
import numpy as np

hdf5_filepath = '2026-07-10_16-38-58_streamLog_tennis_S16.hdf5'

# A helper to extract and format timestamped activity label data.
# This can be used for experimental activities or calibration activities, since both have the same format.
def get_labeled_activity_times(device_name, stream_name, is_calibration=False):
  # Get the timestamped activitiy label data.
  # As described in the HDF5 metadata, each row has entries for ['Activity', 'Start/Stop', 'Valid', 'Notes'].
  activity_datas = h5_file[device_name][stream_name]['data']
  activity_times_s = h5_file[device_name][stream_name]['time_s']
  activity_times_s = np.squeeze(np.array(activity_times_s))  # squeeze (optional) converts from a list of single-element lists to a 1D list
  # Convert to strings for convenience.
  activity_datas = [[x.decode('utf-8') for x in datas] for datas in activity_datas]

  # Combine start/stop rows to single activity entries with start/stop times.
  #   Each row is either the start or stop of the label.
  #   The notes and ratings fields are the same for the start/stop rows of the label, so only need to check one.
  exclude_bad_labels = True # some activities may have been marked as 'Bad' or 'Maybe' by the experimenter; submitted notes with the activity typically give more information
  activities_labels = []
  activities_start_times_s = []
  activities_end_times_s = []
  activities_ratings = []
  activities_notes = []
  activities_calibration_fields = []
  for (row_index, time_s) in enumerate(activity_times_s):
    if not is_calibration:
      label    = activity_datas[row_index][0]
      is_start = activity_datas[row_index][1] == 'Start'
      is_stop  = activity_datas[row_index][1] == 'Stop'
      rating   = activity_datas[row_index][2]
      notes    = activity_datas[row_index][3]
      calibration_fields = []
    else:
      label    = '%s-calibration' % stream_name
      is_start = activity_datas[row_index][0] == 'Start'
      is_stop  = activity_datas[row_index][0] == 'Stop'
      rating   = activity_datas[row_index][1]
      notes    = activity_datas[row_index][2]
      calibration_fields = activity_datas[row_index][3:]
    if exclude_bad_labels and rating in ['Bad', 'Maybe']:
      continue
    # Record the start of a new activity.
    if is_start:
      activities_labels.append(label)
      activities_start_times_s.append(time_s)
      activities_ratings.append(rating)
      activities_notes.append(notes)
      activities_calibration_fields.append(calibration_fields)
    # Record the end of the previous activity.
    if is_stop:
      activities_end_times_s.append(time_s)
  return {
    'labels': activities_labels,
    'start_times_s': activities_start_times_s,
    'end_times_s': activities_end_times_s,
    'ratings': activities_ratings,
    'notes': activities_notes,
    'calibration_fields': activities_calibration_fields,
  }

##########################################################
# Open the file.
##########################################################
h5_file = h5py.File(hdf5_filepath, 'r')

##########################################################
# Get experimental activities.
##########################################################
activities_info = get_labeled_activity_times(
  device_name='experiment-activities',
  stream_name='activities'
)
print()
print('See the following experimental activities and their start/end times since the recording started:')
recording_start_time_s = np.squeeze(h5_file['xsens-segments']['body_orientation_quaternion_wijk']['time_s'])[0]
for activity_index in range(len(activities_info['labels'])):
  print('  Activity index %2d | time bounds [%7.2f, %7.2f] | %s' % (
    activity_index, 
    activities_info['start_times_s'][activity_index] - recording_start_time_s, 
    activities_info['end_times_s'][activity_index] - recording_start_time_s,
    activities_info['labels'][activity_index],
  ))

##########################################################
# Get calibration activities.
##########################################################
calibrations_info = get_labeled_activity_times(
  device_name='experiment-calibration',
  stream_name='body',
  is_calibration=True,
)
print()
print('See the following body calibrations and their start/end times since the recording started:')
recording_start_time_s = np.squeeze(h5_file['xsens-segments']['body_orientation_quaternion_wijk']['time_s'])[0]
for activity_index in range(len(calibrations_info['labels'])):
  print('  Activity index %2d | time bounds [%7.2f, %7.2f] | %s | %s' % (
    activity_index, 
    calibrations_info['start_times_s'][activity_index] - recording_start_time_s, 
    calibrations_info['end_times_s'][activity_index] - recording_start_time_s,
    calibrations_info['labels'][activity_index],
    ' | '.join(calibrations_info['calibration_fields'][activity_index]),
  ))

##########################################################
# Get Xsens data during calibration activities.
##########################################################
print()
print('Loading all Xsens data')

# Get Xsens data for all times.
xsens_times_s = np.squeeze(h5_file['xsens-joints']['body_joint_angles_eulerXZY_xyz_rad']['time_s'])
body_joint_angles_eulerXZY_xyz_rad = np.squeeze(h5_file['xsens-joints']['body_joint_angles_eulerXZY_xyz_rad']['data'])
body_joint_angles_eulerZXY_xyz_rad = np.squeeze(h5_file['xsens-joints']['body_joint_angles_eulerZXY_xyz_rad']['data'])
body_acceleration_xyz_m_ss = np.squeeze(h5_file['xsens-segments']['body_acceleration_xyz_m_ss']['data'])
body_angular_acceleration_xyz_rad_ss =  np.squeeze(h5_file['xsens-segments']['body_angular_acceleration_xyz_rad_ss']['data'])
body_angular_velocity_xyz_rad_s = np.squeeze(h5_file['xsens-segments']['body_angular_velocity_xyz_rad_s']['data'])
body_orientation_eulerZXY_xyz_rad = np.squeeze(h5_file['xsens-segments']['body_orientation_eulerZXY_xyz_rad']['data'])
body_orientation_quaternion_wijk = np.squeeze(h5_file['xsens-segments']['body_orientation_quaternion_wijk']['data'])
body_position_xyz_m = np.squeeze(h5_file['xsens-segments']['body_position_xyz_m']['data'])
body_velocity_xyz_m_s = np.squeeze(h5_file['xsens-segments']['body_velocity_xyz_m_s']['data'])

# Demonstrate that all streams have the same timestamps.
assert np.array_equal(np.squeeze(h5_file['xsens-joints']['body_joint_angles_eulerXZY_xyz_rad']['time_s']), xsens_times_s)
assert np.array_equal(np.squeeze(h5_file['xsens-joints']['body_joint_angles_eulerZXY_xyz_rad']['time_s']), xsens_times_s)
assert np.array_equal(np.squeeze(h5_file['xsens-segments']['body_angular_acceleration_xyz_rad_ss']['time_s']), xsens_times_s)
assert np.array_equal(np.squeeze(h5_file['xsens-segments']['body_angular_velocity_xyz_rad_s']['time_s']), xsens_times_s)
assert np.array_equal(np.squeeze(h5_file['xsens-segments']['body_orientation_eulerZXY_xyz_rad']['time_s']), xsens_times_s)
assert np.array_equal(np.squeeze(h5_file['xsens-segments']['body_orientation_quaternion_wijk']['time_s']), xsens_times_s)
assert np.array_equal(np.squeeze(h5_file['xsens-segments']['body_position_xyz_m']['time_s']), xsens_times_s)
assert np.array_equal(np.squeeze(h5_file['xsens-segments']['body_velocity_xyz_m_s']['time_s']), xsens_times_s)

# Extract data from each calibration pose period.
print('Getting Xsens data during N and T poses')
for activity_index in range(len(calibrations_info['labels'])):
  is_n_pose = calibrations_info['calibration_fields'][activity_index][-1] == 'N-Pose'
  is_t_pose = calibrations_info['calibration_fields'][activity_index][-1] == 'T-Pose'
  if not is_n_pose and not is_t_pose:
    continue
  start_time_s = calibrations_info['start_times_s'][activity_index]
  end_time_s = calibrations_info['end_times_s'][activity_index]
  start_index = np.searchsorted(a=xsens_times_s, v=start_time_s)
  end_index = np.searchsorted(a=xsens_times_s, v=end_time_s)
  calibration_body_joint_angles_eulerXZY_xyz_rad = body_joint_angles_eulerXZY_xyz_rad[start_index:end_index+1, ...]
  calibration_body_joint_angles_eulerZXY_xyz_rad = body_joint_angles_eulerZXY_xyz_rad[start_index:end_index+1, ...]
  calibration_body_acceleration_xyz_m_ss = body_acceleration_xyz_m_ss[start_index:end_index+1, ...]
  calibration_body_angular_acceleration_xyz_rad_ss = body_angular_acceleration_xyz_rad_ss[start_index:end_index+1, ...]
  calibration_body_angular_velocity_xyz_rad_s = body_angular_velocity_xyz_rad_s[start_index:end_index+1, ...]
  calibration_body_orientation_eulerZXY_xyz_rad = body_orientation_eulerZXY_xyz_rad[start_index:end_index+1, ...]
  calibration_body_orientation_quaternion_wijk = body_orientation_quaternion_wijk[start_index:end_index+1, ...]
  calibration_body_position_xyz_m = body_position_xyz_m[start_index:end_index+1, ...]
  calibration_body_velocity_xyz_m_s = body_velocity_xyz_m_s[start_index:end_index+1, ...]

# Load data of a synthetic T-Pose that the Xsens computes.
# Load T-Pose data (standing with legs neutral, looking straight ahead, arms outstetched horizontally at the sides to form a T).
# This is used by some programs for calibration and whatnot.
xsens_synthetic_tpose = {
   'body_orientation_TposeISB_quaternion_wijk': np.array(h5_file['xsens-segments-tpose']['body_orientation_TposeISB_quaternion_wijk']),
   'body_orientation_Tpose_quaternion_wijk': np.array(h5_file['xsens-segments-tpose']['body_orientation_Tpose_quaternion_wijk']),
   'body_orientation_identity_quaternion_wijk': np.array(h5_file['xsens-segments-tpose']['body_orientation_identity_quaternion_wijk']),
   'body_position_TposeISB_xyz_m': np.array(h5_file['xsens-segments-tpose']['body_position_TposeISB_xyz_m']),
   'body_position_Tpose_xyz_m': np.array(h5_file['xsens-segments-tpose']['body_position_Tpose_xyz_m']),
   'body_position_identity_xyz_m': np.array(h5_file['xsens-segments-tpose']['body_position_identity_xyz_m']),
}

##########################################################
# Clean up.
##########################################################
h5_file.close()
print()
print('Done!')
print()





