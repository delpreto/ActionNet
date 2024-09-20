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

from learning_trajectories.helpers.configuration import *
from learning_trajectories.helpers.printing import *

#===========================================================
# Resample all data to a consistent rate.
def resample_bodyPath_data(time_s_byTrial, bodyPath_data_byTrial):
  if not isinstance(bodyPath_data_byTrial, (list, tuple)):
    bodyPath_data_byTrial = [bodyPath_data_byTrial]
    time_s_byTrial = [time_s_byTrial]
    return_single_trial = True
  else:
    return_single_trial = False
    
  for trial_index in range(len(time_s_byTrial)):
    time_s = time_s_byTrial[trial_index]
    time_s_resampled = np.arange(time_s[0], time_s[-1]+1/resampled_fs_hz/2, 1/resampled_fs_hz)
    for data_type in bodyPath_data_byTrial[trial_index].keys():
      for (body_segment, data) in bodyPath_data_byTrial[trial_index][data_type].items():
        fn_interpolate_data = interpolate.interp1d(
            time_s, # x values
            data,   # y values
            axis=0,        # axis of the data along which to interpolate
            kind='linear', # interpolation method, such as 'linear', 'zero', 'nearest', 'quadratic', 'cubic', etc.
            fill_value='extrapolate' # how to handle x values outside the original range
        )
        data_resampled = fn_interpolate_data(time_s_resampled)
        bodyPath_data_byTrial[trial_index][data_type][body_segment] = data_resampled
    time_s_byTrial[trial_index] = time_s_resampled
  
  # Return the tranformed body path data.
  if return_single_trial:
    return (time_s_byTrial[0], bodyPath_data_byTrial[0])
  else:
    return (time_s_byTrial, bodyPath_data_byTrial)

#===========================================================
# Infer the hand position and orientation at a stationary point (such as when water is being poured)
def infer_stationary_poses(time_s_byTrial, bodyPath_data_byTrial, use_variance, hand_segment_key):
  if not isinstance(bodyPath_data_byTrial, (list, tuple)):
    bodyPath_data_byTrial = [bodyPath_data_byTrial]
    time_s_byTrial = [time_s_byTrial]
    return_single_trial = True
  else:
    return_single_trial = False
  
  stationary_pose_byTrial = []
  stationary_time_s_byTrial = []
  for (trial_index, bodyPath_data) in enumerate(bodyPath_data_byTrial):
    body_position_m = bodyPath_data['position_m']
    body_quaternion_wijk = bodyPath_data['quaternion_wijk']
    time_s = time_s_byTrial[trial_index]
    num_timesteps = time_s.shape[0]
    stationary_position_minIndex = round(stationary_position_min_ratio*num_timesteps)
    stationary_position_maxIndex = round(stationary_position_max_ratio*num_timesteps)
    Fs_hz = (num_timesteps-1)/(time_s[-1] - time_s[0])
    if use_variance:
      if time_s[-1] - time_s[0] < 2*stationary_position_buffer_duration_s:
        # Add a dummy stationary position for now
        # TODO do something better?
        stationary_pose_byTrial.append({
        'position_m':
          dict([(name, 0) for (name, position_m) in body_position_m.items()]),
        'quaternion_wijk':
          dict([(name, 0) for (name, quaternion) in body_position_m.items()]),
        })
        stationary_time_s_byTrial.append(0)
        continue
      # Initialize state.
      min_average_distance_m = None
      min_average_distance_buffer_start_index = None
      min_average_distance_buffer_end_index = None
      body_position_stationary_buffer_m = None
      body_quaternion_wijk_stationary_buffer = None
      # Find the most stationary buffer.
      for buffer_start_index in range(num_timesteps):
        buffer_start_time_s = time_s[buffer_start_index]
        buffer_end_time_s = buffer_start_time_s + stationary_position_buffer_duration_s
        if buffer_end_time_s > time_s[-1]:
          break
        buffer_end_index = np.where(time_s <= buffer_end_time_s)[0][-1]
        if buffer_start_index < stationary_position_minIndex:
          continue
        if buffer_end_index > stationary_position_maxIndex:
          continue
        body_position_buffers_m = dict([(name, position_m[buffer_start_index:buffer_end_index, :]) for (name, position_m) in body_position_m.items()])
        body_quaternion_wijk_buffers = dict([(name, quaternion_wijk[buffer_start_index:buffer_end_index, :]) for (name, quaternion_wijk) in body_quaternion_wijk.items()])
        median_hand_position_m = np.median(body_position_buffers_m[hand_segment_key], axis=0)
        distances_m = np.linalg.norm(body_position_buffers_m[hand_segment_key] - median_hand_position_m, axis=1)
        average_distance_m = np.mean(distances_m, axis=0)
        if min_average_distance_m is None or average_distance_m < min_average_distance_m:
          min_average_distance_m = average_distance_m
          min_average_distance_buffer_start_index = buffer_start_index
          min_average_distance_buffer_end_index = buffer_end_index
          body_position_stationary_buffer_m = body_position_buffers_m
          body_quaternion_wijk_stationary_buffer = body_quaternion_wijk_buffers
      # print(time_s[min_average_distance_buffer_start_index] - min(time_s),
      #       time_s[min_average_distance_buffer_end_index] - min(time_s))
      stationary_pose_byTrial.append({
        'position_m':
          dict([(name, np.median(position_m, axis=0)) for (name, position_m) in body_position_stationary_buffer_m.items()]),
        'quaternion_wijk':
          dict([(name, quaternion_wijk[int(quaternion_wijk.shape[0]/2),:]) for (name, quaternion_wijk) in body_quaternion_wijk_stationary_buffer.items()]),
        'time_index':
          int(np.mean([min_average_distance_buffer_start_index, min_average_distance_buffer_end_index])),
        'start_time_index': min_average_distance_buffer_start_index,
        'end_time_index': min_average_distance_buffer_end_index,
        })
      stationary_time_s_byTrial.append(time_s[int(np.mean([min_average_distance_buffer_start_index,
                                                    min_average_distance_buffer_end_index]))]-time_s[0])
    # use hard-coded time fraction instead of computing the variance
    else:
      stationary_position_time_s = time_s[0] + (time_s[-1] - time_s[0])*stationary_position_hardcoded_time_fraction
      stationary_position_index = time_s.searchsorted(stationary_position_time_s)
      buffer_length = Fs_hz * stationary_position_buffer_duration_s
      buffer_start_index = round(stationary_position_index - buffer_length/2)
      buffer_end_index = round(stationary_position_index + buffer_length/2)
      body_position_stationary_buffer_m = dict([(name, position_cm[buffer_start_index:buffer_end_index, :]) for (name, position_cm) in body_position_m.items()])
      body_quaternion_wijk_stationary_buffer = dict([(name, quaternion[buffer_start_index:buffer_end_index, :]) for (name, quaternion) in body_quaternion_wijk.items()])
      stationary_pose_byTrial.append({
        'position_m':
          dict([(name, np.median(position_m, axis=0)) for (name, position_m) in body_position_stationary_buffer_m.items()]),
        'quaternion_wijk':
          dict([(name, quaternion_wijk[int(quaternion_wijk.shape[0]/2),:]) for (name, quaternion_wijk) in body_quaternion_wijk_stationary_buffer.items()]),
        'time_index':
          int(np.mean([buffer_start_index, buffer_end_index])),
        'start_time_index': buffer_start_index,
        'end_time_index': buffer_end_index,
        })
      stationary_time_s_byTrial.append(time_s[int(np.mean([buffer_start_index,
                                                    buffer_end_index]))]-time_s[0])
  
  if return_single_trial:
    return (stationary_time_s_byTrial[0], stationary_pose_byTrial[0])
  else:
    return (stationary_time_s_byTrial, stationary_pose_byTrial)
