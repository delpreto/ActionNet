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
from scipy import interpolate # for resampling
from scipy.signal import butter, lfilter # for filtering
import pandas as pd
from matplotlib import pyplot as plt
import pickle
from collections import OrderedDict
import os, glob
script_dir = os.path.dirname(os.path.realpath(__file__))

from helpers import *
from utils.print_utils import *
from utils.dict_utils import *
from utils.time_utils import *

#######################################
############ CONFIGURATION ############
#######################################

# Define where outputs will be saved.
output_dir = os.path.join(script_dir, '..', '..', '..', 'data_processed')
output_filepath = os.path.join(output_dir, 'data_processed_allStreams_10s_10hz_5subj_ex20-20_allActs.hdf5')
# output_filepath = None

# Define the modalities to use.
# Each entry is (device_name, stream_name, extraction_function)
#  where extraction_function can select a subset of the stream columns.
device_streams_for_features = [
  ('eye-tracking-gaze', 'position', lambda data: data),
  ('myo-left', 'emg', lambda data: data),
  ('myo-right', 'emg', lambda data: data),
  ('tactile-glove-left', 'tactile_data', lambda data: data),
  ('tactile-glove-right', 'tactile_data', lambda data: data),
  ('xsens-joints', 'rotation_xzy_deg', lambda data: data[:,0:22,:]), # exclude fingers by using the first 22 joints
]

# Specify the input data.
data_root_dir = os.path.join(script_dir, '..', '..', '..', 'data', 'experiments')
data_folders_bySubject = OrderedDict([
  ('S00', os.path.join(data_root_dir, '2022-06-07_experiment_S00')),
  ('S02', os.path.join(data_root_dir, '2022-06-13_experiment_S02')),
  ('S03', os.path.join(data_root_dir, '2022-06-14_experiment_S03')),
  ('S04', os.path.join(data_root_dir, '2022-06-14_experiment_S04')),
  ('S05', os.path.join(data_root_dir, '2022-06-14_experiment_S05')),
])

# Specify the labels to include.  These should match the labels in the HDF5 files.
baseline_label = 'None'
activities_to_classify = [
  baseline_label,
  'Get/replace items from refrigerator/cabinets/drawers',
  'Peel a cucumber',
  'Clear cutting board',
  'Slice a cucumber',
  'Peel a potato',
  'Slice a potato',
  'Slice bread',
  'Spread almond butter on a bread slice',
  'Spread jelly on a bread slice',
  'Open/close a jar of almond butter',
  'Pour water from a pitcher into a glass',
  'Clean a plate with a sponge',
  'Clean a plate with a towel',
  'Clean a pan with a sponge',
  'Clean a pan with a towel',
  'Get items from cabinets: 3 each large/small plates, bowls, mugs, glasses, sets of utensils',
  'Set table: 3 each large/small plates, bowls, mugs, glasses, sets of utensils',
  'Stack on table: 3 each large/small plates, bowls',
  'Load dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils',
  'Unload dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils',
  ]
baseline_index = activities_to_classify.index(baseline_label)
# Some older experiments may have had different labels.
#  Each entry below maps the new name to a list of possible old names.
activities_renamed = {
  'Open/close a jar of almond butter': ['Open a jar of almond butter'],
  'Get/replace items from refrigerator/cabinets/drawers': ['Get items from refrigerator/cabinets/drawers'],
}

# Define segmentation parameters.
resampled_Fs = 10 # define a resampling rate for all sensors to interpolate
num_segments_per_subject = 20
num_baseline_segments_per_subject = 20 # num_segments_per_subject*(max(1, len(activities_to_classify)-1))
segment_duration_s = 10
segment_length = int(round(resampled_Fs*segment_duration_s))
buffer_startActivity_s = 2
buffer_endActivity_s = 2

# Define filtering parameters.
filter_cutoff_emg_Hz = 5
filter_cutoff_tactile_Hz = 2
filter_cutoff_gaze_Hz = 5
num_tactile_rows_aggregated = 4
num_tactile_cols_aggregated = 4

# Make the output folder if needed.
if output_dir is not None:
  os.makedirs(output_dir, exist_ok=True)
  print('\n')
  print('Saving outputs to')
  print(output_filepath)
  print('\n')
  
################################################
############ INTERPOLATE AND FILTER ############
################################################

# Will filter each column of the data.
def lowpass_filter(data, cutoff, Fs, order=5):
  nyq = 0.5 * Fs
  normal_cutoff = cutoff / nyq
  b, a = butter(order, normal_cutoff, btype='low', analog=False)
  y = lfilter(b, a, data.T).T
  return y

# Load the original data.
data_bySubject = {}
for (subject_id, data_folder) in data_folders_bySubject.items():
  print()
  print('Loading data for subject %s' % subject_id)
  data_bySubject[subject_id] = []
  hdf_filepaths = glob.glob(os.path.join(data_folder, '**/*.hdf5'), recursive=True)
  for hdf_filepath in hdf_filepaths:
    if 'archived' in hdf_filepath:
      continue
    data_bySubject[subject_id].append({})
    hdf_file = h5py.File(hdf_filepath, 'r')
    print(hdf_filepath)
    # Add the activity label information.
    have_all_streams = True
    try:
      device_name = 'experiment-activities'
      stream_name = 'activities'
      data_bySubject[subject_id][-1].setdefault(device_name, {})
      data_bySubject[subject_id][-1][device_name].setdefault(stream_name, {})
      for key in ['time_s', 'data']:
        data_bySubject[subject_id][-1][device_name][stream_name][key] = hdf_file[device_name][stream_name][key][:]
      num_activity_entries = len(data_bySubject[subject_id][-1][device_name][stream_name]['time_s'])
      if num_activity_entries == 0:
        have_all_streams = False
      elif data_bySubject[subject_id][-1][device_name][stream_name]['time_s'][0] == 0:
        have_all_streams = False
    except KeyError:
      have_all_streams = False
    # Load data for each of the streams that will be used as features.
    for (device_name, stream_name, _) in device_streams_for_features:
      data_bySubject[subject_id][-1].setdefault(device_name, {})
      data_bySubject[subject_id][-1][device_name].setdefault(stream_name, {})
      for key in ['time_s', 'data']:
        try:
          data_bySubject[subject_id][-1][device_name][stream_name][key] = hdf_file[device_name][stream_name][key][:]
        except KeyError:
          have_all_streams = False
    if not have_all_streams:
      data_bySubject[subject_id].pop()
      print('  Ignoring HDF5 file:', hdf_filepath)
    hdf_file.close()
    
# Filter data.
print()
for (subject_id, file_datas) in data_bySubject.items():
  print('Filtering data for subject %s' % subject_id)
  for (data_file_index, file_data) in enumerate(file_datas):
    print(' Data file index', data_file_index)
    # Filter EMG data.
    for myo_key in ['myo-left', 'myo-right']:
      if myo_key in file_data:
        t = file_data[myo_key]['emg']['time_s']
        Fs = (t.size - 1) / (t[-1] - t[0])
        print(' Filtering %s with Fs %0.1f Hz to cutoff %f' % (myo_key, Fs, filter_cutoff_emg_Hz))
        data_stream = file_data[myo_key]['emg']['data'][:, :]
        y = np.abs(data_stream)
        y = lowpass_filter(y, filter_cutoff_emg_Hz, Fs)
        # plt.plot(t-t[0], data_stream[:,0])
        # plt.plot(t-t[0], y[:,0])
        # plt.show()
        file_data[myo_key]['emg']['data'] = y
    # Filter tactile data.
    for tactile_key in ['tactile-glove-left', 'tactile-glove-right']:
      if tactile_key in file_data:
        t = file_data[tactile_key]['tactile_data']['time_s']
        Fs = (t.size - 1) / (t[-1] - t[0])
        print(' Filtering %s with Fs %0.1f Hz to cutoff %f' % (tactile_key, Fs, filter_cutoff_tactile_Hz))
        data_stream = file_data[tactile_key]['tactile_data']['data'][:, :]
        y = data_stream
        y = lowpass_filter(y, filter_cutoff_tactile_Hz, Fs)
        # Eliminate ringing at beginning or end.
        y[0:int(Fs*30),:,:] = np.mean(y, axis=0)
        y[y.shape[0]-int(Fs*30):y.shape[0]+1,:,:] = np.mean(y, axis=0)
        # plt.plot(t-t[0], data_stream[:,0,0])
        # plt.plot(t-t[0], y[:,0,0])
        # plt.xlim(10,t[-1]-t[0])
        # plt.ylim(550,570)
        # plt.show()
        file_data[tactile_key]['tactile_data']['data'] = y
    # Filter eye-gaze data.
    if 'eye-tracking-gaze' in file_data:
      t = file_data['eye-tracking-gaze']['position']['time_s']
      Fs = (t.size - 1) / (t[-1] - t[0])
      
      data_stream = file_data['eye-tracking-gaze']['position']['data'][:, :]
      y = data_stream
      
      # Apply a ZOH to remove clipped values.
      #  The gaze position is already normalized to video coordinates,
      #   so anything outside [0,1] is outside the video.
      print(' Holding clipped values in %s' % ('eye-tracking-gaze'))
      clip_low = 0.05
      clip_high = 0.95
      y = np.clip(y, clip_low, clip_high)
      y[y == clip_low] = np.nan
      y[y == clip_high] = np.nan
      y = pd.DataFrame(y).interpolate(method='zero').to_numpy()
      # Replace any remaining NaNs with a dummy value,
      #  in case the first or last timestep was clipped (interpolate() does not extrapolate).
      y[np.isnan(y)] = 0.5
      # plt.plot(t-t[0], data_stream[:,0], '*-')
      # plt.plot(t-t[0], y[:,0], '*-')
      # plt.ylim(-2,2)
      
      # Filter to smooth.
      print('   Filtering %s with Fs %0.1f Hz to cutoff %f' % ('eye-tracking-gaze', Fs, filter_cutoff_gaze_Hz))
      y = lowpass_filter(y, filter_cutoff_gaze_Hz, Fs)
      # plt.plot(t-t[0], y[:,0])
      # plt.ylim(-2,2)
      # plt.show()
      file_data['eye-tracking-gaze']['position']['data'] = y
    data_bySubject[subject_id][data_file_index] = file_data

# Normalize data.
print()
for (subject_id, file_datas) in data_bySubject.items():
  print('Normalizing data for subject %s' % subject_id)
  for (data_file_index, file_data) in enumerate(file_datas):
    # Normalize EMG data.
    for myo_key in ['myo-left', 'myo-right']:
      if myo_key in file_data:
        data_stream = file_data[myo_key]['emg']['data'][:, :]
        y = data_stream
        print(' Normalizing %s with min/max [%0.1f, %0.1f]' % (myo_key, np.amin(y), np.amax(y)))
        # Normalize them jointly.
        y = y / ((np.amax(y) - np.amin(y))/2)
        # Jointly shift the baseline to -1 instead of 0.
        y = y - np.amin(y) - 1
        file_data[myo_key]['emg']['data'] = y
        print('   Now has range [%0.1f, %0.1f]' % (np.amin(y), np.amax(y)))
        # plt.plot(y.reshape(y.shape[0], -1))
        # plt.show()
    # Normalize tactile data.
    # NOTE: Will clip here, but will normalize later after aggregating.
    for tactile_key in ['tactile-glove-left', 'tactile-glove-right']:
      if tactile_key in file_data:
        data_stream = file_data[tactile_key]['tactile_data']['data'][:, :]
        y = data_stream
        min_val = np.amin(y)
        max_val = np.amax(y)
        print(' Normalizing %s with min/max [%0.1f, %0.1f]' % (tactile_key, min_val, max_val))
        # Clip the values based on the distribution of values across all channels.
        mean_val = np.mean(y)
        std_dev = np.std(y)
        clip_low = mean_val - 2*std_dev # shouldn't be much below the mean, since the mean should be rest basically
        clip_high = mean_val + 3*std_dev
        print('  Clipping to [%0.1f, %0.1f]' % (clip_low, clip_high))
        # heatmap(np.mean(y, axis=0), 'Pre clipping')
        y = np.clip(y, clip_low, clip_high)
        # heatmap(np.mean(y, axis=0), 'Post clipping')
        # input()
        # Store the result.
        file_data[tactile_key]['tactile_data']['data'] = y
    # Normalize Xsens joints.
    if 'xsens-joints' in file_data:
      data_stream = file_data['xsens-joints']['rotation_xzy_deg']['data'][:, :]
      y = data_stream
      min_val = -180
      max_val = 180
      print(' Normalizing %s with forced min/max [%0.1f, %0.1f]' % ('xsens-joints', min_val, max_val))
      # Normalize all at once since using fixed bounds anyway.
      # Preserve relative bends, such as left arm being bent more than the right.
      y = y / ((max_val - min_val)/2)
      # for i in range(20):
      #   plt.plot(y[:,i])
      #   plt.ylim(-1,1)
      #   plt.show()
      file_data['xsens-joints']['rotation_xzy_deg']['data'] = y
      print('   Now has range [%0.1f, %0.1f]' % (np.amin(y), np.amax(y)))
      # plt.plot(y.reshape(y.shape[0], -1))
      # plt.show()
    # Normalize eye-tracking gaze.
    if 'eye-tracking-gaze' in file_data:
      data_stream = file_data['eye-tracking-gaze']['position']['data'][:]
      t = file_data['eye-tracking-gaze']['position']['time_s'][:]
      y = data_stream
      # The gaze position is already normalized to video coordinates,
      #  so anything outside [0,1] is outside the video.
      clip_low = 0.05
      clip_high = 0.95
      print(' Clipping %s to [%0.1f, %0.1f]' % ('eye-tracking-gaze', clip_low, clip_high))
      y = np.clip(y, clip_low, clip_high)
      # Put in range [-1, 1] for extra resolution.
      y = (y - np.mean([clip_low, clip_high]))/((clip_high-clip_low)/2)
      # plt.plot(t-t[0], y)
      # plt.show()
      file_data['eye-tracking-gaze']['position']['data'] = y
      print('   Now has range [%0.1f, %0.1f]' % (np.amin(y), np.amax(y)))
      # plt.plot(y.reshape(y.shape[0], -1))
      # plt.show()
      
    data_bySubject[subject_id][data_file_index] = file_data

# Aggregate data (and normalize if needed).
print()
for (subject_id, file_datas) in data_bySubject.items():
  print('Aggregating data for subject %s' % subject_id)
  for (data_file_index, file_data) in enumerate(file_datas):
    # Aggregate EMG data.
    for myo_key in ['myo-left', 'myo-right']:
      if myo_key in file_data:
        pass
    # Aggregate tactile data.
    for tactile_key in ['tactile-glove-left', 'tactile-glove-right']:
      if tactile_key in file_data:
        data_stream = file_data[tactile_key]['tactile_data']['data'][:, :]
        y = data_stream
        # Make a smaller grid of values, averaging the channels they contain.
        num_rows = y.shape[1]
        num_cols = y.shape[2]
        row_stride = int(num_rows / num_tactile_rows_aggregated + 0.5)
        col_stride = int(num_rows / num_tactile_cols_aggregated + 0.5)
        data_aggregated = np.zeros(shape=(y.shape[0], num_tactile_rows_aggregated, num_tactile_cols_aggregated))
        for r, row_offset in enumerate(range(0, num_rows, row_stride)):
          for c, col_offset in enumerate(range(0, num_cols, col_stride)):
            mask = np.zeros(shape=(num_rows, num_cols))
            mask[row_offset:(row_offset+row_stride), col_offset:(col_offset+col_stride)] = 1
            data_aggregated[:,r,c] = np.sum(y*mask, axis=(1,2))/np.sum(mask)
        y = data_aggregated
        # # De-mean each channel individually.
        # y = y - np.mean(y, axis=0)
        # Normalize all channels jointly.
        y = y / ((np.amax(y) - np.amin(y))/2)
        # Shift baseline to -1 jointly.
        y = y - np.amin(y) - 1
        # Store the result.
        file_data[tactile_key]['tactile_data']['data'] = y
        print('  Tactile now has shape %s and now has range [%0.1f, %0.1f]' % (y.shape, np.amin(y), np.amax(y)))
        # plt.plot(y.reshape(y.shape[0], -1))
        # plt.show()
    # Aggregate Xsens joints.
    if 'xsens-joints' in file_data:
      pass
    # Aggregate eye-tracking gaze.
    if 'eye-tracking-gaze' in file_data:
      pass
    
    data_bySubject[subject_id][data_file_index] = file_data
    
# Resample data.
print()
for (subject_id, file_datas) in data_bySubject.items():
  print('Resampling data for subject %s' % subject_id)
  for (data_file_index, file_data) in enumerate(file_datas):
    for (device_name, stream_name, _) in device_streams_for_features:
      data = np.squeeze(np.array(file_data[device_name][stream_name]['data']))
      time_s = np.squeeze(np.array(file_data[device_name][stream_name]['time_s']))
      target_time_s = np.linspace(time_s[0], time_s[-1],
                                  num=int(round(1+resampled_Fs*(time_s[-1] - time_s[0]))),
                                  endpoint=True)
      fn_interpolate = interpolate.interp1d(
          time_s, # x values
          data,   # y values
          axis=0,              # axis of the data along which to interpolate
          kind='linear',       # interpolation method, such as 'linear', 'zero', 'nearest', 'quadratic', 'cubic', etc.
          fill_value='extrapolate' # how to handle x values outside the original range
      )
      data_resampled = fn_interpolate(target_time_s)
      if np.any(np.isnan(data_resampled)):
        print('\n'*5)
        print('='*50)
        print('='*50)
        print('FOUND NAN')
        print(subject_id, device_name, stream_name)
        timesteps_have_nan = np.any(np.isnan(data_resampled), axis=tuple(np.arange(1,np.ndim(data_resampled))))
        print('Timestep indexes with NaN:', np.where(timesteps_have_nan)[0])
        print_var(data_resampled)
        # input('Press enter to continue ')
        print('\n'*5)
        time.sleep(10)
        data_resampled[np.isnan(data_resampled)] = 0
      file_data[device_name][stream_name]['time_s'] = target_time_s
      file_data[device_name][stream_name]['data'] = data_resampled
    data_bySubject[subject_id][data_file_index] = file_data

#########################################
############ CREATE FEATURES ############
#########################################

def get_feature_matrices(experiment_data, label_start_time_s, label_end_time_s, count=num_segments_per_subject):
  # Determine start/end times for each example segment.
  start_time_s = label_start_time_s + buffer_startActivity_s
  end_time_s = label_end_time_s - buffer_endActivity_s
  segment_start_times_s = np.linspace(start_time_s, end_time_s - segment_duration_s,
                                      num=count,
                                      endpoint=True)
  # Create a feature matrix by concatenating each desired sensor stream.
  feature_matrices = []
  for segment_start_time_s in segment_start_times_s:
    # print('Processing segment starting at %f' % segment_start_time_s)
    segment_end_time_s = segment_start_time_s + segment_duration_s
    feature_matrix = np.empty(shape=(segment_length, 0))
    for (device_name, stream_name, extraction_fn) in device_streams_for_features:
      # print(' Adding data from [%s][%s]' % (device_name, stream_name))
      data = np.squeeze(np.array(experiment_data[device_name][stream_name]['data']))
      time_s = np.squeeze(np.array(experiment_data[device_name][stream_name]['time_s']))
      time_indexes = np.where((time_s >= segment_start_time_s) & (time_s <= segment_end_time_s))[0]
      # Expand if needed until the desired segment length is reached.
      time_indexes = list(time_indexes)
      while len(time_indexes) < segment_length:
        print(' Increasing segment length from %d to %d for %s %s for segment starting at %f' % (len(time_indexes), segment_length, device_name, stream_name, segment_start_time_s))
        if time_indexes[0] > 0:
          time_indexes = [time_indexes[0]-1] + time_indexes
        elif time_indexes[-1] < len(time_s)-1:
          time_indexes.append(time_indexes[-1]+1)
        else:
          raise AssertionError
      while len(time_indexes) > segment_length:
        print(' Decreasing segment length from %d to %d for %s %s for segment starting at %f' % (len(time_indexes), segment_length, device_name, stream_name, segment_start_time_s))
        time_indexes.pop()
      time_indexes = np.array(time_indexes)
      
      # Extract the data.
      time_s = time_s[time_indexes]
      data = data[time_indexes,:]
      data = extraction_fn(data)
      # print('  Got data of shape', data.shape)
      # Add it to the feature matrix.
      data = np.reshape(data, (segment_length, -1))
      if np.any(np.isnan(data)):
        print('\n'*5)
        print('='*50)
        print('='*50)
        print('FOUND NAN')
        print(device_name, stream_name, segment_start_time_s)
        timesteps_have_nan = np.any(np.isnan(data), axis=tuple(np.arange(1,np.ndim(data))))
        print('Timestep indexes with NaN:', np.where(timesteps_have_nan)[0])
        print_var(data)
        # input('Press enter to continue ')
        print('\n'*5)
        time.sleep(10)
        data[np.isnan(data)] = 0
      feature_matrix = np.concatenate((feature_matrix, data), axis=1)
    feature_matrices.append(feature_matrix)
  # print(len(feature_matrices), feature_matrices[0].shape)
  return feature_matrices

#########################################
############ CREATE EXAMPLES ############
#########################################

# Will store intermediate examples from each file.
example_matrices_byLabel = {}
# Then will create the following 'final' lists with the correct number of examples.
example_labels = []
example_label_indexes = []
example_matrices = []
example_subject_ids = []
print()
for (subject_id, file_datas) in data_bySubject.items():
  print()
  print('Processing data for subject %s' % subject_id)
  noActivity_matrices = []
  for (data_file_index, file_data) in enumerate(file_datas):
    # Get the timestamped label data.
    # As described in the HDF5 metadata, each row has entries for ['Activity', 'Start/Stop', 'Valid', 'Notes'].
    device_name = 'experiment-activities'
    stream_name = 'activities'
    activity_datas = file_data[device_name][stream_name]['data']
    activity_times_s = file_data[device_name][stream_name]['time_s']
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
    # Loop through each activity that is designated for classification.
    for (label_index, activity_label) in enumerate(activities_to_classify):
      if label_index == baseline_index:
        continue
      # Extract num_segments_per_subject examples from each instance of the activity.
      # Then later, will select num_segments_per_subject in total from all instances.
      file_label_indexes = [i for (i, label) in enumerate(activities_labels) if label==activity_label]
      if len(file_label_indexes) == 0 and activity_label in activities_renamed:
        for alternate_label in activities_renamed[activity_label]:
          file_label_indexes = [i for (i, label) in enumerate(activities_labels) if label==alternate_label]
          if len(file_label_indexes) > 0:
            print('  Found renamed activity from "%s"' % alternate_label)
            break
      print('  Found %d instances of %s' % (len(file_label_indexes), activity_label))
      for file_label_index in file_label_indexes:
        start_time_s = activities_start_times_s[file_label_index]
        end_time_s = activities_end_times_s[file_label_index]
        duration_s = end_time_s -  start_time_s
        # Extract example segments and generate a feature matrix for each one.
        # num_examples = int(num_segments_per_subject/len(file_label_indexes))
        # if file_label_index == file_label_indexes[-1]:
        #   num_examples = num_segments_per_subject - num_examples*(len(file_label_indexes)-1)
        num_examples = num_segments_per_subject
        print('  Extracting %d examples from activity "%s" with duration %0.2fs' % (num_examples, activity_label, duration_s))
        feature_matrices = get_feature_matrices(file_data,
                                                start_time_s, end_time_s,
                                                count=num_examples)
        example_matrices_byLabel.setdefault(activity_label, [])
        example_matrices_byLabel[activity_label].extend(feature_matrices)
    
    # Generate matrices for not doing any activity.
    # Will generate one matrix for each inter-activity portion,
    #  then later select num_baseline_segments_per_subject of them.
    for (label_index, activity_label) in enumerate(activities_labels):
      if label_index == len(activities_labels)-1:
        continue
      print('  Getting baseline examples between activity "%s"' % (activity_label))
      noActivity_start_time_s = activities_end_times_s[label_index]
      noActivity_end_time_s = activities_start_times_s[label_index+1]
      duration_s = noActivity_end_time_s -  noActivity_start_time_s
      if duration_s < segment_duration_s:
        continue
      # Extract example segments and generate a feature matrix for each one.
      feature_matrices = get_feature_matrices(file_data,
                                              noActivity_start_time_s,
                                              noActivity_end_time_s,
                                              count=10)
      noActivity_matrices.extend(feature_matrices)
  
  # Choose a subset of the examples of each label, so the correct number is retained.
  # Will evenly distribute the selected indexes over all possibilities.
  for (activity_label_index, activity_label) in enumerate(activities_to_classify):
    if activity_label_index == baseline_index:
      continue
    print(' Selecting %d examples for subject %s of activity "%s"' % (num_segments_per_subject, subject_id, activity_label))
    if activity_label not in example_matrices_byLabel:
      print('\n'*5)
      print('='*50)
      print('='*50)
      print('  No examples found!')
      # print('  Press enter to continue ')
      print('\n'*5)
      time.sleep(10)
      continue
    feature_matrices = example_matrices_byLabel[activity_label]
    example_indexes = np.round(np.linspace(0, len(feature_matrices)-1,
                                              endpoint=True,
                                              num=num_segments_per_subject,
                                              dtype=int))
    for example_index in example_indexes:
      example_labels.append(activity_label)
      example_label_indexes.append(activity_label_index)
      example_matrices.append(feature_matrices[example_index])
      example_subject_ids.append(subject_id)
    
  # Choose a subset of the baseline examples.
  print(' Selecting %d examples for subject %s of activity "%s"' % (num_baseline_segments_per_subject, subject_id, baseline_label))
  noActivity_indexes = np.round(np.linspace(0, len(noActivity_matrices)-1,
                                            endpoint=True,
                                            num=num_baseline_segments_per_subject,
                                            dtype=int))
  for noActivity_index in noActivity_indexes:
    example_labels.append(baseline_label)
    example_label_indexes.append(baseline_index)
    example_matrices.append(noActivity_matrices[noActivity_index])
    example_subject_ids.append(subject_id)
    
  
print()

#########################################
############# SAVE RESULTS  #############
#########################################

if output_filepath is not None:
  with h5py.File(output_filepath, 'w') as hdf_file:
    metadata = OrderedDict()
    metadata['output_dir'] = output_dir
    metadata['data_root_dir'] = data_root_dir
    metadata['data_folders_bySubject'] = data_folders_bySubject
    metadata['activities_to_classify'] = activities_to_classify
    metadata['device_streams_for_features'] = device_streams_for_features
    metadata['segment_duration_s'] = segment_duration_s
    metadata['segment_length'] = segment_length
    metadata['num_segments_per_subject'] = num_segments_per_subject
    metadata['num_baseline_segments_per_subject'] = num_baseline_segments_per_subject
    metadata['buffer_startActivity_s'] = buffer_startActivity_s
    metadata['buffer_endActivity_s'] = buffer_endActivity_s
    metadata['filter_cutoff_emg_Hz'] = filter_cutoff_emg_Hz
    metadata['filter_cutoff_tactile_Hz'] = filter_cutoff_tactile_Hz
    metadata['filter_cutoff_gaze_Hz'] = filter_cutoff_gaze_Hz

    metadata = convert_dict_values_to_str(metadata, preserve_nested_dicts=False)

    hdf_file.create_dataset('example_labels', data=example_labels)
    hdf_file.create_dataset('example_label_indexes', data=example_label_indexes)
    hdf_file.create_dataset('example_matrices', data=example_matrices)
    hdf_file.create_dataset('example_subject_ids', data=example_subject_ids)

    hdf_file.attrs.update(metadata)
    
    print()
    print('Saved processed data to', output_filepath)
    print()













