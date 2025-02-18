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

from sensor_streamer_handlers.DataLogger import DataLogger
from sensor_streamers.SensorStreamer import SensorStreamer
from utils.time_utils import *
from utils.dict_utils import *
import h5py
import numpy as np
import traceback
import os
import glob

from scipy.signal import butter, filtfilt

#######################################################

# Define the log directory, which should contain the HDF5 file from ActionSense streaming data.
script_dir = os.path.dirname(os.path.realpath(__file__))
actionsense_root_dir = script_dir
while os.path.split(actionsense_root_dir)[-1] != 'ActionSense':
  actionsense_root_dir = os.path.realpath(os.path.join(actionsense_root_dir, '..'))
log_dir_root = os.path.realpath(os.path.join(actionsense_root_dir, 'data',
                                             'experiments', '2025-02-14_experiment_S11'))
iterate_log_subdirs_depth = 1 # 0 if log_dir_root is a log folder directly (e.g. contains an HDF5 file), then add 1 for each level up

# Define the EEG data files.
eeg_dir = os.path.join(log_dir_root)
eeg_output_filepath = os.path.join(log_dir_root, 'eeg_data.hdf5')

# Define the device to use as a reference time for bounds of the experiment.
reference_time_device_name = 'myo-left'
reference_time_stream_name = 'emg'

eeg_highpass_cutoff_hz = 0.5
eeg_highpass_order = 4

#######################################################
# Extract the EEG data.
print('Loading EEG data')
eeg_filepaths = glob.glob(os.path.join(eeg_dir, '*.easy'))
eeg_data_nv = []
eeg_time_s = []
eeg_files_start_indexes = [0]
for eeg_filepath in eeg_filepaths:
  fin = open(eeg_filepath, 'r')
  for eeg_line in fin.readlines():
    eeg_line_data = [float(x) for x in eeg_line.split('\t')]
    eeg_data_nv.append(eeg_line_data[0:32])
    eeg_time_s.append(eeg_line_data[-1]/1e3)
  eeg_files_start_indexes.append(len(eeg_time_s))
  fin.close()
eeg_files_start_indexes = eeg_files_start_indexes[0:-1]
eeg_data_nv = np.array(eeg_data_nv)
eeg_time_s = np.atleast_2d(eeg_time_s).T
print('  Loaded EEG data with the following shape: ', eeg_data_nv.shape)

# Get the EEG channel names.
eeg_info_filepaths = glob.glob(os.path.join(eeg_dir, '*.info'))
eeg_channel_names = ['x']*eeg_data_nv.shape[1]
eeg_fs_hz = None
for eeg_info_filepath in eeg_info_filepaths:
  fin = open(eeg_info_filepath, 'r')
  found_names = False
  for eeg_info_line in fin.readlines():
    try:
      fs = int(eeg_info_line.split('EEG sampling rate:')[1].strip().split(' ')[0])
      if eeg_fs_hz is not None:
        assert eeg_fs_hz == fs
      eeg_fs_hz = fs
    except:
      pass
    try:
      line_split = eeg_info_line.split('Channel ')[-1].split(':')
      channel_index = int(line_split[0].strip())-1
      channel_name = line_split[1].strip()
    except:
      continue
    if eeg_channel_names[channel_index] != 'x':
      assert eeg_channel_names[channel_index] == channel_name
    eeg_channel_names[channel_index] = channel_name
  fin.close()
print('  Found channel names: %s' % eeg_channel_names)
print('  Found sampling rate: %d Hz' % eeg_fs_hz)


# Filter the EEG data.
print('  Filtering the EEG data')
def butter_highpass(cutoff, fs, order=4):
  nyquist = 0.5 * fs
  normal_cutoff = cutoff / nyquist
  b, a = butter(order, normal_cutoff, btype='high', analog=False)
  return b, a
b, a = butter_highpass(eeg_highpass_cutoff_hz, eeg_fs_hz, eeg_highpass_order)
eeg_data_nv_filtered = np.apply_along_axis(lambda x: filtfilt(b, a, x), axis=0, arr=eeg_data_nv)

#######################################################
# Save an HDF5 file with all EEG data
if eeg_output_filepath is not None:
  print('  Saving an HDF5 file with all EEG data')
  h5_file = h5py.File(eeg_output_filepath, 'w')
  h5_compression_options = {'compression': 'gzip', 'compression_opts': 1}
  metadata = {}
  metadata[SensorStreamer.metadata_data_headings_key] = eeg_channel_names
  metadata['eeg_highpass_cutoff_hz'] = eeg_highpass_cutoff_hz
  metadata['eeg_highpass_order'] = eeg_highpass_order
  metadata['eeg_fs_hz'] = eeg_fs_hz
  metadata = convert_dict_values_to_str(metadata)
  for (file_index, start_index) in enumerate(eeg_files_start_indexes):
    h5_group = h5_file.create_group('recording_%02d' % file_index)
    if file_index+1 < len(eeg_files_start_indexes):
      end_index = eeg_files_start_indexes[file_index+1]-1
    else:
      end_index = eeg_data_nv.shape[0]-1
    eeg_data_nv_forFile = eeg_data_nv[start_index:end_index+1, :]
    eeg_data_nv_filtered_forFile = eeg_data_nv_filtered[start_index:end_index+1, :]
    eeg_time_s_forFile = eeg_time_s[start_index:end_index+1, :]
    h5_group.create_dataset('eeg_data_nV', data=eeg_data_nv_forFile, **h5_compression_options)
    h5_group.create_dataset('eeg_data_filtered_nV', data=eeg_data_nv_filtered_forFile, **h5_compression_options)
    h5_group.create_dataset('time_s', data=eeg_time_s_forFile, **h5_compression_options)
    h5_group.create_dataset('time_str', data=[[get_time_str(t[0], '%Y-%m-%d %H:%M:%S.%f')] for t in eeg_time_s_forFile], **h5_compression_options)
    h5_group.attrs.update(metadata)
  h5_file.close()

#######################################################

# Merge the data!
# Process all subdirectories in the provided log directory,
#   or treat it as the main log directory.
if iterate_log_subdirs_depth is not None and iterate_log_subdirs_depth > 0:
  log_dirs = []
  dirs_to_iterate = next(os.walk(log_dir_root))[1]
  dirs_to_iterate = [os.path.join(log_dir_root, sub_dir) for sub_dir in dirs_to_iterate]
  for i in range(iterate_log_subdirs_depth-1):
    new_dirs_to_iterate = []
    for sub_dir in dirs_to_iterate:
      sub_dirs = next(os.walk(sub_dir))[1]
      new_dirs_to_iterate.extend([os.path.join(sub_dir, sub_sub_dir) for sub_sub_dir in sub_dirs])
    dirs_to_iterate = new_dirs_to_iterate
  log_dirs = dirs_to_iterate
else:
  log_dirs = [log_dir_root]
  
print('Log directories that will be processed:')
for log_dir in log_dirs:
  print(' ', log_dir)
print()
input('Press enter to continue')
print('\n')

for log_dir in log_dirs:
  if os.path.exists(log_dir):
    print('\n\n' + '='*75)
    print('Adding EEG data for log directory %s\n' % log_dir)
    hdf5_filepaths = glob.glob(os.path.join(log_dir, '*.hdf5'))
    if len(hdf5_filepaths) == 0:
      print('  Skipping since no HDF5 file was found')
      continue
    if len(hdf5_filepaths) > 1:
      print('  Skipping since more than 1 HDF5 file was found')
      continue
    hdf5_filepath = hdf5_filepaths[0]
    print('  Opening file %s' % hdf5_filepath)
    h5_file = h5py.File(hdf5_filepath, 'a')
    
    # Get the bounds of this HDF5 file.
    if reference_time_device_name not in h5_file:
      print('  Skipping since the reference time device [%s] was not found in the HDF5 file' % (reference_time_device_name))
      h5_file.close()
      continue
    reference_time_s = np.squeeze(h5_file[reference_time_device_name][reference_time_stream_name]['time_s'])
    start_time_s = reference_time_s[0]
    end_time_s = reference_time_s[-1]
    print('  See HDF5 data spanning %0.3f seconds / %0.3f minutes' % (
      (end_time_s - start_time_s), (end_time_s - start_time_s)/60
    ))
    # Find the EEG indexes that correspond to this HDF5 file.
    indexes = np.argwhere(eeg_time_s >= start_time_s)
    eeg_start_index = indexes[0][0] if indexes.size > 0 else None
    indexes = np.argwhere(eeg_time_s <= end_time_s)
    eeg_end_index = indexes[-1][0] if indexes.size > 0 else None
    if eeg_start_index is None or eeg_end_index is None or (eeg_end_index - eeg_start_index) < 1:
      print('  Skipping this log directory since no overlapping EEG data was found')
      continue
    # Slice the EEG data.
    eeg_data_nv_forFile = eeg_data_nv[eeg_start_index:eeg_end_index+1, :]
    eeg_data_nv_filtered_forFile = eeg_data_nv_filtered[eeg_start_index:eeg_end_index+1, :]
    eeg_time_s_forFile = eeg_time_s[eeg_start_index:eeg_end_index+1, :]
    eeg_time_str_forFile = [[get_time_str(t[0], '%Y-%m-%d %H:%M:%S.%f')] for t in eeg_time_s_forFile]
    eeg_duration_s_forFile = eeg_time_s_forFile[-1][0] - eeg_time_s_forFile[0][0]
    print('  Adding data from %d EEG timesteps (%0.3f seconds / %0.3f minutes)' % (
      eeg_time_s_forFile.size, eeg_duration_s_forFile, eeg_duration_s_forFile/60))
    # Add the EEG data.
    if 'eeg' in h5_file:
      del h5_file['eeg']
    eeg_device_group = h5_file.create_group('eeg')
    eeg_stream_group = eeg_device_group.create_group('all_channels')
    eeg_stream_group.create_dataset('data', data=eeg_data_nv_forFile)
    eeg_stream_group.create_dataset('time_s', data=eeg_time_s_forFile)
    eeg_stream_group.create_dataset('time_str', data=eeg_time_str_forFile)
    # Add the filtered EEG data.
    eeg_filtered_stream_group = eeg_device_group.create_group('all_channels_filtered')
    eeg_filtered_stream_group.create_dataset('data', data=eeg_data_nv_filtered_forFile)
    eeg_filtered_stream_group.create_dataset('time_s', data=eeg_time_s_forFile)
    eeg_filtered_stream_group.create_dataset('time_str', data=eeg_time_str_forFile)
    # Add metadata.
    metadata = {}
    metadata[SensorStreamer.metadata_class_name_key] = 'EEGStreamer'
    metadata = convert_dict_values_to_str(metadata)
    eeg_device_group.attrs.update(metadata)
    metadata = {}
    metadata[SensorStreamer.metadata_data_headings_key] = eeg_channel_names
    metadata['eeg_highpass_cutoff_hz'] = eeg_highpass_cutoff_hz
    metadata['eeg_highpass_order'] = eeg_highpass_order
    metadata['eeg_fs_hz'] = eeg_fs_hz
    metadata = convert_dict_values_to_str(metadata)
    eeg_stream_group.attrs.update(metadata)
    eeg_filtered_stream_group.attrs.update(metadata)
    
    # Clean up.
    h5_file.close()
  else:
    print('\n\nLog directory does not exist: %s\n' % log_dir)
print('\n\n')











