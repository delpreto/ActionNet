
import h5py
import numpy as np
import os

#########################################################
# Configuration
#########################################################

data_root_dir = 'path_to_folder_with_dropbox_folder_structure'

#########################################################
# Helpers
#########################################################

# Recursively print the groups and datasets in an HDF5 file.
def print_hdf5_contents(h5_item, item_name='root', indent_str='', max_subitem_name_length=0):
  if isinstance(h5_item, h5py._hl.dataset.Dataset):
    data_type_str = str(h5_item.dtype).ljust(10)
    data_size_str = '[%s]' % ', '.join(['%4d' % x for x in h5_item.shape]).ljust(18)
    item_name_str = item_name.ljust(max_subitem_name_length+1)
    print('%s%s | type %s | shape %s' % (indent_str, item_name_str, data_type_str, data_size_str))
  else:
    print('%s%s' % (indent_str, item_name))
    max_key_length = max([len(key) for key in h5_item])
    for item_name in h5_item:
      print_hdf5_contents(h5_item[item_name], item_name=item_name,
                          indent_str=indent_str+'. ', max_subitem_name_length=max_key_length)

#########################################################
# Load activity times
#########################################################

print()
print()
print('='*75)
print('Loading activity times')
print('='*75)

# Open the file.
npy_filepath = os.path.join(data_root_dir, 'continuous_data', 'pouring_activity_times_S11-2.npy')
if not os.path.exists(npy_filepath):
  print()
  print('The file does not exist:', npy_filepath)
else:
  activity_times_s = np.load(npy_filepath)
  
  # Parse the matrix columns.
  activity_start_times_s = activity_times_s[:, 0]
  activity_end_times_s = activity_times_s[:, 1]
  activity_durations_s = activity_end_times_s - activity_start_times_s
  
  print()
  print('See time bounds for %d activities' % activity_start_times_s.size)
  print('Average activity duration: %0.2fs +- %0.2fs (range %0.2fs to %0.2fs)' % (
    np.mean(activity_durations_s), np.std(activity_durations_s),
    np.min(activity_durations_s), np.max(activity_durations_s)))

print()
input('Press Enter to continue')

#########################################################
# Load all EEG data
#########################################################

print()
print()
print('='*75)
print('Loading all EEG data')
print('='*75)

# Open the file.
h5_filepath = os.path.join(data_root_dir, 'continuous_data', 'eeg_data.hdf5')
if not os.path.exists(h5_filepath):
  print()
  print('The file does not exist:', h5_filepath)
  print()
  input('Press Enter to continue')
else:
  h5_file = h5py.File(h5_filepath, 'r')
  
  # Print the contents.
  print()
  input('Press Enter to print the HDF5 structure')
  print_hdf5_contents(h5_file)
  
  # Load some data.
  eeg_data_nV = np.squeeze(h5_file['recording_01']['eeg_data_nV'])
  eeg_data_filtered_nV = np.squeeze(h5_file['recording_01']['eeg_data_filtered_nV'])
  time_s = np.squeeze(h5_file['recording_01']['time_s'])
  
  # Load some metadata.
  metadata = dict(h5_file['recording_01'].attrs.items())
  eeg_channel_names = metadata['Data headings']
  print('\nChannel names:', eeg_channel_names)
  
  # Clean up.
  h5_file.close()

#########################################################
# Load EEG data alongside all other wearable data
#########################################################

print()
print()
print('='*75)
print('Loading EEG data saved alongside other wearable data')
print('='*75)

# Open the file.
h5_filepath = os.path.join(data_root_dir, 'continuous_data', '2025-02-14_14-53-26_streamLog_actionSense_S11.hdf5')
if not os.path.exists(h5_filepath):
  print()
  print('The file does not exist:', h5_filepath)
  print()
  input('Press Enter to continue')
else:
  h5_file = h5py.File(h5_filepath, 'r')
  
  # Print the contents.
  print()
  input('Press Enter to print the HDF5 structure')
  print_hdf5_contents(h5_file)
  
  # Load some data.
  eeg_data_nV = np.squeeze(h5_file['eeg']['all_channels']['data'])
  eeg_data_filtered_nV = np.squeeze(h5_file['eeg']['all_channels_filtered']['data'])
  time_s = np.squeeze(h5_file['eeg']['all_channels']['time_s'])
  
  # Load some metadata.
  metadata = dict(h5_file['eeg']['all_channels'].attrs.items())
  eeg_channel_names = metadata['Data headings']
  print('\nChannel names:', eeg_channel_names)
  
  # Clean up.
  h5_file.close()


#########################################################
# Load EEG and body pose data sliced to individual trials
# before resampling to a consistent vector length.
#########################################################

print()
print()
print('='*75)
print('Loading EEG and body pose data sliced to individual trials')
print('BEFORE resampling to a consistent vector length')
print('='*75)

# Open the file.
h5_filepath = os.path.join(data_root_dir, 'data_per_trial', 'pouring_paths_humans_S11-2.hdf5')
if not os.path.exists(h5_filepath):
  print()
  print('The file does not exist:', h5_filepath)
  print()
  input('Press Enter to continue')
else:
  h5_file = h5py.File(h5_filepath, 'r')
  
  # Print the contents.
  print()
  input('Press Enter to print the HDF5 structure')
  print_hdf5_contents(h5_file)
  
  # Load some data.
  eeg_data_nV = np.squeeze(h5_file['subject_11']['trial_000']['eeg']['all_channels'])
  eeg_data_filtered_nV = np.squeeze(h5_file['subject_11']['trial_000']['eeg']['all_channels_filtered'])
  time_s = np.squeeze(h5_file['subject_11']['trial_000']['eeg']['time_s'])
  
  # Clean up.
  h5_file.close()


#########################################################
# Load EEG and body pose data sliced to individual trials
# after resampling to a consistent vector length.
#########################################################

print()
print()
print('='*75)
print('Loading EEG and body pose data sliced to individual trials')
print('AFTER resampling to a consistent vector length')
print('='*75)

# Open the file.
h5_filepath = os.path.join(data_root_dir, 'data_per_trial_resampledForTraining', 'pouring_trainingData_S11-2.hdf5')
if not os.path.exists(h5_filepath):
  print()
  print('The file does not exist:', h5_filepath)
  print()
  input('Press Enter to continue')
else:
  h5_file = h5py.File(h5_filepath, 'r')
  
  # Print the contents.
  print()
  input('Press Enter to print the HDF5 structure')
  print_hdf5_contents(h5_file)
  
  # Load some data.
  eeg_data_nV = np.squeeze(h5_file['eeg']['all_channels'])
  eeg_data_filtered_nV = np.squeeze(h5_file['eeg']['all_channels_filtered'])
  time_s = np.squeeze(h5_file['eeg']['time_s'])
  
  # Clean up.
  h5_file.close()

#########################################################
#########################################################

print()
print('='*75)
print('Happy analyzing!')
print('='*75)
print()


