
import h5py
import numpy as np
import os
from utils.dict_utils import *

################################
# CONFIGURATION
################################

data_root_dir = 'C:/Users/jdelp/Desktop/ActionSense/data/experiments/'
hdf5_filepath_to_copy_metadata = os.path.join(data_root_dir, '2024-12-20_experiment_S15',
                                    '2024-12-20_17-46-00_actionSense_S15_scoop',
                                    '2024-12-20_17-46-30_streamLog_actionSense_S15.hdf5')
hdf5_filepath_to_fix = os.path.join(data_root_dir, '2024-12-20_experiment_S15',
                                    '2024-12-20_18-11-12_actionSense_S15_stir',
                                    '2024-12-20_18-11-46_streamLog_actionSense_S15.hdf5')

print('hdf5_filepath_to_copy_metadata')
print(hdf5_filepath_to_copy_metadata)
print()
print('hdf5_filepath_to_fix')
print(hdf5_filepath_to_fix)
print()

################################
# PROCESSING
################################

h5_file_to_fix = h5py.File(hdf5_filepath_to_fix, 'a')

# Copy metadata.
print('Copying metadata')
def copy_metadata(source_item, dest_item):
  # Nothing to do if it is a dataset.
  if isinstance(source_item, h5py.Dataset):
    return
  # Copy the metadata for this group.
  if isinstance(source_item, h5py.Group):
    source_metadata = dict(source_item.attrs.items())
    dest_item.attrs.update(convert_dict_values_to_str(source_metadata, preserve_nested_dicts=False))
  # Iterate the next level.
  for key in source_item:
    if key not in dest_item:
      continue
    print('  Calling copy_metadata for key [%s]' % key)
    copy_metadata(source_item[key], dest_item[key])

h5_file_to_copy_metadata = h5py.File(hdf5_filepath_to_copy_metadata, 'r')
copy_metadata(h5_file_to_copy_metadata, h5_file_to_fix)
h5_file_to_copy_metadata.close()
print()

# Trim datasets.
print('Trimming datasets')
def trim_datasets(root_item):
  # Trim all datasets based on the time array.
  if isinstance(root_item, h5py.Group) and 'time_s' in root_item:
    time_s = np.array(root_item['time_s'])
    try:
      first_blank_index = np.where(time_s == 0)[0][0]
      for key in root_item:
        if isinstance(root_item[key], h5py.Dataset):
          data = np.array(root_item[key])
          del root_item[key]
          root_item[key] = data[0:first_blank_index]
          print('    Trimmed [%s] to max index [%d] out of [%d]' % (key, first_blank_index, data.shape[0]))
    except IndexError:
      pass
  # Iterate the next level.
  if isinstance(root_item, h5py.Group):
    for key in root_item:
      print('  Calling trim_datasets for key [%s]' % key)
      trim_datasets(root_item[key])
trim_datasets(h5_file_to_fix)

# Clean up.
h5_file_to_fix.close()

















