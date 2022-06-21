
import h5py
import glob
import os
import shutil
import numpy as np
from utils.print_utils import *

# For each HDF5 file on the 'original' directory, will look for one
#  with the same name in the 'adjusted' directory.
#  Will then copy the experiment-activities group from the adjusted HDF5 to the original.

originals_dir = 'P:/MIT/Lab/Wearativity/data/experiments/to_process'
adjusteds_dir = 'D:/MIT/Lab/Wearativity/data/experiments/02 - POST MERGING - UPDATED LABELS'

hdf5_filepaths_originals = glob.glob(os.path.join(originals_dir, '**/*.hdf5'), recursive=True)
hdf5_filepaths_adjusteds = glob.glob(os.path.join(adjusteds_dir, '**/*.hdf5'), recursive=True)
print_var(hdf5_filepaths_originals, 'hdf5_filepaths_originals')
print_var(hdf5_filepaths_adjusteds, 'hdf5_filepaths_adjusteds')

for hdf5_filepath_original in hdf5_filepaths_originals:
  filename_original = os.path.split(hdf5_filepath_original)[1]
  print(filename_original)
  if '_originalPreUpdateLabels' in filename_original:
    continue
  filepath_adjusted = None
  for hdf5_filepath_adjusted in hdf5_filepaths_adjusteds:
    filename_adjusted = os.path.split(hdf5_filepath_adjusted)[1]
    if filename_adjusted == filename_original:
      filepath_adjusted = hdf5_filepath_adjusted
      break
  if filepath_adjusted is None:
    print('COULD NOT FIND ADJUSTED FILE FOR ORIGINAL', hdf5_filepath_original)
    
  backup_filepath = hdf5_filepath_original.replace('.hdf5', '_originalPreUpdateLabels.hdf5')
  if not os.path.exists(backup_filepath):
    shutil.copy(hdf5_filepath_original, backup_filepath)
  
  h5_original = h5py.File(hdf5_filepath_original, 'a')
  h5_adjusted = h5py.File(filepath_adjusted, 'r')
  activities_adjusted = h5_adjusted['experiment-activities']
  
  print()
  print('original path:', hdf5_filepath_original)
  print('adjusted path:', filepath_adjusted)
  print_var(np.array(activities_adjusted['activities']['data']))
  if 'experiment-activities' in h5_original:
    del h5_original['experiment-activities']

  h5_adjusted.copy(activities_adjusted, h5_original,
                    name=None, shallow=False,
                    expand_soft=True, expand_external=True, expand_refs=True,
                    without_attrs=False)
  activities_group_metadata = dict(activities_adjusted.attrs.items())
  h5_original['experiment-activities'].attrs.update(activities_group_metadata)
  
  h5_original.close()
  h5_adjusted.close()
  
print()