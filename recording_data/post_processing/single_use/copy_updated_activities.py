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