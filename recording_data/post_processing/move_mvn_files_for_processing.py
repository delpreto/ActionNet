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

import os
import glob
import shutil

from utils.time_utils import *
from utils.print_utils import *

print()
print()

user_input = input('Collect MVN files for processing, or restore them to original locations? [collect/restore]: ')
collecting_files = user_input.lower().strip() == 'collect'
restoring_files = user_input.lower().strip() == 'restore'
assert (collecting_files or restoring_files)

##########################################
##########################################
if collecting_files:
  # Define the top-level folder to recursively search.
  log_dir_root = input('Enter the directory to recursively search for MVN files: ')
  # script_dir = os.path.dirname(os.path.realpath(__file__))
  
  # Define where to put MVN files to process.
  mvn_processing_dir = os.path.join(log_dir_root, 'mvn_processing')
  if os.path.exists(mvn_processing_dir):
    appended_count = 1
    while os.path.exists(mvn_processing_dir):
      mvn_processing_dir = os.path.join(log_dir_root, 'mvn_processing_%d' % appended_count)
      appended_count = appended_count+1
  os.makedirs(mvn_processing_dir)
  # Also create a directory for the outputs of reprocessing/exporting.
  os.makedirs(os.path.join(mvn_processing_dir, 'processed'))
  
  # Find all MVN files, and move them to the temporary processing folder.
  filepaths = glob.glob(os.path.join(log_dir_root, '**', '*.mvn'), recursive=True)
  print()
  print('Will move the following %d files:')
  for filepath in filepaths:
    print('  ', filepath)
  input('Press Enter to continue')
  print()
  for filepath in filepaths:
    filepath_relative = os.path.relpath(filepath, log_dir_root)
    new_filename = filepath_relative.replace(os.path.sep, '__sep__')
    new_filepath = os.path.join(mvn_processing_dir, new_filename)
    shutil.move(filepath, new_filepath)
    
  # Print completion.
  print()
  print('Moved %d MVN files to the following directory:' % len(filepaths))
  print(mvn_processing_dir)
  try:
    import pyperclip
    pyperclip.copy(mvn_processing_dir)
  except ModuleNotFoundError:
    pass
  print()
  print()

##########################################
##########################################
if restoring_files:
  mvn_processing_dir = input('Enter the MVN processing directory: ')
  log_dir_root = os.path.realpath(os.path.join(mvn_processing_dir, '..'))
  append_to_exports = input('Enter postfix to add to exported files: ').strip()
  append_to_mvns = input('Enter postfix to add to MVN   files: ').strip()
  
  print()
  filepaths = glob.glob(os.path.join(mvn_processing_dir, '**', '*'), recursive=True)
  source_filepaths = []
  restored_filepaths = []
  moved_count = 0
  for filepath in filepaths:
    filename = os.path.basename(filepath)
    filename_split = filename.split('__sep__')
    restored_filepath = os.path.join(log_dir_root, *filename_split)
    if len(append_to_exports) > 0 and '.xlsx' in restored_filepath.lower():
      restored_filepath = restored_filepath.replace('.xlsx', '_%s.xlsx' % append_to_exports.lstrip('_'))
    elif len(append_to_exports) > 0 and '.mvnx' in restored_filepath.lower():
      restored_filepath = restored_filepath.replace('.mvnx', '_%s.mvnx' % append_to_exports.lstrip('_'))
    elif len(append_to_mvns) > 0 and '.mvn' in restored_filepath.lower():
      restored_filepath = restored_filepath.replace('.mvn', '_%s.mvn' % append_to_mvns.lstrip('_'))
    if len(os.path.splitext(filepath)[1]) == 0:
      continue # folder rather than a file
    source_filepaths.append(filepath)
    restored_filepaths.append(restored_filepath)
  existing_destination_filepaths = [f for f in restored_filepaths if os.path.exists(f)]
  if len(existing_destination_filepaths) > 0:
    print('No files will be copied, since the following destinations already exist:')
    print('\n'.join(existing_destination_filepaths))
  else:
    print()
    print('Will move the following %d files:')
    for i in range(len(source_filepaths)):
      print('  %02d: %s' % (i, source_filepaths[i]))
      print(  '    > %s' % (restored_filepaths[i]))
    input('Press Enter to continue')
    print()
    for i in range(len(source_filepaths)):
      print('Moving file %02d: %s > %s' % (i, source_filepaths[i], restored_filepaths[i]))
      shutil.move(source_filepaths[i], restored_filepaths[i])
      moved_count = 1 + moved_count
    
  # Delete the folder if it is empty.
  try:
    os.rmdir(os.path.join(mvn_processing_dir, 'processed'))
  except:
    pass
  try:
    os.rmdir(mvn_processing_dir)
  except:
    pass
  
  # Print completion.
  print()
  print('Moved %d files to their original locations' % moved_count)
  print()
  print()
  
  














