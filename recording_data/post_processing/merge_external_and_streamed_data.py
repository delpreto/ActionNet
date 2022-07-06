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
import os
import traceback

# Define the log directory, which should contain:
#  the HDF5 file from streaming data;
#  any videos saved from streaming data;
#  a folder named 'externally_recorded_data' with external recordings.
script_dir = os.path.dirname(os.path.realpath(__file__))
log_dir_root = os.path.realpath(os.path.join(script_dir, '..', '..', 'data',
          # '2022-01-30 test eye external recording', '2022-01-30_17-16-54_notes-eye'
          # '2022-02-01 test xsens streams', '2022-02-01_18-53-18_notes-xsens'
          # '2022-02-02 test xsens streams', '2022-02-02_21-06-26_notes-xsens_withFingers'
          # '2022-02-02 test xsens streams', '2022-02-02_21-09-45_notes-xsens_noFingers'
          # '2022-02-03 test xsens streams', '2022-02-03_14-38-31_notes-xsens_noFingers'
          # '2022-02-03 test xsens streams', '2022-02-03_14-32-25_notes-xsens_withFingers'
          # '2022-02-03 test xsens streams', '2022-02-03_14-38-31_notes-xsens_noFingers'
          # '2022-03-10 testing eye logging', '2022-03-10_12-28-58_test-eye'
          # '2022-04-14 test all sensors', '2022-04-14_19-13-29_test-all-kitchen-withAV'
          # '2022-05-08 test streaming', '2022-05-08_20-54-31_test-eyeDirect-tactileHub'
          # '2022-05-24 test all in kitchen'
          # '2022-05-29 test xsens update and myos', '2022-05-29_17-45-52_test-myo-xsens'
          # 'experiments', '2022-06-07_experiment_S00'
          'experiments', 'to_process_batch2'
          ))
iterate_log_subdirs_depth = 2

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
    print('Merging external data for log directory %s\n' % log_dir)
    try:
      DataLogger.merge_external_data(log_dir, print_status=True, print_debug=True)
    except:
      print('\n'*5)
      print('x'*50)
      print('ERORR')
      print(traceback.format_exc())
      print('x'*50)
      print('\n'*5)
  else:
    print('\n\nLog directory does not exist: %s\n' % log_dir)
print('\n\n')











