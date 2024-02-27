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

from sensor_streamer_handlers.SensorManager import SensorManager
from sensor_streamer_handlers.DataVisualizer import DataVisualizer
from sensor_streamers.SensorStreamer import SensorStreamer
from sensor_streamers.TouchStreamerESP import TouchStreamerESP

import h5py
import numpy as np
from utils.numpy_scipy_utils import convolve2d_strided
import time
import traceback
from utils.time_utils import *
from utils.dict_utils import *
import sys
import os
import glob
script_dir = os.path.dirname(os.path.realpath(__file__))

# Note that multiprocessing requires the __main__ check.
if __name__ == '__main__':
  # Configure printing to the console.
  print_status = True
  print_debug = True
  
  if len(sys.argv) <= 1:
    # Define the log(s) to replay.
    data_dir = os.path.realpath(os.path.join(script_dir, '..', '..', 'data'))
    experiments_dir = os.path.join(data_dir, 'tests')
    log_dirs = [
      # os.path.join(experiments_dir, '2023-01-10_tactileShear_testing', '2023-01-10_20-07-25_tactileShear_testing'),
      os.path.join(experiments_dir, '2023-02-02 insole shear data from mike', '2023-02-02_14-55-00_my_log_tag'),
    ]
  else:
    log_dirs = sys.argv[1:]
  
  # Loop through each specified log directory to process.
  for (i, log_dir) in enumerate(log_dirs):
    print('\n\n' + '='*75)
    print('Converting tactile to shear for log directory %s' % log_dir)
    
    hdf5_filepath = glob.glob(os.path.join(log_dir, '*.hdf5'))[0]
    hdf5_file = h5py.File(hdf5_filepath, 'a')
    tactile_stream = hdf5_file['tactile-shoe-right']['tactile_data']
    tactile_data = np.array(tactile_stream['data'])
    time_s = np.array(tactile_stream['time_s'])
    time_str = np.array(tactile_stream['time_str'])
    
    shear_group = hdf5_file.create_group('shear-shoe-right')
    # Raw tactile data.
    tactile_raw_group = shear_group.create_group('tactile_data')
    tactile_raw_group.create_dataset('time_s', data=time_s)
    tactile_raw_group.create_dataset('time_str', data=time_str, dtype='S26')
    tactile_raw_group.create_dataset('data', data=tactile_data)
    # Tiled data.
    tactile_tiled = []
    tactile_force_vector = []
    touchStreamer = TouchStreamerESP(com_ports={'dummy':'COMXX'}, is_shear_sensor=True, print_status=False, print_debug=False)
    for frame_index in range(time_s.size):
      if frame_index % round(time_s.size/10) == 0:
        print('  Converting frame %d/%d' % (frame_index+1, time_s.size))
      data_matrix = np.squeeze(tactile_data[frame_index,:,:])
      # Compute shear-specific quantities.
      (time_s, data_matrix, data_matrix_tiled_magnitude,
       data_matrix_tiled_shearAngle_rad, data_matrix_tiled_shearMagnitude) \
        = touchStreamer._compute_shear(time_s, data_matrix)
      # Add to the new arrays.
      tactile_tiled.append(data_matrix_tiled_magnitude)
      tactile_force_vector.append(np.stack((data_matrix_tiled_shearMagnitude,
                                            data_matrix_tiled_shearAngle_rad),
                                           axis=0))
    tactile_raw_group = shear_group.create_group('tactile_tiled')
    tactile_raw_group.create_dataset('time_s', data=time_s)
    tactile_raw_group.create_dataset('time_str', data=time_str, dtype='S26')
    tactile_raw_group.create_dataset('data', data=np.array(tactile_tiled))
    tactile_raw_group = shear_group.create_group('force_vector')
    tactile_raw_group.create_dataset('time_s', data=time_s)
    tactile_raw_group.create_dataset('time_str', data=time_str, dtype='S26')
    tactile_raw_group.create_dataset('data', data=np.array(tactile_force_vector))
    
    # Add metadata.
    metadata = {}
    metadata[SensorStreamer.metadata_class_name_key] = 'TouchStreamerESP'
    metadata = convert_dict_values_to_str(metadata)
    shear_group.attrs.update(metadata)
    
    print('  Wrote converted shear data to %s' % hdf5_filepath)
    
    have_video = 'video' in hdf5_file.keys()
    hdf5_file.close()
    
    