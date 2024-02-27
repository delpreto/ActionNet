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
      os.path.join(experiments_dir, '2023-01-10_tactileShear_testing', 'test_data_fromXilinx'),
    ]
  else:
    log_dirs = sys.argv[1:]
  
  # Define offsets for start/end of processing.
  #  Should be a list with an entry for each log directory.
  #  Each entry can be a time in seconds, or None to use all log data.
  start_offsets_s = [None]*len(log_dirs) # for 2022-05-24: [12*60+30] # 5*60+34, 1*60+40, 2*60+3
  end_offsets_s = [None]*len(log_dirs) # for 2022-05-24: [(22*60+47) - (16*60+45)] # (14*60+24) - (13*60+32), (14*60+1) - (13*60+15), (5*60+58) - (3*60+55)
  # start_end_times = [['12:30', '16:45'], ['5:34', '13:32'], ['1:40', '13:15'], ['2:03', '3:55']]
  
  # Loop through each specified log directory to process.
  for (i, log_dir) in enumerate(log_dirs):
    print('\n\n' + '='*75)
    print('Creating composite visualization for log directory %s\n' % log_dir)
    print()
    
    # Convert data in the Xilinx HDF5 format to ActionNet data if needed.
    hdf5_filepath = glob.glob(os.path.join(log_dir, '*.hdf5'))[0]
    hdf5_file = h5py.File(hdf5_filepath, 'r')
    if 'frame_count' in hdf5_file.keys():
      print('Converting Xlinix data format to ActionNet dataset format')
      tactile_readings = np.array(hdf5_file['pressure'])
      time_us = np.array(hdf5_file['ts'])
      # Convert time from us to s.
      time_s = time_us/1000000.0
      # Start once the timestamps are consistent (they seem to be small sometimes then start being 'real').
      # And stop once the timestamps become 0 (since the HDF5 files aren't trimmed to fit the data).
      try:
        end_index = np.where(time_s == 0)[0][0]-1
      except:
        end_index = len(time_s)
      try:
        target_dt_s = np.mean(np.diff(time_s[end_index//2:end_index]))
        is_near_target_dt = abs((np.diff(time_s[0:end_index]) - target_dt_s)/target_dt_s) <= 0.2
        last_bad_dt = np.where(~is_near_target_dt)[0][-1]
        start_index = last_bad_dt+1
      except:
        start_index = 0
      print('Only using data between indexes [%s, %s]' % (start_index, end_index))
      tactile_readings = tactile_readings[start_index:end_index+1]
      time_s = np.atleast_2d(time_s[start_index:end_index+1]).T
      times_str = [get_time_str(t, format='%Y-%m-%d %H:%M:%S.%f').encode('utf-8') for t in list(np.squeeze(time_s))]
      
      # Create ActionNet dataset.
      # Will create a subfolder for it, and make that the new log directory.
      hdf5_filename_old_noExt = os.path.splitext(os.path.split(hdf5_filepath)[-1])[0]
      hdf5_filename_new = '%s_actionNetFormat.hdf5' % hdf5_filename_old_noExt
      log_dir = os.path.join(log_dir, 'actionNetFormat_%s' % hdf5_filename_old_noExt)
      os.makedirs(log_dir, exist_ok=True)
      hdf5_filepath_new = os.path.join(log_dir, hdf5_filename_new)
      hdf5_file_new = h5py.File(hdf5_filepath_new, 'w')
      shear_group = hdf5_file_new.create_group('shear-sensor')
      # Raw tactile data.
      tactile_raw_group = shear_group.create_group('tactile_raw')
      tactile_raw_group.create_dataset('time_s', data=time_s)
      tactile_raw_group.create_dataset('time_str', data=times_str, dtype='S26')
      tactile_raw_group.create_dataset('data', data=tactile_readings)
      # Tiled data.
      tactile_tiled = []
      tactile_force_vector = []
      for frame_index in range(time_s.size):
        data_matrix = np.squeeze(tactile_readings[frame_index,:,:])
        # The below code is copied from TouchStreamerFPGA.
        # Compute the total force in each shear square.
        toConvolve_tiled_magnitude = np.array([[1,1],[1,1]])
        data_matrix_tiled_magnitude = convolve2d_strided(data_matrix, toConvolve_tiled_magnitude, stride=2)
        # Compute the force angle in each shear square.
        toConvolve_tiled_x = np.array([[-1,1],[-1,1]])
        toConvolve_tiled_y = np.array([[1,1],[-1,-1]])
        data_matrix_tiled_x = convolve2d_strided(data_matrix, toConvolve_tiled_x, stride=2)
        data_matrix_tiled_y = convolve2d_strided(data_matrix, toConvolve_tiled_y, stride=2)
        data_matrix_tiled_shearAngle_rad = np.arctan2(data_matrix_tiled_y, data_matrix_tiled_x)
        data_matrix_tiled_shearMagnitude = np.linalg.norm(np.stack([data_matrix_tiled_y, data_matrix_tiled_x], axis=0), axis=0)
        # Add to the new arrays.
        tactile_tiled.append(data_matrix_tiled_magnitude)
        tactile_force_vector.append(np.stack((data_matrix_tiled_shearMagnitude,
                                              data_matrix_tiled_shearAngle_rad),
                                             axis=0))
      tactile_raw_group = shear_group.create_group('tactile_tiled')
      tactile_raw_group.create_dataset('time_s', data=time_s)
      tactile_raw_group.create_dataset('time_str', data=times_str, dtype='S26')
      tactile_raw_group.create_dataset('data', data=np.array(tactile_tiled))
      tactile_raw_group = shear_group.create_group('force_vector')
      tactile_raw_group.create_dataset('time_s', data=time_s)
      tactile_raw_group.create_dataset('time_str', data=times_str, dtype='S26')
      tactile_raw_group.create_dataset('data', data=np.array(tactile_force_vector))
      
      # Add metadata.
      metadata = {}
      metadata[SensorStreamer.metadata_class_name_key] = 'TouchStreamerFPGA'
      metadata = convert_dict_values_to_str(metadata)
      shear_group.attrs.update(metadata)
      
      have_video = False
      hdf5_file_new.close()
      print('Wrote converted data to %s' % hdf5_filepath_new)
      
    have_video = 'video' in hdf5_file.keys()
    hdf5_file.close()





    log_player_options = {
      'log_dir': log_dir,
      'load_datasets_into_memory': False,
    }
    start_offset_s = start_offsets_s[i]
    end_offset_s = end_offsets_s[i]

    # Configure where and how to save sensor data.
    datalogging_options = None

    # Configure visualizations to be shown as a simulation of real-time streaming.
    visualization_options = None
    composite_video_filepath = os.path.join(log_dir,
                                            'composite_visualization_postProcessed')

    sensor_streamer_specs = [
      # Stream from one or more tactile shear sensors.
      {'class': 'TouchStreamerFPGA',
       'sensor_names': [
         'shear-sensor'
       ],
       'is_shear_sensor': True,
       'downsampling_factor': 1,
       'print_debug': print_debug, 'print_status': print_status
       },
      # Stream from one or more cameras.
      {'class': 'CameraStreamer',
       'cameras_to_stream': { # map camera names (usable as device names in the HDF5 file) to capture device indexes
         'video': 1,
       },
       'print_debug': print_debug, 'print_status': print_status
       },
    ]
    if not have_video:
      sensor_streamer_specs = [spec for spec in sensor_streamer_specs if spec['class'] != 'CameraStreamer']

    # Create a sensor manager.
    sensor_manager = SensorManager(sensor_streamer_specs=sensor_streamer_specs,
                                   log_player_options=log_player_options,
                                   data_logger_options=datalogging_options,
                                   data_visualizer_options=visualization_options,
                                   kill_other_python_processes=False, # may be launching multiple visualizers
                                   print_status=print_status, print_debug=print_debug)

    # Load streams from the saved logs for each streamer.
    sensor_manager.connect()

    # Visualize!
    frame_size = (1800, 3000) # height, width # (1800, 3000)
    composite_col_width = int(frame_size[1] / 2)
    composite_row_height = int(frame_size[0] / 2)
    visualizer = DataVisualizer(sensor_streamers=sensor_manager.get_streamers(),
                                update_period_s = 0.05,
                                use_composite_video=True,
                                composite_video_layout = [
                                  [ # row  0
                                    # {'device_name':'dummy', 'stream_name':'nothing', 'rowspan':1, 'colspan':1, 'width':composite_col_width, 'height':composite_row_height},
                                    {'device_name':'video', 'stream_name':'frame', 'rowspan':1, 'colspan':1, 'width':composite_col_width, 'height':composite_row_height},
                                    {'device_name':'shear-sensor', 'stream_name':'tactile_raw', 'rowspan':1, 'colspan':1, 'width':composite_col_width, 'height':composite_row_height},
                                  ],
                                  [ # row  1
                                    {'device_name':'shear-sensor', 'stream_name':'tactile_tiled', 'rowspan':1, 'colspan':1, 'width':composite_col_width, 'height':composite_row_height},
                                    {'device_name':'shear-sensor', 'stream_name':'force_vector', 'rowspan':1, 'colspan':1, 'width':composite_col_width, 'height':composite_row_height},
                                    # {'device_name':'dummy', 'stream_name':'nothing', 'rowspan':1, 'colspan':1, 'width':composite_col_width, 'height':composite_row_height},
                                  ],
                                ],
                                composite_video_filepath=composite_video_filepath,
                                print_status=print_status, print_debug=print_debug)
    visualizer.visualize_logged_data(start_offset_s=start_offset_s, end_offset_s=end_offset_s,
                                     duration_s=None,
                                     hide_composite=False, realtime=False)
    print('Done visualizing!')
    # time.sleep(5)
    visualizer.close_visualizations()
    print('Closed visualizations!')
    # time.sleep(5)
    # del visualizer
    # del sensor_manager
    # print('Deleted objects!')
    # time.sleep(5)





