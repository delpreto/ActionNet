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

import time
import traceback
from utils.time_utils import *
import sys
import os
script_dir = os.path.dirname(os.path.realpath(__file__))

# Note that multiprocessing requires the __main__ check.
if __name__ == '__main__':
  # Configure printing to the console.
  print_status = True
  print_debug = False
  
  if len(sys.argv) <= 1:
    # Define the log(s) to replay.
    data_dir = os.path.realpath(os.path.join(script_dir, '..', '..', 'data'))
    experiments_dir = os.path.join(data_dir, 'experiments')
    # log_dirs = [os.path.join(data_dir, '2022-04-14 test all sensors', '2022-04-14_19-13-29_test-all-kitchen-withAV')]
    # log_dirs = [os.path.join(data_dir, '2022-05-08 test streaming', '2022-05-08_20-54-31_test-eyeDirect-tactileHub')]
    # log_dirs = [os.path.join(data_dir, '2022-05-13 test all but manus - long duration', '2022-05-13_21-13-04_test-not-worn')]
    # log_dir_root = os.path.join(data_dir, '2022-05-24 test all in kitchen')
    # log_dir_root = os.path.join(data_dir, 'experiments', '2022-06-07_experiment_S00')
    # log_dirs = [
    #   # os.path.join(log_dir_root, '2022-05-24_21-09-52_test-all - peel cucumber'),
    #   # os.path.join(log_dir_root, '2022-05-24_22-22-44_test-all - potatoes bread spread'),
    #   # os.path.join(log_dir_root, '2022-05-24_22-42-15_test-all - jar plate'),
    #   # os.path.join(log_dir_root, '2022-05-24_20-30-53_test-all - tactile calibration'),
    #   # os.path.join(log_dir_root, '2022-05-24_20-30-53_test-all - tactile calibration'),
    #   # os.path.join(log_dir_root, '2022-06-07_17-18-17_actionNet-wearables_S00'),
    #   os.path.join(log_dir_root, '2022-06-07_18-10-55_actionNet-wearables_S00'),
    # ]
    log_dirs = [
      # os.path.join(experiments_dir, '2022-06-07_experiment_S00', '2022-06-07_17-18-17_actionNet-wearables_S00'),
      # os.path.join(experiments_dir, '2022-06-07_experiment_S00', '2022-06-07_18-10-55_actionNet-wearables_S00'),
      os.path.join(experiments_dir, '2022-06-13_experiment_S01_recordingStopped', '2022-06-13_18-13-12_actionNet-wearables_S01'),
      os.path.join(experiments_dir, '2022-06-13_experiment_S02', '2022-06-13_21-39-50_actionNet-wearables_S02'),
      # os.path.join(experiments_dir, '2022-06-13_experiment_S02', '2022-06-13_21-47-57_actionNet-wearables_S02'),
      # os.path.join(experiments_dir, '2022-06-13_experiment_S02', '2022-06-13_22-34-45_actionNet-wearables_S02'),
      # os.path.join(experiments_dir, '2022-06-13_experiment_S02', '2022-06-13_23-16-47_actionNet-wearables_S02'),
      # os.path.join(experiments_dir, '2022-06-13_experiment_S02', '2022-06-13_23-22-21_actionNet-wearables_S02'),
      # os.path.join(experiments_dir, '2022-06-14_experiment_S03', '2022-06-14_13-01-32_actionNet-wearables_S03'),
      # os.path.join(experiments_dir, '2022-06-14_experiment_S03', '2022-06-14_13-11-44_actionNet-wearables_S03'),
      # os.path.join(experiments_dir, '2022-06-14_experiment_S03', '2022-06-14_13-52-21_actionNet-wearables_S03'),
      # os.path.join(experiments_dir, '2022-06-14_experiment_S04', '2022-06-14_16-38-18_actionNet-wearables_S04'),
      # os.path.join(experiments_dir, '2022-06-14_experiment_S05', '2022-06-14_20-36-27_actionNet-wearables_S05'),
      # os.path.join(experiments_dir, '2022-06-14_experiment_S05', '2022-06-14_20-45-43_actionNet-wearables_S05'),
    ]
  else:
    log_dirs = sys.argv[1:]
  
  # start_end_times = [['12:30', '16:45'], ['5:34', '13:32'], ['1:40', '13:15'], ['2:03', '3:55']]
  start_offsets_s = [None]*len(log_dirs) # for 2022-05-24: [12*60+30] # 5*60+34, 1*60+40, 2*60+3
  end_offsets_s = [None]*len(log_dirs) # for 2022-05-24: [(22*60+47) - (16*60+45)] # (14*60+24) - (13*60+32), (14*60+1) - (13*60+15), (5*60+58) - (3*60+55)

  # Loop through each specified log directory to process.
  for (i, log_dir) in enumerate(log_dirs):
    print('\n\n' + '='*75)
    print('Creating composite visualization for log directory %s\n' % log_dir)
    print()

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
                                  'composite_visualization_postProcessed_0-1s')

    # Create a sensor manager.
    sensor_manager = SensorManager(sensor_streamer_specs=None,
                                   log_player_options=log_player_options,
                                   data_logger_options=datalogging_options,
                                   data_visualizer_options=visualization_options,
                                   kill_other_python_processes=False, # may be launching multiple visualizers
                                   print_status=print_status, print_debug=print_debug)

    # Load streams from the saved logs for each streamer.
    sensor_manager.connect()

    # Visualize!
    frame_size = (768, 1280) # height, width # (1800, 3000)
    col_width = int(frame_size[1]/3)
    row_height = int(frame_size[0]/3)
    visualizer = DataVisualizer(sensor_streamers=sensor_manager.get_streamers(),
                                update_period_s = 0.1,
                                use_composite_video=True,
                                composite_video_layout = [
                                  [ # row 0
                                    {'device_name':'tactile-glove-left', 'stream_name':'tactile_data',    'rowspan':1, 'colspan':1, 'width':col_width, 'height':row_height},
                                    {'device_name':'eye-tracking-video-worldGaze', 'stream_name':'frame', 'rowspan':1, 'colspan':1, 'width':col_width, 'height':row_height},
                                    {'device_name':'tactile-glove-right', 'stream_name':'tactile_data',   'rowspan':1, 'colspan':1, 'width':col_width, 'height':row_height},
                                    # {'device_name':'dummy-line', 'stream_name':'dummy-stream',    'width':1000, 'height':600},
                                    # {'device_name':'dummy-line', 'stream_name':'dummy-stream', 'width':1000, 'height':600},
                                    # {'device_name':'dummy-line', 'stream_name':'dummy-stream',   'width':1000, 'height':600},
                                  ],
                                  [ # row  1
                                    {'device_name':'myo-left', 'stream_name':'emg',               'rowspan':1, 'colspan':1, 'width':col_width, 'height':   row_height},
                                    {'device_name':'xsens-segments', 'stream_name':'position_cm', 'rowspan':2, 'colspan':1, 'width':col_width, 'height': 2*row_height},
                                    {'device_name':'myo-right', 'stream_name':'emg',              'rowspan':1, 'colspan':1, 'width':col_width, 'height':   row_height},
                                  ],
                                  [ # row 2
                                    {'device_name':'myo-left', 'stream_name':'acceleration_g',  'rowspan':1, 'colspan':1, 'width':col_width, 'height': row_height},
                                    {'device_name':None, 'stream_name':None,                    'rowspan':0, 'colspan':0, 'width':        0, 'height':          0},
                                    {'device_name':'myo-right', 'stream_name':'acceleration_g', 'rowspan':1, 'colspan':1, 'width':col_width, 'height': row_height},
                                  ],
                                ],
                                composite_video_filepath = composite_video_filepath,
                                print_status=print_status, print_debug=print_debug)
    visualizer.visualize_logged_data(start_offset_s=None, end_offset_s=end_offset_s,
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





