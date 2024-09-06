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
    data_dir = os.path.realpath(os.path.join(script_dir, '..', '..', '..', 'data'))
    experiments_dir = os.path.join(data_dir, 'tests')
    log_dirs = [
      os.path.join(experiments_dir, '2024-09-06_testing_robotHand_setup',
                   '2024-09-06_19-17-11_testing_robotHand_setup'),
    ]
  else:
    log_dirs = sys.argv[1:]

  # Define offsets for start/end of processing.
  #  Should be a list with an entry for each log directory.
  #  Each entry can be a time in seconds, or None to use all log data.
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
                                  'composite_visualization_postProcessed_10fps')

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
    composite_frame_size = (1800, 3000) # height, width # (1800, 3000)
    composite_col_width = int(composite_frame_size[1]/2)
    composite_row_height = int(composite_frame_size[0]/2)
    visualizer = DataVisualizer(sensor_streamers=sensor_manager.get_streamers(),
                                update_period_s = 0.1,
                                use_composite_video=True,
                                composite_video_layout = [
                                  [ # row 0
                                    {'device_name':'myo-left', 'stream_name':'emg', 'rowspan':1, 'colspan':1, 'width':composite_col_width, 'height':   composite_row_height},
                                    {'device_name':'camera', 'stream_name':'frame', 'rowspan':1, 'colspan':1, 'width':composite_col_width, 'height':composite_row_height},
                                  ],
                                  [ # row 1
                                    {'device_name':'myoProcessed-left', 'stream_name':'emg_envelope', 'rowspan':1, 'colspan':1, 'width':composite_col_width, 'height':   composite_row_height},
                                    {'device_name':'myoProcessed-left', 'stream_name':'emg_stiffness', 'rowspan':1, 'colspan':1, 'width':composite_col_width, 'height':   composite_row_height},
                                  ],
                                ],
                                #   [ # row  0
                                #     {'device_name':'xsens-segments', 'stream_name':'position_cm', 'rowspan':1, 'colspan':1, 'width':composite_col_widths[0], 'height': composite_row_height},
                                #     {'device_name':'shear-shoe-right', 'stream_name':'tactile_data', 'rowspan':1, 'colspan':1, 'width':composite_col_widths[1], 'height':composite_row_height},
                                #   ],
                                #   [ # row  1
                                #     {'device_name':'shear-shoe-right', 'stream_name':'tactile_tiled', 'rowspan':1, 'colspan':1, 'width':composite_col_widths[0], 'height':composite_row_height},
                                #     {'device_name':'shear-shoe-right', 'stream_name':'force_vector', 'rowspan':1, 'colspan':1, 'width':composite_col_widths[1], 'height':composite_row_height},
                                #     # {'device_name':'dummy', 'stream_name':'nothing', 'rowspan':1, 'colspan':1, 'width':composite_col_width, 'height':composite_row_height},
                                #   ],
                                # ],
                                # composite_video_layout = [
                                #   [ # row  0
                                #     {'device_name':'xsens-segments', 'stream_name':'position_cm', 'rowspan':1, 'colspan':1, 'width':composite_col_widths[0], 'height': composite_row_height},
                                #     {'device_name':'tactile-shoe-right', 'stream_name':'tactile_data',    'rowspan':1, 'colspan':1, 'width':composite_col_widths[1], 'height':composite_row_height},
                                #   ],
                                # ],
                                # composite_video_layout = [
                                #   [ # row  0
                                #     # {'device_name':'dummy', 'stream_name':'nothing', 'rowspan':1, 'colspan':1, 'width':composite_col_width, 'height':composite_row_height},
                                #     {'device_name':'video', 'stream_name':'frame', 'rowspan':1, 'colspan':1, 'width':composite_col_width, 'height':composite_row_height},
                                #     {'device_name':'shear-sensor', 'stream_name':'tactile_data', 'rowspan':1, 'colspan':1, 'width':composite_col_width, 'height':composite_row_height},
                                #   ],
                                #   [ # row  1
                                #     {'device_name':'shear-sensor', 'stream_name':'tactile_tiled', 'rowspan':1, 'colspan':1, 'width':composite_col_width, 'height':composite_row_height},
                                #     {'device_name':'shear-sensor', 'stream_name':'force_vector', 'rowspan':1, 'colspan':1, 'width':composite_col_width, 'height':composite_row_height},
                                #     # {'device_name':'dummy', 'stream_name':'nothing', 'rowspan':1, 'colspan':1, 'width':composite_col_width, 'height':composite_row_height},
                                #   ],
                                # ],
                                composite_video_filepath=composite_video_filepath,
                                print_status=print_status, print_debug=print_debug)
    # frame_size = (760, 1280) # height, width # (1800, 3000)
    # composite_col_width = int(frame_size[1] / 3)
    # composite_row_height = int(frame_size[0] / 3)
    # visualizer = DataVisualizer(sensor_streamers=sensor_manager.get_streamers(),
    #                             update_period_s = 0.1,
    #                             use_composite_video=True,
    #                             composite_video_layout = [
    #                               [ # row 0
    #                                 {'device_name':'tactile-glove-left', 'stream_name':'tactile_data',    'rowspan':1, 'colspan':1, 'width':col_width, 'height':row_height},
    #                                 {'device_name':'eye-tracking-video-worldGaze', 'stream_name':'frame', 'rowspan':1, 'colspan':1, 'width':col_width, 'height':row_height},
    #                                 {'device_name':'tactile-glove-right', 'stream_name':'tactile_data',   'rowspan':1, 'colspan':1, 'width':col_width, 'height':row_height},
    #                               ],
    #                               [ # row  1
    #                                 {'device_name':'myo-left', 'stream_name':'emg',               'rowspan':1, 'colspan':1, 'width':col_width, 'height':   row_height},
    #                                 {'device_name':'xsens-segments', 'stream_name':'position_cm', 'rowspan':2, 'colspan':1, 'width':col_width, 'height': 2*row_height},
    #                                 {'device_name':'myo-right', 'stream_name':'emg',              'rowspan':1, 'colspan':1, 'width':col_width, 'height':   row_height},
    #                               ],
    #                               [ # row 2
    #                                 {'device_name':'myo-left', 'stream_name':'acceleration_g',  'rowspan':1, 'colspan':1, 'width':col_width, 'height': row_height},
    #                                 {'device_name':None, 'stream_name':None,                    'rowspan':0, 'colspan':0, 'width':        0, 'height':          0},
    #                                 {'device_name':'myo-right', 'stream_name':'acceleration_g', 'rowspan':1, 'colspan':1, 'width':col_width, 'height': row_height},
    #                               ],
    #                             ],
    #                             composite_video_filepath=composite_video_filepath,
    #                             print_status=print_status, print_debug=print_debug)
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





