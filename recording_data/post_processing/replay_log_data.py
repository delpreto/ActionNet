
from sensor_streamer_handlers.SensorManager import SensorManager

import time
import os
import traceback
from utils.time_utils import *

# Note that multiprocessing requires the __main__ check.
if __name__ == '__main__':
  # Configure printing to the console.
  print_status = True
  print_debug = False

  # Define the log to replay.
  log_player_options = {
    # 'log_dir': 'C:/Users/jdelp/Desktop/MIT/Lab/Wearativity/code/data_logs/2021-12-21_01-01-16_testing_all - Copy',
    # 'log_dir': 'C:/Users/jdelp/Desktop/MIT/Lab/Wearativity/code/data_logs/2022-01-07_14-59-11_testing - Copy',
    # 'log_dir': 'C:/Users/jdelp/Desktop/MIT/Lab/Wearativity/code/data_logs/2022-01-07_14-42-33_testing - Copy',
    'log_dir': 'P:/MIT/Lab/Wearativity/data/2022-01-24 test all worn/2022-01-24_19-37-17_notes-xsens-gloves-eye-touch-myo_good',
    'pause_to_replay_in_realtime': True,
    'skip_timesteps_to_replay_in_realtime': True,
    'load_datasets_into_memory': False
  }

  # Configure where and how to save sensor data.
  datalogging_options = None

  # Configure visualization.
  visualization_options = {
    'visualize_streaming_data'       : True,
    'visualize_all_data_when_stopped': False,
    'wait_while_visualization_windows_open': True,
    'update_period_s': 0.5,
    'classes_to_visualize': [
      # 'DummyStreamer',
      'MyoStreamer',
      'EyeStreamer',
      'TouchStreamer',
      'XsensStreamer',
      ],
  }

  # Create a sensor manager.
  sensor_manager = SensorManager(sensor_streamer_specs=None,
                                 log_player_options=log_player_options,
                                 data_logger_options=datalogging_options,
                                 data_visualizer_options=visualization_options,
                                 print_status=print_status, print_debug=print_debug)

  # Run!
  # Will wait for the log to finish replaying or for a timeout duration.
  sensor_manager.connect()
  print('\n'*5)
  sensor_manager.run(duration_s=1200)
  sensor_manager.stop()




