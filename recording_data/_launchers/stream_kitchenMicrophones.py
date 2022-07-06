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

import time
import os
import traceback
from utils.time_utils import *
from utils.print_utils import *

# Note that multiprocessing requires the __main__ check.
if __name__ == '__main__':
  # Configure printing and logging.
  print_status = True
  print_debug = False
  
  # Helper methods for logging/printing.
  def _log_status(msg, *extra_msgs, **kwargs):
    write_log_message(msg, *extra_msgs, source_tag='launcher',
                      print_message=print_status, filepath=log_history_filepath, **kwargs)
  def _log_debug(msg, *extra_msgs, **kwargs):
    write_log_message(msg, *extra_msgs, source_tag='launcher',
                      print_message=print_debug, debug=True, filepath=log_history_filepath, **kwargs)
  def _log_error(msg, *extra_msgs, **kwargs):
    write_log_message(msg, *extra_msgs, source_tag='launcher',
                      print_message=True, error=True, filepath=log_history_filepath, **kwargs)
  def _log_warn(msg, *extra_msgs, **kwargs):
    write_log_message(msg, *extra_msgs, source_tag='launcher',
                      print_message=True, warning=True, filepath=log_history_filepath, **kwargs)
  def _log_userAction(msg, *extra_msgs, **kwargs):
    write_log_message(msg, *extra_msgs, source_tag='launcher',
                      print_message=True, userAction=True, filepath=log_history_filepath, **kwargs)
  
  # Define the streamers to use.
  sensor_streamer_specs = [
    # # Allow the experimenter to label data and enter notes.
    # {'class': 'ExperimentControlStreamer',
    #  'print_debug': print_debug, 'print_status': print_status
    #  },
    # Allow the experimenter to record timestamped notes at any time.
    {'class': 'NotesStreamer',
     'print_debug': print_debug, 'print_status': print_status
     },
    # # Stream from the Myo device including EMG, IMU, and gestures.
    # {'class': 'MyoStreamer',
    #  'num_myos': 1,
    #  'print_debug': print_debug, 'print_status': print_status
    #  },
    # # Stream from the Xsens body tracking and Manus gloves.
    # {'class': 'XsensStreamer',
    #  'print_debug': print_debug, 'print_status': print_status
    #  },
    # # Stream from one or more tactile sensors, such as the ones on the gloves.
    # # See the __init__ method of TouchStreamer to configure settings such as
    # #  what sensors are available and their COM ports.
    # {'class': 'TouchStreamer',
    #  'print_debug': print_debug, 'print_status': print_status
    #  },
    # # Stream from the Pupil Labs eye tracker, including gaze and video data.
    # {'class': 'EyeStreamer',
    #  'stream_video_world'    : False, # the world video
    #  'stream_video_worldGaze': True, # the world video with gaze indication overlayed
    #  'stream_video_eye'      : False, # video of the eye
    #  'print_debug': print_debug, 'print_status': print_status
    #  },
    # # Stream from the Dymo M25 scale.
    # {'class': 'ScaleStreamer',
    #  'print_debug': print_debug, 'print_status': print_status
    #  },
    # Stream from one or more microphones.
    {'class': 'MicrophoneStreamer',
     'device_names_withAudioKeywords':
       {
         'overhead': '(USB audio CODEC)',
         'sink'    : '(USB PnP Audio Device)',
       },
     'print_debug': print_debug, 'print_status': print_status
     },
    # # Dummy data.
    # {'class': 'DummyStreamer',
    #  'update_period_s': 2,
    #  'print_debug': print_debug, 'print_status': print_status
    #  },
  ]
  
  # Configure where and how to save sensor data.
  enable_data_logging = True
  if enable_data_logging:
    script_dir = os.path.dirname(os.path.realpath(__file__))
    (log_time_str, log_time_s) = get_time_str(return_time_s=True)
    log_tag = 'test-mic'
    log_dir_root = os.path.join(script_dir, '..', '..', 'data', '2022-04-13 test microphones in kitchen')
    log_subdir = '%s_%s' % (log_time_str, log_tag)
    log_dir = os.path.join(log_dir_root, log_subdir)
    datalogging_options = {
      'log_dir': log_dir, 'log_tag': log_tag,
      'use_external_recording_sources': True,
      'videos_in_hdf5': False,
      'audio_in_hdf5': False,
      # Choose whether to periodically write data to files.
      'stream_csv'  : False,
      'stream_hdf5' : True,
      'stream_video': False,
      'stream_audio': True,
      'stream_period_s': 15,
      'clear_logged_data_from_memory': True, # ignored if dumping is also enabled
      # Choose whether to write all data at the end.
      'dump_csv'  : False,
      'dump_hdf5' : False,
      'dump_video': False,
      'dump_audio': False,
      # Additional configuration.
      'videos_format': 'avi', # mp4 occasionally gets openCV errors about a tag not being supported?
      'audio_format' : 'wav', # currently only supports WAV
      'print_status': print_status, 'print_debug': print_debug
    }
    # Initialize a file for writing the log history of all printouts/messages.
    log_history_filepath = os.path.join(log_dir, '%s_log_history.txt' % log_time_str)
    os.makedirs(log_dir, exist_ok=True)
  else:
    log_dir = None
    log_history_filepath = None
    datalogging_options = None
  
  # Configure visualization.
  # visualization_options = None
  visualization_options = {
    'visualize_streaming_data'       : False,
    'visualize_all_data_when_stopped': False,
    'wait_while_visualization_windows_open': True,
    'update_period_s': 0.5,
    # 'classes_to_visualize': ['TouchStreamer']
  }
  
  # Create a sensor manager.
  sensor_manager = SensorManager(sensor_streamer_specs=sensor_streamer_specs,
                                 data_logger_options=datalogging_options,
                                 data_visualizer_options=visualization_options,
                                 print_status=print_status, print_debug=print_debug,
                                 log_history_filepath=log_history_filepath)
  
  # Define a callback to print FPS for a certain device.
  # print_fps = False # Use this to disable FPS printing
  streamer_for_fps = sensor_manager.get_streamers(class_name=None)[-1]
  fps_start_time_s = None
  fps_start_num_timesteps = 0
  fps_num_timesteps = 0
  fps_last_print_time_s = 0
  def print_fps():
    global fps_start_time_s, fps_last_print_time_s, fps_start_num_timesteps, fps_num_timesteps
    device_for_fps = streamer_for_fps.get_device_names()[0]
    stream_for_fps = streamer_for_fps.get_stream_names(device_for_fps)[0]
    num_timesteps = streamer_for_fps.get_num_timesteps(device_for_fps, stream_for_fps)
    if fps_start_time_s is None or num_timesteps < fps_num_timesteps:
      fps_start_time_s = time.time()
      fps_start_num_timesteps = num_timesteps
      fps_num_timesteps = num_timesteps - fps_start_num_timesteps
      fps_last_print_time_s = time.time()
    elif time.time() - fps_last_print_time_s > 5:
      fps_duration_s = time.time() - fps_start_time_s
      fps_num_timesteps = num_timesteps - fps_start_num_timesteps
      _log_status('Status for %s %s: %4d timesteps in %0.2fs -> %0.1f Hz' %
                  (device_for_fps, stream_for_fps, fps_num_timesteps,
                   fps_duration_s, (fps_num_timesteps-1)/fps_duration_s))
      fps_last_print_time_s = time.time()
  
  # Define a callback that checks whether the user has entered a quit keyword.
  try:
    control_streamer = sensor_manager.get_streamers(class_name='ExperimentControlStreamer')[0]
    def check_if_user_quit():
      if callable(print_fps):
        print_fps()
      return not control_streamer.experiment_is_running()
  except:
    try:
      notes_streamer = sensor_manager.get_streamers(class_name='NotesStreamer')[0]
      def check_if_user_quit():
        last_notes = notes_streamer.get_last_notes()
        if last_notes is not None:
          last_notes = last_notes.lower().strip()
        if callable(print_fps):
          print_fps()
        return last_notes in ['quit', 'q']
    except:
      def check_if_user_quit():
        return False
  
  
  # print()
  # print('Enter \'quit\' or \'q\' as an experiment note to end the program')
  # print()
  
  # Run!
  sensor_manager.connect()
  sensor_manager.run(duration_s=7200, stopping_condition_fn=check_if_user_quit)
  sensor_manager.stop()




