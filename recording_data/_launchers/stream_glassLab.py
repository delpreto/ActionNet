############
#
# Copyright (c) 2024 MIT CSAIL and Joseph DelPreto
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
# See https://action-sense.csail.mit.edu for more usage information.
# Created 2021-2022 for the MIT ActionSense project by Joseph DelPreto [https://josephdelpreto.com].
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
  
  # TODO: Define the streamers to use.
  #   Then configure settings for each class in sensor_streamer_specs below.
  sensor_streamers_enabled = dict([
    # Use one of the following to control the experiment (enter notes, quit, etc)
    ('ExperimentControlStreamer', True),  # A GUI to label activities/calibrations and enter notes
    ('NotesStreamer',             False),  # A command-line based way to submit notes during the experiment (but not label activities explicitly)
    # Sensors!
    ('MyoStreamer',        True),  # One or more Myo EMG/IMU armbands
    ('TouchStreamer',      False),  # Custom tactile sensors streaming via an Arduino
    ('XsensStreamer',      False),  # The Xsens body tracking system (includes the Manus finger-tracking gloves if connected to Xsens)
    ('EyeStreamer',        False),  # The Pupil Labs eye-tracking headset
    ('ScaleStreamer',      False),  # The Dymo M25 digital postal scale
    ('MoticonStreamer',    False),  # Moticon insole pressure sensors
    ('MicrophoneStreamer', False),  # One or more microphones
    ('CameraStreamer',     True),  # One or more cameras
    ('SerialStreamer',     False),  # One or more serial streams
    ('ResistanceStreamer', False),  # One or more resistance readings
    ('DummyStreamer',      False),  # Dummy data (no hardware required)
  ])
  sensor_streamer_specs = [
    # Allow the experimenter to label data and enter notes.
    {'class': 'ExperimentControlStreamer',
     'activities': [ # TODO: Enter activities that you want to label during the experiment
       'Heat glass',
       'Rotate glass',
       'Shape glass',
       'Kiln',
       'Soak',
       'Apply burn cream',
     ],
     'print_debug': print_debug, 'print_status': print_status
     },
    # Allow the experimenter to record timestamped notes at any time.
    {'class': 'NotesStreamer',
     'print_debug': print_debug, 'print_status': print_status
     },
    # Stream from the Myo device including EMG, IMU, and gestures.
    # TODO specify the number of Myos that are connected via Myo Connect
    {'class': 'MyoStreamer',
     'num_myos': 1,
     'print_debug': print_debug, 'print_status': print_status
     },
    # Stream from the Xsens body tracking and Manus gloves.
    {'class': 'XsensStreamer',
     'print_debug': print_debug, 'print_status': print_status
     },
    # Stream from one or more tactile sensors, such as the ones on the gloves.
    # See the __init__ method of TouchStreamer to configure settings such as
    #  what sensors are available and their COM ports.
    {'class': 'TouchStreamer',
     'print_debug': print_debug, 'print_status': print_status
     },
    # Stream from the Pupil Labs eye tracker, including gaze and video data.
    {'class': 'EyeStreamer',
     'stream_video_world'    : False, # the world video
     'stream_video_worldGaze': True, # the world video with gaze indication overlayed
     'stream_video_eye'      : False, # video of the eye
     'print_debug': print_debug, 'print_status': print_status
     },
    # Stream from the Dymo M25 scale.
    {'class': 'ScaleStreamer',
     'print_debug': print_debug, 'print_status': print_status
     },
    # Stream from one or more microphones.
    {'class': 'MicrophoneStreamer',
     'device_names_withAudioKeywords': {'microphone_conference': 'USB audio CODEC'},
     'print_debug': print_debug, 'print_status': print_status
     },
    # Stream from one or more cameras.
    # TODO specify the device ID of the camera to stream.
    #  A built-in laptop camera is 0, and a USB camera is 1 (or higher if multiple cameras are connected).
    {'class': 'CameraStreamer',
     'cameras_to_stream': { # map camera names (usable as device names in the HDF5 file) to capture device indexes
       'laptop_camera': 0,
     },
     'print_debug': print_debug, 'print_status': print_status
     },
    # Stream from one or more generic serial streams.
    {'class': 'SerialStreamer',
     'com_ports':         {'all-serial-sensors': 'COM11'},
     'baud_rates_bps':    {'all-serial-sensors': 1000000},
     'sampling_rates_hz': {'all-serial-sensors': 55},
     'sample_sizes':      {'all-serial-sensors': [2]},
     'value_delimiters':  {'all-serial-sensors': ' '},
     'sensors_send_debug_values': {'all-serial-sensors': False},
     },
    # Stream from one or more resistance voltage dividers.
    {'class': 'ResistanceStreamer',
     'com_ports':             {'electrode-resistance': 'COM5'},
     'baud_rates_bps':        {'electrode-resistance': 1000000},
     'sampling_rates_hz':     {'electrode-resistance': 100},
     'divider_resistors_ohm': {'electrode-resistance': 1000},
     'divider_references_v':  {'electrode-resistance': 3.3},
     'adc_maxes':             {'electrode-resistance': 4095},
     'sample_sizes':          {'electrode-resistance': [1]},
     'value_delimiters':      {'electrode-resistance': ' '},
     'sensors_send_debug_values': {'electrode-resistance': False},
     },
    # Dummy data.
    {'class': 'DummyStreamer',
     'update_period_s': 0.1,
     'print_debug': print_debug, 'print_status': print_status
     },
  ]
  # Remove disabled streamers.
  sensor_streamer_specs = [spec for spec in sensor_streamer_specs
                           if spec['class'] in sensor_streamers_enabled
                           and sensor_streamers_enabled[spec['class']]]
  
  # TODO: Configure where and how to save sensor data.
  #       Adjust "enable_data_logging", "log_dir_root", "log_tag" as desired.
  #       log_dir_root is the folder to save data.
  #         A subfolder will be created in here each time the program is run.
  #       log_tag is an identifier for the upcoming series of runs.
  #         The subfolders will have this tag in the name, preceded by the current date/time.
  enable_data_logging = True # If False, no data will be logged and the below directory settings will be ignored
  if enable_data_logging:
    script_dir = os.path.dirname(os.path.realpath(__file__))
    (log_time_str, log_time_s) = get_time_str(return_time_s=True)
    log_dir_root = os.path.join(script_dir, '..', '..', 'data',
                                'tests', # recommend 'tests' and 'experiments' for testing vs "real" data
                                '%s_testing_glassLab_sensors' % get_time_str(format='%Y-%m-%d'))
    log_tag = 'testing_glassLab_sensors'
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
      'stream_video': True,
      'stream_audio': True,
      'stream_period_s': 300, # Will dump data from memory to files with this period
      'clear_logged_data_from_memory': True, # ignored if dumping is also enabled below
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
  
  # TODO: Configure visualization.
  #   This composite video will be shown during the experiment and also saved as a video file.
  #   There is a post-processing script to generate these videos at a higher frame rate if desired (or adjust the layout, streams, formats, etc).
  #
  composite_frame_size = (1200, 1800) # height, width # TODO adjust so the video looks reasonable on your screen
  composite_col_width = int(composite_frame_size[1]/2) # the column size of the grid of plots; the divisor should match the number of columns you specify below
  composite_row_height = int(composite_frame_size[0]/2) # the row size of the grid of plots; the divisor should match the number of rows you specify below
  visualization_options = {
    'visualize_streaming_data'       : True,
    'visualize_all_data_when_stopped': False,
    'wait_while_visualization_windows_open': False,
    'update_period_s': 0.25, # defines the frame rate of the video (if too fast, it won't keep up in real time)
    'use_composite_video': True,
    'composite_video_filepath': os.path.join(log_dir, 'composite_visualization') if log_dir is not None else None,
    # Specify what streams to show and how to lay them out in a grid.
    'composite_video_layout':
      [
        [ # row 0; each entry is a column in the first row
          {'device_name':'myo-left', 'stream_name':'emg', 'rowspan':1, 'colspan':1, 'width':composite_col_width, 'height':   composite_row_height},
          {'device_name':'myo-right', 'stream_name':'emg', 'rowspan':1, 'colspan':1, 'width':composite_col_width, 'height':   composite_row_height},
          # {'device_name':'eye-tracking-video-worldGaze', 'stream_name':'frame', 'rowspan':1, 'colspan':1, 'width':composite_col_width, 'height':composite_row_height},
        ],
        [ # row 1; each entry is a column in the second row
          {'device_name':'laptop_camera', 'stream_name':'frame', 'rowspan':1, 'colspan':1, 'width':composite_col_width, 'height':composite_row_height},
          {'device_name':'dummy', 'stream_name':'nothing', 'rowspan':1, 'colspan':1, 'width':composite_col_width, 'height':composite_row_height},
          # {'device_name':'all-serial-sensors', 'stream_name':'serial_data', 'rowspan':1, 'colspan':1, 'width':composite_col_width, 'height':composite_row_height},
        ],
      ],
  }
  
  # Create a sensor manager.
  sensor_manager = SensorManager(sensor_streamer_specs=sensor_streamer_specs,
                                 data_logger_options=datalogging_options,
                                 data_visualizer_options=visualization_options,
                                 print_status=print_status, print_debug=print_debug,
                                 log_history_filepath=log_history_filepath)
  
  # Define a callback to print FPS for a certain device.
  # print_fps = False # Use this to disable FPS printing
  classes_to_exclude_for_fps = ['ExperimentControlStreamer', 'NotesStreamer']
  streamers_for_fps = sensor_manager.get_streamers(class_name=None)
  streamers_for_fps = [streamer for streamer in streamers_for_fps if True not in [exclude in type(streamer).__name__ for exclude in classes_to_exclude_for_fps]]
  fps_start_time_s = [None]*len(streamers_for_fps)
  fps_start_num_timesteps = [0]*len(streamers_for_fps)
  fps_num_timesteps = [0]*len(streamers_for_fps)
  fps_last_print_time_s = 0
  def print_fps():
    global fps_start_time_s, fps_last_print_time_s, fps_start_num_timesteps, fps_num_timesteps
    printed_fps = False
    for (streamer_index, streamer) in enumerate(streamers_for_fps):
      device_for_fps = streamer.get_device_names()[0]
      stream_for_fps = streamer.get_stream_names(device_for_fps)[0]
      num_timesteps = streamer.get_num_timesteps(device_for_fps, stream_for_fps)
      if fps_start_time_s[streamer_index] is None or num_timesteps < fps_num_timesteps[streamer_index]:
        fps_start_time_s[streamer_index] = time.time()
        fps_start_num_timesteps[streamer_index] = num_timesteps
        fps_num_timesteps[streamer_index] = num_timesteps - fps_start_num_timesteps[streamer_index]
        fps_last_print_time_s = time.time()
      elif time.time() - fps_last_print_time_s > 5:
        printed_fps = True
        fps_duration_s = time.time() - fps_start_time_s[streamer_index]
        fps_num_timesteps[streamer_index] = num_timesteps - fps_start_num_timesteps[streamer_index]
        _log_status('Status: %5.1f Hz (%4d timesteps in %6.2fs) for %s: %s' %
                    ((fps_num_timesteps[streamer_index]-1)/fps_duration_s,
                     fps_num_timesteps[streamer_index], fps_duration_s,
                     device_for_fps, stream_for_fps))
    if printed_fps:
      fps_last_print_time_s = time.time()
  # Define a callback to print averaged data when notes are entered.
  average_duration_s = 15
  prev_num_notes = 0
  control_streamer = sensor_manager.get_streamers(class_name='ExperimentControlStreamer')[0]
  # serial_streamer = sensor_manager.get_streamers(class_name='SerialStreamer')[0]
  def print_data():
    global average_duration_s, prev_num_notes, control_streamer#, serial_streamer
    notes = control_streamer.get_data('experiment-notes', 'notes')
    if notes is None:
      prev_num_notes = 0
      return
    num_notes = len(notes['data'])
    if num_notes > prev_num_notes:
      note = notes['data'][-1]
      note_t = notes['time_s'][-1]
      # serial_data = serial_streamer.get_data(device_name='all-sensors', stream_name='serial_data',
      #                                        starting_time_s=note_t-average_duration_s)
      # # serial_t = np.squeeze(np.array(serial_data['time_s']))
      # serial_data = np.squeeze(np.array(serial_data['data']))
      # print(serial_data.shape)
      # # indexes = np.where(serial_t >= note_t - average_duration_s)[0]
      # # serial_data_averaged = np.mean(serial_data[indexes, :], dim=0)
      # serial_data_averaged = np.mean(serial_data, axis=0)
      # serial_data_averaged_std = np.std(serial_data, axis=0)
      # serial_data_averaged_str = str(list(serial_data_averaged)).replace(', ', '\t').replace('[','').replace(']','')
      # serial_data_averaged_std_str = str(list(serial_data_averaged_std)).replace(', ', '\t').replace('[','').replace(']','')
      # _log_status('\t%f\t%s\t%s\t%s' % (note_t, note, serial_data_averaged_str, serial_data_averaged_std_str))
      _log_status('\t%f\t%s' % (note_t, note,))
    prev_num_notes = num_notes
  
  # Define a callback that checks whether the user has entered a quit keyword.
  try:
    control_streamer = sensor_manager.get_streamers(class_name='ExperimentControlStreamer')[0]
    def check_if_user_quit():
      if callable(print_fps):
        print_fps()
      if callable(print_data):
        print_data()
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
  sensor_manager.run(duration_s=36000, stopping_condition_fn=check_if_user_quit)
  sensor_manager.stop()




