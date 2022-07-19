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


    # TODO: Define the streamers to use.
    #   Configure settings for each class in sensor_streamer_specs.
    sensor_streamers_enabled = dict([
        # Use one of the following to control the experiment (enter notes, quit, etc)
        ('ExperimentControlStreamer', True),  # A GUI to label activities/calibrations and enter notes
        ('NotesStreamer',             False),  # A command-line based way to submit notes during the experiment (but not label activities explicitly)
        # Sensors!
        ('MyoStreamer',        True),  # One or more Myo EMG/IMU armbands
        ('TouchStreamer',      False),  # Custom tactile sensors streaming via an Arduino
        ('XsensStreamer',      True),  # The Xsens body tracking system (includes the Manus finger-tracking gloves if connected to Xsens)
        ('EyeStreamer',        True),  # The Pupil Labs eye-tracking headset
        ('E4Streamer',         True),  # Moticon insole pressure sensors
        ('MoticonStreamer',    True),  # Moticon insole pressure sensors
        ('ScaleStreamer',      False),  # The Dymo M25 digital postal scale
        ('MicrophoneStreamer', False),  # One or more microphones
        ('CameraStreamer',     True),  # One or more cameras
        ('DummyStreamer',      False),  # Dummy data (no hardware required)
    ])
    sensor_streamer_specs = [
        # Allow the experimenter to label data and enter notes.
        {'class': 'ExperimentControlStreamer',
         'activities': [ # TODO: Enter your activities that you want to label
             'Smashing (with birdie)',
             'Smashing (no birdie)',
             'Backhand',
             'Forehand',
         ],
         'print_debug': print_debug, 'print_status': print_status
         },
        # Allow the experimenter to record timestamped notes at any time.
        {'class': 'NotesStreamer',
         'print_debug': print_debug, 'print_status': print_status
         },
        # Moticon insole pressure sensors.
        {'class': 'MoticonStreamer',
         # Add any keyword arguments here that you added to __init__()
         'print_debug': print_debug, 'print_status': print_status
         },
        # The E4 smart watch.
        {'class': 'E4Streamer',
         # Add any keyword arguments here that you added to __init__()
         'print_debug': print_debug, 'print_status': print_status
         },
        # Stream from one or more tactile sensors, such as the ones on the gloves.
        # See the __init__ method of TouchStreamer to configure settings such as
        #  what sensors are available and their COM ports.
        {'class': 'TouchStreamer',
         'com_ports': {
           'tactile-glove-left' : 'COM3', # None
           'tactile-glove-right': 'COM6', # None
         },
         'print_debug': print_debug, 'print_status': print_status
         },
        # Stream from the Xsens body tracking and Manus gloves.
        {'class': 'XsensStreamer',
         'print_debug': print_debug, 'print_status': print_status
         },
        # Stream from the Myo device including EMG, IMU, and gestures.
        {'class': 'MyoStreamer',
         'num_myos': 2,
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
        {'class': 'CameraStreamer',
         'cameras_to_stream': { # map camera names (usable as device names in the HDF5 file) to capture device indexes
           'camera-usb': 1,
         },
         'print_debug': print_debug, 'print_status': print_status
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
    #       Adjust enable_data_logging, log_tag, and log_dir_root as desired.
    enable_data_logging = True # If False, no data will be logged and the below directory settings will be ignored
    if enable_data_logging:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        (log_time_str, log_time_s) = get_time_str(return_time_s=True)
        log_tag = 'testing in holodeck'
        log_dir_root = os.path.join(script_dir, '..', '..', 'data',
                                    'tests', # recommend 'tests' and 'experiments' for testing vs "real" data
                                    '%s_badminton_test' % get_time_str(format='%Y-%m-%d'))
        log_subdir = '%s_%s' % (log_time_str, log_tag)
        log_dir = os.path.join(log_dir_root, log_subdir)
        datalogging_options = {
            'log_dir': log_dir, 'log_tag': log_tag,
            'use_external_recording_sources': True,
            'videos_in_hdf5': False,
            'audio_in_hdf5': False,
            # Choose whether to periodically write data to files.
            'stream_hdf5': True,  # recommended over CSV since it creates a single file
            'stream_csv': False,  # will create a CSV per stream
            'stream_video': True,
            'stream_audio': True,
            'stream_period_s': 15,  # how often to save streamed data to disk
            'clear_logged_data_from_memory': True,  # ignored if dumping is also enabled below
            # Choose whether to write all data at the end.
            'dump_csv': False,
            'dump_hdf5': False,
            'dump_video': False,
            'dump_audio': False,
            # Additional configuration.
            'videos_format': 'avi',  # mp4 occasionally gets openCV errors about a tag not being supported?
            'audio_format': 'wav',  # currently only supports WAV
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
    composite_frame_size = (1800, 3000)  # height, width # (1800, 3000)
    composite_col_width = int(composite_frame_size[1] / 3)
    composite_row_height = int(composite_frame_size[0] / 4)
    visualization_options = {
        'visualize_streaming_data': True,
        'visualize_all_data_when_stopped': False,
        'wait_while_visualization_windows_open': False,
        'update_period_s': 0.5,
        # 'classes_to_visualize': ['TemplateStreamer']
        'use_composite_video': True,
        'composite_video_filepath': os.path.join(log_dir, 'composite_visualization') if log_dir is not None else None,
      'composite_video_layout':
        [
          [ # row 0
            {'device_name':'camera-usb',  'stream_name':'frame', 'rowspan':1, 'colspan':1, 'width':composite_col_width, 'height':composite_row_height},
            {'device_name':'xsens-segments', 'stream_name':'position_cm', 'rowspan':2, 'colspan':1, 'width':composite_col_width, 'height': 2*composite_row_height},
            {'device_name':'eye-tracking-video-worldGaze', 'stream_name':'frame', 'rowspan':1, 'colspan':1, 'width':composite_col_width, 'height':composite_row_height},
          ],
          [  # row  1
            {'device_name': 'insole-moticon-left', 'stream_name': 'pressure_values_N_cm2', 'rowspan': 1, 'colspan': 1, 'width': composite_col_width, 'height': composite_row_height},
            {'device_name':None, 'stream_name':None, 'rowspan':0, 'colspan':0, 'width':0, 'height': 0},
            {'device_name': 'insole-moticon-right', 'stream_name': 'pressure_values_N_cm2', 'rowspan': 1, 'colspan': 1, 'width': composite_col_width, 'height': composite_row_height},
    
            # {'device_name': 'xsens-segments', 'stream_name': 'position_cm', 'rowspan': 1, 'colspan': 1, 'width': composite_col_width, 'height': composite_row_height},
          ],
          [ # row 2
            {'device_name':'myo-left', 'stream_name':'emg',               'rowspan':1, 'colspan':1, 'width':composite_col_width, 'height':   composite_row_height},
            {'device_name': 'ACC-empatica_e4', 'stream_name': 'acc-values', 'rowspan': 1, 'colspan': 1, 'width': composite_col_width, 'height': composite_row_height},
            {'device_name':'myo-right', 'stream_name':'emg',              'rowspan':1, 'colspan':1, 'width':composite_col_width, 'height':   composite_row_height},
          ],
          [ # row 3
            {'device_name': 'BVP-empatica_e4', 'stream_name': 'bvp-values', 'rowspan': 1, 'colspan': 1, 'width': composite_col_width, 'height': composite_row_height},
            {'device_name': 'GSR-empatica_e4', 'stream_name': 'gsr-values', 'rowspan': 1, 'colspan': 1, 'width': composite_col_width, 'height': composite_row_height},
            {'device_name': 'Tmp-empatica_e4', 'stream_name': 'tmp-values', 'rowspan': 1, 'colspan': 1, 'width': composite_col_width, 'height': composite_row_height},
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
    streamers_for_fps = [streamer for streamer in streamers_for_fps if
                         True not in [exclude in type(streamer).__name__ for exclude in classes_to_exclude_for_fps]]
    fps_start_time_s = [None] * len(streamers_for_fps)
    fps_start_num_timesteps = [0] * len(streamers_for_fps)
    fps_num_timesteps = [0] * len(streamers_for_fps)
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
                            ((fps_num_timesteps[streamer_index] - 1) / fps_duration_s,
                             fps_num_timesteps[streamer_index], fps_duration_s,
                             device_for_fps, stream_for_fps))
        if printed_fps:
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

    # Run!
    sensor_manager.connect()
    sensor_manager.run(duration_s=36000, stopping_condition_fn=check_if_user_quit)
    sensor_manager.stop()




