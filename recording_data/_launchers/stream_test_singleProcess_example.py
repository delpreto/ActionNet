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

from sensor_streamer_handlers.DataLogger     import DataLogger
from sensor_streamer_handlers.DataVisualizer import DataVisualizer
from sensor_streamers.MyoStreamer   import MyoStreamer
from sensor_streamers.XsensStreamer import XsensStreamer
from sensor_streamers.TouchStreamer import TouchStreamer
from sensor_streamers.EyeStreamer   import EyeStreamer
from sensor_streamers.NotesStreamer import NotesStreamer
from sensor_streamers.ExperimentControlStreamer import ExperimentControlStreamer

import time
import os
from utils.time_utils import *


# Configure printing to the console.
print_debug = False
print_status = True

# Configure where to save sensor data.
script_dir = os.path.dirname(os.path.realpath(__file__))
(log_time_str, log_time_s) = get_time_str(return_time_s=True)
log_tag = 'test-experimentController_singleProcess'
log_dir_root = os.path.join(script_dir, '..', 'data', '2022-03-04 test experiment controller')
log_subdir = '%s_%s' % (log_time_str, log_tag)
log_dir = os.path.join(log_dir_root, log_subdir)


# Create the streamers.
control_streamer = ExperimentControlStreamer(print_debug=print_debug, print_status=print_status)
# notes_streamer = NotesStreamer(print_debug=print_debug, print_status=print_status)
# myo_streamer = MyoStreamer(num_myos=1, print_debug=print_debug, print_status=print_status)
# xsens_streamer = XsensStreamer(print_debug=print_debug, print_status=print_status)
# touch_streamer = TouchStreamer(print_debug=print_debug, print_status=print_status)
# eye_streamer = EyeStreamer(stream_video_world=True, stream_video_worldGaze=True,
#                             stream_video_eye=True,
#                             print_debug=print_debug, print_status=print_status)


# Create a logger to record the streaming data.
# Put data from all sensors into the same HDF5 file for now.
#   Alternatively, multiple DataLoggers could be created with
#   various subsets of the streamers.
# Note that the stream_* arguments are whether to periodically save data,
#   and the dump_* arguments are whether to wait until the end and then save all data.
#   Any combination of these options can be enabled.
streamers = [
    control_streamer,
    # notes_streamer,
    # myo_streamer,
    # xsens_streamer,
    # touch_streamer,
    # eye_streamer,
  ]
logger = DataLogger(streamers, log_dir=log_dir, log_tag=log_tag,
                    # Choose whether to periodically write data to files.
                    stream_csv=False, stream_hdf5=True, stream_video=True,
                    stream_period_s=5, clear_logged_data_from_memory=True,
                    # Choose whether to write all data at the end.
                    dump_csv=False, dump_hdf5=False, dump_video=False,
                    # Additional configuration.
                    videos_format='avi', # mp4 occasionally gets openCV errors about a tag not being supported?
                    print_status=print_status, print_debug=print_debug)

# Connect and start the streamers.
print()
print('Connecting and starting streamers')
for (streamer_index, streamer) in enumerate(streamers):
  if print_status: print('\nConnecting streamer %d/%d (class %s)' % (streamer_index+1, len(streamers), type(streamer).__name__))
  connected = streamer.connect()
  if not connected:
    raise AssertionError('Error connecting the streamer')
for (streamer_index, streamer) in enumerate(streamers):
  if print_status: print('Starting streamer %d/%d' % (streamer_index+1, len(streamers)))
  streamer.run()

# Create a visualizer to display data.
# visualizer = None
visualizer = DataVisualizer(streamers,
                            update_period_s=0.5,
                            print_status=print_status, print_debug=print_debug)

# Stream data!
print()
print('Starting data logger')
print('Enter \'quit\' or \'q\' as an experiment note to end the program')
print()
logger.run()

# Define a callback that checks whether the user has entered a quit keyword.
try:
  def check_if_user_quit():
    return not control_streamer.experiment_is_running()
  check_if_user_quit()
except:
  try:
    def check_if_user_quit():
      last_notes = notes_streamer.get_last_notes()
      if last_notes is not None:
        last_notes = last_notes.lower().strip()
      return last_notes in ['quit', 'q']
    check_if_user_quit()
  except:
    def check_if_user_quit():
      return False
    
duration_s = 1200
if visualizer is not None:
  visualizer.visualize_streaming_data(duration_s=duration_s,
                                      stopping_condition_fn=check_if_user_quit)
else:
  time_start_s = time.time()
  while not check_if_user_quit() and time.time() - time_start_s < duration_s:
    time.sleep(1)

# Stop streamers and data logging.
print()
print('Main loop terminated - stopping data logger and streamers')
for (streamer_index, streamer) in enumerate(streamers):
    if print_status: print('\nStopping streamer %d/%d of class %s' % (streamer_index+1, len(streamers), type(streamers[streamer_index]).__name__))
    streamer.stop()
logger.stop()






