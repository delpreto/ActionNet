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

from sensor_streamers.SensorStreamer import SensorStreamer
from visualizers.LinePlotVisualizer import LinePlotVisualizer
from visualizers.HeatmapVisualizer import HeatmapVisualizer
from visualizers.VideoVisualizer import VideoVisualizer
from visualizers.XsensSkeletonVisualizer import XsensSkeletonVisualizer

import time
import numpy as np
import traceback

################################################
################################################
# A class to stream dummy data for testing.
################################################
################################################
class DummyStreamer(SensorStreamer):

  ########################
  ###### INITIALIZE ######
  ########################

  def __init__(self, streams_info=None, log_player_options=None,
                visualization_options=None,
                streamer_tag='',
                sample_size=(1,), update_period_s=1,
                visualization_type='line', # note that 'video' may override sample_size
                print_status=True, print_debug=False, log_history_filepath=None):
    SensorStreamer.__init__(self, streams_info,
                              log_player_options=log_player_options,
                              visualization_options=visualization_options,
                              print_status=print_status, print_debug=print_debug,
                              log_history_filepath=log_history_filepath)

    self._log_source_tag = 'dummy'
    
    self._visualization_type = visualization_type.lower()
    self._sample_size = list(sample_size)
    self._update_period_s = update_period_s

    # Update frame size if showing a video.
    if self._visualization_type == 'video':
      if len(self._sample_size) != 3:
        self._sample_size = (480, 640, 3)
      data_type = 'uint8'
    else:
      data_type = 'int'

    # Create the device and stream names.
    self._device_name = ('dummy-%s-%s' % (streamer_tag, self._visualization_type)).strip('-').replace('--', '-')
    self._stream_name = 'dummy-stream'

    # Create the stream unless an existing log is being replayed
    #  (in which case SensorStreamer will create the stream automatically).
    if not self._replaying_data_logs:
      self.add_stream(device_name=self._device_name, stream_name=self._stream_name,
                      data_type=data_type, sample_size=self._sample_size, data_notes=None,
                      sampling_rate_hz=1/self._update_period_s)

  def _connect(self, timeout_s=10):
    return True

  # Specify how the stream should be visualized.
  def get_default_visualization_options(self, visualization_options=None):
    processed_options = {self._device_name: {self._stream_name: {}}}
    if self._visualization_type == 'video':
      processed_options[self._device_name][self._stream_name] = {
        'class': VideoVisualizer
        }
    elif self._visualization_type == 'line':
      # Specify default options.
      processed_options[self._device_name][self._stream_name] = {
        'class': LinePlotVisualizer,
        'single_graph': True,
        'plot_duration_s': 30,
        }
      # Override with any provided options.
      if isinstance(visualization_options, dict):
        for (k, v) in visualization_options.items():
          processed_options[self._device_name][self._stream_name][k] = v
    elif self._visualization_type == 'heatmap':
      # Specify default options.
      processed_options[self._device_name][self._stream_name] = {
        'class': HeatmapVisualizer,
        }
      # Override with any provided options.
      if isinstance(visualization_options, dict):
        for (k, v) in visualization_options.items():
          processed_options[self._device_name][self._stream_name][k] = v
    elif self._visualization_type == 'xsens-skeleton':
      # Specify default options.
      processed_options[self._device_name][self._stream_name] = {
        'class': XsensSkeletonVisualizer,
        }
      # Override with any provided options.
      if isinstance(visualization_options, dict):
        for (k, v) in visualization_options.items():
          processed_options[self._device_name][self._stream_name][k] = v
    return processed_options

  #####################
  ###### RUNNING ######
  #####################

  # Loop until self._running is False
  def _run(self):
    num_elements = np.prod(self._sample_size)
    if self._visualization_type == 'video':
      data = np.zeros(self._sample_size, dtype='uint8')
    else:
      elements = np.arange(0, num_elements, dtype='int')
      data = elements.reshape(self._sample_size)
    next_update_time_s = time.time() + self._update_period_s
    try:
      while self._running:
        self.append_data(self._device_name, self._stream_name, time.time(), data)
        if self._visualization_type == 'video':
          data = (data + 20) % (255)
        else:
          data = data + num_elements
        next_update_time_s = next_update_time_s + self._update_period_s
        time.sleep(max(0, next_update_time_s - time.time()))
    except KeyboardInterrupt: # The program was likely terminated
      pass
    except:
      self._log_error('\n\n***ERROR RUNNING DummyStreamer:\n%s\n' % traceback.format_exc())
    finally:
      pass

  # Clean up and quit
  def quit(self):
    self._log_debug('DummyStreamer quitting')
    SensorStreamer.quit(self)














