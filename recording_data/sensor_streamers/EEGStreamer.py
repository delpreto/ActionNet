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
class EEGStreamer(SensorStreamer):

  ########################
  ###### INITIALIZE ######
  ########################

  def __init__(self, streams_info=None, log_player_options=None,
                visualization_options=None,
                print_status=True, print_debug=False, log_history_filepath=None):
    SensorStreamer.__init__(self, streams_info,
                              log_player_options=log_player_options,
                              visualization_options=visualization_options,
                              print_status=print_status, print_debug=print_debug,
                              log_history_filepath=log_history_filepath)

    self._log_source_tag = 'eeg'
    
    self._device_name = 'eeg'
    self._stream_names = ['all_channels', 'all_channels_filtered']
    
    if not self._replaying_data_logs:
      raise AssertionError('The EEG streamer currently only supports replaying existing logs (not streaming data in real time)')

  def _connect(self, timeout_s=10):
    return True

  # Specify how the stream should be visualized.
  def get_default_visualization_options(self, visualization_options=None):
    processed_options = {self._device_name: {}}
    for stream_name in self._stream_names:
      # Specify default options.
      processed_options[self._device_name][stream_name] = {
        'class': LinePlotVisualizer,
        'single_graph': False,
        'show_yaxis_labels': True,
        'plot_duration_s': 15,
        'downsample_factor': 1,
        }
    # Override with any provided options.
    if isinstance(visualization_options, dict):
      for (k, v) in visualization_options.items():
        for stream_name in self._stream_names:
          processed_options[self._device_name][stream_name][k] = v
    return processed_options

  #####################
  ###### RUNNING ######
  #####################

  # Loop until self._running is False
  def _run(self):
    raise AssertionError('The EEG streamer currently only supports replaying existing logs (not streaming data in real time)')

  # Clean up and quit
  def quit(self):
    self._log_debug('EEGStreamer quitting')
    SensorStreamer.quit(self)














