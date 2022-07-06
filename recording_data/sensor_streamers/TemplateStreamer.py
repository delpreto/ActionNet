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

import numpy as np
import time
from collections import OrderedDict
import traceback

from utils.print_utils import *

################################################
################################################
# A template class for implementing a new sensor.
################################################
################################################
class TemplateStreamer(SensorStreamer):
  
  ########################
  ###### INITIALIZE ######
  ########################
  
  # Initialize the sensor streamer.
  # @param visualization_options Can be used to specify how data should be visualized.
  #   It should be a dictionary with the following keys:
  #     'visualize_streaming_data': Whether or not visualize any data during streaming.
  #     'update_period_s': How frequently to update the visualizations during streaming.
  #     'visualize_all_data_when_stopped': Whether to visualize a summary of data at the end of the experiment.
  #     'wait_while_visualization_windows_open': After the experiment finishes, whether to automatically close visualization windows or wait for the user to close them.
  #     'classes_to_visualize': [optional] A list of class names that should be visualized (others will be suppressed).  For example, ['TouchStreamer', 'MyoStreamer']
  #     'use_composite_video': Whether to combine visualizations from multiple streamers into a single tiled visualization.  If not, each streamer will create its own window.
  #     'composite_video_filepath': If using composite video, can specify a filepath to save it as a video.
  #     'composite_video_layout': If using composite video, can specify which streamers should be included and how to arrange them. See some of the launch files for examples.
  # @param log_player_options Can be used to replay data from an existing log instead of streaming real-time data.
  #   It should be a dictionary with the following keys:
  #     'log_dir': The directory with log data to replay (should directly contain the HDF5 file).
  #     'pause_to_replay_in_realtime': If reading from the logs is faster than real-time, can wait between reads to keep the replay in real time.
  #     'skip_timesteps_to_replay_in_realtime': If reading from the logs is slower than real-time, can skip timesteps as needed to remain in real time.
  #     'load_datasets_into_memory': Whether to load all data into memory before starting the replay, or whether to read from the HDF5 file each timestep.
  # @param print_status Whether or not to print messages with level 'status'
  # @param print_debug Whether or not to print messages with level 'debug'
  # @param log_history_filepath A filepath to save log messages if desired.
  def __init__(self,
               log_player_options=None, visualization_options=None,
               print_status=True, print_debug=False, log_history_filepath=None):
    SensorStreamer.__init__(self, streams_info=None,
                            visualization_options=visualization_options,
                            log_player_options=log_player_options,
                            print_status=print_status, print_debug=print_debug,
                            log_history_filepath=log_history_filepath)
    
    ## TODO: Add a tag here for your sensor that can be used in log messages.
    #        Try to keep it under 10 characters long.
    #        For example, 'myo' or 'scale'.
    self._log_source_tag = 'template'
    
    ## TODO: Initialize any state that your sensor needs.
    
    ## TODO: Add devices and streams to organize data from your sensor.
    #        Data is organized as devices and then streams.
    #        For example, a Myo device may have streams for EMG and Acceleration.
    #        If desired, this could also be done in the connect() method instead.
    self._device_name = 'template-sensor-device'
    self.add_stream(device_name=self._device_name,
                    stream_name='stream_1',
                    data_type='float32',
                    sample_size=[2],     # the size of data saved for each timestep; here, we expect a 2-element vector per timestep
                    sampling_rate_hz=65, # the expected sampling rate for the stream
                    extra_data_info={}, # can add extra information beyond the data and the timestamp if needed (probably not needed, but see MyoStreamer for an example if desired)
                    # Notes can add metadata about the stream,
                    #  such as an overall description, data units, how to interpret the data, etc.
                    # The SensorStreamer.metadata_data_headings_key is special, and is used to
                    #  describe the headings for each entry in a timestep's data.
                    #  For example - if the data was saved in a spreadsheet with a row per timestep, what should the column headings be.
                    data_notes=OrderedDict([
                      ('Description', 'A template class to show how to add new sensors '
                                      'by implementing the SensorStreamer class.'
                                      ),
                      ('Units', 'm/s'),
                      (SensorStreamer.metadata_data_headings_key, ['first_column', 'second_column']),
                    ]))
    # Add as many devices/streams as you need!
    self.add_stream(device_name=self._device_name,
                    stream_name='stream_2',
                    data_type='int',
                    sample_size=[6, 5],
                    sampling_rate_hz=45,
                    extra_data_info={},
                    data_notes=OrderedDict([
                      ('Description', 'More data!'),
                      ('Units', 'm/s/s'),
                      (SensorStreamer.metadata_data_headings_key,
                       ['entry_%02d-%02d' % (matrix_row, matrix_col)
                        for matrix_row in range(6) for matrix_col in range(5)]),
                    ]))

  #######################################
  # Connect to the sensor.
  # @param timeout_s How long to wait for the sensor to respond.
  def _connect(self, timeout_s=10):
    
    ## TODO: Add code for connecting to your sensor.
    #        Then return True or False to indicate whether connection was successful.

    self._log_status('Successfully connected to the template streamer.')
    return True
  
  #######################################
  ###### INTERFACE WITH THE SENSOR ######
  #######################################
  
  ## TODO: Add functions to control your sensor and acquire data.
  #        [Optional but probably useful]
  
  # A function to read a timestep of data for the first stream.
  def _read_data_stream1(self):
    # For example, may want to return the data for the timestep
    #  and the time at which it was received.
    my_data = np.random.rand(2)
    time_s = time.time()
    return (time_s, my_data)
  
  # A function to read a timestep of data for the second stream.
  def _read_data_stream2(self):
    # For example, may want to return the data for the timestep
    #  and the time at which it was received.
    my_data = np.random.rand(6, 5)
    time_s = time.time()
    return (time_s, my_data)
  
  
  #####################
  ###### RUNNING ######
  #####################
  
  ## TODO: Continuously read data from your sensor.
  # Loop until self._running is False.
  # Acquire data from your sensor as desired, and for each timestep
  #  call self.append_data(device_name, stream_name, time_s, data).
  def _run(self):
    try:
      while self._running:
        # Read and store data for stream 1.
        (time_s, data) = self._read_data_stream1()
        self.append_data(self._device_name, 'stream_1', time_s, data)
        
        # Read and store data for stream 2.
        # Use a random number to simulate data not being ready yet (a slower sampling rate for this stream).
        if np.random.rand() < 0.7:
          (time_s, data) = self._read_data_stream2()
          self.append_data(self._device_name, 'stream_2', time_s, data)
        
        # Wait to slow down the template sampling rate for demo purposes.
        time.sleep(0.01)
        
    except KeyboardInterrupt: # The program was likely terminated
      pass
    except:
      self._log_error('\n\n***ERROR RUNNING TemplateStreamer:\n%s\n' % traceback.format_exc())
    finally:
      ## TODO: Disconnect from the sensor if desired.
      pass
  
  # Clean up and quit
  def quit(self):
    ## TODO: Add any desired clean-up code.
    self._log_debug('TemplateStreamer quitting')
    SensorStreamer.quit(self)

  ###########################
  ###### VISUALIZATION ######
  ###########################

  # Specify how the streams should be visualized.
  # Return a dict of the form options[device_name][stream_name] = stream_options
  #  Where stream_options is a dict with the following keys:
  #   'class': A subclass of Visualizer that should be used for the specified stream.
  #   Any other options that can be passed to the chosen class.
  def get_default_visualization_options(self, visualization_options=None):
    # Start by not visualizing any streams.
    processed_options = {}
    for (device_name, device_info) in self._streams_info.items():
      processed_options.setdefault(device_name, {})
      for (stream_name, stream_info) in device_info.items():
        processed_options[device_name].setdefault(stream_name, {'class': None})
    
    ## TODO: Specify whether some streams should be visualized.
    #        Examples of a line plot and a heatmap are below.
    #        To not visualize data, simply omit the following code and just leave each streamer mapped to the None class as shown above.
    # Use a line plot to visualize the weight.
    processed_options[self._device_name]['stream_1'] = \
      {'class': LinePlotVisualizer,
       'single_graph': True,   # Whether to show each dimension on a subplot or all on the same plot.
       'plot_duration_s': 15,  # The timespan of the x axis (will scroll as more data is acquired).
       'downsample_factor': 1, # Can optionally downsample data before visualizing to improve performance.
      }
    processed_options[self._device_name]['stream_2'] = \
      {'class': HeatmapVisualizer,
       'colorbar_levels': 'auto', # The range of the colorbar.
                                  # Can be a 2-element list [min, max] to use hard-coded bounds,
                                  # or 'auto' to determine them dynamically based on a buffer of the data.
      }
  
    # Override the above defaults with any provided options.
    if isinstance(visualization_options, dict):
      for (device_name, device_info) in self._streams_info.items():
        if device_name in visualization_options:
          device_options = visualization_options[device_name]
          # Apply the provided options for this device to all of its streams.
          for (stream_name, stream_info) in device_info.items():
            for (k, v) in device_options.items():
              processed_options[device_name][stream_name][k] = v
  
    return processed_options


#####################
###### TESTING ######
#####################
if __name__ == '__main__':
  # Configuration.
  duration_s = 30
  
  # Connect to the device(s).
  template_streamer = TemplateStreamer(print_status=True, print_debug=False)
  template_streamer.connect()
  
  # Run for the specified duration and periodically print the sample rate.
  print('\nRunning for %gs!' % duration_s)
  template_streamer.run()
  start_time_s = time.time()
  try:
    while time.time() - start_time_s < duration_s:
      time.sleep(2)
      # Print the sampling rates.
      msg = ' Duration: %6.2fs' % (time.time() - start_time_s)
      for device_name in template_streamer.get_device_names():
        stream_names = template_streamer.get_stream_names(device_name=device_name)
        for stream_name in stream_names:
          num_timesteps = template_streamer.get_num_timesteps(device_name, stream_name)
          msg += ' | %s-%s: %6.2f Hz (%4d Timesteps)' % \
                 (device_name, stream_name, ((num_timesteps)/(time.time() - start_time_s)), num_timesteps)
      print(msg)
  except:
    pass
  
  # Stop the streamer.
  template_streamer.stop()
  print('\n'*2)
  print('='*75)
  print('Done!')
  print('\n'*2)
  
  
  
  
  











