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

from sensor_streamers.SerialStreamer import SerialStreamer
from sensor_streamers.SensorStreamer import SensorStreamer
from visualizers.LinePlotVisualizer import LinePlotVisualizer

import serial
from threading import Thread
import numpy as np
import time
from collections import OrderedDict
import traceback


################################################
################################################
# A class to interface with a generic Serial data stream.
# Any number of COM ports can be specified, to interface with multiple sensors.
# See corresponding Arduino code in SerialStreamer_arduino/SerialStreamer_arduino.ino
################################################
################################################
class GloveStreamer(SerialStreamer):
  
  ########################
  ###### INITIALIZE ######
  ########################
  
  def __init__(self, streams_info=None,
               log_player_options=None, visualization_options=None,
               com_ports=None, baud_rates_bps=None,
               sampling_rates_hz=None, sample_sizes=None, value_delimiters=None,
               sensors_send_debug_values=False,
               print_status=True, print_debug=False, log_history_filepath=None):
    
    self._log_source_tag = 'glove'

    self._device_name = list(com_ports.keys())[0]
    self._num_strains = 16
    self._num_accels = 3
    
    # Apply default settings for the glove.
    baud_rates_bps = baud_rates_bps or {self._device_name: 460800}
    sampling_rates_hz = sampling_rates_hz or {self._device_name: 100}
    sample_sizes = sample_sizes or {self._device_name: [self._num_strains+self._num_accels]}
    value_delimiters = value_delimiters or {self._device_name: ', '}
    
    # Initialize.
    SerialStreamer.__init__(self, streams_info,
                            log_player_options=log_player_options, visualization_options=visualization_options,
                            com_ports=com_ports, baud_rates_bps=baud_rates_bps,
                            sampling_rates_hz=sampling_rates_hz, sample_sizes=sample_sizes, value_delimiters=value_delimiters,
                            sensors_send_debug_values=sensors_send_debug_values,
                            print_status=print_status, print_debug=print_debug, log_history_filepath=log_history_filepath)
    
  def _connect(self, timeout_s=10):
    # Use the parent method to set up the serial communication.
    if not SerialStreamer._connect(self, timeout_s=timeout_s):
      return False
    
    # Remove the default streams and add glove-specific ones.
    self.set_streams_info(streams_info=None)
    self.add_stream(device_name=self._device_name,
                    stream_name='strain',
                    data_type='int',
                    sample_size=[self._num_strains],
                    sampling_rate_hz=self._sampling_rates_hz[self._device_name],
                    extra_data_info={},
                    data_notes=OrderedDict([
                      ('Description', 'Raw ADC values from each strain channel'),
                      ('Units', 'Raw ADC values in the range [0, 65535]'),
                      (SensorStreamer.metadata_data_headings_key, ['Channel %d' % x for x in range(1,self._num_strains+1)])
                    ]))
    self.add_stream(device_name=self._device_name,
                    stream_name='acceleration',
                    data_type='int',
                    sample_size=[self._num_accels],
                    sampling_rate_hz=self._sampling_rates_hz[self._device_name],
                    extra_data_info={},
                    data_notes=OrderedDict([
                      ('Description', 'Raw ADC values from each accelerometer axis'),
                      ('Units', 'Raw ADC values shifted up by 32768, so they are in the range [0, 65535] with a value of 32768 being 0 acceleration'),
                      (SensorStreamer.metadata_data_headings_key, ['X', 'Y', 'Z'])
                    ]))
    
    return True
    
  
  ###########################
  ###### VISUALIZATION ######
  ###########################
  
  # Specify how the streams should be visualized.
  # visualization_options can have entries for each sensor name as defined in self._com_ports.
  def get_default_visualization_options(self, visualization_options=None):
    # Add default options for all devices/streams.
    processed_options = {}
    for (device_name, device_info) in self._streams_info.items():
      processed_options.setdefault(device_name, {})
      for (stream_name, stream_info) in device_info.items():
        processed_options[device_name].setdefault(stream_name,
                                                  {
                                                    'class': LinePlotVisualizer,
                                                    'single_graph': True,
                                                    'plot_duration_s': 30,
                                                    'downsample_factor': 1,
                                                  })
    
    # Override with any provided options.
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
  ###### RUNNING ######
  #####################
  
  # Will start a new run thread for each sensor.
  # Otherwise, the serial reads seem to interfere with each other
  #  and cause corrupt data that needs to be discarded
  #  (enough to reduce the sampling rate from just over 20 Hz per sensor
  #   to just over 18 Hz, over the course of a 30-second experiment).
  def _run_for_sensor(self, sensor_name):
    try:
      # Note that warnings will be suppressed for the first few reads, since they
      #  typically contain a few incomplete data lines before the reading becomes
      #  aligned with the Arduino streaming cadence.
      count = 0
      while self._running:
        try:
          (time_s, data) = self._read_sensor(sensor_name, suppress_printing=(count < 10))
          if time_s is not None and data is not None:
            self.append_data(sensor_name, 'strain', time_s, data[0:self._num_strains])
            self.append_data(sensor_name, 'acceleration', time_s, data[self._num_strains:(self._num_strains+self._num_accels)])
          count = count + 1
        except serial.serialutil.SerialException:
          self._log_error('*** Could not read from glove sensor %s:\n%s\n' % (sensor_name, traceback.format_exc()))
          try:
            self._sensor_serials[sensor_name] = serial.Serial(
                self._com_ports[sensor_name],
                baudrate=self._baud_rates_bps[sensor_name],
                timeout=1.0)
            time.sleep(0.5)
          except serial.serialutil.SerialException:
            self._log_error('*** Could not reconnect to glove sensor %s - waiting a bit then retrying' % (sensor_name))
            time.sleep(5)
      
      for sensor_name in self._sensor_names_active:
        self._sensor_serials[sensor_name].close()
    except KeyboardInterrupt: # The program was likely terminated
      pass
    except:
      self._log_error('\n\n***ERROR RUNNING GloveStreamer for sensor %s:\n%s\n' % (sensor_name, traceback.format_exc()))
    finally:
      pass
  
#####################
###### TESTING ######
#####################
if __name__ == '__main__':
  # Configuration.
  duration_s = 7200
  
  # Connect to the device(s).
  glove_streamer = GloveStreamer(
      com_ports={
        'test-glove': 'COM8',
      },
      sensors_send_debug_values={
        'test-glove': False,
      },
      print_status=True, print_debug=False)
  glove_streamer.connect()
  
  # Run for the specified duration and periodically print the sample rate.
  print('\nRunning for %gs!' % duration_s)
  glove_streamer.run()
  start_time_s = time.time()
  try:
    while time.time() - start_time_s < duration_s:
      time.sleep(2)
      fps_msg = ' Duration: %6.2fs' % (time.time() - start_time_s)
      for device_name in glove_streamer.get_device_names():
        stream_name = glove_streamer.get_stream_names(device_name=device_name)[0]
        num_timesteps = glove_streamer.get_num_timesteps(device_name, stream_name)
        fps_msg += ' | %s: %4d Timesteps (Fs = %6.2f Hz)' % \
                   (device_name, num_timesteps, ((num_timesteps)/(time.time() - start_time_s)))
      print(fps_msg)
  except:
    pass
  glove_streamer.stop()
  print('\nDone!\n')
  
  
  
  











