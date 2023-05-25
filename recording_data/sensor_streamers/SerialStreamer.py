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
class SerialStreamer(SensorStreamer):
  
  ########################
  ###### INITIALIZE ######
  ########################
  
  def __init__(self, streams_info=None,
               log_player_options=None, visualization_options=None,
               com_ports=None, baud_rates_bps=None,
               sampling_rates_hz=None, sample_sizes=None, value_delimiters=None,
               sensors_send_debug_values=False,
               print_status=True, print_debug=False, log_history_filepath=None):
    SensorStreamer.__init__(self, streams_info,
                            log_player_options=log_player_options,
                            visualization_options=visualization_options,
                            print_status=print_status, print_debug=print_debug,
                            log_history_filepath=log_history_filepath)
    
    self._log_source_tag = 'serial'
    
    # Define the connected sensors as a dictionary mapping device name to com port.
    # Port configurations should be checked on the operating system.
    # Setting a port to None will ignore that sensor.
    self._com_ports = com_ports
    self._sampling_rates_hz = sampling_rates_hz
    self._sensors_send_debug_values = sensors_send_debug_values
    if isinstance(self._sensors_send_debug_values, bool) and self._com_ports is not None:
      self._sensors_send_debug_values = dict([(sensor_name, self._sensors_send_debug_values) for sensor_name in self._com_ports])
    
    # Configurations that should match settings in Arduino code.
    self._baud_rates_bps = baud_rates_bps
    self._sample_sizes = sample_sizes
    self._value_delimiters = value_delimiters
    
    # Initialize state.
    self._sensor_names = list(self._com_ports.keys()) if self._com_ports is not None else []
    self._sensor_names_active = []
    self._sensor_serials = {}
    self._run_threads = {}
  
  
  def _connect(self, timeout_s=10):
    # Try to connect to each specified serial port.
    # If the port is active, start a data stream for the sensor.
    sensor_names_connected = []
    for sensor_name in self._sensor_names:
      if self._com_ports[sensor_name] is None:
        continue
      try:
        self._sensor_serials[sensor_name] = serial.Serial(
            self._com_ports[sensor_name],
            baudrate=self._baud_rates_bps[sensor_name],
            timeout=1.0)
        # Make the buffers large enough to accommodate unexpected/stochastic delays
        #  from the operating system that may cause temporary backlogs of data to process.
        self._sensor_serials[sensor_name].set_buffer_size(rx_size=50*self._sample_sizes[sensor_name][0],
                                                          tx_size=50*self._sample_sizes[sensor_name][0])
        self.add_stream(device_name=sensor_name,
                        stream_name='serial_data',
                        data_type='float32',
                        sample_size=self._sample_sizes[sensor_name],
                        sampling_rate_hz=self._sampling_rates_hz[sensor_name],
                        extra_data_info={},
                        data_notes=OrderedDict([]))
        sensor_names_connected.append(sensor_name)
      except:
        self._sensor_serials[sensor_name] = None
    self._log_status('Found the following serial sensors connected: %s' % sensor_names_connected)
    
    # Wait for the sensors to become active.
    # For example, the Arduino may restart upon serial connection.
    self._sensor_names_active = []
    for sensor_name in sensor_names_connected:
      self._log_status('Waiting for the serial sensor %s to start streaming data' % sensor_name)
      wait_start_time_s = time.time()
      while time.time() - wait_start_time_s < 10:
        (time_s, data) = self._read_sensor(sensor_name, suppress_printing=True)
        if data is not None:
          self._sensor_names_active.append(sensor_name)
          break
        time.sleep(0.05)
    self._log_status('Found the following serial sensors active: %s' % self._sensor_names_active)
    
    # Return success if all desired sensors were found to be active.
    if len(self._sensor_names_active) == len([sensor_name for (sensor_name, port) in self._com_ports.items() if port is not None]):
      return True
    else:
      return False
  
  #######################################
  ###### INTERFACE WITH THE SENSOR ######
  #######################################
  
  # Read from the sensor.
  #  Each row of data will be sent as a new line, with streams separated by the specified delimiter.
  def _read_sensor(self, sensor_name, suppress_printing=False):
    sensor_serial = self._sensor_serials[sensor_name]
    if sensor_serial is None or not sensor_serial.is_open:
      return (None, None)
    
    # Receive data from the sensor
    try:
      data = sensor_serial.readline().decode('utf-8').strip()
    except:
      return (None, None)
    data_time_s = time.time()
    
    # Parse the streams and ensure that all values are numeric.
    data_row = data.split(self._value_delimiters[sensor_name])
    data_row = [data_entry for data_entry in data_row if len(data_entry) > 0]
    try:
      data_row = [float(data_entry) for data_entry in data_row]
    except:
      self._log_warn('WARNING: Serial sensor [%s] sent non-float values. Ignoring the data.' % (sensor_name))
      return (None, None)
    
    # Validate the length of the data.
    if len(data_row) != self._sample_sizes[sensor_name][0]:
      if not suppress_printing:
        self._log_warn('WARNING: Serial sensor [%s] sent %d values instead of %s values. Ignoring the data.' % (sensor_name, len(data_row), self._sample_sizes[sensor_name]))
      return (None, None)
    
    if (not suppress_printing) and self._print_debug:
      self._log_debug('Received data from %s with length %d and min/max %d/%d: \n%s' % (sensor_name, len(data_row), min(data_row), max(data_row), str(data_row)))
    
    # Validate the data if debug values are being used.
    if self._sensors_send_debug_values[sensor_name]:
      target_data = np.reshape(np.arange(np.prod(self._sample_sizes[sensor_name])), self._sample_sizes[sensor_name])
      if not np.all(np.array(data_row) == target_data):
        if not suppress_printing:
          self._log_warn('WARNING: Serial sensor [%s] sent data that does not match the expected debug values: \n%s' % (sensor_name, data_row))
    
    # Return the data!
    return (data_time_s, data_row)
  
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
                                                    'plot_duration_s': 60,
                                                    # 'plot_duration_s': 2700, # oven tests
                                                    # 'plot_duration_s': 200, # vacuum tests
                                                    # 'plot_duration_s': 600, # spreader tests
                                                    'downsample_factor': 1,
                                                    'y_lim': None,
                                                    'x_label': 'Time [s]',
                                                    'y_label': None,
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
            self.append_data(sensor_name, 'serial_data', time_s, data)
          count = count + 1
        except serial.serialutil.SerialException:
          self._log_error('*** Could not read from serial sensor %s:\n%s\n' % (sensor_name, traceback.format_exc()))
          try:
            self._sensor_serials[sensor_name] = serial.Serial(
                self._com_ports[sensor_name],
                baudrate=self._baud_rates_bps[sensor_name],
                timeout=1.0)
            time.sleep(0.5)
          except serial.serialutil.SerialException:
            self._log_error('*** Could not reconnect to serial sensor %s - waiting a bit then retrying' % (sensor_name))
            time.sleep(5)
      
      for sensor_name in self._sensor_names_active:
        self._sensor_serials[sensor_name].close()
    except KeyboardInterrupt: # The program was likely terminated
      pass
    except:
      self._log_error('\n\n***ERROR RUNNING SerialStreamer for sensor %s:\n%s\n' % (sensor_name, traceback.format_exc()))
    finally:
      pass
  
  # Launch the per-sensor threads.
  def _run(self):
    # Create and start a thread for each sensor.
    for sensor_name in self._sensor_names_active:
      self._run_threads[sensor_name] = Thread(target=self._run_for_sensor,
                                              args=(),
                                              kwargs={'sensor_name': sensor_name})
      self._run_threads[sensor_name].daemon = False
      self._run_threads[sensor_name].start()
    # Join the threads to wait until all are done.
    for sensor_name in self._sensor_names_active:
      self._run_threads[sensor_name].join()
  
  # Clean up and quit
  def quit(self):
    self._log_debug('SerialStreamer quitting')
    SensorStreamer.quit(self)


#####################
###### TESTING ######
#####################
if __name__ == '__main__':
  # Configuration.
  duration_s = 7200
  
  # Connect to the device(s).
  serial_streamer = SerialStreamer(
                                    com_ports={
                                      'test-device': 'COM5',
                                    },
                                    baud_rates_bps={
                                      'test-device': 1000000,
                                    },
                                    sampling_rates_hz={
                                      'test-device': 100,
                                    }, sample_sizes={
                                      'test-device': [2],
                                    },
                                    value_delimiters={
                                      'test-device': ' ',
                                    },
                                    sensors_send_debug_values={
                                      'test-device': False,
                                    },
                                   print_status=True, print_debug=False)
  serial_streamer.connect()
  
  # Run for the specified duration and periodically print the sample rate.
  print('\nRunning for %gs!' % duration_s)
  serial_streamer.run()
  start_time_s = time.time()
  try:
    while time.time() - start_time_s < duration_s:
      time.sleep(2)
      fps_msg = ' Duration: %6.2fs' % (time.time() - start_time_s)
      for device_name in serial_streamer.get_device_names():
        stream_name = serial_streamer.get_stream_names(device_name=device_name)[0]
        num_timesteps = serial_streamer.get_num_timesteps(device_name, stream_name)
        fps_msg += ' | %s: %4d Timesteps (Fs = %6.2f Hz)' % \
                   (device_name, num_timesteps, ((num_timesteps)/(time.time() - start_time_s)))
      print(fps_msg)
  except:
    pass
  serial_streamer.stop()
  print('\nDone!\n')
  
  
  
  











