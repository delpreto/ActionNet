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
from visualizers.HeatmapVisualizer import HeatmapVisualizer
from visualizers.FlowFieldVisualizer import FlowFieldVisualizer
from utils.numpy_scipy_utils import convolve2d_strided

import socket
from threading import Thread
import numpy as np
import time
from collections import OrderedDict
import traceback


################################################
################################################
# A class to interface with the tactile shear sensors via the FPGA board.
################################################
################################################
class TouchShearStreamer(SensorStreamer):
  
  ########################
  ###### INITIALIZE ######
  ########################
  
  def __init__(self, streams_info=None,
               log_player_options=None, visualization_options=None,
               sensor_names=None, downsampling_factor=1,
               print_status=True, print_debug=False, log_history_filepath=None):
    SensorStreamer.__init__(self, streams_info,
                            log_player_options=log_player_options,
                            visualization_options=visualization_options,
                            print_status=print_status, print_debug=print_debug,
                            log_history_filepath=log_history_filepath)
    
    self._log_source_tag = 'shear'
    
    # Define the connected sensors.
    if sensor_names is not None:
      self._sensor_sockets = OrderedDict([(sensor_name, None) for sensor_name in sensor_names])
    else:
      self._sensor_sockets = {'shear-sensor': None}
    # Configurations that should match settings in Arduino code.
    self._sensor_waits_for_request = False # Should match setting in Xilinx code
    self._sensor_sends_debug_values = False # Should match setting in Xilinx code
    self._sensor_streams_rows = False # whether each message is a row of data or the entire matrix of data
    self._tactile_sample_size = (32, 32) # (height, width)
    self._tiled_sample_size = tuple([x // 2 for x in self._tactile_sample_size]) # (height, width)
    self._sensor_header_length = 12
    self._data_length_expected_perMatrix = int(2 * (np.prod(self._tactile_sample_size)) + self._sensor_header_length) # each uint16 byte will be sent as two consecutive uint8 bytes
    self._data_length_expected_perRow    = int(2 * (self._tactile_sample_size[0]) + self._sensor_header_length) # each uint16 byte will be sent as two consecutive uint8 bytes
    if self._sensor_streams_rows:
      self._data_length_expected = self._data_length_expected_perRow
    else:
      self._data_length_expected = self._data_length_expected_perMatrix
    
    # Initialize state.
    self._downsampling_factor = downsampling_factor
    self._downsampling_counters = dict([(sensor_name, 0) for sensor_name in sensor_names])
    self._sensor_names = list(self._sensor_sockets.keys())
    self._sensor_names_active = []
    self._matrix_indexes = {}
    self._run_threads = {}
  
  
  def _connect(self, timeout_s=10):
    # Try to connect to each specified sensor.
    # If the sensor is active, start a data stream for the sensor.
    sensor_names_connected = []
    for sensor_name in self._sensor_names:
      try:
        self._sensor_sockets[sensor_name] = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self._sensor_sockets[sensor_name].setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sensor_sockets[sensor_name].bind(('', 10000))
        self.add_stream(device_name=sensor_name,
                        stream_name='tactile_data',
                        data_type='float32',
                        sample_size=self._tactile_sample_size,
                        sampling_rate_hz=None,
                        extra_data_info={},
                        data_notes=OrderedDict([
                          ('Description', 'ADC readings from the matrix of tactile sensors.'),
                          ('Range', '[0, 65536]'),
                        ]))
        self.add_stream(device_name=sensor_name,
                        stream_name='tactile_tiled',
                        data_type='float32',
                        sample_size=self._tiled_sample_size,
                        sampling_rate_hz=None,
                        extra_data_info={},
                        data_notes=OrderedDict([
                          ('Description', 'ADC readings from the matrix of tactile sensors, '
                                          'grouped by the readings under each physical tile.'),
                        ]))
        self.add_stream(device_name=sensor_name,
                        stream_name='force_vector',
                        data_type='float32',
                        sample_size=(2, *self._tiled_sample_size), # magnitude and angle
                        sampling_rate_hz=None,
                        extra_data_info={},
                        data_notes=OrderedDict([
                          ('Description', 'An estimate of the magnitude and angle of the force '
                                          'vector under each physical tile of the sensor.'),
                          ('Matrix ordering', 'Force magnitude is the first matrix dimension '
                                              '(e.g. data[frame_index][0][:,:]) and the angle '
                                              'is the second dimension.'),
                          ('Units_for_angles', 'Radians'),
                        ]))
        sensor_names_connected.append(sensor_name)
        # Initialize the counter that will store the last received matrix index.
        self._matrix_indexes[sensor_name] = None
      except:
        self._sensor_sockets[sensor_name] = None
    self._log_status('Found the following shear sensors connected: %s' % sensor_names_connected)
    
    # Wait for the sensors to become active.
    self._sensor_names_active = []
    for sensor_name in sensor_names_connected:
      self._log_status('Waiting for the shear sensor %s to start streaming data' % sensor_name)
      wait_start_time_s = time.time()
      while time.time() - wait_start_time_s < 10:
        (time_s, data_matrix, _, _, _) = self._read_sensor(sensor_name, suppress_printing=False)
        if data_matrix is not None:
          self._sensor_names_active.append(sensor_name)
          break
        time.sleep(0.05)
    self._log_status('Found the following shear sensors active: %s' % self._sensor_names_active)
    
    # Return success if all desired sensors were found to be active.
    if len(self._sensor_names_active) == len([sensor_name for (sensor_name, socket) in self._sensor_sockets.items() if socket is not None]):
      return True
    else:
      return False
  
  #######################################
  ###### INTERFACE WITH THE SENSOR ######
  #######################################
  
  # Helper to switch between data transfer paradigms that the Arduino might use.
  def _read_sensor(self, sensor_name, suppress_printing=False):
    if self._sensor_waits_for_request:
      return self._read_sensor_requestParadigm(sensor_name, suppress_printing=suppress_printing)
    else:
      return self._read_sensor_streamParadigm(sensor_name, suppress_printing=suppress_printing)
  
  # Read from the sensor using the stream paradigm,
  #  in which the Arduino constantly sends newline-terminated data lines.
  #  Each row of data will be sent as a new line, with the format [matrix_index][row_index][row_data][\n].
  def _read_sensor_streamParadigm(self, sensor_name, suppress_printing=False):
    sensor_socket = self._sensor_sockets[sensor_name]
    if sensor_socket is None:
      return (None, None, None, None, None)
    
    # Read a matrix of data from the sensor.
    data = sensor_socket.recv(self._data_length_expected)
    time_s = time.time()
    # Validate the length of the data.
    if len(data) != self._data_length_expected:
      if not suppress_printing:
        self._log_warn('WARNING: Shear sensor [%s] sent %d values instead of %d values. Ignoring the data.' % (sensor_name, len(data), self._data_length_expected))
      return (None, None, None, None, None)
    # Parse the data.
    data_header = data[0:self._sensor_header_length]
    data_header = np.frombuffer(data_header, dtype=np.uint16).astype(np.uint16)
    data_matrix = data[self._sensor_header_length:]
    data_matrix = np.frombuffer(data_matrix, dtype=np.uint16).astype(np.uint16)
    data_matrix = data_matrix.reshape(*self._tactile_sample_size)
    # data_matrix = data_matrix.transpose(1, 0)//16
    
    if (not suppress_printing) and self._print_debug:
      self._log_debug('Received data from %s with size %s and min/max %d/%d: \n%s' % (sensor_name, data_matrix.shape, np.min(data_matrix), np.max(data_matrix), data_matrix))
    
    # Compute the total force in each shear square.
    toConvolve_tiled_magnitude = np.array([[1,1],[1,1]])
    data_matrix_tiled_magnitude = convolve2d_strided(data_matrix, toConvolve_tiled_magnitude, stride=2)
    
    # Compute the force angle in each shear square.
    toConvolve_tiled_x = np.array([[-1,1],[-1,1]])
    toConvolve_tiled_y = np.array([[1,1],[-1,-1]])
    data_matrix_tiled_x = convolve2d_strided(data_matrix, toConvolve_tiled_x, stride=2)
    data_matrix_tiled_y = convolve2d_strided(data_matrix, toConvolve_tiled_y, stride=2)
    data_matrix_tiled_shearAngle_rad = np.arctan2(data_matrix_tiled_y, data_matrix_tiled_x)
    data_matrix_tiled_shearMagnitude = np.linalg.norm(np.stack([data_matrix_tiled_y, data_matrix_tiled_x], axis=0), axis=0)
    
    # Return the data!
    return (time_s, data_matrix, data_matrix_tiled_magnitude,
            data_matrix_tiled_shearAngle_rad, data_matrix_tiled_shearMagnitude)
  
  # Read from the sensor using the request paradigm,
  #  in which this program explicitly requests every sample.
  #  The whole matrix of data will be sent as a single line, with the format [matrix_index][matrix_data][\n].
  def _read_sensor_requestParadigm(self, sensor_name, suppress_printing=False):
    raise AssertionError('Request paradigm is not implemented for FPGA-based shear sensors')
  
  
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
        if stream_name == 'tactile_data':
          processed_options[device_name].setdefault(stream_name,
                                                    {
                                                      'class': HeatmapVisualizer,
                                                      'colorbar_levels': 'auto', # a 2-element list, 'auto', or omitted
                                                    })
        elif stream_name == 'tactile_tiled':
          processed_options[device_name].setdefault(stream_name,
                                                    {
                                                      'class': HeatmapVisualizer,
                                                      'colorbar_levels': 'auto', # a 2-element list, 'auto', or omitted
                                                    })
        elif stream_name == 'force_vector':
          processed_options[device_name].setdefault(stream_name,
                                                    {
                                                      'class': FlowFieldVisualizer,
                                                      'magnitude_normalization': 'auto',
                                                      'magnitude_normalization_min': 1,
                                                      'magnitude_normalization_buffer_duration_s': 30,
                                                      'magnitude_normalization_update_period_s': 10,
                                                    })
        else:
          processed_options[device_name].setdefault(stream_name, {'class':None})
    
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
          (time_s, data_matrix, data_matrix_tiled_magnitude,
           data_matrix_tiled_shearAngle_rad, data_matrix_tiled_shearMagnitude) \
            = self._read_sensor(sensor_name, suppress_printing=(count < 10))
          if time_s is not None and data_matrix is not None:
            self._downsampling_counters[sensor_name] += 1
            if self._downsampling_counters[sensor_name] == self._downsampling_factor:
              self._downsampling_counters[sensor_name] = 0
              self.append_data(sensor_name, 'tactile_data', time_s, data_matrix)
              self.append_data(sensor_name, 'tactile_tiled', time_s, data_matrix_tiled_magnitude)
              self.append_data(sensor_name, 'force_vector', time_s,
                               np.stack((data_matrix_tiled_shearMagnitude,
                                         data_matrix_tiled_shearAngle_rad),
                                        axis=0))
          count = count + 1
        except:
          self._log_error('*** Could not read from shear sensor %s - waiting a bit then retrying:\n%s\n' % (sensor_name, traceback.format_exc()))
          time.sleep(5)
      
      for sensor_name in self._sensor_names_active:
        self._sensor_sockets[sensor_name].close()
    except KeyboardInterrupt: # The program was likely terminated
      pass
    except:
      self._log_error('\n\n***ERROR RUNNING TouchShearStreamer for sensor %s:\n%s\n' % (sensor_name, traceback.format_exc()))
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
    self._log_debug('TouchShearStreamer quitting')
    SensorStreamer.quit(self)


#####################
###### TESTING ######
#####################
if __name__ == '__main__':
  # Configuration.
  duration_s = 7200
  
  # Connect to the device(s).
  touchShear_streamer = TouchShearStreamer(sensor_names=['shear-sensor'],
                                           downsampling_factor=10,
                                           print_status=True, print_debug=False)
  touchShear_streamer.connect()
  
  # Run for the specified duration and periodically print the sample rate.
  print('\nRunning for %gs!' % duration_s)
  touchShear_streamer.run()
  start_time_s = time.time()
  try:
    while time.time() - start_time_s < duration_s:
      time.sleep(2)
      fps_msg = ' Duration: %6.2fs' % (time.time() - start_time_s)
      for device_name in touchShear_streamer.get_device_names():
        stream_name = touchShear_streamer.get_stream_names(device_name=device_name)[0]
        num_timesteps = touchShear_streamer.get_num_timesteps(device_name, stream_name)
        fps_msg += ' | %s: %4d Timesteps (Fs = %6.2f Hz)' % \
                   (device_name, num_timesteps, ((num_timesteps)/(time.time() - start_time_s)))
      print(fps_msg)
  except:
    print('ERROR')
    print(traceback.format_exc())
  touchShear_streamer.stop()
  print('\nDone!\n')
  
  
  
  











