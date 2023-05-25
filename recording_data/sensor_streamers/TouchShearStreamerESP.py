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
from sensor_streamers.TouchStreamer import TouchStreamer
from visualizers.HeatmapVisualizer import HeatmapVisualizer
from visualizers.FlowFieldVisualizer import FlowFieldVisualizer

import serial
from threading import Thread
import numpy as np
from utils.numpy_scipy_utils import convolve2d_strided
import time
from collections import OrderedDict
import traceback


################################################
################################################
# A class to interface with the tactile sensors via the Arduino board.
# Any number of COM ports can be specified, to interface with multiple sensors.
# See the corresponding Arduino code in TouchStreamer_arduino/TouchStreamer_arduino.ino
################################################
################################################
class TouchShearStreamerESP(TouchStreamer):
  
  ########################
  ###### INITIALIZE ######
  ########################
  
  def __init__(self, streams_info=None,
               log_player_options=None, visualization_options=None,
               com_ports=None,
               print_status=True, print_debug=False, log_history_filepath=None):
    TouchStreamer.__init__(self, streams_info,
                            log_player_options=log_player_options,
                            visualization_options=visualization_options,
                            com_ports=com_ports,
                            print_status=print_status, print_debug=print_debug,
                            log_history_filepath=log_history_filepath)
    
    self._log_source_tag = 'shear'
    
    # Configurations that should match settings in Arduino code.
    self._sensor_waits_for_request = True # Should match setting in Arduino code
    self._sensor_sends_debug_values = False # Should match setting in Arduino code
    self._sensor_bitshift = 6
    self._sensor_streams_rows = not self._sensor_waits_for_request # whether each message is a row of data or the entire matrix of data
    self._baud_rate_bps = 1000000
    self._sensor_sample_size = (32, 32) # (height, width)
    self._data_length_expected_perMatrix = int(2*np.prod(self._sensor_sample_size)) # each uint16 byte will be sent as two consecutive uint8 bytes
    self._data_length_expected_perRow    = int(2*self._sensor_sample_size[0]) # each uint16 byte will be sent as two consecutive uint8 bytes
    if self._sensor_streams_rows:
      self._data_length_expected = self._data_length_expected_perRow
    else:
      self._data_length_expected = self._data_length_expected_perMatrix
    
    # Shear-specific configuration.
    self._tiled_sample_size = tuple([x // 2 for x in self._sensor_sample_size]) # (height, width)
    # State for subtracting an average of initial readings.
    self._calibration_duration_s = None # None to not calibrate
    self._calibration_startTime_s = None
    self._calibration_matrices = []
    self._calibration_matrix = np.zeros(shape=self._sensor_sample_size)
    self._calibration_completed = False if self._calibration_duration_s is not None else True
    
  
  def _connect(self, timeout_s=10):
    if TouchStreamer._connect(self, timeout_s=timeout_s):
      # Add shear-specific streams.
      for sensor_name in self._sensor_names_active:
        self.add_stream(device_name=sensor_name,
                        stream_name='tactile_data_calibrated',
                        data_type='float32',
                        sample_size=self._sensor_sample_size,
                        sampling_rate_hz=None,
                        extra_data_info={},
                        data_notes=OrderedDict([
                          ('Description', 'ADC readings from the matrix of tactile sensors '
                                          'on the glove.  Higher readings indicate '
                                          'higher pressures.  See the calibration periods '
                                          'for more information about conversions.'),
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
      return True
    else:
      return False
    
  #######################################
  ###### INTERFACE WITH THE SENSOR ######
  #######################################
  
  # Compute shear-specific quantities from a tactile data matrix.
  def _compute_shear(self, time_s, data_matrix):
    # Compute shear-specific quantities.
    if time_s is not None:
      # Subtract an average of initial readings as a rough calibration.
      if not self._calibration_completed:
        # Record the initial matrices.
        if self._calibration_startTime_s is None:
          self._calibration_startTime_s = time.time()
        if time.time() - self._calibration_startTime_s < self._calibration_duration_s:
          self._calibration_matrices.append(data_matrix)
        else:
          # Aggregate the initial matrices to create a calibration matrix.
          self._calibration_matrix = np.median(np.array(self._calibration_matrices), axis=0)
          self._calibration_completed = True
      data_matrix_calibrated = data_matrix - self._calibration_matrix # will do nothing if the matrix hasn't been computed or if calibration is disabled
      
      # Compute the total force in each shear square.
      toConvolve_tiled_magnitude = np.array([[1,1],[1,1]])
      data_matrix_tiled_magnitude = convolve2d_strided(data_matrix_calibrated, toConvolve_tiled_magnitude, stride=2)
    
      # Compute the force angle in each shear square.
      toConvolve_tiled_x = np.array([[-1,1],[-1,1]])
      toConvolve_tiled_y = np.array([[1,1],[-1,-1]])
      data_matrix_tiled_x = convolve2d_strided(data_matrix_calibrated, toConvolve_tiled_x, stride=2)
      data_matrix_tiled_y = convolve2d_strided(data_matrix_calibrated, toConvolve_tiled_y, stride=2)
      data_matrix_tiled_shearAngle_rad = np.arctan2(data_matrix_tiled_y, data_matrix_tiled_x)
      data_matrix_tiled_shearMagnitude = np.linalg.norm(np.stack([data_matrix_tiled_y, data_matrix_tiled_x], axis=0), axis=0)
    
      # Return the data!
      return (time_s, data_matrix, data_matrix_calibrated, data_matrix_tiled_magnitude,
              data_matrix_tiled_shearAngle_rad, data_matrix_tiled_shearMagnitude)
    return (None, None, None, None, None, None)
  
    
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
        if stream_name == 'tactile_data_calibrated':
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
          (time_s, data_matrix) \
            = self._read_sensor(sensor_name, suppress_printing=(count < 10))
          (time_s, data_matrix, data_matrix_calibrated, data_matrix_tiled_magnitude,
           data_matrix_tiled_shearAngle_rad, data_matrix_tiled_shearMagnitude) \
            = self._compute_shear(time_s, data_matrix)
          if time_s is not None and data_matrix is not None:
            self.append_data(sensor_name, 'tactile_data', time_s, data_matrix)
            self.append_data(sensor_name, 'tactile_data_calibrated', time_s, data_matrix_calibrated)
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
      self._log_error('\n\n***ERROR RUNNING TouchShearStreamerESP for sensor %s:\n%s\n' % (sensor_name, traceback.format_exc()))
    finally:
      pass
  

#####################
###### TESTING ######
#####################
if __name__ == '__main__':
  # Configuration.
  duration_s = 7200
  
  # Connect to the device(s).
  touch_streamer = TouchShearStreamerESP(com_ports={
    'shear-test' : 'COM14',
  }, print_status=True, print_debug=False)
  touch_streamer.connect()
  
  # Run for the specified duration and periodically print the sample rate.
  print('\nRunning for %gs!' % duration_s)
  touch_streamer.run()
  start_time_s = time.time()
  try:
    while time.time() - start_time_s < duration_s:
      time.sleep(2)
      fps_msg = ' Duration: %6.2fs' % (time.time() - start_time_s)
      for device_name in touch_streamer.get_device_names():
        stream_name = touch_streamer.get_stream_names(device_name=device_name)[0]
        num_timesteps = touch_streamer.get_num_timesteps(device_name, stream_name)
        fps_msg += ' | %s: %4d Timesteps (Fs = %6.2f Hz)' % \
                   (device_name, num_timesteps, ((num_timesteps)/(time.time() - start_time_s)))
        fps_msg += ' | matrix index %d' % touch_streamer._matrix_indexes[device_name]
      print(fps_msg)
  except:
    pass
  touch_streamer.stop()
  print('\nDone!\n')
  
  
  
  











