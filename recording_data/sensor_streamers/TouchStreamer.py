
from sensor_streamers.SensorStreamer import SensorStreamer
from visualizers.HeatmapVisualizer import HeatmapVisualizer

import serial
from threading import Thread
import numpy as np
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
class TouchStreamer(SensorStreamer):

  ########################
  ###### INITIALIZE ######
  ########################

  def __init__(self, streams_info=None,
                log_player_options=None, visualization_options=None,
                print_status=True, print_debug=False, log_history_filepath=None):
    SensorStreamer.__init__(self, streams_info,
                              log_player_options=log_player_options,
                              visualization_options=visualization_options,
                              print_status=print_status, print_debug=print_debug,
                              log_history_filepath=log_history_filepath)

    self._log_source_tag = 'tactile'
    
    # Define the connected sensors.
    # Port configurations should be checked on the operating system.
    # Setting a port to None will ignore that sensor.
    self._com_ports = {
      'tactile-glove-left' : 'COM3', # None
      'tactile-glove-right': 'COM6', # None
    }
    # Configurations that should match settings in Arduino code.
    self._sensor_waits_for_request = True # Should match setting in Arduino code
    self._sensor_sends_debug_values = False # Should match setting in Arduino code
    self._sensor_bitshift = 5
    self._sensor_streams_rows = not self._sensor_waits_for_request # whether each message is a row of data or the entire matrix of data
    self._baud_rate_bps = 1000000
    self._sensor_sample_size = (32, 32) # (height, width)
    self._data_length_expected_perMatrix = int(2*np.prod(self._sensor_sample_size)) # each uint16 byte will be sent as two consecutive uint8 bytes
    self._data_length_expected_perRow    = int(2*self._sensor_sample_size[0]) # each uint16 byte will be sent as two consecutive uint8 bytes
    if self._sensor_streams_rows:
      self._data_length_expected = self._data_length_expected_perRow
    else:
      self._data_length_expected = self._data_length_expected_perMatrix
    
    # Initialize state.
    self._sensor_names = list(self._com_ports.keys())
    self._sensor_names_active = []
    self._sensor_serials = {}
    self._matrix_indexes = {}
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
                                      baudrate=self._baud_rate_bps,
                                      timeout=1.0)
        # Make the buffers large enough to accommodate unexpected/stochastic delays
        #  from the operating system that may cause temporary backlogs of data to process.
        self._sensor_serials[sensor_name].set_buffer_size(rx_size=50*self._data_length_expected,
                                                          tx_size=50*self._data_length_expected)
        self.add_stream(device_name=sensor_name,
                    stream_name='tactile_data',
                    data_type='float32',
                    sample_size=self._sensor_sample_size,
                    sampling_rate_hz=None,
                    extra_data_info={},
                    data_notes=OrderedDict([
                      ('Description', 'ADC readings from the matrix of tactile sensors '
                                      'on the glove.  Higher readings indicate '
                                      'higher pressures.  See the calibration periods '
                                      'for more information about conversions.'),
                      ('Range', '[0, 4095]'),
                    ]))
        sensor_names_connected.append(sensor_name)
        # Initialize the counter that will store the last received matrix index.
        self._matrix_indexes[sensor_name] = None
      except:
        self._sensor_serials[sensor_name] = None
    self._log_status('Found the following tactile sensors connected: %s' % sensor_names_connected)

    # Wait for the sensors to become active.
    # For example, the Arduino may restart upon serial connection.
    self._sensor_names_active = []
    for sensor_name in sensor_names_connected:
      self._log_status('Waiting for the tactile sensor %s to start streaming data' % sensor_name)
      wait_start_time_s = time.time()
      while time.time() - wait_start_time_s < 10:
        (time_s, data) = self._read_sensor(sensor_name, suppress_printing=True)
        if data is not None:
          self._sensor_names_active.append(sensor_name)
          break
        time.sleep(0.05)
    self._log_status('Found the following tactile sensors active: %s' % self._sensor_names_active)

    # Return success if all desired sensors were found to be active.
    if len(self._sensor_names_active) == len([sensor_name for (sensor_name, port) in self._com_ports.items() if port is not None]):
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
    sensor_serial = self._sensor_serials[sensor_name]
    if sensor_serial is None or not sensor_serial.is_open:
      return (None, None)

    # sensor_serial.reset_input_buffer()
    # prev_row_index = None
    # while True:
    #   data = sensor_serial.readline()
    #   try:
    #     matrix_index = data[0] - (ord('\n')+1)
    #     row_index = data[1] - (ord('\n')+1)
    #   except:
    #     print(len(data), data)
    #     raise
    #   print('%3d %2d' % (matrix_index, row_index))
    #   if prev_row_index is not None:
    #     if not ((row_index == 0 and prev_row_index == 31) or (row_index > 0 and row_index == prev_row_index + 1)):
    #       print(len(data), data)
    #       raise
    #   prev_row_index = row_index

    # Read and process rows of data until an entire matrix is received.
    matrix_time_s = None
    data_matrix = np.zeros(shape=self._sensor_sample_size)
    last_row_index = None
    while last_row_index is None or last_row_index < (self._sensor_sample_size[0] - 1):
      # Receive data from the sensor
      data = sensor_serial.readline()
      time_s = time.time()
      # Validate that the length is at least enough for metadata (the rest will be validated later).
      if len(data) < 3:
        if not suppress_printing:
          self._log_warn('WARNING: Tactile sensor [%s] sent less than 3 bytes of data. Ignoring the data.' % (sensor_name))
        return (None, None)
      
      # Extract the matrix index, row index, and the row data.
      matrix_index = data[0] - (ord('\n')+1)
      row_index = data[1] - (ord('\n')+1)
      data_row = data[2:-1] # ignore the newline at the end
      # print(len(data), 'mat:', np.frombuffer(data, dtype=np.uint8).astype(np.uint16)[0], matrix_index, 'row:', np.frombuffer(data, dtype=np.uint8).astype(np.uint16)[1], row_index)
      
      # Validate the row index, which should always increment by 1 (or be 0 if it's the first read).
      if last_row_index is None:
        if row_index != 0:
          if not suppress_printing:
            self._log_warn('WARNING: Tactile sensor [%s] started with row index %d instead of 0 for matrix index %d. Ignoring the data.' % (sensor_name, row_index, matrix_index))
          return (None, None)
      else:
        if row_index != last_row_index+1:
          if not suppress_printing:
            self._log_warn('WARNING: Tactile sensor [%s] sent row index %d, but expected %d, for matrix index %d. Ignoring the data.' % (sensor_name, row_index, last_row_index+1, matrix_index))
          return (None, None)
      # Validate the matrix index, which should match the previous one unless this is the first row.
      if last_row_index is None:
        if matrix_index == self._matrix_indexes[sensor_name]:
          if not suppress_printing:
            self._log_warn('WARNING: Tactile sensor [%s] sent a starting row for matrix index %d which was already processed. Ignoring the data.' % (sensor_name, matrix_index))
          return (None, None)
      else:
        if matrix_index != self._matrix_indexes[sensor_name]:
          if not suppress_printing:
            self._log_warn('WARNING: Tactile sensor [%s] sent row index %d for matrix index %d, but was expecting matrix index %d. Ignoring the data.' % (sensor_name, row_index, matrix_index, self._matrix_indexes[sensor_name]))
          return (None, None)
      # Validate the length of the data.
      if len(data_row) != self._data_length_expected:
        if not suppress_printing:
          self._log_warn('WARNING: Tactile sensor [%s] sent %d values instead of %d values for row index %d of matrix index %d. Ignoring the data.' % (sensor_name, len(data_row), self._data_length_expected, row_index, matrix_index))
        return (None, None)

      # Parse the row data:
      #  Combine consecutive uint8 bytes into single uint16 bytes.
      #  Subtract the offset added in the ESP ('\n'+1) that ensures data never contains a newline.
      data_row = np.frombuffer(data_row, dtype=np.uint8).astype(np.uint16)
      data_row = data_row - (ord('\n')+1)
      data_row = data_row[0::2]*(2**self._sensor_bitshift) + data_row[1::2]
      # if (not suppress_printing) and self._print_debug:
      #   self._log_debug('Received data from %s with size %s and min/max %d/%d for matrix index %d and row index %d: \n%s' % (sensor_name, data_row.shape, np.min(data_row), np.max(data_row), matrix_index, row_index, data_row))
      
      # Record the data, and the timestamp if this is the first row.
      data_matrix[row_index, :] = data_row
      if row_index == 0:
        matrix_time_s = time_s
      
      # Update state.
      last_row_index = row_index
      self._matrix_indexes[sensor_name] = matrix_index
      
    if (not suppress_printing) and self._print_debug:
      self._log_debug('Received data from %s with size %s and min/max %d/%d for matrix index %d: \n%s' % (sensor_name, data_matrix.shape, np.min(data_matrix), np.max(data_matrix), matrix_index, data_matrix))
    
    # Validate the data if debug values are being used.
    if self._sensor_sends_debug_values:
      target_matrix = np.reshape(np.arange(np.prod(self._sensor_sample_size)), self._sensor_sample_size)
      if not np.all(data_matrix == target_matrix):
        if not suppress_printing:
          self._log_warn('WARNING: Tactile sensor [%s] sent data that does not match the expected debug values for matrix index %d: \n%s' % (sensor_name, matrix_index, data_matrix))
    # Return the data!
    return (matrix_time_s, data_matrix)
    
  # Read from the sensor using the request paradigm,
  #  in which this program explicitly requests every sample.
  #  The whole matrix of data will be sent as a single line, with the format [matrix_index][matrix_data][\n].
  def _read_sensor_requestParadigm(self, sensor_name, suppress_printing=False):
    sensor_serial = self._sensor_serials[sensor_name]
    if sensor_serial is None or not sensor_serial.is_open:
      return (None, None)

    attempts = 0
    data = b''
    while len(data) == 0 and attempts < 3:
      # Remove any existing characters in the buffer.
      sensor_serial.reset_input_buffer()

      # Request a sensor reading
      if not suppress_printing:
        self._log_debug('Requesting tactile data from %s' % sensor_name)
      sensor_serial.write('a'.encode('utf-8'))
    
      # Receive data from the sensor
      data = sensor_serial.readline() # sensor_serial.read(self._data_length_expected)
      time_s = time.time() # record a timestamp for the data
      attempts += 1
      if len(data) == 0:
        time.sleep(1)
    # Unpack the data.
    matrix_index = data[0] - (ord('\n')+1)
    data_matrix = data[1:-1] # ignore the newline character at the end
    
    # Validate the matrix index, which should be different than the previous one.
    if matrix_index == self._matrix_indexes[sensor_name]:
      if not suppress_printing:
        self._log_warn('WARNING: Tactile sensor [%s] sent a data for matrix index %d which was already processed. Ignoring the data.' % (sensor_name, matrix_index))
      return (None, None)
    # Validate the length of the data.
    if len(data_matrix) != self._data_length_expected:
      if not suppress_printing:
        self._log_warn('WARNING: Tactile sensor [%s] sent %d values instead of %d values for matrix index %d. Ignoring the data.' % (sensor_name, len(data), self._data_length_expected, matrix_index))
      return (None, None)

    # Parse the matrix data:
    #  Combine consecutive uint8 bytes into single uint16 bytes.
    #  Subtract the offset added in the ESP ('\n'+1) that ensures data never contains a newline.
    data_matrix = np.frombuffer(data_matrix, dtype=np.uint8).astype(np.uint16)
    data_matrix = data_matrix - (ord('\n')+1)
    data_matrix = data_matrix[0::2]*(2**self._sensor_bitshift) + data_matrix[1::2]
    data_matrix = data_matrix.reshape(self._sensor_sample_size)
    if (not suppress_printing) and self._print_debug:
      self._log_debug('Received data from %s with size %s and min/max %d/%d for matrix index %d: \n%s' % (sensor_name, data_matrix.shape, np.min(data_matrix), np.max(data_matrix), matrix_index, data_matrix))
    
    # Update state.
    self._matrix_indexes[sensor_name] = matrix_index

    # Validate the data if debug values are being used.
    if self._sensor_sends_debug_values:
      target_matrix = np.reshape(np.arange(np.prod(self._sensor_sample_size)), self._sensor_sample_size)
      if not np.all(data_matrix == target_matrix):
        if not suppress_printing:
          self._log_warn('WARNING: Tactile sensor [%s] sent data that does not match the expected debug values for matrix index %d: \n%s' % (sensor_name, matrix_index, data_matrix))

    # Return the data!
    return (time_s, data_matrix)


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
                                                    'class': HeatmapVisualizer,
                                                    'colorbar_levels': 'auto', # a 2-element list, 'auto', or omitted
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
            self.append_data(sensor_name, 'tactile_data', time_s, data)
          count = count + 1
        except serial.serialutil.SerialException:
          self._log_error('*** Could not read from tactile sensor %s:\n%s\n' % (sensor_name, traceback.format_exc()))
          try:
            self._sensor_serials[sensor_name] = serial.Serial(
                self._com_ports[sensor_name],
                baudrate=self._baud_rate_bps,
                timeout=1.0)
            time.sleep(0.5)
          except serial.serialutil.SerialException:
            self._log_error('*** Could not reconnect to tactile sensor %s - waiting a bit then retrying' % (sensor_name))
            time.sleep(5)
          
      for sensor_name in self._sensor_names_active:
        self._sensor_serials[sensor_name].close()
    except KeyboardInterrupt: # The program was likely terminated
      pass
    except:
      self._log_error('\n\n***ERROR RUNNING TouchStreamer for sensor %s:\n%s\n' % (sensor_name, traceback.format_exc()))
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
    # Join the threads to wait until are all done.
    for sensor_name in self._sensor_names_active:
      self._run_threads[sensor_name].join()
    
  # Clean up and quit
  def quit(self):
    self._log_debug('TouchStreamer quitting')
    SensorStreamer.quit(self)


#####################
###### TESTING ######
#####################
if __name__ == '__main__':
  # Configuration.
  duration_s = 7200
  
  # Connect to the device(s).
  touch_streamer = TouchStreamer(print_status=True, print_debug=False)
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
        fps_msg += ' | %s: %4d Timesteps (Fs = %6.2f Hz)' %  \
                    (device_name, num_timesteps, ((num_timesteps)/(time.time() - start_time_s)))
        fps_msg += ' | matrix index %d' % touch_streamer._matrix_indexes[device_name]
      print(fps_msg)
  except:
    pass
  touch_streamer.stop()
  print('\nDone!\n')
  
  
  
  











