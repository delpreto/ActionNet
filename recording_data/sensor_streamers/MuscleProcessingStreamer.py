############
#
# Copyright (c) 2024 MIT CSAIL and Joseph DelPreto
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
# Created 2021-2024 for the MIT ActionNet project by Joseph DelPreto [https://josephdelpreto.com].
#
############

import numpy as np
from scipy import interpolate
from scipy.signal import butter, lfilter
import json
import pickle
from threading import Thread
from threading import Lock
import copy
import time
import socket

from sensor_streamers.SensorStreamer import SensorStreamer
from sensor_streamers.MyoStreamer import MyoStreamer

################################################
################################################
# A class to perform real-time processing on streaming Myo data.
################################################
################################################
class MuscleProcessingStreamer(MyoStreamer):
  
  ########################
  ###### INITIALIZE ######
  ########################
  
  def __init__(self, streams_info=None, num_myos=1,
                log_player_options=None, visualization_options=None,
                print_status=True, print_debug=False, log_history_filepath=None):
    MyoStreamer.__init__(self, streams_info=streams_info, num_myos=num_myos,
                log_player_options=log_player_options, visualization_options=visualization_options,
                print_status=print_status, print_debug=print_debug, log_history_filepath=log_history_filepath)
    
    self._log_source_tag = 'muscle'
    
    # Define EMG processing.
    self._lowpass_cutoff_emg_Hz = 5
    
    # Define buffers of streaming data.
    # For example, they can be accessed as self._buffers['data']['emg'][time_index]
    self._buffer_duration_s = 30
    self._device_name_to_buffer = 'myo-left'
    self._device_name_processed_data = 'myoProcessed-left'
    self._stream_keys = [
      'emg',
      'acceleration_g',
      'angular_velocity_deg_s',
      'orientation_quaternion',
      'gesture',
      'battery',
      'synced',
      'connected',
      # 'rssi',
      ]
    self._buffers = {
      'time_s': dict((stream_key, []) for stream_key in self._stream_keys),
      'data': dict((stream_key, []) for stream_key in self._stream_keys),
    }
    self._buffers_mutex = Lock()
    
    # Create a thread to periodically process the buffers of data.
    self._process_data_period_s = 1/1
    self._processData_thread = Thread(target=self._process_data, args=())
    self._processData_thread_is_running = True
    self._processData_thread.start()
    
    # Create a thread to handle UDP communication of processed data.
    self._receiver_ip_address = '128.30.10.150'
    self._socket_ports = {
      'emg_envelope': 65432,
      'emg_stiffness': 65431,
    }
    self._udp_buffers = dict((stream_key, []) for stream_key in self._socket_ports)
    self._max_udp_buffer_length = 1000
    self._udp_buffers_mutex = Lock()
    self._udp_thread_polling_period_s = 0.05
    self._udp_thread = Thread(target=self._send_data_udp, args=())
    self._udp_thread_is_running = True
    self._udp_thread.start()
    
  # Add new processed data streams when the device connects.
  def on_arm_synced(self, event):
    MyoStreamer.on_arm_synced(self, event)
    if self._device_name_to_buffer in self._metadata:
      # Add streams for processed EMG data.
      self.add_stream(self._device_name_processed_data, stream_name='emg_envelope',
                      data_type='int32', sample_size=[8], sampling_rate_hz=200,
                      data_notes=None)
      self.add_stream(self._device_name_processed_data, stream_name='emg_stiffness',
                      data_type='int32', sample_size=[1], sampling_rate_hz=200,
                      data_notes=None)
      self._metadata[self._device_name_processed_data] = self._metadata[self._device_name_to_buffer].copy()
    
  # Intercept data as it is appended to the log.
  def append_data(self, device_name, stream_name, time_s, data, extra_data=None):
    # Call the parent method to handle the logging, visualization, etc.
    MyoStreamer.append_data(self, device_name=device_name, stream_name=stream_name,
                            time_s=time_s, data=data, extra_data=extra_data)
    # Maintain our custom buffers of data.
    if device_name == self._device_name_to_buffer and stream_name in self._stream_keys:
      self._buffers_mutex.acquire()
      # Add the new data to the end of the buffers.
      self._buffers['time_s'][stream_name].append(time_s)
      self._buffers['data'][stream_name].append(data)
      # Remove old entries from the beginning of the buffer if needed.
      while self._buffers['time_s'][stream_name][-1] - self._buffers['time_s'][stream_name][0] > self._buffer_duration_s:
        del self._buffers['time_s'][stream_name][0]
        del self._buffers['data'][stream_name][0]
      self._buffers_mutex.release()
  
  # Process the buffers of streaming data.
  def _process_data(self):
    last_process_time_s = time.time()
    while self._processData_thread_is_running:
      # Process the data if it is time to do so.
      if time.time() - last_process_time_s >= self._process_data_period_s:
        # Get a snapshot of the buffers, and convert them to numpy arrays.
        self._buffers_mutex.acquire()
        emg_time_s = np.array(self._buffers['time_s']['emg'])
        emg_data = np.array(self._buffers['data']['emg'])
        acceleration_g_time_s = np.array(self._buffers['time_s']['acceleration_g'])
        acceleration_g_data = np.array(self._buffers['data']['acceleration_g'])
        gestures_time_s = np.array(self._buffers['time_s']['gesture'])
        gestures = np.array(self._buffers['data']['gesture'])
        self._buffers_mutex.release()
        
        # Skip if there is no data.
        if len(emg_time_s) == 0:
          last_process_time_s = time.time()
          continue
          
        # Example of upsampling one of the IMU data streams to match the EMG time vector.
        fn_interpolate_acceleration = interpolate.interp1d(
                                        acceleration_g_time_s, # x values
                                        acceleration_g_data,   # y values
                                        axis=0,              # axis of the data along which to interpolate
                                        kind='linear',       # interpolation method, such as 'linear', 'zero', 'nearest', 'quadratic', 'cubic', etc.
                                        fill_value='extrapolate' # how to handle x values outside the original range
                                        )
        acceleration_g_data = fn_interpolate_acceleration(emg_time_s)
        acceleration_g_time_s = emg_time_s
        
        # Example of doing something with a recent gesture detection.
        if len(gestures_time_s) > 0 and np.any(gestures_time_s > last_process_time_s):
          new_gesture_indexes = np.where(gestures_time_s > last_process_time_s)[0]
          new_gestures = gestures[new_gesture_indexes]
          new_gestures = [gesture for gesture in new_gestures if gesture != 'rest']
          new_gestures_time_s = gestures_time_s[new_gesture_indexes]
          if len(new_gestures) > 0:
            print('Detected new gestures: %s' % new_gestures)
        
        # Compute the EMG envelope.
        emg_Fs_Hz = (len(emg_time_s)-1)/(emg_time_s[-1] - emg_time_s[0])
        emg_data_rectified = np.abs(emg_data)
        emg_data_envelope = lowpass_filter(emg_data_rectified, self._lowpass_cutoff_emg_Hz, emg_Fs_Hz)
        
        # Estimate overall stiffness.
        emg_stiffness = np.sum(emg_data_envelope, axis=1)
        # emg_stiffness = np.prod(emg_data_envelope, axis=1)
        
        # Add processed data as streams that can be saved and visualized.
        for (time_index, time_s) in enumerate(emg_time_s):
          if time_s > last_process_time_s:
            SensorStreamer.append_data(self,
                                       device_name=self._device_name_processed_data,
                                       stream_name='emg_envelope',
                                       time_s=emg_time_s[time_index],
                                       data=list(emg_data_envelope[time_index, :]))
            SensorStreamer.append_data(self,
                                       device_name=self._device_name_processed_data,
                                       stream_name='emg_stiffness',
                                       time_s=emg_time_s[time_index],
                                       data=[emg_stiffness[time_index]])
        
        # Send results via UDP.
        for (time_index, time_s) in enumerate(emg_time_s):
          if time_s > last_process_time_s:
            self._udp_buffers_mutex.acquire()
            data_to_send = (emg_time_s[time_index], emg_data_envelope[time_index, :].tolist())
            self._udp_buffers['emg_envelope'].append(data_to_send)
            data_to_send = (emg_time_s[time_index], emg_stiffness[time_index])
            self._udp_buffers['emg_stiffness'].append(data_to_send)
            while len(self._udp_buffers['emg_envelope']) > self._max_udp_buffer_length:
              del self._udp_buffers['emg_envelope'][0]
              del self._udp_buffers['emg_stiffness'][0]
            self._udp_buffers_mutex.release()
            
        # Update loop state.
        last_process_time_s = time.time()
        
      # Sleep until the next processing time.
      target_processing_time_s = last_process_time_s + self._process_data_period_s
      sleep_duration_s = target_processing_time_s - time.time()
      if sleep_duration_s > 0.01:
        time.sleep(sleep_duration_s)
  
  # Send data via UDP when it is ready.
  def _send_data_udp(self):
    # Create sockets to send data via udp.
    self._sockets = {}
    for (stream_name, port) in self._socket_ports.items():
      self._sockets[stream_name] = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Periodically send any new data that has been processed.
    while self._udp_thread_is_running:
      self._udp_buffers_mutex.acquire()
      for (stream_name, buffer) in self._udp_buffers.items():
        for data in buffer:
          self._sockets[stream_name].sendto(json.dumps(data).encode(), (self._receiver_ip_address, self._socket_ports[stream_name]))
      self._udp_buffers = dict((stream_key, []) for stream_key in self._socket_ports)
      self._udp_buffers_mutex.release()
      time.sleep(self._udp_thread_polling_period_s)
    for (stream_name, stream_socket) in self._sockets.items():
      stream_socket.close()
      
  
  ###########################
  ###### VISUALIZATION ######
  ###########################

  # Specify how the stream should be visualized.
  # visualization_options can have entries for 'emg' and 'imu'.
  # The same settings will be used for all connected Myos.
  def get_default_visualization_options(self, visualization_options=None):
    processed_options = MyoStreamer.get_default_visualization_options(self, visualization_options=visualization_options)
    # Copy the emg graph options.
    emg_envelope_options = processed_options[self._device_name_to_buffer]['emg'].copy()
    emg_stiffness_options = processed_options[self._device_name_to_buffer]['emg'].copy()
    
    # Override with any provided options.
    if isinstance(visualization_options, dict):
      if 'emg_envelope' in visualization_options:
        for (k, v) in visualization_options['emg'].items():
          emg_envelope_options[k] = v
      if 'emg_stiffness' in visualization_options:
        for (k, v) in visualization_options['emg'].items():
          emg_stiffness_options[k] = v

    # Set all devices to the same options.
    processed_options[self._device_name_processed_data] = {
      'emg_envelope': emg_envelope_options,
      'emg_stiffness': emg_stiffness_options
    }

    return processed_options
  
  # Clean up and quit
  def __del__(self):
    self.quit()
    
  def quit(self):
    # Stop the processing thread and wait for it to finish.
    try:
      self._processData_thread_is_running = False
      self._udp_thread_is_running = False
      self._processData_thread.join()
      self._udp_thread.join()
    except:
      pass
    MyoStreamer.quit(self)


# Will filter each column of the data.
def lowpass_filter(data, cutoff, Fs, order=5):
  nyq = 0.5 * Fs
  normal_cutoff = cutoff / nyq
  b, a = butter(order, normal_cutoff, btype='low', analog=False)
  y = lfilter(b, a, data.T).T
  return y






















