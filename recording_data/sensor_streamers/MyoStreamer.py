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
from visualizers.OrientationVisualizer import OrientationVisualizer

# Add the myo path and then import the library.
# Use a path relative to this script directory (rather than the current execution directory, which could be arbitrary).
import os
import sys
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_dir, '..', 'myo_library', 'myo-python-master'))
import myo

from threading import Lock, Thread
from collections import OrderedDict
import time
import traceback

from utils.dict_utils import *
from utils.time_utils import *


################################################
################################################
# A class to interface with the Myo armband.
# Multiple Myos may be connected.
# Myo Connect must be running and connected to all desired armbands.
# Will stream the following data:
#   EMG (200Hz)
#   3-axis acceleration (50Hz)
#   3-axis angular velocity (50Hz)
#   Quaternion orientations (50Hz)
#   Gesture classifications (asynchronous)
#   Battery and RSSI levels (at start/stop or on demand)
################################################
################################################
class MyoStreamer(myo.ApiDeviceListener, SensorStreamer):

  ########################
  ###### INITIALIZE ######
  ########################

  def __init__(self, streams_info=None, num_myos=1,
                log_player_options=None, visualization_options=None,
                print_status=True, print_debug=False, log_history_filepath=None):
    myo.ApiDeviceListener.__init__(self)
    SensorStreamer.__init__(self, streams_info,
                              log_player_options=log_player_options,
                              visualization_options=visualization_options,
                              print_status=print_status, print_debug=print_debug,
                              log_history_filepath=log_history_filepath)

    self._log_source_tag = 'myo'
    
    # Handles for the actual Myo class interface and devices
    self._myo_hub = None
    self._deviceProxies = OrderedDict()

    # Configuration
    self._target_num_myos = num_myos
    self._battery_rssi_polling_period_s = 120

    # Define data notes that will be associated with streams created below.
    self._define_data_notes()
    
    # Threading variables
    self._connectAndRun_thread = None
    self._connectAndRunThread_running = False
    self._locks = OrderedDict()
    self._logging = False # will ignore Myo data callbacks if not logging yet
    
    # Store an offset that converts from Myo event timestamps to epoch timestamps.
    # Will store an offset for each device in case they are unique.
    self._event_timestamp_offset_toAdd = {}

  # Initiate connections to the Myos, and wait for the desired number of devices.
  # @return True or False based on whether the desired number
  #          of Myos connected and synced within the timeout period.
  def _connect(self, timeout_s=10):
    # Start a thread that will wait for connections
    #  (and it will also do the main execution loop when run() is called).
    # First, check that the thread is not already running.
    if self._connectAndRunThread_running:
      self._log_error('')
      self._log_error('XXXXXXXXXX')
      self._log_error('WARNING: MyoStreamer connect() is being ignored since it was already called.')
      self._log_error('XXXXXXXXXX')
      self._log_error('')
      return True
    self._connectAndRun_thread = Thread(target=self._connect_and_run_threadFn, args=())
    self._connectAndRunThread_running = True
    self._connectAndRun_thread.start()

    # Wait for connection(s)
    num_myos = self._target_num_myos
    self._log_status('Waiting for connection to %d %s.' % (num_myos, 'Myo' if num_myos == 1 else 'Myos'))
    self._log_status(' Make sure Myo Connect is running and is connected to the %s.' % ('Myo' if num_myos == 1 else 'Myos'))
    start_time_s = time.time()
    while self.get_num_devices() < num_myos and time.time() - start_time_s < timeout_s:
      time.sleep(0.1)
    num_myos_connected = self.get_num_devices()
    all_connected = num_myos_connected == num_myos
    # Wait for each device to be synced
    while False in [self._metadata[device_name]['arm'] is not None for device_name in self.get_device_names()] \
        and time.time() - start_time_s < timeout_s:
      time.sleep(0.1)
    num_myos_synced = sum([self._metadata[device_name]['arm'] is not None for device_name in self.get_device_names()])
    all_synced = num_myos_synced == num_myos

    # Stop the thread and raise an exception if connection was unsuccessful.
    if not (all_connected and all_synced):
      self._log_error('')
      self._log_error('XXXXXXXXXX')
      self._log_error('ERROR: Connected %d/%d %s and synced %d/%d %s.'
                      % (num_myos_connected, num_myos, 'Myo' if num_myos == 1 else 'Myos',
                         num_myos_synced, num_myos, 'Myo' if num_myos == 1 else 'Myos'))
      self._log_error('XXXXXXXXXX')
      self._log_error('')
      raise AssertionError('Unsuccessful Myo connection')

    # Print success
    self._log_status('Successfully connected and synced %d %s' % (num_myos, 'Myo' if num_myos == 1 else 'Myos'))
    return True

  #############################
  ###### GETTERS/SETTERS ######
  #############################

  # Get a unique device ID from an event object
  def _get_event_device_id(self, event, check_existence=True):
    device_id = str(event.deviceProxy._device.handle) # note that it seems better to use something like MAC address, but some event types yield an error when trying to fetch that
    if check_existence and device_id not in self._get_device_ids():
      msg = 'Received an event for unknown device %s' % device_id
      self._log_error('\nXXXXX\n%s\nXXXXXX\n' % msg)
      raise Exception(msg)
    return device_id

  # Get the device name currently associated with the device in an event object
  def _get_event_device_name(self, event, check_existence=True):
    device_id = self._get_event_device_id(event, check_existence=check_existence)
    for (device_name, device_info) in self._metadata.items():
      if device_info['device_id'] == device_id:
        return device_name
    return None

  # Get known device IDs
  def _get_device_ids(self):
    device_ids = []
    for device_name in self.get_device_names():
      device_ids.append(self._metadata[device_name]['device_id'])
    return device_ids

  # Convert among device IDs, names, and indexes
  def _get_device_index(self, device_id=None, device_name=None):
    if device_id is not None:
      return self._get_device_ids().index(device_id)
    if device_name is not None:
      return self.get_device_names().index(device_id)

  def _get_device_id(self, device_index=None, device_name=None):
    if device_name is None:
      device_name = self.get_device_names()[device_index]
    return self._metadata[device_name]['device_id']

  # Rename a device
  def rename_device(self, old_device_name, new_device_name, acquire_lock=True):
    # Acquire locks if desired (should do so unless the caller already acquired them)
    if acquire_lock:
      self._locks[old_device_name].acquire()
    # Rename the device in Myo-specific dictionaries
    if new_device_name in self._locks:
      raise AssertionError('ERROR: More than one Myo will have the same name. They probably think they are on the same arm, so you should re-sync them.')
    self._locks = rename_dict_key(self._locks, old_device_name, new_device_name)
    self._deviceProxies = rename_dict_key(self._deviceProxies, old_device_name, new_device_name)
    # Rename the device in any SensorStreamer variables
    SensorStreamer.rename_device(self, old_device_name, new_device_name)
    # Release locks if they were acquired
    if acquire_lock:
      self._locks[new_device_name].release()

  ########################################
  ###### MYO SETUP/STATUS CALLBACKS ######
  ########################################

  def on_paired(self, event):
    pass
  def on_unpaired(self, event):
    pass

  # When a new Myo is connected, initialize variables and data streams.
  def on_connected(self, event):
    # Get the unique device identifier
    device_id = self._get_event_device_id(event, check_existence=False)
    # Set the device name to the ID for now (can be renamed later)
    device_name = device_id
    self._log_status('Connected to Myo %s' % device_name)

    # Check whether this device is already stored, based on the MAC address.
    # If so, do nothing since the device names should not change mid-experiment.
    is_new_device = True
    mac_new = event.mac_address
    for (device_name_toCheck, device_info_toCheck) in self._metadata.items():
      if str(device_info_toCheck['mac_address']) == str(mac_new):
        is_new_device = False
        device_name = device_name_toCheck
        self._log_status('The Myo was already known, so will use the existing device entry %s' % device_name)

    # Initialize the device and its streams if it is new.
    if is_new_device:
      self._locks[device_name] = Lock()
      with self._locks[device_name]:
        # Record metadata about the device
        self._metadata[device_name] = OrderedDict([
          ('arm', None),
          ('x_direction', None),
          ('mac_address', event.mac_address),
          ('device_id', device_id),
          ('device_handle', event.deviceProxy._device.handle),
          ('firmware_version', event.deviceProxy.firmware_version),
          ('connect_time', event.deviceProxy.connect_time),
          ('pair_time', event.deviceProxy.pair_time),
        ])
        self._log_debug('Device metadata:')
        self._log_debug(str(self._metadata[device_name]))
        # Record a handle to the device itself
        self._deviceProxies[device_name] = event.deviceProxy
        # Add data streams for the device
        self.add_stream(device_name, stream_name='emg',                    data_type='int32',   sample_size=[8], sampling_rate_hz=200,  timesteps_before_solidified=1, extra_data_info={'time_s_original': {'data_type':'float64', 'sample_size':[1]}}, data_notes=self._data_notes['emg'])
        self.add_stream(device_name, stream_name='acceleration_g',         data_type='float64', sample_size=[3], sampling_rate_hz=50,   data_notes=self._data_notes['acceleration_g'])
        self.add_stream(device_name, stream_name='angular_velocity_deg_s', data_type='float64', sample_size=[3], sampling_rate_hz=50,   data_notes=self._data_notes['angular_velocity_deg_s'])
        self.add_stream(device_name, stream_name='orientation_quaternion', data_type='float64', sample_size=[4], sampling_rate_hz=50,   data_notes=self._data_notes['orientation_quaternion'])
        self.add_stream(device_name, stream_name='gesture',                data_type='S25',     sample_size=[1], sampling_rate_hz=None, data_notes=self._data_notes['gesture'])
        self.add_stream(device_name, stream_name='battery',                data_type='int32',   sample_size=[1], sampling_rate_hz=None, data_notes=self._data_notes['battery'])
        self.add_stream(device_name, stream_name='rssi',                   data_type='int32',   sample_size=[1], sampling_rate_hz=None, data_notes=self._data_notes['rssi'])
        self.add_stream(device_name, stream_name='synced',                 data_type='int32',   sample_size=[1], sampling_rate_hz=None, data_notes=self._data_notes['synced'])
        self.add_stream(device_name, stream_name='connected',              data_type='int32',   sample_size=[1], sampling_rate_hz=None, data_notes=self._data_notes['connected'])
  
      # Record that the Myo is not yet known to be synced.
      self.append_data(device_name, 'synced', time.time(), 0)
    
    # Record that the Myo is connected.
    self.append_data(device_name, 'connected', time.time(), 1)
    
    # Initialize an offset that converts from Myo event timestamps to epoch timestamps.
    # Will reset this each time the device connects, in case the offset changes.
    self._event_timestamp_offset_toAdd[device_name] = None
    
    # Initialize streaming.
    self.vibrate(device_name=device_name, duration='medium')
    event.deviceProxy.stream_emg(True)
    self.request_rssi(device_name)
    self.request_battery_level(device_name)
    
  def on_disconnected(self, event):
    device_name = self._get_event_device_name(event, check_existence=True)
    self.append_data(device_name, 'connected', time.time(), 0)
    # Print an update.
    self._log_status('Myo %s arm DISCONNECTED' % device_name)
    print('\n'*10)
    print('WARNING: MYO %s DISCONNECTED' % (device_name))
    print('\n'*10)
  
  def on_warmup_completed(self, event):
    pass
  # Note: locked refers to whether the built-in pose classifications are allowed to update or not
  def on_unlocked(self, event):
    pass
  def on_locked(self, event):
    pass

  # When the arm is synced, record the information and rename the Myo device accordingly.
  # @param event contains:
  #   timestamp: microseconds since epoch
  #   arm
  #   x_direction
  def on_arm_synced(self, event):
    device_name = self._get_event_device_name(event, check_existence=True)
    with self._locks[device_name]:
      # Check whether the device has been synced before.
      is_first_sync = self._metadata[device_name]['arm'] is None
      # Record the new sync information.
      arm = str(event.arm).split('.')[-1]
      x_direction = str(event.x_direction).split('.')[-1]
      self._metadata[device_name]['arm'] = arm
      self._metadata[device_name]['x_direction'] = x_direction
      # Record that the Myo has synced.
      self.append_data(device_name, 'synced', time.time(), 1)
      # Rename the device to include the arm if this is the first sync.
      if is_first_sync:
        new_device_name = 'myo-%s' % (arm)
        self.rename_device(device_name, new_device_name, acquire_lock=False)
        device_name = new_device_name
      # Print an update.
      self._log_status('Myo %s synced: arm %s | x_direction %s' % (device_name, arm, x_direction))
      if not is_first_sync:
        print('\n'*10)
        print('WARNING: MYO %s RE-SYNCED - arm is now %s, %s' % (device_name, arm, x_direction))
        print('\n'*10)

  def on_arm_unsynced(self, event):
    device_name = self._get_event_device_name(event, check_existence=True)
    with self._locks[device_name]:
      # Note: no longer remove the arm information since we assume that the Myo
      #  is still on the same arm, but it has just been active enough to trigger a re-sync.
      #  If this is changed, the sync callback should be updated with a different way of knowing whether to rename the device.
      # self._metadata[device_name]['arm'] = None
      # self._metadata[device_name]['x_direction'] = None
      # Record that the Myo has unsynced.
      self.append_data(device_name, 'synced', time.time(), 0)
      # Print an update.
      self._log_status('Myo %s arm UNsynced' % device_name)
      print('\n'*10)
      print('WARNING: MYO %s UNSYNCED' % (device_name))
      print('\n'*10)
  
  # Extract an epoch timestamp from a Myo event.
  def _get_time_s_from_event(self, device_name, event):
    # Get the timestamp associated with the event object.
    event_time_s = event.timestamp/1000000.0
    # If this is the first event being processed for this device,
    #  record an offset that converts to standard timestamp format.
    self._event_timestamp_offset_toAdd.setdefault(device_name, None)
    if self._event_timestamp_offset_toAdd[device_name] is None:
      self._event_timestamp_offset_toAdd[device_name] = time.time() - event_time_s
      self._log_status('Will use the following timestamp offset for Myo device [%s]: %f' % (device_name, self._event_timestamp_offset_toAdd[device_name]))
    # Add the stored offset to the event timestamp.
    return event_time_s + self._event_timestamp_offset_toAdd[device_name]
  
  ################################
  ###### MYO STATUS/CONTROL ######
  ################################

  # Ask the Myo to send its battery level, and optionally wait for the response.
  def request_battery_level(self, device_name, wait_for_response=False):
    # Record how many responses have been receieved so far, in case we want to wait for a new one
    num_battery_levels_start = len(self._data[device_name]['battery']['data'])
    # Request a new response
    self._deviceProxies[device_name].request_battery_level()
    # Wait for a new response if desired
    if wait_for_response:
      time_s_start = time.time()
      timeout_s = 5
      while len(self._data[device_name]['battery']['data']) == num_battery_levels_start and time.time() - time_s_start < timeout_s:
        time.sleep(0.01)

  # Ask the Myo to send its RSSI level, and optionally wait for the response.
  def request_rssi(self, device_name, wait_for_response=False):
    # Record how many responses have been receieved so far, in case we want to wait for a new one
    num_rssis_start = len(self._data[device_name]['rssi']['data'])
    # Request a new response
    self._deviceProxies[device_name].request_rssi()
    # Wait for a new response if desired
    if wait_for_response:
      time_s_start = time.time()
      timeout_s = 5
      while len(self._data[device_name]['rssi']['data']) == num_rssis_start and time.time() - time_s_start < timeout_s:
        time.sleep(0.01)

  # Ask all connected Myos to send battery and RSSI levels, and optionally wait for all responses.
  def request_battery_rssi_allDevices(self, wait_for_response=False):
    # Record how many responses have been receieved so far, in case we want to wait for a new one
    num_battery_levels_start = {}
    num_rssis_start = {}
    for device_name in self.get_device_names():
      num_battery_levels_start[device_name] = len(self._data[device_name]['battery']['data'])
      num_rssis_start[device_name] = len(self._data[device_name]['rssi']['data'])
    # Request the battery and RSSI levels
    for device_name in self.get_device_names():
      self.request_rssi(device_name, wait_for_response=False)
      self.request_battery_level(device_name, wait_for_response=False)
    # Wait for a new response from each device
    if wait_for_response:
      time_s_start = time.time()
      timeout_s = 5
      for device_name in self.get_device_names():
        while len(self._data[device_name]['battery']['data']) == num_battery_levels_start[device_name] and time.time() - time_s_start < timeout_s:
          time.sleep(0.01)
        while len(self._data[device_name]['rssi']['data']) == num_rssis_start[device_name] and time.time() - time_s_start < timeout_s:
          time.sleep(0.01)

  # Vibrate a Myo.
  # @param duration can be 'short', 'medium', or 'long'
  def vibrate(self, device_name, duration='medium'):
    if duration.lower().strip() == 'short':
      self._deviceProxies[device_name].vibrate(myo.VibrationType.short)
    elif duration.lower().strip() == 'medium':
      self._deviceProxies[device_name].vibrate(myo.VibrationType.medium)
    elif duration.lower().strip() == 'long':
      self._deviceProxies[device_name].vibrate(myo.VibrationType.long)

  ################################
  ###### MYO DATA CALLBACKS ######
  ################################

  # Extract data from an event object and append it to the data log.
  # @param event_data_key the field of event that will be used as data.
  # @param time_s_original The original timestamp of the data.
  #   Will only be relevant for streams that may modify timestamps
  #   instead of using the one contained in the event.
  def append_data_fromEvent(self, event, stream_name, event_data_key, time_s_original=None):
    # If logging is not enabled yet, ignore the data.
    if not self._logging:
      return
    # Extract the data and add it to the log.
    device_name = self._get_event_device_name(event, check_existence=True)
    time_s = self._get_time_s_from_event(device_name, event)
    data = getattr(event, event_data_key)
    if time_s_original is not None:
      extra_data = OrderedDict([('time_s_original', time_s_original)])
    else:
      extra_data = None
    self.append_data(device_name, stream_name, time_s, data, extra_data)

  # Record a new battery level.
  # @param event contains:
  #   timestamp: microseconds since epoch
  #   battery_level: level from 0-100
  # Should first call request_battery_level()
  def on_battery_level(self, event):
    device_name = self._get_event_device_name(event, check_existence=True)
    with self._locks[device_name]:
      self._log_debug('Myo %s battery level: %d' % (device_name, event.battery_level))
      self.append_data_fromEvent(event, 'battery', 'battery_level')

  # Record a new RSSI level.
  # @param event contains:
  #   timestamp: microseconds since epoch
  #   rssi: rssi level
  # Should first call request_rssi()
  def on_rssi(self, event):
    device_name = self._get_event_device_name(event, check_existence=True)
    with self._locks[device_name]:
      self._log_debug('Myo %s RSSI: %d' % (device_name, event.rssi))
      self.append_data_fromEvent(event, 'rssi', 'rssi')

  # Record new EMG data.
  # Note that stream_emg(True) should be called on the device to enable this callback.
  # Note that the Myo interface will call this method back-to-back
  #   with samples from two timesteps, and provide the same timestamp for each call.
  #  To compensate for this, the below method will manually adjust the first timestamp
  #   to be the average of the previous one and the new one.
  # @param event contains:
  #   timestamp: microseconds since epoch
  #   emg: 8-element list
  def on_emg(self, event):
    device_name = self._get_event_device_name(event, check_existence=True)
    with self._locks[device_name]:
      time_s = self._get_time_s_from_event(device_name, event)
      emg_timestamps_s = self._data[device_name]['emg']['time_s']
      # Adjust the previous timestamp if it equals the new timestamp.
      if len(emg_timestamps_s) > 0 and emg_timestamps_s[-1] == time_s:
        # If possible, change the previous timestamp to the average of the ones around it.
        # (Assume the previous one arrived equally spaced between its predecessor and this new one.)
        if len(emg_timestamps_s) > 1:
          updated_prev_time_s = (emg_timestamps_s[-2] + time_s)/2.0
        # If not enough samples to average, use a nominal sample rate.
        else:
          nominal_Ts = 1.0/self.get_stream_info(device_name, 'emg')['sampling_rate_hz']
          updated_prev_time_s = (time_s - nominal_Ts)
        updated_prev_time_str = get_time_str(updated_prev_time_s, '%Y-%m-%d %H:%M:%S.%f')
        self._data[device_name]['emg']['time_s'][-1] = updated_prev_time_s
        self._data[device_name]['emg']['time_str'][-1] = updated_prev_time_str
      # Store the new timestamped data
      self.append_data_fromEvent(event, 'emg', 'emg', time_s_original=time_s)

  # Record new IMU data including the accelerometer, gyroscope, and quaternion estimate.
  # @param event contains:
  #   timestamp: microseconds since epoch
  #   orientation: Quaternion object
  #   gyroscope: Vector object
  #   acceleration: Vector object
  def on_orientation(self, event):
    device_name = self._get_event_device_name(event, check_existence=True)
    with self._locks[device_name]:
      # Convert to lists
      event.acceleration_asList = list(event.acceleration)
      event.gyroscope_asList = list(event.gyroscope)
      event.orientation_asList = list(event.orientation)
      # Store the data
      self.append_data_fromEvent(event, 'acceleration_g', 'acceleration_asList')
      self.append_data_fromEvent(event, 'angular_velocity_deg_s', 'gyroscope_asList')
      self.append_data_fromEvent(event, 'orientation_quaternion', 'orientation_asList')

  # Record a new pose classification.
  # @param event contains:
  #   timestamp: microseconds since epoch
  #   pose: Pose object
  def on_pose(self, event):
    device_name = self._get_event_device_name(event, check_existence=True)
    with self._locks[device_name]:
      # Convert to a string
      event.pose_asStr = str(event.pose).split('.')[-1]
      # Store the prediction
      self.append_data_fromEvent(event, 'gesture', 'pose_asStr')
      self._log_debug('Myo %s gesture: %s' % (device_name, event.pose_asStr))


  ###########################
  ###### VISUALIZATION ######
  ###########################

  # Specify how the stream should be visualized.
  # visualization_options can have entries for 'emg' and 'imu'.
  # The same settings will be used for all connected Myos.
  def get_default_visualization_options(self, visualization_options=None):
    # Specify default options.
    emg_options = {
      'class': LinePlotVisualizer,
      'single_graph': False,
      'plot_duration_s': 15,
      'downsample_factor': 1,
      }
    imu_options = {
      'class': LinePlotVisualizer,
      'single_graph': True,
      'plot_duration_s': 15,
      'downsample_factor': 1,
      }
    orientation_options = {
      'class': OrientationVisualizer,
      }
    # Override with any provided options.
    if isinstance(visualization_options, dict):
      if 'emg' in visualization_options:
        for (k, v) in visualization_options['emg'].items():
          emg_options[k] = v
      if 'imu' in visualization_options:
        for (k, v) in visualization_options['emg'].items():
          imu_options[k] = v
      if 'orientation' in visualization_options:
        for (k, v) in visualization_options['orientation'].items():
          orientation_options[k] = v

    # Set all devices to the same options.
    processed_options = {}
    for (device_name, device_info) in self._streams_info.items():
      processed_options[device_name] = {}
      for (stream_name, stream_info) in device_info.items():
        if stream_name == 'emg':
          processed_options[device_name][stream_name] = emg_options
        elif stream_name == 'acceleration_g':
          processed_options[device_name][stream_name] = imu_options
        elif stream_name == 'angular_velocity_deg_s':
          processed_options[device_name][stream_name] = imu_options
        elif stream_name == 'orientation_quaternion':
          processed_options[device_name][stream_name] = orientation_options
        else:
          processed_options[device_name][stream_name] = {'class':None}

    return processed_options

  #####################
  ###### RUNNING ######
  #####################

  # Make a single function to wait for connections and then run,
  #  instead of using the separate connect() and run() methods.
  # This accommodates the need for a single hub.run_in_background handler.
  def _connect_and_run_threadFn(self):
    # Initialize the Myo interface
    self._log_status('Initializing the Myo interface')
    myo.init()
    self._myo_hub = myo.Hub()

    try:
      with self._myo_hub.run_in_background(self.on_event):
        self._log_debug('Myo thread waiting to run!')
        # Wait for the main running signal.
        # Note that in the background, the hub will be connecting to devices
        #  and calling the relevant callbacks.
        while not self._running and self._connectAndRunThread_running:
          time.sleep(0.005)
        if self._connectAndRunThread_running:
          # Check that all connected devices are synced
          for device_name in self.get_device_names():
            if self._metadata[device_name]['arm'] is None:
              self._log_error('')
              self._log_error('XXXXXXXXXX')
              self._log_error('WARNING: Myo device "%s" is not synced!' % device_name)
              self._log_error('XXXXXXXXXX')
              self._log_error('')
          # Start recording data
          self._logging = True
          # Wait while the main work happens.
          # Data will be streamed from the Myo in the background,
          #  and the hub will call the relevant callback methods.
          self._log_status('Myo thread is running!')
          last_battery_poll_time_s = None
          while self._running and self._connectAndRunThread_running:
            time.sleep(0.005)
            # Periodically request battery and RSSI information
            if last_battery_poll_time_s is None or time.time() - last_battery_poll_time_s > self._battery_rssi_polling_period_s:
              self.request_battery_rssi_allDevices()
              last_battery_poll_time_s = time.time()
          # Get the final battery/RSSI levels
          self.request_battery_rssi_allDevices(wait_for_response=True)
    except KeyboardInterrupt: # The program was likely terminated
      pass
    except:
      self._log_error('\n\n***ERROR RUNNING MyoStreamer:\n%s\n' % traceback.format_exc())
    finally:
      # All done!
      self._log_status('Myo thread has stopped running')
      self._logging = False
      self._myo_hub = None
      self._connectAndRunThread_running = False

  # Loop until self._running is False
  def _run(self):
    # The combined connect-and-run thread is the heavy lifter in this class.
    # Start it running if it is not already running.
    if not self._connectAndRunThread_running:
      self.connect()
    # Join the thread so it becomes the run thread (which is handled by SensorStreamer).
    self._connectAndRun_thread.join()

  # Clean up and quit
  def quit(self):
    SensorStreamer.quit(self)


  #####################################
  ###### DATA NOTES AND HEADINGS ######
  #####################################

  def _define_data_notes(self):
    self._data_notes = {}
  
    self._data_notes['emg'] = OrderedDict([
      ('Units', 'Normalized'),
      ('Range', '[-128, 127]'),
      (SensorStreamer.metadata_data_headings_key, ['Channel %d' % x for x in range(1,8+1)]),
    ])
    self._data_notes['acceleration_g'] = OrderedDict([
      ('Units', 'g'),
      ('Coordinate frame', 'The Myo\'s x axis points along the user\'s forearm, '
                           'either towards the wrist or the elbow - see this '
                           'device\'s metadata field \'x_direction\'. '
                           'The y and z directions depend on the worn orientation; '
                           'the calibration periods can help by inferring gravity '
                           'during a stationary known pose. '
                           'The coordinate frame is right-handed.'),
      (SensorStreamer.metadata_data_headings_key, ['x','y','z']),
    ])
    self._data_notes['angular_velocity_deg_s'] = OrderedDict([
      ('Units', 'degrees/s'),
      ('Coordinate frame', 'The Myo\'s x axis points along the user\'s forearm, '
                           'either towards the wrist or the elbow - see this '
                           'device\'s metadata field \'x_direction\'. '
                           'The y and z directions depend on the worn orientation; '
                           'the calibration periods can help by inferring gravity '
                           'during a stationary known pose. '
                           'The coordinate frame is right-handed.'),
      (SensorStreamer.metadata_data_headings_key, ['x','y','z']),
    ])
    self._data_notes['orientation_quaternion'] = OrderedDict([
      ('Description', 'A unit quaternion that described how the Myo is currently oriented '
                      'relative to a fixed reference frame'),
      ('Format', 'Represented as \'x*i + y*j + z*k + w\''),
      ('Reference frame', 'The orientation is relative to an arbitrary frame '
                          'that is defined as the Myo\'s fixed frame (technically '
                          'the pose when it is powered on / unplugged from USB, '
                          'but effectively arbitrary for most purposes). '
                          'Mapping this fixed frame to the task/world is facilitated '
                          'by the calibration period with a stationary known pose.'),
      (SensorStreamer.metadata_data_headings_key, ['x','y','z','w']),
    ])
    self._data_notes['gesture'] = OrderedDict([
      ('Description', 'Gestures/poses detected by the Myo\'s built-in classifier.'),
    ])
    self._data_notes['synced'] = OrderedDict([
      ('Description', 'Indicates when the Myo is synced, i.e. when it knows '
                      'which arm it\'s on and how it\'s oriented on the arm.  '
                      'Only changes in the sync status are recorded; so for example, '
                      'it is synced for all times after a \'1\' entry until a '
                      '\'0\' entry is recorded.'),
    ])
    self._data_notes['connected'] = OrderedDict([
      ('Description', 'Indicates when the Myo is connected (via Myo Connect). '
                      'Only changes in the connection status are recorded; so for example, '
                      'it is connected for all times after a \'1\' entry until a '
                      '\'0\' entry is recorded.'),
    ])
    self._data_notes['battery'] = OrderedDict([
      ('Units', 'Percent charged, in range [0, 100]'),
    ])
    self._data_notes['rssi'] = OrderedDict([
      ('Units', 'dB'),
    ])


#####################
###### TESTING ######
#####################
if __name__ == '__main__':
  duration_s = 30
  myo_streamer = MyoStreamer(print_status=True, print_debug=False)
  myo_streamer.connect()
  myo_streamer.run()
  time.sleep(duration_s)
  myo_streamer.stop()
  
  fps_device_name = myo_streamer.get_device_names()[0]
  fps_stream_name = 'emg'
  num_timesteps = myo_streamer.get_num_timesteps(fps_device_name, fps_stream_name)
  print('Duration: ', duration_s)
  print('Num timesteps for %s %s: %d -> Fs=%g' %
        (fps_device_name, fps_stream_name, num_timesteps, (num_timesteps-1)/duration_s))







