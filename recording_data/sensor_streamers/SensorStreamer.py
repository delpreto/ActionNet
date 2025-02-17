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

from abc import ABC, abstractmethod

from threading import Thread, Lock
import copy
from collections import OrderedDict
import os
import h5py
import cv2
import decord
import numpy as np
import traceback

from utils.time_utils import *
from utils.dict_utils import *
from utils.print_utils import *


################################################
################################################
# An abstract class to interface with a particular sensor.
#  For example, may be a superclass for a Myo, Glove, or Webcam class.
#  May contain multiple streams from the sensor, such as EMG/IMU from the Myo.
# Structure of data and streams_info:
#  Dict with device names as keys, each of which maps to:
#    Dict with name of streams, each of which maps to:
#      for data: 'data', 'time_s', 'time_str', and others if desired
#      for streams_info: 'data_type', 'sample_size', 'sampling_rate_hz', 'timesteps_before_solidified', 'extra_data_info'
################################################
################################################
class SensorStreamer(ABC):

  ########################
  ###### INITIALIZE ######
  ########################

  # Will store the class name of each sensor in HDF5 metadata,
  #  to facilitate recreating classes when replaying the logs later.
  # The following is the metadata key to store that information.
  metadata_class_name_key = 'SensorStreamer class name'
  # Will look for a special metadata key that labels data channels,
  #  to use for logging purposes and general user information.
  metadata_data_headings_key = 'Data headings'
  
  # @param streams_info see self.streams_info for more details
  def __init__(self, streams_info=None,
                      log_player_options=None, visualization_options=None,
                      print_status=True, print_debug=False, log_history_filepath=None):
    self._data = OrderedDict()
    self._data_lock = Lock()
    self._metadata = OrderedDict()

    self.set_streams_info(streams_info)

    self._running = False
    self._run_thread = None
    self._wait_after_stopping = True
    self._always_run_in_main_process = False
    self._print_status = print_status
    self._print_debug = print_debug
    self._log_source_tag = type(self).__name__
    self._log_history_filepath = log_history_filepath
    
    # Record whether existing data logs will be replayed instead of streaming from sensors.
    if log_player_options is not None:
      self._replaying_data_logs = True
      self._log_player_options = log_player_options
    else:
      self._replaying_data_logs = False
      self._log_player_options = None

    # Record the visualization options, but do not use them yet
    #  since the streamer may not be ready for them (may not have created devices).
    self._user_visualization_options = visualization_options

  ##############################
  ###### GETTERS/SSETTERS ######
  ##############################

  # Get/set metadata
  def get_metadata(self, device_name=None, only_str_values=False):
    # Get metadata for all devices or for the specified device.
    if device_name is None:
      metadata = self._metadata
    elif device_name in self._metadata:
      metadata = self._metadata[device_name]
    else:
      metadata = OrderedDict()

    # Add the class name.
    class_name = type(self).__name__
    if device_name is None:
      for device_name_toUpdate in metadata:
        metadata[device_name_toUpdate][SensorStreamer.metadata_class_name_key] = class_name
    else:
      metadata[SensorStreamer.metadata_class_name_key] = class_name

    # Convert values to strings if desired.
    if only_str_values:
      metadata = convert_dict_values_to_str(metadata)
    return metadata

  def set_metadata(self, new_metadata):
    self._metadata = new_metadata

  def get_num_devices(self):
    return len(self._streams_info)

  # Get the names of streaming devices
  def get_device_names(self):
    return list(self._streams_info.keys())

  # Rename a device (and keep the device indexing order the same).
  # If the name already exists, will not do anything.
  def rename_device(self, old_device_name, new_device_name):
    self._log_debug('Renaming device "%s" to "%s"' % (old_device_name, new_device_name))
    with self._data_lock:
      self._streams_info = rename_dict_key(self._streams_info, old_device_name, new_device_name)
      self._metadata = rename_dict_key(self._metadata, old_device_name, new_device_name)
      self._data = rename_dict_key(self._data, old_device_name, new_device_name)

  # Get the names of streams.
  # If device_name is None, will assume streams are the same for every device.
  def get_stream_names(self, device_name=None):
    if device_name is None:
      device_name = self.get_device_names()[0]
    return list(self._streams_info[device_name].keys())

  # Get information about a stream.
  # Returned dict will have keys data_type, sample_size, sampling_rate_hz, timesteps_before_solidified, extra_data_info
  def get_stream_info(self, device_name, stream_name):
    return self._streams_info[device_name][stream_name]

  # Get all information about all streams.
  def get_all_stream_infos(self):
    return copy.deepcopy(self._streams_info)

  # Get a list of values for a particular type of stream info (decoupled from device/stream names).
  # info_key can be data_type, sample_size, sampling_rate_hz, timesteps_before_solidified, extra_data_info
  def get_stream_attribute_allStreams(self, stream_attribute):
    infos = []
    for (device_name, device_info) in self._streams_info.items():
      for (stream_name, stream_info) in device_info.items():
        infos.append(stream_info[stream_attribute])

  # Get a list of data keys associated with a stream.
  # Will include 'data', 'time_s', 'time_str', and any extra ones defined.
  def get_stream_data_keys(self, device_name, stream_name):
    return list(self._data[device_name][stream_name].keys())

  # Add a new device stream.
  # @param timesteps_before_solidified allows indication that data/timestamps for
  #   previous timesteps may be altered when a new sample is received.
  #   For example, some sensors may send two samples at the same time, and the
  #     streamer class may infer a timestamp for the previous reading when
  #     a new pair of readings arrive.  A timestamp is thus not 'final' until the next timestep.
  # @param extra_data_info is used to specify additional data keys that will be streamed
  #   along with the default 'data', 'time_s', and 'time_str'.
  #   It should be a dict, where each extra data key maps to a dict with at least 'data_type' and 'sample_size'.
  # @param data_notes can be a string or a dict of relevant info.
  #   If it's a dict, the key 'Data headings' is recommended and will be used by DataLogger for headers.
  #     In that case, 'Data headings' should map to a list of strings of length sample_size.
  def add_stream(self, device_name, stream_name, data_type, sample_size, sampling_rate_hz, data_notes=None, is_video=False, is_audio=False, timesteps_before_solidified=0, extra_data_info=None):
    self._log_debug('Adding stream %s.%s: type %s | size %s | sampling %s Hz | solidification %s' % (device_name, stream_name, str(data_type), str(sample_size), str(sampling_rate_hz), str(timesteps_before_solidified)))
    self._streams_info.setdefault(device_name, OrderedDict())
    if not isinstance(sample_size, (list, tuple)):
      sample_size = [sample_size]
    self._streams_info[device_name][stream_name] = OrderedDict([
                                                    ('data_type', data_type),
                                                    ('sample_size', sample_size),
                                                    ('data_notes', data_notes or {}),
                                                    ('sampling_rate_hz', sampling_rate_hz),
                                                    ('is_video', is_video),
                                                    ('is_audio', is_audio),
                                                    ('timesteps_before_solidified', timesteps_before_solidified),
                                                    ('extra_data_info', extra_data_info or {}),
                                                    ])
    self.clear_data(device_name, stream_name)

  # Define all devices/streams at once.
  # @param streams_info should be a dict mapping device names to dicts.
  #   Each nested dict should map a stream name to a dict of stream info.
  #   See add_stream() for information about the parameters that should be contained in each stream info.
  def set_streams_info(self, streams_info):
    if streams_info is None:
      self._streams_info = OrderedDict()
      self._data = OrderedDict()
      return
    # Validate the format
    required_info_keys = ['data_type', 'sample_size', 'sampling_rate_hz']
    try:
      for (device_name, device_info) in streams_info.items():
        for (stream_name, stream_info) in device_info.items():
          if False in [info_key in stream_info for info_key in required_info_keys]:
            raise AssertionError('Provided stream information for %s.%s does not contain all required keys' % (device_name, stream_name))
    except AssertionError:
      raise
    except:
      raise ValueError('Provided stream information does not have the correct format')
    # Set the new streams
    for (device_name, device_info) in streams_info.items():
      for (stream_name, stream_info) in device_info.items():
        stream_info.setdefault('timesteps_before_solidified', 0)
        stream_info.setdefault('extra_data_info', {})
        stream_info.setdefault('data_notes', None)
        stream_info.setdefault('is_video', False)
        stream_info.setdefault('is_audio', False)
        if not isinstance(stream_info['sample_size'], (list, tuple)):
          stream_info['sample_size'] = [stream_info['sample_size']]
        self.add_stream(device_name=device_name,
                        stream_name=stream_name,
                        data_type=stream_info['data_type'],
                        sample_size=stream_info['sample_size'],
                        data_notes=stream_info['data_notes'],
                        is_video=stream_info['is_video'],
                        is_audio=stream_info['is_audio'],
                        sampling_rate_hz=stream_info['sampling_rate_hz'],
                        timesteps_before_solidified=stream_info['timesteps_before_solidified'],
                        extra_data_info=stream_info['extra_data_info'])

  # A helper to get some information about threading configuration,
  # since class attributes are not accessible from proxy objects when multiprocessing.
  def get_threading_config(self, config_name):
    if config_name == 'always_run_in_main_process':
      return self._always_run_in_main_process
    if config_name == 'wait_after_stopping':
      return self._wait_after_stopping
    return None

  ##################
  ###### DATA ######
  ##################

  # Clear data for a stream (and add the stream if it doesn't exist).
  # Optionally, can clear only data before a specified index.
  def clear_data(self, device_name, stream_name, first_index_to_keep=None):
    with self._data_lock:
      # Create the device/stream entry if it doesn't exist
      self._data.setdefault(device_name, OrderedDict())
      self._data[device_name].setdefault(stream_name, OrderedDict())

      # Get the data keys that have been populated so far.
      data_keys = list(self._data[device_name][stream_name].keys())
      # Get the expected data keys in the stream,
      #  in case data hasn't been received for some of them yet.
      data_keys.extend(['time_s', 'time_str', 'data'])
      data_keys.extend(list(self._streams_info[device_name][stream_name]['extra_data_info'].keys()))
      data_keys = list(set(data_keys))
      # Clear data for each known or expected data key.
      for data_key in data_keys:
        if first_index_to_keep is None:
          self._data[device_name][stream_name][data_key] = []
        else:
          self._data[device_name][stream_name][data_key] = \
            self._data[device_name][stream_name][data_key][first_index_to_keep:]

  def clear_data_all(self):
    for (device_name, device_info) in self._streams_info.items():
      for (stream_name, stream_info) in device_info.items():
        self.clear_data(device_name, stream_name)

  # Add a single timestep of data to the data log.
  # @param time_s and @param data should each be a single value.
  # @param extra_data should be a dict mapping each extra data key to a single value.
  #   The extra data keys should match what was specified in add_stream() or set_streams_info().
  def append_data(self, device_name, stream_name, time_s, data, extra_data=None):
    # self._log_debug('Appending data to %s.%s: %f | %s' % (device_name, stream_name, time_s, str(data)))
    with self._data_lock:
      time_str = get_time_str(time_s, '%Y-%m-%d %H:%M:%S.%f')
      self._data[device_name][stream_name]['time_s'].append(time_s)
      self._data[device_name][stream_name]['time_str'].append(time_str)
      self._data[device_name][stream_name]['data'].append(data)
      if extra_data is not None:
        for (extra_key, extra_value) in extra_data.items():
          if extra_key not in self._data[device_name][stream_name]:
            self._log_error('ERROR: Unknown extra data key: %s' % extra_key)
          self._data[device_name][stream_name][extra_key].append(extra_value)

  # Get the number of timesteps recorded so far.
  def get_num_timesteps(self, device_name, stream_name):
    times_s = self._get_times_s(device_name, stream_name)
    try: # If reading from a stored HDF5 file, avoid loading the array into memory
      return times_s.shape[0]
    except:
      return len(times_s)

  # Get the start time of data recorded so far.
  def get_start_time_s(self, device_name, stream_name):
    try:
      start_time_s = self._get_times_s(device_name, stream_name)[0]
      if isinstance(start_time_s, np.ndarray):
        start_time_s = start_time_s[0]
      return start_time_s
    except IndexError:
      return None

  # Get the end time of data recorded so far.
  def get_end_time_s(self, device_name, stream_name):
    try:
      end_time_s = self._get_times_s(device_name, stream_name)[-1]
      if isinstance(end_time_s, np.ndarray):
        end_time_s = end_time_s[0]
      return end_time_s
    except IndexError:
      return None
  
  # Get the duration of data recorded so far.
  def get_duration_s(self, device_name, stream_name):
    start_time_s = self.get_start_time_s(device_name, stream_name)
    end_time_s = self.get_end_time_s(device_name, stream_name)
    if end_time_s is not None and start_time_s is not None:
      return end_time_s - start_time_s
    return None

  # Get data.
  # Can get all data, or only data to/from a specified timestep.
  # @starting_index the desired first index to return, or None to start at the beginning.
  # @ending_index the desired ending index to return, or None to go until the end.
  #   If provided, will be used with standard Python list indexing,
  #    so the provided index will be *excluded* from the returned data.
  #   Note that it can also be negative, in accordance with standard Python list indexing.
  # @starting_time_s and @ending_time_s are alternatives to starting_index and ending_index.
  #   If not None, will be used instead of the corresponding index argument.
  # If no data exists based on the specified timesteps, will return None.
  # @param return_deepcopy Whether to return a deep copy of the data (safer)
  #                         or to potentially include pointers to the data record (faster).
  def get_data(self, device_name, stream_name,
                    starting_index=None, ending_index=None,
                    starting_time_s=None, ending_time_s=None,
                    return_deepcopy=True):
    with self._data_lock:
      # Convert desired times to indexes if applicable.
      if starting_time_s is not None:
        starting_index = self.get_index_for_time_s(device_name, stream_name, starting_time_s, 'after')
        if starting_index is None:
          return None
      if ending_time_s is not None:
        ending_index = self.get_index_for_time_s(device_name, stream_name, ending_time_s, 'before')
        if ending_index is None:
          return None
        else:
          ending_index += 1 # since will use as a list index and thus exclude the specified index
      # Update default starting/ending indexes.
      if starting_index is None:
        starting_index = 0
      if ending_index is None:
        ending_index = self.get_num_timesteps(device_name, stream_name)
      
      # Use streaming logs or existing logs as needed.
      if not self._replaying_data_logs:
        # Get the desired subset of data.
        # Note that this also creates a copy of each array, so the returned data remains static as new data is collected.
        res = dict([(key, values[starting_index:ending_index]) for (key, values) in self._data[device_name][stream_name].items()])
      else:
        # Use existing logs.
        # Create a data array that mimics a streaming log,
        #  but that is only populated with the desired subset of data.
        # print('Getting saved data for', device_name, stream_name, 'indexes', starting_index, ending_index)
        if device_name in self._hdf5_stream_groups and stream_name in self._hdf5_stream_groups[device_name]:
          # Get data from the HDF5 file.
          # print('Getting HDF5 group')
          res = {}
          # Extract the data subset and convert to a list (possibly a list of lists if N-D)
          #  instead of numpy arrays that come from the HDF5.
          for (key, value) in self._hdf5_stream_groups[device_name][stream_name].items():
            res[key] = value[starting_index:ending_index]
            res[key] = [x.squeeze().tolist() for x in res[key]]
            # if not isinstance(res[key], list): # was probably a single element
            #   res[key] = [res[key]]
        else:
          # Get data from the video file.
          # print('Getting video data', starting_index, ending_index)
          res = {'data':[], 'time_s':[], 'time_str':[]}
          video_reader = self._video_readers[device_name][stream_name]
          video_time_s_group = self._video_time_s_stream_groups[device_name][stream_name]
          for index in range(starting_index, ending_index):
            try:
              frame = cv2.cvtColor(video_reader[index].asnumpy(), cv2.COLOR_RGB2BGR)
            except:
              continue
            time_s = video_time_s_group['time_s'][index][0]
            res['data'].append(frame)
            res['time_s'].append(time_s)
            res['time_str'].append(get_time_str(time_s, format='%Y-%m-%d %H:%M:%S.%f'))
        
      # Return the result
      if len(list(res.values())[0]) == 0:
        return None
      elif return_deepcopy:
        return copy.deepcopy(res)
      else:
        return res
        
  # Helper to get the time array for a specified stream.
  # Will use the streaming data or the loaded log data as applicable.
  def _get_times_s(self, device_name, stream_name):
    if not self._replaying_data_logs:
      return self._data[device_name][stream_name]['time_s']
    else:
      try:
        return self._hdf5_stream_groups[device_name][stream_name]['time_s']
      except KeyError:
        return self._video_time_s_stream_groups[device_name][stream_name]['time_s']
    
  # Helper to get the sample index that best corresponds to the specified time.
  # Will return the index of the last sample that is <= the specified time
  #  or the index of the first sample that is >= the specified time.
  def get_index_for_time_s(self, device_name, stream_name, target_time_s, target_type='before'):
    # Get the sample times streamed so far, or loaded from existing logs.
    times_s = np.array(self._get_times_s(device_name, stream_name))
    # Get the last sample before the specified time.
    if 'before' == target_type.lower().strip():
      indexes = np.argwhere(times_s <= target_time_s)
      return np.max(indexes) if indexes.size > 0 else None
    # Get the first sample after the specified time.
    elif 'after' == target_type.lower().strip():
      indexes = np.argwhere(times_s >= target_time_s)
      return np.min(indexes) if indexes.size > 0 else None
    else:
      raise ValueError

  #########################################
  ###### REPLAYING EXISTING DATA LOG ######
  #########################################

  # Initialize the streamer using existing logs instead of an active sensor.
  # Will create streams based on what is present in any HDF5 files in the directory.
  # Will create video readers as applicable based on video files in the directory.
  def _init_from_data_log(self):
    self._log_status('%s initializing from existing data logs' % type(self).__name__)
    # Will store a dict mapping device and stream names to an HDF5 group.
    self._hdf5_stream_groups = {}
    # Will also store similar dicts for video readers
    #  and for corresponding frame timestamps.
    self._video_readers = {}
    self._video_time_s_stream_groups = {}

    # Find all HDF5 files in the provided folder.
    hdf5_filepaths = []
    for file in os.listdir(self._log_player_options['log_dir']):
      if file.endswith('.hdf5'):
        hdf5_filepaths.append(os.path.join(self._log_player_options['log_dir'], file))

    # Create devices/streams based on what was logged by any HDF5 file.
    for hdf5_filepath in hdf5_filepaths:
      self._log_debug('Opening the following HDF5 file:', hdf5_filepath)
      hdf5_file = h5py.File(hdf5_filepath, 'r')
      # Get a list of devices in the HDF5 file that belong to the current streamer
      #  (to the current SensorStreamer subclass running this method).
      device_names = []
      for (device_name, device_group) in hdf5_file.items():
        metadata = dict(device_group.attrs.items())
        try:
          streamer_class_name = metadata[SensorStreamer.metadata_class_name_key]
        except KeyError: # legacy log that didn't have this metadata yet
          if 'experiment-notes' in device_name:
            streamer_class_name = 'NotesStreamer'
          elif 'myo' in device_name:
            streamer_class_name = 'MyoStreamer'
          elif 'eye-tracking' in device_name:
            streamer_class_name = 'EyeStreamer'
          elif 'glove' in device_name:
            streamer_class_name = 'TouchStreamer'
          elif 'xsens' in device_name:
            streamer_class_name = 'XsensStreamer'
          else:
            raise AssertionError('Unknown streamer type in HDF5 file')
        if streamer_class_name == type(self).__name__:
          device_names.append(device_name)
      # Create a stream for each device stream in the file.
      self._log_debug('Found the following devices:', device_names)
      for device_name in device_names:
        device_group = hdf5_file[device_name]
        self._hdf5_stream_groups[device_name] = {}
        for (stream_name, stream_group) in device_group.items():
          try:
            data = stream_group['data']
            time_s = stream_group['time_s']
          except:
            continue
          # Extract information about the data so it can be recreated.
          data_type = str(data.dtype)
          if data_type.find('|S') == 0:
            data_type = data_type[1:]
          if data.shape[0] > 0:
            sample_size = data[0].shape
          else:
            sample_size = [1]
          # Get timing information, discarding dummy all-0 entries if there are any.
          sampling_rate_hz = None
          if time_s.shape[0] > 1:
            start_index = 0
            while time_s[start_index] == 0 and start_index < time_s.shape[0]:
              start_index += 1
            end_index = -1
            while time_s[end_index] == 0 and time_s.shape[0] + end_index >= 0:
              end_index -= 1
            if time_s[end_index] - time_s[start_index] > 0:
              sampling_rate_hz = (time_s.shape[0]-1)/(time_s[end_index] - time_s[start_index])
              sampling_rate_hz = sampling_rate_hz[0] # above returned a numpy ndarray
              sampling_rate_hz = round(sampling_rate_hz, 2)
          # Get notes.
          data_notes = dict(stream_group.attrs.items()) or None
          try:
            # Lists of channel names would have been stored as a string in the HDF5.
            if isinstance(data_notes[SensorStreamer.metadata_data_headings_key], str):
              data_notes[SensorStreamer.metadata_data_headings_key] = eval(data_notes[SensorStreamer.metadata_data_headings_key])
          except (KeyError, TypeError):
            pass
          # Get whether the stream is video or audio.
          is_video = False # assume videos weren't saved to HDF5 files
          is_audio = 'Microphone' in type(self).__name__ and stream_name == 'chunked_data' # assume MicrophoneStreamer is the only one to save audio
          timesteps_before_solidified = 0 # timestamps have already been adjusted
          # Extract information about any extra keys streamed with the data.
          extra_data_keys = [key for key in stream_group.keys()
                             if key not in ['data', 'time_s', 'time_str']]
          if len(extra_data_keys) > 0:
            extra_data_info = {}
            for extra_data_key in extra_data_keys:
              extra_data_type = str(stream_group[extra_data_key].dtype)
              if extra_data_type.find('|S') == 0:
                extra_data_type = extra_data_type[1:]
              if len(stream_group[extra_data_key]) > 0:
                extra_sample_size = stream_group[extra_data_key][0].shape
              else:
                extra_sample_size = [1]
              extra_data_info[extra_data_key] = {
                'data_type': extra_data_type,
                'sample_size': extra_sample_size
              }
          else:
            extra_data_info = None
          # Add the stream!
          self.add_stream(device_name, stream_name,
                          data_type, sample_size, sampling_rate_hz,
                          data_notes=data_notes, is_video=is_video, is_audio=is_audio,
                          timesteps_before_solidified=timesteps_before_solidified,
                          extra_data_info=extra_data_info)
          if self._print_debug:
            debug_msg  = 'Added the following stream:\n'
            debug_msg += '  Device name: %s\n' % device_name
            debug_msg += '  Stream name: %s\n' % stream_name
            debug_msg += '  Data type    : %s\n' % str(data_type)
            debug_msg += '  Sample size  : %s\n' % str(sample_size)
            debug_msg += '  Sampling rate: %s\n' % str(sampling_rate_hz)
            debug_msg += '  Data notes   : %s\n' % str(data_notes)
            debug_msg += '  Is video     : %s\n' % str(is_video)
            debug_msg += '  Is audio     : %s\n' % str(is_audio)
            debug_msg += '  Timesteps before solidified: %s\n' % str(timesteps_before_solidified)
            debug_msg += '  Extra data info: %s\n' % str(extra_data_info)
            self._log_debug(debug_msg.strip())
          # Store the stream group to get data for the stream.
          # Will store the HDF5 group directly if reading from disk,
          #  or will fetch all data now if loading into memory.
          # Either approach will allow data to be accessed via a dict-like interface.
          if self._log_player_options['load_datasets_into_memory']:
            self._hdf5_stream_groups[device_name][stream_name] = {}
            for (data_key, dataset) in stream_group.items():
              self._hdf5_stream_groups[device_name][stream_name][data_key] = dataset[:]
          else:
            self._hdf5_stream_groups[device_name][stream_name] = stream_group
    # Find video files in the provided folder as well.
    # These will be determined by the subclasses directly.
    videos_info = self.get_videos_info_from_log_dir()
    for (device_name, device_videos_info) in videos_info.items():
      self._log_debug('Opening the following video files:', get_var_str(device_videos_info))
      self._video_readers[device_name] = {}
      self._video_time_s_stream_groups[device_name] = {}
      for (stream_name, video_info) in device_videos_info.items():
        # Open a video reader to extract frames.
        video_reader = decord.VideoReader(video_info['video_filepath'])
        self._video_readers[device_name][stream_name] = video_reader
        # Extract information about the video.
        video_frame = cv2.cvtColor(video_reader[0].asnumpy(), cv2.COLOR_RGB2BGR)
        data_type = str(video_frame.dtype)
        sample_size = video_frame.shape
        sampling_rate_hz = video_reader.get_avg_fps()
        frame_count = len(video_reader)
        # Create streams for the video.
        self.add_stream(device_name, stream_name,
                          data_type, sample_size, sampling_rate_hz,
                          data_notes=None, is_video=True,
                          timesteps_before_solidified=0,
                          extra_data_info=None)
        if self._print_debug:
          debug_msg =  'Added the following video stream:\n'
          debug_msg += '  Device name: %s\n' % device_name
          debug_msg += '  Stream name: %s\n' % stream_name
          debug_msg += '  Data type    : %s\n' % str(data_type)
          debug_msg += '  Sample size  : %s\n' % str(sample_size)
          debug_msg += '  Sampling rate: %s\n' % str(sampling_rate_hz)
          self._log_debug(debug_msg.strip())
        # Store the hdf5 stream corresponding to frame timestamps.
        time_s_stream_group = self._hdf5_stream_groups[video_info['time_s_stream_device_name']][video_info['time_s_stream_name']]
        if time_s_stream_group['time_s'].shape[0] != frame_count:
          msg = 'Video from device %s stream %s has %d frames,' \
                  % (device_name, stream_name, frame_count)
          msg += ' but the timestamp stream from device %s stream %s has %d timesteps.' \
                  % (video_info['time_s_stream_device_name'],
                     video_info['time_s_stream_name'],
                     time_s_stream_group['time_s'].shape[0])
          self._log_warn('\n\nWARNING: %s\n\n' % msg)
        self._video_time_s_stream_groups[device_name][stream_name] = time_s_stream_group
        if self._print_debug:
          debug_msg = '  Added the following video timestamp stream:\n'
          debug_msg += '    Device name: %s\n' % video_info['time_s_stream_device_name']
          debug_msg += '    Stream name: %s\n' % video_info['time_s_stream_name']
          self._log_debug(debug_msg.strip())

  # A method that subclasses can define to determine whether videos in the
  #  log directory correspond to video streams from this sensor.
  # Returns a dict with structure videos_info[device_name][stream_name] = video_info
  #  where video_info has keys 'video_filepath', 'time_s_stream_device_name', and 'time_s_stream_name'
  #  that indicate the video filepath and the HDF5 stream that contains frame timestamps.
  def get_videos_info_from_log_dir(self):
    return {}

  # A run() method that streams data from logs rather than the sensors.
  # Will run until self._running is False or until all log data has been streamed.
  # Will stream from HDF5 streams and from video files.
  #   Note that dedicated streams for frame timestamps will be treated independently of the video,
  #   so they may be out of step with each other if skipping timesteps to stay real-time.
  #   The time_s values stored in both frames should still match though, even if there are different numbers of entries in each.
  # @param time_to_start_s optionally indicates when the streaming should being.
  #   Simulated log time will be calculated relative to this start time.
  #   This can allow multiple SensorStreamers to synchronize their streaming.
  def _run_from_logs(self, time_to_start_s=None):
    # Determine the earliest/latest timestamp in any of the datasets (not just this sensor).
    log_start_time_s = None
    log_end_time_s = None
    for file in os.listdir(self._log_player_options['log_dir']):
      if file.endswith('.hdf5'):
        hdf5_file = h5py.File(os.path.join(self._log_player_options['log_dir'], file), 'r')
        for (device_name, device_group) in hdf5_file.items():
          for (stream_name, stream_group) in device_group.items():
            time_s = stream_group['time_s']
            time_s = np.array(time_s)
            time_s = time_s[time_s > 0] # ignore invalid times, such as if the log was not trimmed and trailing entries are 0
            if len(time_s) == 0: # no data was received for this stream
              continue
            start_time_s = min(time_s)
            end_time_s = max(time_s)
            if log_start_time_s is None or start_time_s < log_start_time_s:
              log_start_time_s = start_time_s
            if log_end_time_s is None or end_time_s > log_end_time_s:
              log_end_time_s = end_time_s
        hdf5_file.close()
    if self._print_debug:
      debug_msg  = '%s found log time bounds across all streamers:\n' % type(self).__name__
      debug_msg += '  start time  : %s (%s)\n' % (get_time_str(log_start_time_s), log_start_time_s)
      debug_msg += '  end time    : %s (%s)\n' % (get_time_str(log_end_time_s), log_end_time_s)
      debug_msg += '  duration [s]: %s\n' % str(log_end_time_s - log_start_time_s)
      self._log_debug(debug_msg.strip())

    # Helper function to convert current time to log time.
    def log_time_s():
      return (time.time() - time_to_start_s) + log_start_time_s

    # Helper function to advance the next index.
    def advance_next_index(next_index, time_s_stream_group, skip_timesteps_to_replay_in_realtime):
      # Skip indexes to remain real-time if desired, otherwise just advance by 1.
      if skip_timesteps_to_replay_in_realtime:
        next_index = next_index+1
        while next_index < time_s_stream_group['time_s'].shape[0] \
            and time_s_stream_group['time_s'][next_index] < log_time_s():
          next_index = next_index+1
      else:
        next_index = next_index+1
      return next_index

    # Initialize counters to record the last used timestep of each stream.
    next_indexes = {}
    for (device_name, device_group) in self._hdf5_stream_groups.items():
      next_indexes.setdefault(device_name, {})
      for (stream_name, stream_group) in device_group.items():
        next_indexes[device_name][stream_name] = 0
    for (device_name, device_video_readers) in self._video_readers.items():
      next_indexes.setdefault(device_name, {})
      for (stream_name, video_reader) in device_video_readers.items():
        next_indexes[device_name][stream_name] = 0

    # Stream from the log!
    try:
      all_streams_complete = False
      while self._running and not all_streams_complete:
        all_streams_complete = True
        # Stream non-video data from HDF5 log files.
        for (device_name, device_group) in self._hdf5_stream_groups.items():
          for (stream_name, stream_group) in device_group.items():
            next_index = next_indexes[device_name][stream_name]
            # Check if all data has been streamed yet for this stream.
            if next_index >= stream_group['time_s'].shape[0]:
              continue
            all_streams_complete = False
            # See if the next timestamp has been reached yet (if replaying in real time).
            if log_time_s() >= stream_group['time_s'][next_index] \
                or (not self._log_player_options['pause_to_replay_in_realtime']):
              # Get the main data and any extra data.
              time_s = stream_group['time_s'][next_index][0]
              data = stream_group['data'][next_index]
              if stream_group['data'].shape[1] == 1 and len(stream_group['data'].shape) == 2:
                data = data[0] # flatten it to a single number instead of an array
              extra_data = {}
              for (data_key, dataset) in stream_group.items():
                if data_key in ['data', 'time_s', 'time_str']:
                  continue
                extra_data[data_key] = dataset[next_index]
                if stream_group[data_key].shape[1] == 1 and len(stream_group[data_key].shape) == 2:
                  extra_data[data_key] = extra_data[data_key][0] # flatten it to a single number instead of an array
              extra_data = extra_data or None
              # Append the data!
              self.append_data(device_name, stream_name, time_s, data, extra_data)
              debug_msg  = 'Time for new log data! Device %s Stream %s\n' % (device_name, stream_name)
              debug_msg += '  Log time %s %f | sim log time %s %f | time since start %f\n' % (get_time_str(time_s), time_s, get_time_str(log_time_s()), log_time_s(), log_time_s() - log_start_time_s)
              # debug_msg += '  %s\n' % type(data)
              # debug_msg += '  %s\n' % str(data.shape)
              # debug_msg += '  %s\n' % str(data)
              debug_msg += '  Next index advancing from %d' % next_index
              # Advance the log index.
              next_index = advance_next_index(next_index, stream_group, self._log_player_options['skip_timesteps_to_replay_in_realtime'])
              next_indexes[device_name][stream_name] = next_index
              if self._print_debug:
                debug_msg += ' to %d' % next_index
                self._log_debug(debug_msg)

        # Stream videos.
        for (device_name, device_video_readers) in self._video_readers.items():
          for (stream_name, video_reader) in device_video_readers.items():
            next_index = next_indexes[device_name][stream_name]
            # Check if all data has been streamed yet for this stream.
            if next_index >= len(video_reader):
              continue
            all_streams_complete = False
            # See if the next timestamp has been reached yet (if replaying in real time).
            time_s_stream_group = self._video_time_s_stream_groups[device_name][stream_name]
            if log_time_s() >= time_s_stream_group['time_s'][next_index] \
                or (not self._log_player_options['pause_to_replay_in_realtime']):
              # Get the desired video frame and its original timestamp.
              frame = cv2.cvtColor(video_reader[next_index].asnumpy(), cv2.COLOR_RGB2BGR)
              time_s = time_s_stream_group['time_s'][next_index][0]
              # Append the data!
              self.append_data(device_name, stream_name, time_s, frame)
              debug_msg  = 'Time for new log data (video frame)! Device %s Stream %s\n' % (device_name, stream_name)
              debug_msg += '  Log time %s %f | sim log time %s %f | time since start %f\n' % (get_time_str(time_s), time_s, get_time_str(log_time_s()), log_time_s(), log_time_s() - log_start_time_s)
              # debug_msg += '  %s\n' % type(frame)
              # debug_msg += '  %s\n' % str(frame.shape)
              # debug_msg += '  %s\n' % str(frame)
              debug_msg += '  Next index advancing from %d' % next_index
              # Advance the log index.
              next_index = advance_next_index(next_index, time_s_stream_group, self._log_player_options['skip_timesteps_to_replay_in_realtime'])
              next_indexes[device_name][stream_name] = next_index
              if self._print_debug:
                debug_msg += ' to %d' % next_index
                self._log_debug(debug_msg)
      # Running was cancelled, or streams in the log have finished.
      self._running = False
    except KeyboardInterrupt: # The program was likely terminated
      pass
    except:
      self._log_error('\n\n***ERROR REPLAYING LOGS FOR %s:\n%s\n' % (type(self).__name__, traceback.format_exc()))
    finally:
      pass


  ###########################
  ###### VISUALIZATION ######
  ###########################

  # Get how each stream should be visualized.
  # Should be overridden by subclasses if desired.
  # Returns a dictionary mapping [device_name][stream_name] to a dict of visualizer options.
  # See DataVisualizer for options of default visualizers, such as line plots and videos.
  def get_default_visualization_options(self, visualization_options=None):
    # Do not show any visualization by default. Subclasses can override this.
    processed_options = {}
    for (device_name, device_info) in self._streams_info.items():
      processed_options[device_name] = {}
      for (stream_name, stream_info) in device_info.items():
        processed_options[device_name][stream_name] = {
          'class': None
        }
    return processed_options

  # Specify how each stream should be visualized.
  # Should be overridden by subclasses if desired.
  # See DataVisualizer for options of default visualizers, such as line plots and videos.
  def get_visualization_options(self, device_name, stream_name):
    return self._visualization_options[device_name][stream_name]

  # Override default visualization options if desired.
  def set_visualization_options(self, device_name, stream_name, options):
    for (k, v) in options.items():
      self._visualization_options[device_name][stream_name][k] = v


  #####################################
  ###### EXTERNAL DATA RECORDING ######
  #####################################
  
  # Start recording data via the sensor's dedicated software.
  def start_external_data_recording(self, data_dir):
    pass
  
  # Whether recording via the sensor's dedicated software will require user action.
  def external_data_recording_requires_user(self):
    pass

  # Stop recording data via the sensor's dedicated software.
  def stop_external_data_recording(self):
    pass
  
  # Process externally recorded data and use it to update the main data log.
  def merge_external_data_with_streamed_data(self,
                                             # Final post-processed outputs
                                             hdf5_file_toUpdate,
                                             data_dir_toUpdate,
                                             # Original streamed and external data
                                             data_dir_streamed,
                                             data_dir_external_original,
                                             # Archives for data no longer needed
                                             data_dir_archived,
                                             hdf5_file_archived):
    pass

  ##############################
  ###### LOGGING/PRINTING ######
  ##############################

  def _log_status(self, msg, *extra_msgs, **kwargs):
    write_log_message(msg, *extra_msgs, source_tag=self._log_source_tag,
                      print_message=self._print_status, filepath=self._log_history_filepath, **kwargs)
  def _log_debug(self, msg, *extra_msgs, **kwargs):
    write_log_message(msg, *extra_msgs, source_tag=self._log_source_tag,
                      print_message=self._print_debug, filepath=self._log_history_filepath, debug=True, **kwargs)
  def _log_error(self, msg, *extra_msgs, **kwargs):
    write_log_message(msg, *extra_msgs, source_tag=self._log_source_tag,
                      print_message=True, error=True, filepath=self._log_history_filepath, **kwargs)
  def _log_warn(self, msg, *extra_msgs, **kwargs):
    write_log_message(msg, *extra_msgs, source_tag=self._log_source_tag,
                      print_message=True, warning=True, filepath=self._log_history_filepath, **kwargs)
  def _log_userAction(self, msg, *extra_msgs, **kwargs):
    write_log_message(msg, *extra_msgs, source_tag=self._log_source_tag,
                      print_message=True, userAction=True, filepath=self._log_history_filepath, **kwargs)
  
  ############################
  ###### INTERFACE FLOW ######
  ############################

  # Connect to the sensor device(s)
  @abstractmethod
  def _connect(self, timeout_s=10):
    pass

  # Should loop until self._running is False.
  # May also want to wrap the loop in a try/except/finally block
  #  in case the program is terminated via Ctrl-C.
  @abstractmethod
  def _run(self):
    pass

  # Clean up and quit
  @abstractmethod
  def quit(self):
    self.stop()

  # Connect using the subclass definition or existing logs.
  def connect(self, timeout_s=10):
    # Connect active sensors or initialize past data logs to replay.
    if not self._replaying_data_logs:
      connection_result = self._connect(timeout_s=timeout_s)
    else:
      self._init_from_data_log()
      connection_result = True

    # Disable visualization if desired, or if connection was unsuccessful.
    disable_visualization = (not connection_result)
    if isinstance(self._user_visualization_options, dict) \
        and ('disable_visualization' in self._user_visualization_options) \
        and (self._user_visualization_options['disable_visualization']):
      disable_visualization = True
    if disable_visualization:
      self._visualization_options = SensorStreamer.get_default_visualization_options(self, None)
    # Set visualization options according to the subclass implementation if there is one.
    else:
      self._visualization_options = self.get_default_visualization_options(self._user_visualization_options)

    # Return whether connection was successful.
    return connection_result

  # Run the defined _run() method in a new thread.
  def run(self, time_to_start_s=None):
    self._log_status('Starting to run!')
    # Create a thread to stream from sensors or from logs.
    if not self._replaying_data_logs:
      self._run_thread = Thread(target=self._run, args=())
      # If stopping later won't wait for the thread to finish,
      #  mark it as a daemon so it will be killed when the main program exits.
      self._run_thread.daemon = (not self._wait_after_stopping)
    else:
      self._run_thread = Thread(target=self._run_from_logs,
                                args=(),
                                kwargs={'time_to_start_s': time_to_start_s})
      self._run_thread.daemon = False
    # Wait for the indicated start time if one was provided.
    # If replaying from logs though, _run_from_logs() will handle this.
    if (time_to_start_s is not None) and (not self._replaying_data_logs):
      while time.time() < time_to_start_s:
        time.sleep(0.001)
    # Start running!
    self._running = True
    self._run_thread.start()

  def is_running(self):
    return self._running

  # Stop the _run() thread.
  # Will wait for the thread to stop unless self._wait_after_stopping is False.
  def stop(self):
    self._log_status('Stopping!')
    self._running = False
    if self._wait_after_stopping and self._run_thread is not None and self._run_thread.is_alive():
      self._log_debug('Joining run thread')
      self._run_thread.join()
      self._log_debug('Run thread finished successfully')
    else:
      self._log_debug('Not waiting for run thread to finish')
    self._run_thread = None






























