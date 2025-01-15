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

import h5py
from threading import Thread
import cv2
import wave
import numpy as np

import os
import traceback
import time
from collections import OrderedDict

from utils.time_utils import *
from utils.dict_utils import *
from utils.print_utils import *


################################################
################################################
# A class to log streaming data to one or more files.
# SensorStreamer instances are passed to the class, and the data
#  that they stream are written to disk periodically and/or at the end.
# If using the periodic option, latest data will be fetched from the streamers
#  and then written in a separate thread to avoid undue performance impact.
#  Any leftover data once the program is stopped will be flushed to the files.
# If using the periodic option, can optionally clear old data from memory to reduce RAM usage.
#  Data will then be cleared each time new data is logged, unless all data is expected to be written at the end.
#  Will treat video/audio data separately, so can choose to stream/clear non-AV data but dump AV data or vice versa.
# Logging currently supports CSV, HDF5, AVI/MP4, and WAV files.
#  If using HDF5, a single file will be created for all of the SensorStreamers.
#  If using CSV, a separate file will be created for each SensorStreamer.
#    N-D data will be unwrapped so that each entry is its own column.
#  Videos can be saved as AVI or MP4 files.
#  Audio can be saved as WAV files.
# Note that the is_video / is_audio flags of each stream will be used to identify video/audio.
#  Classes with audio streams will also require a method get_audioStreaming_info()
#   that returns a dict with keys num_channels, sample_width, sampling_rate.
#   See MicrophoneStreamer for an example of defining these attributes.
################################################
################################################
class DataLogger:

  ########################
  ###### INITIALIZE ######
  ########################

  def __init__(self, sensor_streamers, log_dir=None, log_tag=None,
                stream_csv=False, stream_hdf5=False, stream_video=False, stream_audio=False,
                stream_period_s=30, clear_logged_data_from_memory=False,
                dump_csv=False, dump_hdf5=False, dump_video=False, dump_audio=False,
                use_external_recording_sources=False,
                videos_in_csv=False, videos_in_hdf5=False, videos_format='avi',
                audio_in_csv=False, audio_in_hdf5=False, audio_format='wav',
                print_status=True, print_debug=False, log_history_filepath=None):

    # Validate the streamer objects, and make a list of them.
    if not isinstance(sensor_streamers, (list, tuple)):
      sensor_streamers = [sensor_streamers]
    if len(sensor_streamers) == 0:
      raise AssertionError('At least one SensorStreamer must be provided to DataLogger')
    self._streamers = list(sensor_streamers)

    # Record the configuration options.
    self._stream_hdf5 = stream_hdf5
    self._stream_csv = stream_csv
    self._stream_video = stream_video
    self._stream_audio = stream_audio
    self._stream_period_s = stream_period_s
    self._clear_logged_data_from_memory = clear_logged_data_from_memory
    self._dump_hdf5 = dump_hdf5
    self._dump_csv = dump_csv
    self._dump_video = dump_video
    self._dump_audio = dump_audio
    self._use_external_recording_sources = use_external_recording_sources
    self._videos_in_csv = videos_in_csv
    self._videos_in_hdf5 = videos_in_hdf5
    self._videos_format = videos_format
    self._audio_in_csv = audio_in_csv
    self._audio_in_hdf5 = audio_in_hdf5
    self._audio_format = audio_format
    self._log_tag = log_tag
    self._log_dir = log_dir
    self._print_status = print_status
    self._print_debug = print_debug
    self._log_source_tag = 'logger'
    self._log_history_filepath = log_history_filepath

    # Initialize a record of what indexes have been logged,
    #  and how many timesteps to stay behind of the most recent step (if needed).
    self._next_data_indexes = [OrderedDict() for i in range(len(self._streamers))]
    self._timesteps_before_solidified = [OrderedDict() for i in range(len(self._streamers))]
    # Each time an HDF5 dataset reaches its limit,
    #  its size will be increased by the following amount.
    self._hdf5_log_length_increment = 10000
    self._next_data_indexes_hdf5 = [OrderedDict() for i in range(len(self._streamers))]

    # Initialize state for the thread that will do stream-logging.
    self._streamLog_thread = None # the thread to periodically write data
    self._streamLogging = False   # whether periodic writing is active
    self._flush_log = False       # whether remaining data at the end should now be flushed

    # Initialize the logging writers.
    self._csv_writers = None
    self._csv_writer_metadata = None
    self._hdf5_file = None
    self._video_writers = None
    self._audio_writers = None
    # Create the log directory if needed.
    if self._stream_csv or self._stream_hdf5 or self._stream_video or self._stream_audio \
        or self._dump_csv or self._dump_hdf5 or self._dump_video or self._dump_audio:
      if self._log_dir is None:
        raise ValueError('A log directory must be provided to save data')
      os.makedirs(self._log_dir, exist_ok=True)


  # Helper to get a base prefix for logging filenames.
  # Will indicate the logging start time and whether it's streaming or post-experiment.
  def _get_filename_base(self, log_time_s, for_streaming=True):
    log_datetime_str = get_time_str(log_time_s, '%Y-%m-%d_%H-%M-%S')
    filename_base = '%s_%sLog' % (log_datetime_str,
                              'stream' if for_streaming else 'postExperiment')
    if self._log_tag is not None:
      filename_base = '%s_%s' % (filename_base, self._log_tag)
    return filename_base

  # Initialize the data indexes to fetch for logging.
  # Will record the next data indexes that should be fetched for each stream,
  #  and the number of timesteps that each streamer needs before data is solidified.
  def _init_log_indexes(self):
    for (streamer_index, streamer) in enumerate(self._streamers):
      for (device_name, streams_info) in streamer.get_all_stream_infos().items():
        self._next_data_indexes[streamer_index][device_name] = OrderedDict()
        self._timesteps_before_solidified[streamer_index][device_name] = OrderedDict()
        for (stream_name, stream_info) in streams_info.items():
          self._next_data_indexes[streamer_index][device_name][stream_name] = 0
          self._timesteps_before_solidified[streamer_index][device_name][stream_name] \
                  = stream_info['timesteps_before_solidified']

  # Create and initialize CSV files.
  # Will have a separate file for each stream of each device.
  # Currently assumes that device names are unique across all streamers.
  def _init_writing_csv(self, log_time_s, for_streaming=True):
    self._log_status('Opening CSV writers')
    filename_base = self._get_filename_base(log_time_s=log_time_s,
                                            for_streaming=for_streaming)

    # Open a writer for each CSV data file.
    csv_writers = [OrderedDict() for i in range(len(self._streamers))]
    for (streamer_index, streamer) in enumerate(self._streamers):
      for (device_name, streams_info) in streamer.get_all_stream_infos().items():
        csv_writers[streamer_index][device_name] = OrderedDict()
        for (stream_name, stream_info) in streams_info.items():
          # Skip saving videos in a CSV.
          if stream_info['is_video'] and not self._videos_in_csv:
            continue
          # Skip saving audio in a CSV.
          if stream_info['is_audio'] and not self._audio_in_csv:
            continue
          filename_csv = '%s_%s_%s.csv' % (filename_base, device_name, stream_name)
          filepath_csv = os.path.join(self._log_dir, filename_csv)
          csv_writers[streamer_index][device_name][stream_name] = open(filepath_csv, 'w')

    # Open a writer for a CSV metadata file.
    filename_csv = '%s__metadata.csv' % (filename_base)
    filepath_csv = os.path.join(self._log_dir, filename_csv)
    csv_writer_metadata = open(filepath_csv, 'w')

    # Write CSV headers.
    csv_headers = [
            'Timestamp',
            'Timestamp [s]',
            ]
    for (streamer_index, streamer) in enumerate(self._streamers):
      for (device_name, stream_writers) in csv_writers[streamer_index].items():
        for (stream_name, stream_writer) in stream_writers.items():
          stream_writer.write(','.join(csv_headers))
          # First check if custom header titles have been specified.
          stream_info = streamer.get_all_stream_infos()[device_name][stream_name]
          sample_size = stream_info['sample_size']
          if isinstance(stream_info['data_notes'], dict) and SensorStreamer.metadata_data_headings_key in stream_info['data_notes']:
            data_headers = stream_info['data_notes'][SensorStreamer.metadata_data_headings_key]
          else:
            # Write a number of data headers based on how many values are in each data sample.
            # Each sample may be a matrix that will be unwrapped into columns,
            #  so label headers as i-j where i is the original matrix row
            #  and j is the original matrix column (and if more than 2D keep adding more).
            data_headers = []
            subs = np.unravel_index(range(0,np.prod(sample_size)), sample_size)
            subs = np.stack(subs).T
            for header_index in range(subs.shape[0]):
              header = 'Data Entry '
              for sub_index in range(subs.shape[1]):
                header += '%d-' % subs[header_index, sub_index]
              header = header.strip('-')
              data_headers.append(header)
          stream_writer.write(',')
          stream_writer.write(','.join(data_headers))
          # Append headers for any extra items in the stream.
          for (extra_data_key, extra_data_info) in streamer.get_stream_info(device_name, stream_name)['extra_data_info'].items():
            sample_size = extra_data_info['sample_size']
            if np.prod(sample_size) == 1:
              # Use the specified data key as the header title.
              extra_data_headers = [extra_data_key]
            else:
              # See previous comments about unwrapping matrices into columns.
              subs = np.unravel_index(range(0,np.prod(sample_size)), sample_size)
              subs = np.stack(subs).T
              for header_index in range(subs.shape[0]):
                header = '%s ' % extra_data_key
                for sub_index in range(subs.shape[1]):
                  header += '%d-' % subs[header_index, sub_index]
                header = header.strip('-')
                extra_data_headers.append(header)
            stream_writer.write(',%s' % ','.join(extra_data_headers))

    # Store the writers
    self._csv_writers = csv_writers
    self._csv_writer_metadata = csv_writer_metadata

  # Create and initialize an HDF5 file.
  # Will have a single file for all streams from all devices.
  # Currently assumes that device names are unique across all streamers.
  def _init_writing_hdf5(self, log_time_s, for_streaming=True):
    self._log_status('Opening HDF5 writer')
    filename_base = self._get_filename_base(log_time_s=log_time_s,
                                            for_streaming=for_streaming)

    # Open an HDF5 file writer
    filename_hdf5 = '%s.hdf5' % filename_base
    filepath_hdf5 = os.path.join(self._log_dir, filename_hdf5)
    num_to_append = 1
    while os.path.exists(filepath_hdf5):
      filename_hdf5 = '%s_%02d.hdf5' % (filename_base, num_to_append)
      filepath_hdf5 = os.path.join(self._log_dir, filename_hdf5)
      num_to_append = num_to_append+1
    hdf5_file = h5py.File(filepath_hdf5, 'w')
    # Create a dataset for each data key of each stream of each device.
    for (streamer_index, streamer) in enumerate(self._streamers):
      for (device_name, streams_info) in streamer.get_all_stream_infos().items():
        device_group = hdf5_file.create_group(device_name)
        self._next_data_indexes_hdf5[streamer_index][device_name] = OrderedDict()
        for (stream_name, stream_info) in streams_info.items():
          # Skip saving videos in the HDF5.
          if stream_info['is_video'] and not self._videos_in_hdf5:
            continue
          # Skip saving audio in the HDF5.
          if stream_info['is_audio'] and not self._audio_in_hdf5:
            continue
          stream_group = device_group.create_group(stream_name)
          self._next_data_indexes_hdf5[streamer_index][device_name][stream_name] = 0
          # Create a dataset for each item in the stream.
          for data_key in streamer.get_stream_data_keys(device_name, stream_name):
            # The main data has specifications defined by stream_info.
            if data_key == 'data':
              sample_size = stream_info['sample_size']
              data_type = stream_info['data_type']
            # Any extra data has its specification in the extra stream info.
            elif data_key in stream_info['extra_data_info']:
              sample_size = stream_info['extra_data_info'][data_key]['sample_size']
              data_type = stream_info['extra_data_info'][data_key]['data_type']
            # The time fields will have their specifications hard-coded below.
            elif data_key == 'time_s':
              sample_size = [1]
              data_type = 'float64'
            elif data_key == 'time_str':
              sample_size = [1]
              data_type = 'S26'
            else:
              raise KeyError('Unknown data specification for %s.%s.%s' % (device_name, stream_name, data_key))
            # Create the dataset!
            stream_group.create_dataset(data_key,
                  (self._hdf5_log_length_increment, *sample_size),
                  maxshape=(None, *sample_size),
                  dtype=data_type,
                  chunks=True)
    # Store the writer.
    self._hdf5_file = hdf5_file

  # Create and initialize video writers.
  def _init_writing_videos(self, log_time_s, for_streaming=True):
    self._log_status('Opening video writers')
    filename_base = self._get_filename_base(log_time_s=log_time_s,
                                            for_streaming=for_streaming)
    # Create a video writer for each video stream of each device.
    self._video_writers = [OrderedDict() for i in range(len(self._streamers))]
    for (streamer_index, streamer) in enumerate(self._streamers):
      for (device_name, streams_info) in streamer.get_all_stream_infos().items():
        for (stream_name, stream_info) in streams_info.items():
          # Skip non-video streams.
          if not stream_info['is_video']:
            continue
          # Create a video writer.
          frame_height = stream_info['sample_size'][0]
          frame_width = stream_info['sample_size'][1]
          fps = stream_info['sampling_rate_hz']
          if self._videos_format.lower() == 'mp4':
            extension = 'mp4'
            fourcc = 'MP4V'
          elif self._videos_format.lower() == 'avi':
            extension = 'avi'
            fourcc = 'MJPG'
          else:
            raise AssertionError('Unsupported video format %s.  Can be mp4 or avi.' % self._videos_format)
          filename_video = '%s_%s_%s.%s' % (filename_base, device_name, stream_name, extension)
          filepath_video = os.path.join(self._log_dir, filename_video)
          video_writer = cv2.VideoWriter(filepath_video,
                              cv2.VideoWriter_fourcc(*fourcc),
                              fps, (frame_width, frame_height))
          # Store the writer.
          if device_name not in self._video_writers[streamer_index]:
            self._video_writers[streamer_index][device_name] = {}
          self._video_writers[streamer_index][device_name][stream_name] = video_writer

  # Create and initialize audio writers.
  def _init_writing_audio(self, log_time_s, for_streaming=True):
    self._log_status('Opening audio writers')
    filename_base = self._get_filename_base(log_time_s=log_time_s,
                                            for_streaming=for_streaming)
    # Create an audio writer for each audio stream of each device.
    self._audio_writers = [OrderedDict() for i in range(len(self._streamers))]
    for (streamer_index, streamer) in enumerate(self._streamers):
      for (device_name, streams_info) in streamer.get_all_stream_infos().items():
        for (stream_name, stream_info) in streams_info.items():
          # Skip non-audio streams.
          if not stream_info['is_audio']:
            continue
          # Create an audio writer.
          filename_audio = '%s_%s_%s.wav' % (filename_base, device_name, stream_name)
          filepath_audio = os.path.join(self._log_dir, filename_audio)
          audioStreaming_info = streamer.get_audioStreaming_info()
          audio_writer = wave.open(filepath_audio, 'wb')
          audio_writer.setnchannels(audioStreaming_info['num_channels'])
          audio_writer.setsampwidth(audioStreaming_info['sample_width'])
          audio_writer.setframerate(audioStreaming_info['sampling_rate'])
          # Store the writer.
          if device_name not in self._audio_writers[streamer_index]:
            self._audio_writers[streamer_index][device_name] = {}
          self._audio_writers[streamer_index][device_name][stream_name] = audio_writer


  #####################
  ###### WRITING ######
  #####################

  # Write provided data to the CSV logs.
  # Note that this can be called during streaming (periodic writing)
  #  or during post-experiment dumping.
  # @param new_data is a dict with 'data', 'time_str', 'time_s',
  #   and any extra fields specified by the streamer.
  #   The data may contain multiple timesteps (each value may be a list).
  def _write_data_CSV(self, streamer_index, device_name, stream_name, new_data):
    streamer = self._streamers[streamer_index]
    num_new_entries = len(new_data['data'])
    try:
      stream_writer = self._csv_writers[streamer_index][device_name][stream_name]
    except KeyError: # a writer was not created for this stream
      return
    for entry_index in range(num_new_entries):
      # Create a list of column entries to write.
      # Note that they should match the heading order in _init_writing_csv().
      to_write = []
      to_write.append(new_data['time_str'][entry_index])
      to_write.append(new_data['time_s'][entry_index])
      data_toWrite = new_data['data'][entry_index]
      if isinstance(data_toWrite, (list, tuple)):
        to_write.extend(data_toWrite)
      elif isinstance(data_toWrite, np.ndarray):
        to_write.extend(list(np.atleast_1d(data_toWrite.reshape(1, -1).squeeze())))
      else:
        to_write.append(data_toWrite)
      for extra_data_header in streamer.get_stream_info(device_name, stream_name)['extra_data_info'].keys():
        to_write.append(new_data[extra_data_header][entry_index])
      # Write the new row
      stream_writer.write('\n')
      stream_writer.write(','.join([str(x) for x in to_write]))
    stream_writer.flush()

  # Write provided data to the HDF5 file.
  # Note that this can be called during streaming (periodic writing)
  #  or during post-experiment dumping.
  # @param new_data is a dict with 'data', probably 'time_str' and 'time_s',
  #   and any extra fields specified by the streamer.
  #   The data may contain multiple timesteps (each value may be a list).
  def _write_data_HDF5(self, streamer_index, device_name, stream_name, new_data):
    try:
      stream_group = self._hdf5_file['/'.join([device_name, stream_name])]
    except KeyError: # a dataset was not created for this stream
      return
    # Get the amount of new data to write and the index to start writing it.
    num_new_entries = len(new_data['data'])
    starting_index = self._next_data_indexes_hdf5[streamer_index][device_name][stream_name]
    num_old_entries = starting_index
    # Write the data!
    for data_key in new_data.keys():
      dataset = stream_group[data_key]
      # Expand the dataset if needed.
      while len(dataset) < num_old_entries + num_new_entries:
        self._log_debug('Expanding dataset %s.%s.%s from %d' % (device_name, stream_name, data_key, len(dataset)))
        dataset.resize((len(dataset) + self._hdf5_log_length_increment, *dataset.shape[1:]))
      # Write the new entries.
      ending_index = starting_index + num_new_entries - 1
      data_toWrite = new_data[data_key]
      if dataset.dtype.char == 'S':
        if len(data_toWrite) > 0 and isinstance(data_toWrite[0], (list, tuple)):
          data_toWrite = [[n.encode("ascii", "ignore") for n in data_row] for data_row in data_toWrite]
        else:
          data_toWrite = [n.encode("ascii", "ignore") for n in data_toWrite]
      dataset[starting_index:ending_index+1,:] = np.array(data_toWrite).reshape((-1, *dataset.shape[1:]))
    self._hdf5_file.flush()
    # Update the next starting index to use.
    next_starting_index = starting_index + num_new_entries
    self._next_data_indexes_hdf5[streamer_index][device_name][stream_name] = next_starting_index

  # Write provided data to the video files.
  # Note that this can be called during streaming (periodic writing)
  #  or during post-experiment dumping.
  # @param new_data is a dict with 'data' (all other keys will be ignored).
  #   The 'data' entry must contain raw frames as a matrix.
  #   The data may contain multiple timesteps (a list of matrices).
  def _write_data_video(self, streamer_index, device_name, stream_name, new_data):
    streamer = self._streamers[streamer_index]
    num_new_entries = len(new_data['data'])
    try:
      video_writer = self._video_writers[streamer_index][device_name][stream_name]
    except KeyError: # a writer was not created for this stream
      return
    for entry_index in range(num_new_entries):
      # Assume the data is the frame.
      video_writer.write(new_data['data'][entry_index])

  # Write provided data to the audio files.
  # Note that this can be called during streaming (periodic writing)
  #  or during post-experiment dumping.
  # @param new_data is a dict with 'data' (all other keys will be ignored).
  #   The 'data' entry should be a list of raw audio readings, so it should be
  #    a list of lists (since audio data at each timestep is assumed to be a list, i.e. a chunk).
  def _write_data_audio(self, streamer_index, device_name, stream_name, new_data):
    streamer = self._streamers[streamer_index]
    try:
      audio_writer = self._audio_writers[streamer_index][device_name][stream_name]
    except KeyError: # a writer was not created for this stream
      return
    # Assume the data is a list of lists (each entry is a list of chunked audio data).
    if len(new_data['data']) > 0:
      audio_writer.writeframes(b''.join([bytearray(x) for x in new_data['data']]))

  # Write metadata from each streamer to CSV and/or HDF5 files.
  # Will include device-level metadata and any lower-level data notes.
  def _log_metadata(self):
    for (streamer_index, streamer) in enumerate(self._streamers):
      for (device_name, streams_info) in streamer.get_all_stream_infos().items():
        # Get metadata for this device.
        # To make it HDF5 compatible,
        #  flatten the dictionary and then
        #  prune objects that can't be converted to a string easily.
        device_metadata = streamer.get_metadata(device_name=device_name, only_str_values=True)
        device_metadata = convert_dict_values_to_str(device_metadata, preserve_nested_dicts=False)
        # Write the device-level metadata.
        if self._csv_writer_metadata is not None:
          self._csv_writer_metadata.write('\n%s,%s\n' % ('='*25,'='*25))
          self._csv_writer_metadata.write('Device Name,%s' % (device_name))
          self._csv_writer_metadata.write('\n%s,%s' % ('='*25,'='*25))
          for (meta_key, meta_value) in device_metadata.items():
            self._csv_writer_metadata.write('\n')
            self._csv_writer_metadata.write('%s,"%s"' % (str(meta_key), str(meta_value)))
          self._csv_writer_metadata.write('\n')
        if self._hdf5_file is not None:
          device_group = self._hdf5_file['/'.join([device_name])]
          device_group.attrs.update(device_metadata)

        # Get data notes for each stream.
        for (stream_name, stream_info) in streams_info.items():
          data_notes = stream_info['data_notes']
          if isinstance(data_notes, dict):
            stream_metadata = data_notes
          else:
            stream_metadata = {'data_notes':data_notes}
          stream_metadata = convert_dict_values_to_str(stream_metadata, preserve_nested_dicts=False)
          # Write the stream-level metadata.
          if self._csv_writer_metadata is not None:
            self._csv_writer_metadata.write('\n')
            self._csv_writer_metadata.write('Stream Name,%s' % (stream_name))
            for (meta_key, meta_value) in stream_metadata.items():
              self._csv_writer_metadata.write('\n')
              self._csv_writer_metadata.write('%s,"%s"' % (str(meta_key), str(meta_value)))
            self._csv_writer_metadata.write('\n')
          if self._hdf5_file is not None:
            try:
              stream_group = self._hdf5_file['/'.join([device_name, stream_name])]
              stream_group.attrs.update(stream_metadata)
            except KeyError: # a writer was not created for this stream
              pass


  # Flush/close CSV, HDF5, and video file writers.
  def close_files(self):
    # Flush/close all of the CSV writers
    if self._csv_writers is not None:
      self._log_status('Closing CSV writers')
      for (streamer_index, streamer) in enumerate(self._streamers):
        for (device_name, stream_writers) in self._csv_writers[streamer_index].items():
          for (stream_name, stream_writer) in stream_writers.items():
            stream_writer.close()
      self._csv_writers = None
      self._csv_writer_metadata = None

    # Flush/close the HDF5 file.
    # Also resize datasets to remove extra empty rows.
    if self._hdf5_file is not None:
      self._log_status('Closing HDF5 writer')
      for (streamer_index, streamer) in enumerate(self._streamers):
        try:
          for (device_name, streams_info) in streamer.get_all_stream_infos().items():
            for (stream_name, stream_info) in streams_info.items():
              try:
                stream_group = self._hdf5_file['/'.join([device_name, stream_name])]
              except KeyError: # a dataset was not created for this stream
                continue
              starting_index = self._next_data_indexes_hdf5[streamer_index][device_name][stream_name]
              ending_index = starting_index - 1
              for data_key in streamer.get_stream_data_keys(device_name, stream_name):
                dataset = stream_group[data_key]
                dataset.resize((ending_index+1, *dataset.shape[1:]))
        except BrokenPipeError:
          print('XXXX Could not close dataset for streamer index %d' % streamer_index)
      self._hdf5_file.close()
      self._hdf5_file = None

    # Flush/close all of the video writers.
    if self._video_writers is not None:
      self._log_status('Closing video writers')
      for (streamer_index, streamer) in enumerate(self._streamers):
        for (device_name, video_writers) in self._video_writers[streamer_index].items():
          for (stream_name, video_writer) in video_writers.items():
            video_writer.release()
      self._video_writers = None

    # Flush/close all of the audio writers.
    if self._audio_writers is not None:
      self._log_status('Closing audio writers')
      for (streamer_index, streamer) in enumerate(self._streamers):
        for (device_name, audio_writers) in self._audio_writers[streamer_index].items():
          for (stream_name, audio_writer) in audio_writers.items():
            audio_writer.close()
      self._audio_writers = None

  ##########################
  ###### DATA LOGGING ######
  ##########################

  # Helper to start the stream-logging thread to periodically write data.
  def _start_stream_logging(self):
    self._log_debug('DataLogger starting stream-logging')
    self._streamLog_thread = Thread(target=self._log_data, args=())
    self._streamLogging = True
    self._flush_log = False
    self._streamLog_thread.start()

  # Helper to stop the stream-logging thread to periodically write data.
  # Will write any outstanding data,
  #  and will log any metadata associated with the streamers.
  # Will wait for the thread to finish before returning.
  def _stop_stream_logging(self):
    self._log_debug('DataLogger stopping stream-logging')
    self._flush_log = True
    self._streamLogging = False
    if self._streamLog_thread is not None and self._streamLog_thread.is_alive():
      self._log_debug('DataLogger joining stream-logging thread')
      try:
        self._streamLog_thread.join()
      except:
        print('XXX Error joining the streamLog thread')
    # Log metadata
    self._log_metadata()
    # Flush/close all files
    self.close_files()

  # Helper method to log all data currently available.
  # This is meant to be called at the end of an experiment, if dump_* options are enabled.
  # Will first terminate any ongoing stream-logging that was happening during the experiment.
  def log_all_data(self, dump_csv=False, dump_hdf5=False, dump_video=False, dump_audio=False, log_time_s=None):
    self._log_status('DataLogger dumping all data')
    # Stop any ongoing stream-logging.
    self._stop_stream_logging()
    # Initialize the file writers.
    log_time_s = log_time_s or time.time()
    if dump_csv:
      self._init_writing_csv(log_time_s, for_streaming=False)
    if dump_hdf5:
      self._init_writing_hdf5(log_time_s, for_streaming=False)
    if dump_video:
      self._init_writing_videos(log_time_s, for_streaming=False)
    if dump_audio:
      self._init_writing_audio(log_time_s, for_streaming=False)
    # Log all data.
    # Will basically enable periodic stream-logging,
    #  but will set self._flush_log and self._streamLogging such that
    #  it seems like the experiment ended and just a final flush is required.
    #  This will cause the stream-logging in self._log_data()
    #  to fetch and write any outstanding data, which is all data since
    #  none is written yet.  It will then exit after the single write.
    # First, save the existing stream-logging options.
    stream_csv = self._stream_csv
    stream_hdf5 = self._stream_hdf5
    stream_video = self._stream_video
    stream_audio = self._stream_audio
    # Then pretend like the dumping options are actually streaming options.
    self._stream_csv = self._dump_csv
    self._stream_hdf5 = self._dump_hdf5
    self._stream_video = self._dump_video
    self._stream_audio = self._dump_audio
    # Configure it to be the final 'periodic' stream-writing flush.
    self._streamLogging = False
    self._flush_log = True
    # Initialize indexes and log all of the data.
    self._init_log_indexes()
    self._log_data()
    # Log metadata.
    self._log_metadata()
    # Flush/close all files.
    self.close_files()
    # Restore the original stream-logging configuration.
    self._stream_csv = stream_csv
    self._stream_hdf5 = stream_hdf5
    self._stream_video = stream_video
    self._stream_audio = stream_audio

  # Poll data from each streamer and log it, either periodically or all at once.
  # The poll period is set by self._stream_period_s.
  # Will loop until self._streamLogging is False, and then
  #  will do one final fetch/log if self._flush_log is True.
  # Usage to periodically poll data from each streamer and log it:
  #   Set self._streamLogging to True and self._flush_log to False
  #     then run this method in a thread.
  #   To finish, set self._streamLogging to False and self._flush_log to True.
  # Usage to log all available data once:
  #   Set self._streamLogging to False and self._flush_log to True
  #     then run this method (in a thread or in the main thread).
  def _log_data(self):
    last_log_time_s = None
    self._log_status('Starting stream-logging thread')
    try:
      flushing_log = False
      while self._streamLogging or self._flush_log:
        # Wait until it is time to write new data, which is either:
        #  This is the first iteration,
        #  it has been at least self._stream_period_s since the last write, or
        #  periodic logging has been deactivated.
        self._log_debug('Stream-logging thread waiting for next write time')
        while last_log_time_s is not None \
                and time.time() - last_log_time_s < self._stream_period_s \
                and self._streamLogging:
          # Sleep a reasonable amount by default to poll the stream-logging,
          #  but sleep shorter as the next log time approaches.
          time.sleep(min(0.1, (last_log_time_s + self._stream_period_s) - time.time()))
        if not self._streamLogging and not self._flush_log:
          continue
        self._log_status('Stream-logging finished waiting for write time')
        # Update the last log time now, before the write actually starts.
        # This will keep the log period more consistent; otherwise, the amount
        #   of time it takes to perform the write would be added to the log period.
        #   This would compound over time, leading to longer delays and more data to write each time.
        #   This becomes more severe as the write duration increases (e.g. videos).
        last_log_time_s = time.time()
        # If the log should be flushed, record that it is happening during this iteration for all streamers.
        if self._flush_log:
          flushing_log = True
        # Write new data for each stream of each device of each streamer.
        for (streamer_index, streamer) in enumerate(self._streamers):
          for (device_name, streams_info) in streamer.get_all_stream_infos().items():
            # self._log_debug('Logging streams for streamer %d device %s' % (streamer_index, device_name))
            for (stream_name, stream_info) in streams_info.items():
              # Fetch data starting with the first timestep that hasn't been logged yet,
              #  and ending at the most recent data (or back by a few timesteps
              #  if the streamer may still edit the most recent timesteps).
              starting_index = self._next_data_indexes[streamer_index][device_name][stream_name]
              ending_index = -self._timesteps_before_solidified[streamer_index][device_name][stream_name]
              if self._flush_log: # Flushing everything, so write it even if it would normally need more time to solidify
                ending_index = None
              if ending_index == 0: # no time is needed to solidify, so fetch up to the most recent data
                ending_index = None
              new_data = streamer.get_data(device_name, stream_name, return_deepcopy=False,
                                            starting_index=starting_index, ending_index=ending_index)
              # Write any new data to files.
              if new_data is not None:
                if self._stream_csv:
                  self._write_data_CSV(streamer_index, device_name, stream_name, new_data)
                if self._stream_hdf5:
                  self._write_data_HDF5(streamer_index, device_name, stream_name, new_data)
                if self._stream_video:
                  self._write_data_video(streamer_index, device_name, stream_name, new_data)
                if self._stream_audio:
                  self._write_data_audio(streamer_index, device_name, stream_name, new_data)
                # Update starting indexes for the next write.
                num_new_entries = len(new_data['data'])
                next_starting_index = starting_index + num_new_entries
                # Clear the logged data if desired.
                if stream_info['is_video']:
                  logged_data = self._stream_video
                  dumping_data = self._dump_video
                elif stream_info['is_audio']:
                  logged_data = self._stream_audio
                  dumping_data = self._dump_audio
                else:
                  logged_data = self._stream_csv or self._stream_hdf5
                  dumping_data = self._dump_csv or self._dump_hdf5
                can_clear_data = logged_data and (not dumping_data)
                if self._clear_logged_data_from_memory and can_clear_data:
                  try:
                    streamer.clear_data(device_name, stream_name, first_index_to_keep=next_starting_index)
                  except EOFError:
                    print('XXX Error trying to clear data from streamer [%s]' % stream_name)
                  next_starting_index = 0
                self._next_data_indexes[streamer_index][device_name][stream_name] = next_starting_index
                self._log_debug('Logged %d new entries for stream %s.%s' % (num_new_entries, device_name, stream_name))
                new_data = None # should be unnecessary, but maybe it helps free the memory?
        self._log_debug('Stream-logging finished write')
        # If stream-logging is disabled, but a final flush had been requested,
        #  record that the flush is complete so streaming can really stop now.
        # Note that it also checks whether the flush was configured to happen for all streamers during this iteration.
        #  Especially if a lot of data was being written (such as with video data),
        #  the self._flush_log flag may have been set sometime during the data writing.
        #  In that case, all streamers would not have known to flush data and some data may be omitted.
        if (not self._streamLogging) and self._flush_log and flushing_log:
          self._log_debug('Stream-logging finished flushing')
          self._flush_log = False
      flushing_log = False
    except KeyboardInterrupt: # The program was likely terminated
      pass
    except:
      self._log_error('\n'*10 + ('***ERROR DURING DATA LOGGING:\n%s' % traceback.format_exc()) + '\n'*10)
      try:
        self._log_warn('\nThe last attempted DataLogger fetch was for %s.%s.\n' % (device_name, stream_name))
      except:
        pass
    finally:
      # Stop all streamers if not done already.
      # Note that this is especially important if an error occurred,
      #  to try to make sure that the files are closed so data is not lost.
      if self._streamLogging:
        try:
          self.stop()
        except:
          # Try to write the metadata at least.
          try:
            self._log_metadata()
          except:
            pass
          # And at least flush/close the main files to save any data logged so far.
          self.close_files()

  #####################################
  ###### EXTERNAL DATA RECORDING ######
  #####################################
  
  # Start recordings from sensor-specific software.
  def start_external_recordings(self):
    if not self._use_external_recording_sources:
      return
    # Create a folder for externally recorded data.
    external_data_dir = os.path.join(self._log_dir, 'externally_recorded_data')
    os.makedirs(external_data_dir, exist_ok=True)
    # Tell each streamer to start external recording if supported.
    #  Do ones that require user action first, so automatic ones aren't recording in the meantime.
    for streamer in self._streamers:
      if streamer.external_data_recording_requires_user():
        streamer.start_external_data_recording(external_data_dir)
    for streamer in self._streamers:
      if not streamer.external_data_recording_requires_user():
        streamer.start_external_data_recording(external_data_dir)
  
  # Stop recordings from sensor-specific software.
  def stop_external_recordings(self):
    if not self._use_external_recording_sources:
      return
    # Tell each streamer to stop external recording if supported.
    #  Do ones that require user action last, so automatic ones aren't recording in the meantime.
    for (streamer_index, streamer) in enumerate(self._streamers):
      if not streamer.external_data_recording_requires_user():
        try:
          streamer.stop_external_data_recording()
        except:
          print('XXX Error trying to stop external data recording for streamer index %d' % streamer_index)
    for (streamer_index, streamer) in enumerate(self._streamers):
      if streamer.external_data_recording_requires_user():
        try:
          streamer.stop_external_data_recording()
        except:
          print('XXX Error trying to stop external data recording for streamer index %d' % streamer_index)
        
  # Merge logged data with recordings from sensor-specific software.
  @staticmethod
  def merge_external_data(log_dir, print_status=False, print_debug=False):
    import glob
    import importlib
    hdf5_filepaths = glob.glob(os.path.join(log_dir, '*.hdf5'))
    if len(hdf5_filepaths) == 0:
      return
    hdf5_filepath = hdf5_filepaths[0]
    hdf5_filename = os.path.basename(hdf5_filepath)
    hdf5_file_toUpdate = h5py.File(hdf5_filepath, 'a')
    data_dir_toUpdate = log_dir
    data_dir_streamed = log_dir
    data_dir_external_original = os.path.join(log_dir, 'externally_recorded_data')
    data_dir_archived = os.path.join(log_dir, 'archived_data_replaced_by_postprocessing')
    os.makedirs(data_dir_archived, exist_ok=True)
    hdf5_filename_archived = '%s_archived.hdf5' % os.path.splitext(hdf5_filename)[0]
    hdf5_file_archived = h5py.File(os.path.join(data_dir_archived, hdf5_filename_archived), 'w')
    
    # Create a streamer object for each streamer that added data to the HDF5 file.
    streamers = []
    class_names = [] # use to detect duplicates - will only use one instance of each streamer type
    for (device_name, device_group) in hdf5_file_toUpdate.items():
      metadata = dict(device_group.attrs.items())
      if 'SensorStreamer class name' not in metadata:
        continue
      class_name = metadata['SensorStreamer class name']
      if class_name not in class_names:
        class_module = importlib.import_module('sensor_streamers.%s' % class_name, class_name)
        class_type = getattr(class_module, class_name)
        streamers.append(class_type(print_status=print_status, print_debug=print_debug))
        class_names.append(class_name)
    
    # Tell each streamer to merge external recordings if supported.
    for streamer in streamers:
      streamer.merge_external_data_with_streamed_data(# Final post-processed outputs
                                                      hdf5_file_toUpdate,
                                                      data_dir_toUpdate,
                                                      # Original streamed and external data
                                                      data_dir_streamed,
                                                      data_dir_external_original,
                                                      # Archives for data no longer needed
                                                      data_dir_archived,
                                                      hdf5_file_archived)
    # Delete the original external data folder if it is now empty.
    try:
      os.rmdir(data_dir_external_original)
    except OSError:
      pass
    # Close the HDF5 files.
    hdf5_file_toUpdate.close()
    hdf5_file_archived.close()
    
  #####################
  ###### RUNNING ######
  #####################

  # The main run method for the DataLogger.
  # Initialize and start stream-logging to periodically write data if desired.
  # The SensorStreamers should already be connected and running when this is called.
  def run(self):
    log_start_time_s = time.time()
    self._log_status('DataLogger running')

    # Set up CSV/HDF5 file writers for stream-logging if desired.
    if self._stream_csv:
      self._init_writing_csv(log_start_time_s, for_streaming=True)
    if self._stream_hdf5:
      self._init_writing_hdf5(log_start_time_s, for_streaming=True)
    if self._stream_video:
      self._init_writing_videos(log_start_time_s, for_streaming=True)
    if self._stream_audio:
      self._init_writing_audio(log_start_time_s, for_streaming=True)
    # Start stream-logging.
    if self._stream_csv or self._stream_hdf5 or self._stream_video or self._stream_audio:
      self._init_log_indexes()
      self._start_stream_logging()

  # Stop all data logging.
  # Will stop stream-logging if it is active.
  # Will dump all data if desired.
  # The SensorStreamers should already be stopped when this is called.
  def stop(self):
    log_stop_time_s = time.time()
    self._log_status('DataLogger stopping')

    # Stop stream-logging and wait for it to finish.
    self._stop_stream_logging()

    # Write all data if desired.
    if self._dump_csv or self._dump_hdf5 or self._dump_video or self._dump_audio:
      self.log_all_data(dump_csv=self._dump_csv, dump_hdf5=self._dump_hdf5,
                        dump_video=self._dump_video, dump_audio=self._dump_audio,
                        log_time_s=log_stop_time_s)
      
    self._log_debug('DataLogger stopped')

  ######################
  ###### PRINTING ######
  ######################

  def _log_status(self, msg, *extra_msgs, **kwargs):
    write_log_message(msg, *extra_msgs, source_tag=self._log_source_tag,
                      print_message=self._print_status, filepath=self._log_history_filepath, **kwargs)
  def _log_debug(self, msg, *extra_msgs, **kwargs):
    write_log_message(msg, *extra_msgs, source_tag=self._log_source_tag,
                      print_message=self._print_debug, debug=True, filepath=self._log_history_filepath, **kwargs)
  def _log_error(self, msg, *extra_msgs, **kwargs):
    write_log_message(msg, *extra_msgs, source_tag=self._log_source_tag,
                      print_message=True, error=True, filepath=self._log_history_filepath, **kwargs)
  def _log_warn(self, msg, *extra_msgs, **kwargs):
    write_log_message(msg, *extra_msgs, source_tag=self._log_source_tag,
                      print_message=True, warning=True, filepath=self._log_history_filepath, **kwargs)
  def _log_userAction(self, msg, *extra_msgs, **kwargs):
    write_log_message(msg, *extra_msgs, source_tag=self._log_source_tag,
                      print_message=True, userAction=True, filepath=self._log_history_filepath, **kwargs)



#####################
###### TESTING ######
#####################
if __name__ == '__main__':

  from sensor_streamers.NotesStreamer      import NotesStreamer
  from sensor_streamers.MyoStreamer        import MyoStreamer
  from sensor_streamers.XsensStreamer      import XsensStreamer
  from sensor_streamers.TouchStreamer      import TouchStreamer
  from sensor_streamers.EyeStreamer        import EyeStreamer
  from sensor_streamers.MicrophoneStreamer import MicrophoneStreamer
  import time

  # Configure printing to the console.
  print_debug = False
  print_status = True

  # Configure where to save sensor data.
  (log_time_str, log_time_s) = get_time_str(return_time_s=True)
  log_tag = 'testing'
  log_dir_root = '../data_logs'
  log_subdir = '%s_%s' % (log_time_str, log_tag)
  log_dir = os.path.join(log_dir_root, log_subdir)

  # Create the streamers.
  notes_streamer = NotesStreamer(print_debug=print_debug, print_status=print_status)
  myo_streamer = MyoStreamer(num_myos=2, print_debug=print_debug, print_status=print_status)
  xsens_streamer = XsensStreamer(print_debug=print_debug, print_status=print_status)
  touch_streamer = TouchStreamer(print_debug=print_debug, print_status=print_status)
  eye_streamer = EyeStreamer(stream_video_world=True, stream_video_worldGaze=True,
                              stream_video_eye=True,
                              print_debug=print_debug, print_status=print_status)
  microphone_streamer = MicrophoneStreamer(device_names_withAudioKeywords={'internal': 'Realtek'},
                                           print_debug=print_debug, print_status=print_status)

  # Create a logger to record the streaming data.
  # Put data from all sensors into the same HDF5 file for now.
  #   Alternatively, multiple DataLoggers could be created with
  #   various subsets of the streamers.
  # Note that the stream_* arguments are whether to periodically save data,
  #   and the dump_* arguments are whether to wait until the end and then save all data.
  #   Any combination of these options can be enabled.
  streamers = [
      notes_streamer,
      myo_streamer,
      xsens_streamer,
      touch_streamer,
      eye_streamer,
      microphone_streamer,
    ]
  logger = DataLogger(streamers, log_dir=log_dir, log_tag=log_tag,
                      # Choose whether to periodically write data to files.
                      stream_csv=False, stream_hdf5=True, stream_video=False, stream_audio=False,
                      stream_period_s=5,
                      # Choose whether to write all data at the end.
                      dump_csv=False, dump_hdf5=False, dump_video=True, dump_audio=True,
                      # Additional configuration.
                      videos_format='avi', # mp4 occasionally gets openCV errors about a tag not being supported?
                      audio_format='wav', # currently only supports WAV
                      print_status=print_status, print_debug=print_debug)


  # Connect and start the streamers.
  for (streamer_index, streamer) in enumerate(streamers):
    if print_status: print('\nConnecting streamer %d/%d (class %s)' % (streamer_index+1, len(streamers), type(streamer).__name__))
    connected = streamer.connect()
    if not connected:
      raise AssertionError('Error connecting the streamer')
  for (streamer_index, streamer) in enumerate(streamers):
    if print_status: print('Starting streamer %d/%d' % (streamer_index+1, len(streamers)))
    streamer.run()

  # Log data!
  print()
  print('Starting data logger and streamers')
  print('Enter \'quit\' or \'q\' as an experiment note to end the program')
  print()
  logger.run()

  # Wait for a set time or until the user quits.
  duration_s = 15
  time_start_s = time.time()
  last_notes = ''
  while last_notes not in ['quit', 'q'] and time.time() - time_start_s < duration_s:
    time.sleep(1)
    last_notes = notes_streamer.get_last_notes()
    if last_notes is not None:
      last_notes = last_notes.lower().strip()

  # Stop streamers and data logging.
  print()
  print('Main loop terminated - stopping data logger and streamers')
  for (streamer_index, streamer) in enumerate(streamers):
      if print_status: print('\nStopping streamer %d/%d of class %s' % (streamer_index+1, len(streamers), type(streamers[streamer_index]).__name__))
      streamer.stop()
  logger.stop()


