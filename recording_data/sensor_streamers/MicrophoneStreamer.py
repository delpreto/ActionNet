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

import pyaudio
import wave
import numpy as np
import array
import sys
from collections import OrderedDict
import time
import traceback

################################################
################################################
# A class to stream from one or more microphones.
################################################
################################################
class MicrophoneStreamer(SensorStreamer):
  
  ########################
  ###### INITIALIZE ######
  ########################
  
  def __init__(self, streams_info=None, log_player_options=None,
               visualization_options=None,
               device_names_withAudioKeywords=None, # dict from audio device keyword to streaming device name
               print_status=True, print_debug=False, log_history_filepath=None):
    SensorStreamer.__init__(self, streams_info,
                            log_player_options=log_player_options,
                            visualization_options=visualization_options,
                            print_status=print_status, print_debug=print_debug,
                            log_history_filepath=log_history_filepath)
    
    self._log_source_tag = 'mic'

    # Initialize PyAudio.
    self._pyaudio = pyaudio.PyAudio()
    
    # Configuration (will use the same settings for all input streams).
    self._bytes_per_sample = 2
    self._num_channels = 1
    self._audio_sampling_rate_hz = 48000
    self._chunk_size = 2048 # number of samples to acquire before running the callback function
    array_decoding_codes = { # based on https://docs.python.org/3/library/array.html
      1: 'B',
      2: 'h',
      4: 'l',
      8: 'q',
    }
    self._array_decoding_code = array_decoding_codes[self._bytes_per_sample]
    data_types = {
      1: 'int8',
      2: 'int16',
      4: 'int32',
      8: 'int64',
    }
    self._data_type = data_types[self._bytes_per_sample]
    self._format = self._pyaudio.get_format_from_width(self._bytes_per_sample)
    self._device_names_withAudioKeywords = device_names_withAudioKeywords
    self._device_names_withAudioIndexes = OrderedDict()
    self._audio_streams = []
    self._num_chunks_received = {}
  
  # Helper to get information about the audio sampling
  # (assumed to be the same for all inputs used).
  def get_audioStreaming_info(self):
    return {
      'num_channels':  self._num_channels,
      'sample_width':  self._pyaudio.get_sample_size(self._format),
      'sampling_rate': self._audio_sampling_rate_hz,
    }
  
  #####################
  ###### HELPERS ######
  #####################
  
  # Discover available input and output devices.
  def _scan_available_audioDevices(self):
    info = self._pyaudio.get_host_api_info_by_index(0)
    num_audioDevices = info.get('deviceCount')
    audioDevices_info = [self._pyaudio.get_device_info_by_host_api_device_index(0, i)
                          for i in range(num_audioDevices)]
    self._input_audioDevices_info = OrderedDict()
    self._output_audioDevices_info = OrderedDict()
    for (audioDevice_index, audioDevice_info) in enumerate(audioDevices_info):
      if audioDevice_info.get('maxInputChannels') > 0:
        self._input_audioDevices_info[audioDevice_index] = audioDevice_info
      elif audioDevice_info.get('maxOutputChannels') > 0:
        self._output_audioDevices_info[audioDevice_index] = audioDevice_info
    if self._print_debug:
      self._log_debug(self._get_available_audioDevices_list_str())
      
  def _get_available_audioDevices_list_str(self):
    msg = 'Input audio devices:'
    for (audioDevice_index, audioDevice_info) in self._input_audioDevices_info.items():
      msg += '\n ID %2d: %s' % (audioDevice_index, audioDevice_info.get('name'))
    msg += '\nOutput audio devices:'
    for (audioDevice_index, audioDevice_info) in self._output_audioDevices_info.items():
      msg += '\n ID %2d: %s' % (audioDevice_index, audioDevice_info.get('name'))
    return msg
  
  def print_available_audioDevices(self):
    print(self._get_available_audioDevices_list_str())
    
  def _device_name_from_audioDevice_name(self, audioDevice_name):
    device_name = audioDevice_name
    device_name = device_name.lower()
    device_name = device_name.replace(' ', '-')
    device_name = device_name.replace('(', '_')
    device_name = device_name.replace(')', '_')
    device_name = device_name.replace('__', '_')
    device_name = device_name.replace('-_', '_')
    device_name = device_name.strip('-').strip('_').strip()
    return device_name
  
  #####################
  ###### CONNECT ######
  #####################
  def _connect(self, timeout_s=10):
    # Discover available devices.
    self._scan_available_audioDevices()
    # Select the desired input device(s).
    self._device_names_withAudioIndexes = OrderedDict()
    # Use the specified device(s).
    if self._device_names_withAudioKeywords is not None:
      for (audioDevice_index, audioDevice_info) in self._input_audioDevices_info.items():
        audioDevice_name = audioDevice_info.get('name')
        for (device_name, audioKeyword) in self._device_names_withAudioKeywords.items():
          if audioKeyword in audioDevice_name:
            self._device_names_withAudioIndexes[device_name] = audioDevice_index
      # Check that all desired devices were found
      if len(self._device_names_withAudioKeywords) != len(self._device_names_withAudioIndexes):
        raise AssertionError('Found %d/%d specified microphones.  The following devices are available:\n%s\n\n'
                             % (len(self._device_names_withAudioIndexes),
                                len(self._device_names_withAudioKeywords),
                                self._get_available_audioDevices_list_str()))
    # Choose a device if none was specified.
    if len(self._device_names_withAudioIndexes) == 0:
      # Try to use any USB microphone(s).
      for (audioDevice_index, audioDevice_info) in self._input_audioDevices_info.items():
        audioDevice_name = audioDevice_info.get('name')
        device_name = self._device_name_from_audioDevice_name(audioDevice_name)
        if 'USB' in audioDevice_name:
          self._device_names_withAudioIndexes[device_name] = audioDevice_index
      # If none were found, just use the first device.
      if len(self._device_names_withAudioIndexes) == 0:
        (audioDevice_index, audioDevice_info) = list(self._input_audioDevices_info.items())[0]
        audioDevice_name = audioDevice_info.get('name')
        device_name = self._device_name_from_audioDevice_name(audioDevice_name)
        self._device_names_withAudioIndexes[device_name] = audioDevice_index
      
    # Print the selected device(s).
    msg = 'Recording from the following %d audio devices' % len(self._device_names_withAudioIndexes)
    for (device_name, audioDevice_index) in self._device_names_withAudioIndexes.items():
      msg += '\n  Audio device ID %2d (%s) as stream %s' % (audioDevice_index, self._input_audioDevices_info[audioDevice_index].get('name'), device_name)
    self._log_status(msg)
    
    # Open a stream for each input device.
    self._audio_streams = []
    for (device_name, audioDevice_index) in self._device_names_withAudioIndexes.items():
      audioDevice_info = self._input_audioDevices_info[audioDevice_index]
      self._audio_streams.append(self._pyaudio.open(
                                  format=self._format,
                                  input=True,
                                  output=False,
                                  input_device_index=audioDevice_index,
                                  output_device_index=None,
                                  channels=self._num_channels,
                                  rate=self._audio_sampling_rate_hz,
                                  frames_per_buffer=self._chunk_size,
                                  stream_callback=lambda *args,
                                                         device_name=device_name, # bind it to the current value
                                                         audioDevice_index=audioDevice_index: # bind it to the current value
                                                      self._mic_data_callback(device_name, audioDevice_index, *args),
                                  ))
        
      self._num_chunks_received[audioDevice_index] = 0
      self.add_stream(device_name=device_name,
                      stream_name='chunked_data',
                      data_type=self._data_type,
                      sample_size=[self._chunk_size],
                      sampling_rate_hz=(self._audio_sampling_rate_hz / self._chunk_size),
                      is_audio=True,
                      extra_data_info={},
                      data_notes=OrderedDict([
                        ('Description', 'Raw audio data recorded from the microhpone.'
                                        'Audio data was chunked into buffers in hardware,'
                                        'then each chunk of samples was logged as a single entry.'
                                        'Each timestamp estimates the system time of the '
                                        '*last* sample in the chunk.'),
                      ]))
      self.add_stream(device_name=device_name,
                      stream_name='chunk_timestamp',
                      data_type='float64',
                      sample_size=[1],
                      sampling_rate_hz=(self._audio_sampling_rate_hz / self._chunk_size),
                      extra_data_info={},
                      data_notes=OrderedDict([
                        ('Description', 'Audio data was chunked into buffers in hardware,'
                                        'then each chunk of samples was logged as a single entry.'
                                        'Each timestamp here estimates the system time of the '
                                        '*last* sample in the chunk.'),
                      ]))
      
    return True

  #######################
  ###### Recording ######
  #######################
  
  # Callback function to process new audio data when it is available.
  def _mic_data_callback(self, device_name, audioDevice_index, data, frame_count, time_info, status):
    # Timestamp the data and advance the chunk counter.
    time_s = time.time()
    self._num_chunks_received[audioDevice_index] = 1 + self._num_chunks_received[audioDevice_index]
    # Ignore the first few chunks to let everything settle.
    if self._num_chunks_received[audioDevice_index] >= 3:
      # Decode the bytes.
      data_decoded = np.array(array.array(self._array_decoding_code, data))
      if sys.byteorder == 'big':
        data_decoded = data_decoded.byteswap()
      # Store the new data.
      self.append_data(device_name, 'chunked_data', time_s, data_decoded)
      self.append_data(device_name, 'chunk_timestamp', time_s, time_s)
    # All done!
    return (data, pyaudio.paContinue)
  
  ###########################
  ###### Visualization ######
  ###########################
  
  # Specify how the stream should be visualized.
  def get_default_visualization_options(self, visualization_options=None):
    processed_options = OrderedDict([(device_name,
                                      {'chunked_data': {'class':None},
                                       'chunk_timestamp': {'class':None}})
                                     for device_name in self.get_device_names()])
    return processed_options
  
  #####################
  ###### RUNNING ######
  #####################
  
  # Loop until self._running is False
  def _run(self):
    try:
      # Start the streams!
      for audio_stream in self._audio_streams:
        audio_stream.start_stream()
      # Wait for the program to end.
      while self._running:
        time.sleep(1)
    except KeyboardInterrupt: # The program was likely terminated
      pass
    except:
      self._log_error('\n\n***ERROR RUNNING MicrophoneStreamer:\n%s\n' % traceback.format_exc())
    finally:
      # Stop the streams.
      for audio_stream in self._audio_streams:
        audio_stream.stop_stream()
  
  # Clean up and quit
  def quit(self):
    self._log_debug('MicrophoneStreamer quitting')
    # Stop the streams.
    for audio_stream in self._audio_streams:
      audio_stream.stop_stream()
      audio_stream.close()
    # Stop PyAudio.
    self._pyaudio.terminate()
    # Call the parent method.
    SensorStreamer.quit(self)

#####################
###### TESTING ######
#####################
if __name__ == '__main__':
  print_status = True
  print_debug = True
  duration_s = 10
  
  import h5py
  
  # Create a microphone streamer.
  microphone_streamer = MicrophoneStreamer(print_status=print_status, print_debug=print_debug,
                                           device_names_withAudioKeywords=
                                            {
                                              'internal': 'Realtek',
                                              'overhead': '(USB audio CODEC)',
                                              'sink'    : '(USB PnP Audio Device)',
                                            })
  
  # Connect and run.
  microphone_streamer.connect()
  microphone_streamer.run()
  time_start_s = time.time()
  time.sleep(duration_s)
  microphone_streamer.stop()
  time_stop_s = time.time()
  duration_s = time_stop_s - time_start_s
  # Print some sample rates.
  print('Stream duration [s]: ', duration_s)
  for device_name in microphone_streamer.get_device_names():
    num_timesteps = microphone_streamer.get_num_timesteps(device_name, 'chunked_data')
    num_frames = num_timesteps*microphone_streamer._chunk_size
    print('  %s: Nchunks=%d, chunkRate=%0.2f | Nframes=%d, frameRate=%0.2f' %
          (device_name,
           num_timesteps, (num_timesteps/duration_s),
           num_frames, (num_frames/duration_s),
           ))
  
  # Save a wav file.
  for device_name in microphone_streamer.get_device_names():
    filepath = 'output_%s.wav' % device_name
    wf = wave.open(filepath, 'wb')
    wf.setnchannels(microphone_streamer._num_channels)
    wf.setsampwidth(microphone_streamer._pyaudio.get_sample_size(microphone_streamer._format))
    wf.setframerate(microphone_streamer._audio_sampling_rate_hz)
    data = microphone_streamer.get_data(device_name, 'chunked_data')['data']
    wf.writeframes(b''.join([bytearray(x) for x in data]))
    wf.close()

    wf = wave.open(filepath, 'rb')
    frames = wf.getnframes()
    rate = wf.getframerate()
    duration = frames / float(rate)
    print('from written WAV file: %d frames, %0.2fHz, duration %gs' % (frames, rate, duration))

  microphone_streamer.quit()












