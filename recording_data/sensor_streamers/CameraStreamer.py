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
from visualizers.VideoVisualizer import VideoVisualizer

import cv2
from threading import Thread
import numpy as np
import time
from collections import OrderedDict
import traceback

from utils.print_utils import *

################################################
################################################
# A class for streaming videos from USB cameras.
################################################
################################################
class CameraStreamer(SensorStreamer):
  
  ########################
  ###### INITIALIZE ######
  ########################
  
  # Initialize the sensor streamer.
  # @param visualization_options Can be used to specify how data should be visualized.
  #   It should be a dictionary with the following keys:
  #     'visualize_streaming_data': Whether or not visualize any data during streaming.
  #     'update_period_s': How frequently to update the visualizations during streaming.
  #     'visualize_all_data_when_stopped': Whether to visualize a summary of data at the end of the experiment.
  #     'wait_while_visualization_windows_open': After the experiment finishes, whether to automatically close visualization windows or wait for the user to close them.
  #     'classes_to_visualize': [optional] A list of class names that should be visualized (others will be suppressed).  For example, ['TouchStreamer', 'MyoStreamer']
  #     'use_composite_video': Whether to combine visualizations from multiple streamers into a single tiled visualization.  If not, each streamer will create its own window.
  #     'composite_video_filepath': If using composite video, can specify a filepath to save it as a video.
  #     'composite_video_layout': If using composite video, can specify which streamers should be included and how to arrange them. See some of the launch files for examples.
  # @param log_player_options Can be used to replay data from an existing log instead of streaming real-time data.
  #   It should be a dictionary with the following keys:
  #     'log_dir': The directory with log data to replay (should directly contain the HDF5 file).
  #     'pause_to_replay_in_realtime': If reading from the logs is faster than real-time, can wait between reads to keep the replay in real time.
  #     'skip_timesteps_to_replay_in_realtime': If reading from the logs is slower than real-time, can skip timesteps as needed to remain in real time.
  #     'load_datasets_into_memory': Whether to load all data into memory before starting the replay, or whether to read from the HDF5 file each timestep.
  # @param print_status Whether or not to print messages with level 'status'
  # @param print_debug Whether or not to print messages with level 'debug'
  # @param log_history_filepath A filepath to save log messages if desired.
  def __init__(self,
               cameras_to_stream=None, # a dict mapping camera names to device indexes
               log_player_options=None, visualization_options=None,
               print_status=True, print_debug=False, log_history_filepath=None):
    SensorStreamer.__init__(self, streams_info=None,
                            visualization_options=visualization_options,
                            log_player_options=log_player_options,
                            print_status=print_status, print_debug=print_debug,
                            log_history_filepath=log_history_filepath)
    
    # A tag that can be used in log messages.
    self._log_source_tag = 'cameras'
    
    # Initialize general state.
    if cameras_to_stream is None:
      cameras_to_stream = {'default': 0}
    self._cameras_to_stream = cameras_to_stream
    self._captures = {}
    self._run_threads = {}
    
  #######################################
  # Connect to the sensor.
  # @param timeout_s How long to wait for the sensor to respond.
  def _connect(self, timeout_s=10):
    
    # Connect to each camera and estimate its frame rate.
    # Add devices and streams for each camera.
    for (camera_name, device_index) in self._cameras_to_stream.items():
      # Connect to the camera.
      self._captures[camera_name] = cv2.VideoCapture(device_index)
      (success, frame) = self._captures[camera_name].read()
      if not success:
        self._log_error('\n\n***ERROR CameraStreamer could not connect to camera %s at device index %d' % (camera_name, device_index))
        return False
      # Get the frame rate.
      fps = self._captures[camera_name].get(cv2.CAP_PROP_FPS)
      # Add a stream for the frames.
      self.add_stream(device_name=camera_name,
                      stream_name='frame',
                      is_video=True,
                      data_type='uint8',
                      sample_size=frame.shape, # the size of data saved for each timestep; here, we expect a 2-element vector per timestep
                      sampling_rate_hz=fps,    # the expected sampling rate for the stream
                      extra_data_info=None,    # can add extra information beyond the data and the timestamp if needed (probably not needed, but see MyoStreamer for an example if desired)
                      # Notes can add metadata about the stream,
                      #  such as an overall description, data units, how to interpret the data, etc.
                      # The SensorStreamer.metadata_data_headings_key is special, and is used to
                      #  describe the headings for each entry in a timestep's data.
                      #  For example - if the data was saved in a spreadsheet with a row per timestep, what should the column headings be.
                      data_notes=OrderedDict([
                        ('Format', 'Frames are in BGR format'),
                      ]))
      # Add a stream for the frames.
      self.add_stream(device_name=camera_name,
                      stream_name='frame_timestamp',
                      is_video=False,
                      data_type='float64',
                      sample_size=[1],         # the size of data saved for each timestep; here, we expect a 2-element vector per timestep
                      sampling_rate_hz=fps,    # the expected sampling rate for the stream
                      extra_data_info=None,    # can add extra information beyond the data and the timestamp if needed (probably not needed, but see MyoStreamer for an example if desired)
                      # Notes can add metadata about the stream,
                      #  such as an overall description, data units, how to interpret the data, etc.
                      # The SensorStreamer.metadata_data_headings_key is special, and is used to
                      #  describe the headings for each entry in a timestep's data.
                      #  For example - if the data was saved in a spreadsheet with a row per timestep, what should the column headings be.
                      data_notes=OrderedDict([
                        ('Format', 'Frames are in BGR format'),
                      ]))
    
    self._log_status('Successfully connected to the camera streamer.')
    return True
  
  #####################
  ###### RUNNING ######
  #####################
  
  # A run function that can be used for a single camera.
  # Will start a new thread for each camera, so they do not slow each other down.
  def _run_for_camera(self, camera_name):
    try:
      while self._running:
        # Read a frame from the camera.
        (success, frame) = self._captures[camera_name].read()
        # Timestamp the frame.
        time_s = time.time()
        # Store the data.
        self.append_data(camera_name, 'frame', time_s, frame)
        self.append_data(camera_name, 'frame_timestamp', time_s, time_s)
    except KeyboardInterrupt: # The program was likely terminated
      pass
    except:
      self._log_error('\n\n***ERROR RUNNING CameraStreamer for camera %s:\n%s\n' % (camera_name, traceback.format_exc()))
    finally:
      try:
        self._captures[camera_name].release()
      except:
        pass

  # Launch the per-camera threads.
  def _run(self):
    # Create and start a thread for each camera.
    for camera_name in self._cameras_to_stream:
      self._run_threads[camera_name] = Thread(target=self._run_for_camera,
                                              args=(),
                                              kwargs={'camera_name': camera_name})
      self._run_threads[camera_name].daemon = False
      self._run_threads[camera_name].start()
    # Join the threads to wait until all are done.
    for camera_name in self._cameras_to_stream:
      self._run_threads[camera_name].join()
      
  # Clean up and quit
  def quit(self):
    self._log_debug('CameraStreamer quitting')
    SensorStreamer.quit(self)
  
  ###########################
  ###### VISUALIZATION ######
  ###########################
  
  # Specify how the streams should be visualized.
  # Return a dict of the form options[device_name][stream_name] = stream_options
  #  Where stream_options is a dict with the following keys:
  #   'class': A subclass of Visualizer that should be used for the specified stream.
  #   Any other options that can be passed to the chosen class.
  def get_default_visualization_options(self, visualization_options=None):
    # Start by not visualizing any streams.
    processed_options = {}
    for (device_name, device_info) in self._streams_info.items():
      processed_options.setdefault(device_name, {})
      for (stream_name, stream_info) in device_info.items():
        processed_options[device_name].setdefault(stream_name, {'class': None})
    
    # Show frames from each camera as a video.
    for camera_name in self._cameras_to_stream:
      processed_options[camera_name]['frame'] = {'class': VideoVisualizer}
    
    # Override the above defaults with any provided options.
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
  ###### HELPERS ######
  #####################
  
# Try to discover cameras and display a frame from each one,
#  to help identify device indexes.
def discover_cameras(display_frames=True):
  device_indexes = []
  for device_index in range(0, 100):
    capture = cv2.VideoCapture(device_index)
    # Try to get a frame to check if the camera exists.
    (success, frame) = capture.read()
    if success:
      device_indexes.append(device_index)
      if display_frames:
        cv2.imshow('Device %d' % device_index, frame)
        cv2.waitKey(1)
    capture.release()
  if display_frames:
    cv2.waitKey(0)
  return device_indexes
  
  

#####################
###### TESTING ######
#####################
if __name__ == '__main__':
  # Configuration.
  duration_s = 30

  # # Discover available cameras.
  # device_indexes = discover_cameras(display_frames=True)
  # print('Available device indexes:', device_indexes)

  # Connect to the device(s).
  camera_streamer = CameraStreamer(cameras_to_stream={
                                    'camera-built-in': 0,
                                    },
                                   print_status=True, print_debug=False)
  camera_streamer.connect()

  # Run for the specified duration and periodically print the sample rate.
  print('\nRunning for %gs!' % duration_s)
  camera_streamer.run()
  start_time_s = time.time()
  try:
    while time.time() - start_time_s < duration_s:
      time.sleep(2)
      # Print the sampling rates.
      msg = ' Duration: %6.2fs' % (time.time() - start_time_s)
      for device_name in camera_streamer.get_device_names():
        stream_names = camera_streamer.get_stream_names(device_name=device_name)
        for stream_name in stream_names:
          num_timesteps = camera_streamer.get_num_timesteps(device_name, stream_name)
          msg += ' | %s-%s: %6.2f Hz (%4d Timesteps)' % \
                 (device_name, stream_name, ((num_timesteps)/(time.time() - start_time_s)), num_timesteps)
      print(msg)
  except:
    pass

  # Stop the streamer.
  camera_streamer.stop()
  print('\n'*2)
  print('='*75)
  print('Done!')
  print('\n'*2)
  
  
  
  
  











