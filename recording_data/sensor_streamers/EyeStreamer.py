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
import numpy as np
import msgpack
from utils.zmq_tools import *

from collections import OrderedDict
import time
import traceback
import os
import glob
import shutil

from utils.print_utils import *
from utils.time_utils import *
from utils.dict_utils import *

################################################
################################################
# A class to interface with the Pupil Labs eye tracker.
# Will stream gaze data in the video and world frames.
# Will stream world video and eye video.
################################################
################################################
class EyeStreamer(SensorStreamer):

  ########################
  ###### INITIALIZE ######
  ########################

  def __init__(self, streams_info=None,
                log_player_options=None, visualization_options=None,
                stream_video_world=True, stream_video_worldGaze=True, stream_video_eye=True,
                print_status=True, print_debug=False, log_history_filepath=None):
    SensorStreamer.__init__(self, streams_info,
                              log_player_options=log_player_options,
                              visualization_options=visualization_options,
                              print_status=print_status, print_debug=print_debug,
                              log_history_filepath=log_history_filepath)

    self._log_source_tag = 'eye'
    
    # Run this streamer on the main process, since otherwise it seems like
    #  RAM usage continues to grow even if the videos are periodically saved
    #  to disk and the data buffers are correspondingly cleared.
    self._always_run_in_main_process = True
    
    self._stream_video_world = stream_video_world
    self._stream_video_worldGaze = stream_video_worldGaze
    self._stream_video_eye = stream_video_eye
    self._pupil_capture_ip = 'localhost'
    self._pupil_capture_port = 50020
    self._video_image_format = 'bgr' # jpeg or bgr
    self._gaze_estimate_stale_s = 0.2 # how long before a gaze estimate is considered stale (changes color in the world-gaze video)

    # Update configuration based on existing data logs if desired.
    if self._replaying_data_logs:
      videos_info = self.get_videos_info_from_log_dir()
      self._stream_video_world = 'eye-tracking-video-world' in videos_info
      self._stream_video_worldGaze = 'eye-tracking-video-worldGaze' in videos_info
      self._stream_video_eye = 'eye-tracking-video-eye' in videos_info

  # Connect to the data streams, detect video frame rates, and detect available data type (e.g. 2D vs 3D).
  def _connect(self, timeout_s=10):
    # Most of the startup code adapted from https://docs.pupil-labs.com/developer/core/network-api/#communicating-with-pupil-service

    # Connect to the Pupil Capture socket.
    self._zmq_context = zmq.Context()
    self._zmq_requester = self._zmq_context.socket(zmq.REQ)
    self._zmq_requester.RCVTIMEO = 2000 # receive timeout in milliseconds
    self._zmq_requester.connect('tcp://%s:%d' % (self._pupil_capture_ip, self._pupil_capture_port))
    # Get the port that will be used for data.
    self._zmq_requester.send_string('SUB_PORT')
    try:
      self._ipc_sub_port = self._zmq_requester.recv_string()
    except:
      self._log_error('ERROR: Eye tracking could not connect; is Pupil Capture running and streaming over the network at %s:%d?' % (self._pupil_capture_ip, self._pupil_capture_port))
      return False
    
    # Sync the Pupil Core clock with the system clock.
    self._sync_pupil_time()
    
    # Subscribe to the desired topics.
    topics = ['notify.', 'gaze.3d.'] # 'logging.' # Note that the logging.debug topic will generate a message whenever the pupil time is requested, leading to ~3kHz messages since we request the time whenever a message is received
    if self._stream_video_world or self._stream_video_worldGaze:
      topics.append('frame.world')
    if self._stream_video_eye:
      topics.append('frame.eye.0')
    self._receiver = Msg_Receiver(self._zmq_context,
                           'tcp://%s:%s' % (self._pupil_capture_ip, self._ipc_sub_port),
                           topics=topics)
    self._log_debug('Subscribed to eye tracking topics')

    # # Start gaze tracking.
    # if self._print_debug: print('Starting pupil services')
    # self._send_pupil_request_dict({'subject': 'eye_process.should_start.0', 'eye_id': 0, 'args': {}})
    # # Activate frame publishing.
    # if self._stream_video_world or self._stream_video_worldGaze:
    #   self._send_pupil_request_dict({'subject': 'stop_plugin',  'name': 'Frame_Publisher', 'args': {}})
    #   self._send_pupil_request_dict({'subject': 'start_plugin', 'name': 'Frame_Publisher', 'args': {'format': self._video_image_format}})

    # Get some sample data to determine what gaze/pupil streams are present.
    # Will validate that all expected streams are present, and will also
    #  determine whether 2D or 3D processing is being used.
    self._log_status('Waiting for initial eye tracking data to determine streams')
    # Temporarily set self._stream_video_world to True if we want self._stream_video_worldGaze.
    #   Gaze data is not stored yet, so video_worldGaze won't be created yet.
    #   So instead, we can at least check that the world is streamed.
    stream_video_world_original = self._stream_video_world
    if self._stream_video_worldGaze:
      self._stream_video_world = True
    gaze_data = None
    pupil_data = None
    video_world_data = None
    video_eye_data = None
    wait_start_time_s = time.time()
    wait_timeout_s = 5
    while (gaze_data is None
            or pupil_data is None
            or (video_world_data is None and self._stream_video_world)
            or (video_world_data is None and self._stream_video_worldGaze) # see above note - gaze data is not stored yet, so video_worldGaze won't be created yet, but can check if the world is streamed at least
            or (video_eye_data is None and self._stream_video_eye)) \
          and (time.time() - wait_start_time_s < wait_timeout_s):
      time_s, data = self._process_pupil_data()
      gaze_data = gaze_data or data['gaze']
      pupil_data = pupil_data or data['pupil']
      video_world_data = video_world_data or data['video-world']
      video_eye_data = video_eye_data or data['video-eye']
    if (time.time() - wait_start_time_s >= wait_timeout_s):
      msg = 'ERROR: Eye tracking did not detect all expected streams as active'
      msg+= '\n Gaze  data  is streaming? %s' % (gaze_data is not None)
      msg+= '\n Pupil data  is streaming? %s' % (pupil_data is not None)
      msg+= '\n Video world is streaming? %s' % (video_world_data is not None)
      msg+= '\n Video eye   is streaming? %s' % (video_eye_data is not None)
      self._log_error(msg)
      raise AssertionError(msg)

    # Estimate the video frame rates
    fps_video_world = None
    fps_video_eye = None
    def get_fps(data_key, duration_s=0.1):
      self._log_status('Estimating the eye-tracking frame rate for %s... ' % data_key, end='')
      # Wait for a new data entry, so we start timing close to a frame boundary.
      data = {data_key: None}
      while data[data_key] is not None:
        time_s, data = self._process_pupil_data()
      # Receive/process messages for the desired duration.
      time_start_s = time.time()
      frame_count = 0
      while time.time() - time_start_s < duration_s:
        time_s, data = self._process_pupil_data()
        if data[data_key] is not None:
          frame_count = frame_count+1
      # Since we both started and ended timing right after a sample, no need to do (frame_count-1).
      frame_rate = frame_count/(time.time() - time_start_s)
      self._log_status('estimated frame rate as %0.2f for %s' % (frame_rate, data_key))
      return frame_rate
    if self._stream_video_world or self._stream_video_worldGaze:
      fps_video_world = get_fps('video-world')
    if self._stream_video_eye:
      fps_video_eye = get_fps('video-eye')

    # Restore the desired stream world setting, now that world-gaze data has been obtained if needed.
    self._stream_video_world = stream_video_world_original
    
    # Define data notes that will be associated with streams created below.
    self._define_data_notes()
    
    # Create a stream for the Pupil Core time, to help evaluate drift and offsets.
    # Note that core time is included with each other stream as well,
    #  but include a dedicated one too just in case there are delays in sending
    #  the other data payloads.
    self.add_stream(device_name='eye-tracking-time', stream_name='pupilCore_time_s',
                    data_type='float64', sample_size=[1],
                    sampling_rate_hz=None, extra_data_info=None,
                    data_notes=self._data_notes['eye-tracking-time']['pupilCore_time_s'])
    # Create streams for gaze data.
    for (stream_name, data) in gaze_data.items():
      sample_size = np.array(data).shape
      if len(sample_size) == 0: # it was probably a scalar
        sample_size = 1
      self.add_stream(device_name='eye-tracking-gaze', stream_name=stream_name,
                        data_type='float64', sample_size=sample_size,
                        sampling_rate_hz=None, extra_data_info=None,
                        data_notes=self._data_notes['eye-tracking-gaze'][stream_name])
    # Create streams for pupil data.
    for (stream_name, data) in pupil_data.items():
      sample_size = np.array(data).shape
      if len(sample_size) == 0: # it was probably a scalar
        sample_size = 1
      self.add_stream(device_name='eye-tracking-pupil', stream_name=stream_name,
                        data_type='float64', sample_size=sample_size,
                        sampling_rate_hz=None, extra_data_info=None,
                        data_notes=self._data_notes['eye-tracking-pupil'][stream_name])
    # Create streams for video data.
    if self._stream_video_world:
      self.add_stream(device_name='eye-tracking-video-world', stream_name='frame_timestamp',
                        data_type='float64', sample_size=(1),
                        sampling_rate_hz=fps_video_world, extra_data_info=None,
                        data_notes=self._data_notes['eye-tracking-video-world']['frame_timestamp'])
      self.add_stream(device_name='eye-tracking-video-world', stream_name='frame',
                        data_type='uint8', sample_size=(video_world_data['frame'].shape),
                        sampling_rate_hz=fps_video_world, extra_data_info=None,
                        data_notes=self._data_notes['eye-tracking-video-world']['frame'],
                        is_video=True)
    if self._stream_video_worldGaze:
      self.add_stream(device_name='eye-tracking-video-worldGaze', stream_name='frame_timestamp',
                        data_type='float64', sample_size=(1),
                        sampling_rate_hz=fps_video_world, extra_data_info=None,
                        data_notes=self._data_notes['eye-tracking-video-worldGaze']['frame_timestamp'])
      self.add_stream(device_name='eye-tracking-video-worldGaze', stream_name='frame',
                        data_type='uint8', sample_size=(video_world_data['frame'].shape),
                        sampling_rate_hz=fps_video_world, extra_data_info=None,
                        data_notes=self._data_notes['eye-tracking-video-worldGaze']['frame'],
                        is_video=True)
    if self._stream_video_eye:
      self.add_stream(device_name='eye-tracking-video-eye', stream_name='frame_timestamp',
                        data_type='float64', sample_size=(1),
                        sampling_rate_hz=fps_video_eye, extra_data_info=None,
                        data_notes=self._data_notes['eye-tracking-video-eye']['frame_timestamp'])
      self.add_stream(device_name='eye-tracking-video-eye', stream_name='frame',
                        data_type='uint8', sample_size=(video_eye_data['frame'].shape),
                        sampling_rate_hz=fps_video_eye, extra_data_info=None,
                        data_notes=self._data_notes['eye-tracking-video-eye']['frame'],
                        is_video=True)

    self._log_status('Started eye tracking streamer')
    return True


  ##############################
  ###### SENSOR INTERFACE ######
  ##############################

  # A helper to clear the Pupil socket receive buffer.
  def _flush_pupilCapture_input_buffer(self):
    flush_completed = False
    while not flush_completed:
      try:
        self._zmq_requester.recv(flags=zmq.NOBLOCK)
        flush_completed = False
      except:
        flush_completed = True

  # A helper method to send data to the pupil system.
  # Payload can be a dict or a simple string.
  # Strings will be sent as-is, while a dict will be sent after a topic message.
  # Returns the response message received.
  def _send_to_pupilCapture(self, payload, topic=None):
    # Try to receive any outstanding messages, since sending
    #  will fail if there are any waiting.
    self._flush_pupilCapture_input_buffer()
    # Send the desired data as a dict or string.
    if isinstance(payload, dict):
      # Send the topic, using a default if needed.
      if topic is None:
        topic = 'notify.%s' % payload['subject']
      # Pack and send the payload.
      payload = msgpack.dumps(payload)
      self._zmq_requester.send_string(topic, flags=zmq.SNDMORE)
      self._zmq_requester.send(payload)
    else:
      # Send the topic if there is one.
      if topic is not None:
        self._zmq_requester.send_string(topic, flags=zmq.SNDMORE)
      # Send the payload as a string.
      self._zmq_requester.send_string(payload)
    # Receive the response.
    return self._zmq_requester.recv_string()

  # Receive data and return a parsed dictionary.
  # The data dict will have keys 'gaze', 'pupil', 'video-world', 'video-worldGaze', and 'video-eye'
  #  where each will map to a dict or to None if it was not applicable.
  # The dict keys correspond to device names after the 'eye-tracking-' prefix.
  #   Each sub-dict has keys that are stream names.
  def _process_pupil_data(self):
    topic, payload = self._receiver.recv()
    time_s = time.time()
    pupilCore_time_s = self._get_pupil_time()

    gaze_items = None
    pupil_items = None
    video_world_items = None
    video_worldGaze_items = None
    video_eye_items = None
    time_items = [
      ('pupilCore_time_s', pupilCore_time_s)
    ]

    # self._log_debug('Received eye-tracking data for topic %s:' % (topic))

    # Process gaze/pupil data
    if topic in ['gaze.2d.0.', 'gaze.3d.0.']:
      pupil_data = payload['base_data'][0] # pupil detection on which the gaze detection was based (just use the first one for now if there were multiple)
      # Record data common to both 2D and 3D formats
      gaze_items = [
        ('timestamp'  , payload['timestamp']),  # seconds from an arbitrary reference time, but should be synced with the video timestamps
        ('position'   , payload['norm_pos']),   # normalized units [0-1]
        ('confidence' , payload['confidence']), # gaze confidence [0-1]
      ]
      pupil_items = [
        ('timestamp'  , pupil_data['timestamp']),  # seconds from an arbitrary reference time, but should be synced with the video timestamps
        ('position'   , pupil_data['norm_pos']),   # normalized units [0-1]
        ('confidence' , pupil_data['confidence']), # [0-1]
        ('diameter'   , pupil_data['diameter']),   # 2D image space, unit: pixel
      ]
      # Add extra data available for 3D formats
      if topic == 'gaze.3d.0.':
        gaze_items.extend([
          ('normal_3d' , payload['gaze_normal_3d']),    # x,y,z
          ('point_3d'  , payload['gaze_point_3d']),     # x,y,z
          ('eye_center_3d' , payload['eye_center_3d']), # x,y,z
        ])
        pupil_items.extend([
          ('polar_theta' , pupil_data['theta']),
          ('polar_phi'   , pupil_data['phi']),
          ('circle3d_radius' , pupil_data['circle_3d']['radius']), # mm in 3D space
          ('circle3d_center' , pupil_data['circle_3d']['center']), # mm in 3D space
          ('circle3d_normal' , pupil_data['circle_3d']['normal']), # mm in 3D space
          ('diameter3d'      , pupil_data['diameter_3d']), # mm in 3D space
          ('sphere_center' , pupil_data['sphere']['center']), # mm in 3D space
          ('sphere_radius' , pupil_data['sphere']['radius']), # mm in 3D space
          ('projected_sphere_center' , pupil_data['projected_sphere']['center']), # pixels in image space
          ('projected_sphere_axes'   , pupil_data['projected_sphere']['axes']),   # pixels in image space
          ('projected_sphere_angle'  , pupil_data['projected_sphere']['angle']),
        ])
      # Add extra data available for 2D formats
      else:
        pupil_items.extend([
          ('ellipse_center'   , pupil_data['ellipse']['center']), # pixels, in image space
          ('ellipse_axes'     , pupil_data['ellipse']['axes']),   # pixels, in image space
          ('ellipse_angle_deg', pupil_data['ellipse']['angle']),  # degrees
        ])

    # Process world video data
    elif topic == 'frame.world':
      frame_timestamp = float(payload['timestamp'])
      img_data = np.frombuffer(payload['__raw_data__'][0], dtype=np.uint8)
      if self._video_image_format == 'bgr':
        img = img_data.reshape(payload['height'], payload['width'], 3)
      else:
        img_data = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        img = img_data.reshape(payload['height'], payload['width'], 3)
      # cv2.imshow('world_frame', img)
      # cv2.waitKey(1) # waits until a key is pressed
      # print(time.time() - frame_timestamp)
      if self._stream_video_world: # we might be here because we want 'worldGaze' and not 'world'
        video_world_items = [
          ('frame_timestamp', frame_timestamp),
          ('frame', img),
        ]

      # Synthesize a stream with the gaze superimposed on the world video.
      try:
        world_img = img
        gaze_norm_pos = self._data['eye-tracking-gaze']['position']['data'][-1]
        gaze_timestamp = self._data['eye-tracking-gaze']['timestamp']['data'][-1]
        world_gaze_time_diff_s = frame_timestamp - gaze_timestamp
        gaze_radius = 10
        gaze_color_outer = (255, 255, 255) # BGR format
        if abs(world_gaze_time_diff_s) < self._gaze_estimate_stale_s: # check if the gaze prediction is recent
          gaze_color_inner = (0, 255, 0) # BGR format
        else: # gaze prediction is stale
          gaze_color_inner = (0, 0, 0) # BGR format
        gaze_norm_pos = np.array(gaze_norm_pos)
        world_with_gaze = world_img.copy()
        gaze_norm_pos[1] = 1 - gaze_norm_pos[1]
        gaze_norm_pos = tuple((gaze_norm_pos * [world_with_gaze.shape[1], world_with_gaze.shape[0]]).astype(int))
        cv2.circle(world_with_gaze, gaze_norm_pos, gaze_radius, gaze_color_outer, -1, lineType=cv2.LINE_AA)
        cv2.circle(world_with_gaze, gaze_norm_pos, round(gaze_radius*0.7), gaze_color_inner, -1, lineType=cv2.LINE_AA)
        # cv2.imshow('world_frame_with_gaze', world_with_gaze)
        # cv2.waitKey(10) # waits until a key is pressed
        video_worldGaze_items = [
          ('frame_timestamp', frame_timestamp),
          ('frame', world_with_gaze),
        ]
      except KeyError: # Streams haven't been configured yet
        pass
      except IndexError: # Data hasn't been received yet
        pass

    # Process eye video data
    elif topic == 'frame.eye.0':
      frame_timestamp = float(payload['timestamp'])
      img_data = np.frombuffer(payload['__raw_data__'][0], dtype=np.uint8)
      if self._video_image_format == 'bgr':
        img = img_data.reshape(payload['height'], payload['width'], 3)
      else:
        img_data = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        img = img_data.reshape(payload['height'], payload['width'], 3)
      # cv2.imshow('eye_frame', img)
      # cv2.waitKey(10) # waits until a key is pressed
      video_eye_items = [
        ('frame_timestamp', frame_timestamp),
        ('frame', img),
      ]

    # Create a data dictionary.
    # The keys should correspond to device names after the 'eye-tracking-' prefix.
    data = OrderedDict([
      ('gaze',  OrderedDict(gaze_items)  if gaze_items  is not None else None),
      ('pupil', OrderedDict(pupil_items) if pupil_items is not None else None),
      ('video-world', OrderedDict(video_world_items) if video_world_items is not None else None),
      ('video-worldGaze', OrderedDict(video_worldGaze_items) if video_worldGaze_items is not None else None),
      ('video-eye', OrderedDict(video_eye_items) if video_eye_items is not None else None),
      ('time', OrderedDict(time_items) if time_items is not None else None),
    ])
    # self._log_debug(get_dict_str(data))
    
    return time_s, data

  # Append processed data to the sensor streams.
  def _append_pupil_data(self, time_s, processed_data):
    for (device_name_key, streams_data) in processed_data.items():
      device_name = 'eye-tracking-%s' % device_name_key
      if streams_data is not None:
        for (stream_name, data) in streams_data.items():
          self.append_data(device_name, stream_name, time_s, data)
  
  # Get the time of the Pupil Core clock.
  # Data exported from the Pupil Capture software uses timestamps
  #  that are relative to a random epoch time.
  def _get_pupil_time(self):
    pupil_time_str = self._send_to_pupilCapture('t')
    return float(pupil_time_str)

  # Set the time of the Pupil Core clock to the system time.
  def _sync_pupil_time(self):
    self._log_status('Syncing the Pupil Core clock with the system clock')
    # Note that the same number of decimals will always be used,
    #  so the length of the message is always the same
    #  (this can help make delay estimates more accurate).
    def set_pupil_time(time_s):
      self._send_to_pupilCapture('T %0.8f' % time_s)
    # Estimate the network delay when sending the set-time command.
    num_samples = 100
    transmit_delays_s = []
    for i in range(num_samples):
      local_time_before = time.time()
      set_pupil_time(time.time())
      local_time_after = time.time()
      # Assume symmetric delays.
      transmit_delays_s.append((local_time_after - local_time_before)/2.0)
    # Set the time!
    transmit_delay_s = np.mean(transmit_delays_s)
    set_pupil_time(time.time() + transmit_delay_s)
    # self._log_debug('Estimated Pupil Core set clock transmit delay [ms]: mean %0.3f | std %0.3f | min %0.3f | max %0.3f' % \
    #                 (np.mean(transmit_delay_s)*1000.0, np.std(transmit_delay_s)*1000.0,
    #                  np.min(transmit_delay_s)*1000.0, np.max(transmit_delay_s)*1000.0))
    # Check that the sync was successful.
    clock_offset_ms = self._measure_pupil_clock_offset_s()/1000.0
    if abs(clock_offset_ms) > 5:
      self._log_warn('WARNING: Pupil Core clock sync may not have been successful. Offset is still %0.3f ms.' % clock_offset_ms)
      return False
    return True
  
  # Measure the offset between the Pupil Core clock (relative to a random epoch)
  #  and the system clock (relative to the standard epoch).
  # See the following for more information:
  #  https://docs.pupil-labs.com/core/terminology/#timestamps
  #  https://github.com/pupil-labs/pupil-helpers/blob/6e2cd2fc28c8aa954bfba068441dfb582846f773/python/simple_realtime_time_sync.py#L119
  def _measure_pupil_clock_offset_s(self, num_samples=100):
    assert num_samples > 0, 'Measuring the Pupil Capture clock offset requires at least one sample'
    clock_offsets_s = []
    for i in range(num_samples):
      # Account for network delays by recording the local time
      #  before and after the call to fetch the pupil time
      #  and then assuming that the clock was measured at the midpoint
      #  (assume symmetric network delays).
      # Note that in practice, this delay is small when
      #  using Pupil Capture via USB (typically 0-1 ms, rarely 5-10 ms).
      local_time_before = time.time()
      pupil_time = self._get_pupil_time()
      local_time_after = time.time()
      local_time = (local_time_before + local_time_after) / 2.0
      clock_offsets_s.append(pupil_time - local_time)
    # Average multiple readings to account for variable network delays.
    self._log_debug('Estimated Pupil Core clock offset [ms]: mean %0.3f | std %0.3f | min %0.3f | max %0.3f' % \
                    (np.mean(clock_offsets_s)*1000.0, np.std(clock_offsets_s)*1000.0,
                     np.min(clock_offsets_s)*1000.0, np.max(clock_offsets_s)*1000.0))
    return np.mean(clock_offsets_s)

  #####################################
  ###### EXTERNAL DATA RECORDING ######
  #####################################

  # Whether recording via the sensor's dedicated software will require user action.
  def external_data_recording_requires_user(self):
    return False
  
  # Start the Pupil Capture software recording functionality.
  def start_external_data_recording(self, recording_dir):
    recording_dir = os.path.join(recording_dir, 'pupil_capture')
    os.makedirs(recording_dir, exist_ok=True)
    self._send_to_pupilCapture('R %s' % recording_dir)
    self._log_status('Started Pupil Capture recording to %s' % recording_dir)

  # Stop the Pupil Capture software recording funtionality.
  def stop_external_data_recording(self):
    self._send_to_pupilCapture('r')
    self._log_status('Stopped Pupil Capture recording')
  
  # Update a streamed data log with data recorded from Pupil Capture.
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
  
    self._log_status('EyeStreamer merging streamed data with Pupil Capture data')
    self._define_data_notes()
    # Will save some state that will be useful later for creating a world-gaze video.
    world_video_filepath = None
    
    # Find the Pupil Capture recording subfolder within the main external data folder.
    # Use the most recent Pupil Capture subfolder in case the folder was used more than once.
    #   Pupil Capture will always create a folder named 000, 001, etc.
    data_dir_external_original = os.path.join(data_dir_external_original, 'pupil_capture')
    data_dir_external_subdirs = next(os.walk(data_dir_external_original))[1]
    numeric_subdirs = [subdir for subdir in data_dir_external_subdirs if subdir.isdigit()]
    if len(numeric_subdirs) == 0:
      self._log_error('\n\nAborting data merge for eye tracking - no externally recorded Pupil Capture folder found in %s\n' % data_dir_external_original)
      return
    recent_subdir = sorted(numeric_subdirs)[-1]
    data_dir_external_original = os.path.join(data_dir_external_original, recent_subdir)
    
    # Move files and deal with HDF5 timestamps.
    video_device_names = [
      'eye-tracking-video-world',
      'eye-tracking-video-worldGaze',
      'eye-tracking-video-eye',
    ]
    for video_device_name in video_device_names:
      # Move the streamed video to the archive folder.
      #  (From data_dir_streamed to data_dir_archived)
      # Also save the streamed video filename, so the Pupil recording can assume it later.
      filepaths = glob.glob(os.path.join(data_dir_streamed, '*%s_frame.*' % (video_device_name)))
      if len(filepaths) > 0:
        filepath = filepaths[0]
        streamed_video_filename = os.path.basename(filepath)
        self._log_debug(' Moving streamed video %s to %s' % (filepath, data_dir_archived))
        shutil.move(filepath, os.path.join(data_dir_archived, streamed_video_filename))
      else:
        streamed_video_filename = None
      
      # Move the Pupil Capture video to the final data folder.
      #  (from data_dir_external_original to data_dir_toUpdate)
      filepath = None
      if video_device_name == 'eye-tracking-video-world':
        filepath = os.path.join(data_dir_external_original, 'world.mp4')
      elif video_device_name == 'eye-tracking-video-eye':
        filepaths = glob.glob(os.path.join(data_dir_external_original, 'eye*.mp4'))
        if len(filepaths) > 0:
          filepath = filepaths[0]
      if isinstance(filepath, str) and os.path.exists(filepath):
        if streamed_video_filename is not None:
          filename = '%s.mp4' % os.path.splitext(streamed_video_filename)[0]
        else:
          filename = '%s_frame.mp4' % video_device_name
        self._log_debug(' Moving Pupil Capture video %s to %s' % (filepath, data_dir_toUpdate))
        shutil.move(filepath, os.path.join(data_dir_toUpdate, filename))
        if video_device_name == 'eye-tracking-video-world':
          world_video_filepath = os.path.join(data_dir_toUpdate, filename)
      
      # Move streamed timestamps, and frame data if applicable, to the archive HDF5 file.
      #  (from hdf5_file_toUpdate to hdf5_file_archived)
      device_group_metadata = {}
      if video_device_name in hdf5_file_toUpdate:
        self._log_debug(' Moving streamed timestamps and frames for %s to archived HDF5' % video_device_name)
        device_group_old = hdf5_file_toUpdate[video_device_name]
        hdf5_file_toUpdate.copy(device_group_old, hdf5_file_archived,
                                name=None, shallow=False,
                                expand_soft=True, expand_external=True, expand_refs=True,
                                without_attrs=False)
        device_group_metadata = dict(device_group_old.attrs.items())
        hdf5_file_archived[video_device_name].attrs.update(device_group_metadata)
        del hdf5_file_toUpdate[video_device_name]

      # Add the Pupil Capture timestamps to the final HDF5 file.
      #  (from files in data_dir_external_original to hdf5_file_toUpdate)
      timestamps_filepath = None
      if video_device_name in ['eye-tracking-video-world', 'eye-tracking-video-worldGaze']:
        timestamps_filepath = os.path.join(data_dir_external_original, 'world_timestamps.npy')
      elif video_device_name == 'eye-tracking-video-eye':
        for eye_index in range(2):
          timestamps_filepath = os.path.join(data_dir_external_original, 'eye%d_timestamps.npy' % eye_index)
          if os.path.exists(timestamps_filepath):
            break
      if isinstance(timestamps_filepath, str) and os.path.exists(timestamps_filepath):
        self._log_debug(' Adding recorded timestamps for %s to updated HDF5' % video_device_name)
        # Load the timestamps, and create user-friendly strings from them.
        timestamps_s = np.load(timestamps_filepath)
        timestamps_str = [get_time_str(t, '%Y-%m-%d %H:%M:%S.%f') for t in timestamps_s]
        num_timestamps = timestamps_s.shape[0]
        # Delete any existing timestamp data in the HDF5 and prepare a new group.
        if video_device_name not in hdf5_file_toUpdate:
          hdf5_file_toUpdate.create_group(video_device_name)
          hdf5_file_toUpdate[video_device_name].attrs.update(device_group_metadata)
        if 'frame_timestamp' in hdf5_file_toUpdate[video_device_name]:
          del hdf5_file_toUpdate[video_device_name]['frame_timestamp']
        hdf5_file_toUpdate[video_device_name].create_group('frame_timestamp')
        stream_group = hdf5_file_toUpdate[video_device_name]['frame_timestamp']
        # Add the new timestamps.
        # Note that for now, the Pupil Capture timestamps are assumed to
        #  be the same as the system time at which they would have been received (time_s).
        # This assumes that the Pupil Core clock was synced with the system time.
        # If further adjustment is desired, to estimate when each frame would have been
        #  received by the streaming system, then the information in
        #  the device eye-tracking-time can be used to interpolate new arrival times,
        #  since it maps from system arrival time to Pupil Core time.
        stream_group.create_dataset('data', [num_timestamps, 1], dtype='float64',
                                    data=timestamps_s)
        stream_group.create_dataset('time_s', [num_timestamps, 1], dtype='float64',
                                    data=timestamps_s)
        stream_group.create_dataset('time_str', [num_timestamps, 1], dtype='S26',
                                    data=timestamps_str)
        stream_group.attrs.update(convert_dict_values_to_str(
            self._data_notes_postprocessed[video_device_name]['frame_timestamp'],
            preserve_nested_dicts=False))

    # Create a world video with a gaze indicator overlaid.
    #  Note that timestamps for it will be added to the HDF5 file below.
    world_video_timestamps_filepath = os.path.join(data_dir_external_original, 'world_timestamps.npy')
    if (world_video_filepath is not None and os.path.exists(world_video_filepath)) \
        and os.path.exists(world_video_timestamps_filepath) \
        and 'eye-tracking-gaze' in hdf5_file_toUpdate:
      video_reader = cv2.VideoCapture(world_video_filepath)
      video_times_s = np.load(world_video_timestamps_filepath)
      gaze_positions = hdf5_file_toUpdate['eye-tracking-gaze']['position']['data']
      gaze_times_s = np.asarray(hdf5_file_toUpdate['eye-tracking-gaze']['timestamp']['data']) # use Pupil Core time instead of system time (['eye-tracking-gaze']['position']['time_s']) to align more precisely with the video frame timestamps copied above
      # Base the new filename on the streamed world video filename if there was one.
      # Note that the streamed videos were moved to the arvhive folder above.
      streamed_video_filepaths = glob.glob(os.path.join(data_dir_archived, '*eye-tracking-video-world_frame.*'))
      if len(streamed_video_filepaths) > 0:
        worldGaze_filename = streamed_video_filepaths[0].replace('eye-tracking-video-world', 'eye-tracking-video-worldGaze')
        worldGaze_filename = os.path.basename(worldGaze_filename)
      else:
        worldGaze_filename = 'eye-tracking-video-worldGaze_frame.avi'
      worldGaze_filepath = os.path.join(data_dir_toUpdate, worldGaze_filename)
      self._log_debug(' Creating world-gaze video based on %s' % world_video_filepath)
      self._log_debug(' Will output synthesized video to   %s' % worldGaze_filepath)
      # Extract information about the video input and create a video writer for the output.
      success, video_frame = video_reader.read()
      video_reader.set(cv2.CAP_PROP_POS_FRAMES, 0) # go back to the beginning
      frame_height = video_frame.shape[0]
      frame_width = video_frame.shape[1]
      data_type = str(video_frame.dtype)
      sampling_rate_hz = video_reader.get(cv2.CAP_PROP_FPS)
      frame_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
      fourcc = 'MJPG'
      video_writer = cv2.VideoWriter(worldGaze_filepath,
                                     cv2.VideoWriter_fourcc(*fourcc),
                                     sampling_rate_hz, (frame_width, frame_height))
      # Loop through each input frame and overlay the gaze indicator.
      previous_video_frame = None
      previous_video_frame_noGaze = None
      for i in range(frame_count):
        # Get the next video frame, and the gaze estimate closest to it in time.
        try:
          success, video_frame = video_reader.read()
          previous_video_frame_noGaze = video_frame
          video_time_s = video_times_s[i]
          gaze_index = (np.abs(gaze_times_s - video_time_s)).argmin()
          gaze_time_s = gaze_times_s[gaze_index][0]
          gaze_position = gaze_positions[gaze_index, :]
          # Draw the gaze indicator on the frame.
          world_gaze_time_diff_s = video_time_s - gaze_time_s
          gaze_radius = 10
          gaze_color_outer = (255, 255, 255) # BGR format
          if abs(world_gaze_time_diff_s) < self._gaze_estimate_stale_s: # check if the gaze prediction is recent
            gaze_color_inner = (0, 255, 0) # BGR format
          else: # gaze prediction is stale
            gaze_color_inner = (0, 0, 0) # BGR format
          gaze_position[1] = 1 - gaze_position[1]
          gaze_position = tuple((gaze_position * [video_frame.shape[1], video_frame.shape[0]]).astype(int))
          cv2.circle(video_frame, gaze_position, gaze_radius, gaze_color_outer, -1, lineType=cv2.LINE_AA)
          cv2.circle(video_frame, gaze_position, round(gaze_radius*0.7), gaze_color_inner, -1, lineType=cv2.LINE_AA)
          # Write the frame.
          video_writer.write(video_frame)
          previous_video_frame = video_frame
        except:
          # Add a dummy frame to stay aligned with the frame timestamps.
          self._log_debug('  Error processing frame %6d/%d. Copying the previous frame instead.' % ((i+1), frame_count))
          if previous_video_frame_noGaze is not None:
            video_frame = previous_video_frame_noGaze
          else:
            video_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
          video_writer.write(video_frame)
        # Print some status updates.
        if self._print_debug and (((i+1) % int(frame_count/10)) == 0 or (i+1) == frame_count):
          self._log_debug('  Processed %6d/%d frames (%0.1f%%)' % ((i+1), frame_count, 100*(i+1)/frame_count))
      video_writer.release()
      video_reader.release()

  #####################
  ###### RUNNING ######
  #####################

  # Loop until self._running is False
  def _run(self):
    try:
      while self._running:
        time_s, data = self._process_pupil_data()
        self._append_pupil_data(time_s, data)
    except KeyboardInterrupt: # The program was likely terminated
      pass
    except:
      self._log_error('\n\n***ERROR RUNNING EyeStreamer:\n%s\n' % traceback.format_exc())
    finally:
      pass

  # Clean up and quit
  def quit(self):
    self._log_debug('PupilStreamer quitting')
    SensorStreamer.quit(self)


  ###########################
  ###### VISUALIZATION ######
  ###########################

  # Specify how the streams should be visualized.
  # visualization_options can have entries for 'video-worldGaze', 'video-eye', and 'video-world'.
  def get_default_visualization_options(self, visualization_options=None):
    # Specify default options.
    processed_options = {
      'eye-tracking-video-worldGaze': {'frame': {'class': VideoVisualizer}},
      'eye-tracking-video-world':     {'frame': {'class': None}},
      'eye-tracking-video-eye':       {'frame': {'class': None}},
    }
    # Override with any provided options.
    if isinstance(visualization_options, dict):
      if 'video-worldGaze' in visualization_options:
        for (k, v) in visualization_options['video-worldGaze'].items():
          processed_options['eye-tracking-video-worldGaze'][k] = v
      if 'video-world' in visualization_options:
        for (k, v) in visualization_options['video-world'].items():
          processed_options['eye-tracking-video-world'][k] = v
      if 'video-eye' in visualization_options:
        for (k, v) in visualization_options['video-eye'].items():
          processed_options['eye-tracking-video-eye'][k] = v

    # Add default options for all other devices/streams.
    for (device_name, device_info) in self._streams_info.items():
      processed_options.setdefault(device_name, {})
      for (stream_name, stream_info) in device_info.items():
        processed_options[device_name].setdefault(stream_name, {'class': None})

    return processed_options

  #########################################
  ###### REPLAYING EXISTING DATA LOG ######
  #########################################

  # A method to determine whether videos in the
  #  log directory correspond to video streams from this sensor.
  # Returns a dict with structure videos_info[device_name][stream_name] = video_info
  #  where video_info has keys 'video_filepath', 'time_s_stream_device_name', and 'time_s_stream_name'
  #  that indicate the video filepath and the HDF5 stream that contains frame timestamps.
  def get_videos_info_from_log_dir(self):
    videos_info = {}
    # The expected device names should match those created in _connect().
    video_device_names = [
      'eye-tracking-video-world',
      'eye-tracking-video-worldGaze',
      'eye-tracking-video-eye'
      ]
    # Check all video files in the log directory.
    for file in os.listdir(self._log_player_options['log_dir']):
      if file.endswith('.avi') or file.endswith('.mp4'):
        video_filepath = os.path.join(self._log_player_options['log_dir'], file)
        # Check if this video is for one from this class.
        for video_device_name in video_device_names:
          if '%s_frame' % video_device_name in file:
            device_name = video_device_name
            stream_name = 'frame'
            video_info = {}
            video_info['video_filepath'] = video_filepath
            video_info['time_s_stream_device_name'] = device_name
            video_info['time_s_stream_name'] = 'frame_timestamp'
            videos_info[device_name] = {}
            videos_info[device_name][stream_name] = video_info
    return videos_info

  #####################################
  ###### DATA NOTES AND HEADINGS ######
  #####################################

  def _define_data_notes(self):
    self._data_notes = {}
    self._data_notes.setdefault('eye-tracking-gaze', {})
    self._data_notes.setdefault('eye-tracking-pupil', {})
    self._data_notes.setdefault('eye-tracking-time', {})
    self._data_notes.setdefault('eye-tracking-video-eye', {})
    self._data_notes.setdefault('eye-tracking-video-world', {})
    self._data_notes.setdefault('eye-tracking-video-worldGaze', {})
    
    # Gaze data
    self._data_notes['eye-tracking-gaze']['confidence'] = OrderedDict([
      ('Range', '[0, 1]'),
      ('Description', 'Confidence of the gaze detection'),
      ('PupilCapture key', 'gaze.Xd.0 > confidence'),
    ])
    self._data_notes['eye-tracking-gaze']['eye_center_3d'] = OrderedDict([
      ('Units', 'mm'),
      ('Notes', 'Maps pupil positions into the world camera coordinate system'),
      (SensorStreamer.metadata_data_headings_key, ['x','y','z']),
      ('PupilCapture key', 'gaze.Xd.0 > eye_center_3d'),
    ])
    self._data_notes['eye-tracking-gaze']['normal_3d'] = OrderedDict([
      ('Units', 'mm'),
      ('Notes', 'Maps pupil positions into the world camera coordinate system'),
      (SensorStreamer.metadata_data_headings_key, ['x','y','z']),
      ('PupilCapture key', 'gaze.3d.0 > gaze_normal_3d'),
    ])
    self._data_notes['eye-tracking-gaze']['point_3d'] = OrderedDict([
      ('Units', 'mm'),
      ('Notes', 'Maps pupil positions into the world camera coordinate system'),
      (SensorStreamer.metadata_data_headings_key, ['x','y','z']),
      ('PupilCapture key', 'gaze.3d.0 > gaze_point_3d'),
    ])
    self._data_notes['eye-tracking-gaze']['position'] = OrderedDict([
      ('Description', 'The normalized gaze position in image space, corresponding to the world camera image'),
      ('Units', 'normalized between [0, 1]'),
      ('Origin', 'bottom left'),
      (SensorStreamer.metadata_data_headings_key, ['x','y']),
      ('PupilCapture key', 'gaze.Xd.0 > norm_pos'),
    ])
    self._data_notes['eye-tracking-gaze']['timestamp'] = OrderedDict([
      ('Description', 'The timestamp recorded by the Pupil Capture software, '
                      'which should be more precise than the system time when the data was received (the time_s field).  '
                      'Note that Pupil Core time was synchronized with system time at the start of recording, accounting for communication delays.'),
      ('PupilCapture key', 'gaze.Xd.0 > timestamp'),
    ])
    
    # Pupil data
    self._data_notes['eye-tracking-pupil']['confidence'] = OrderedDict([
      ('Range', '[0, 1]'),
      ('Description', 'Confidence of the pupil detection'),
      ('PupilCapture key', 'gaze.Xd.0 > base_data > confidence'),
    ])
    self._data_notes['eye-tracking-pupil']['circle3d_center'] = OrderedDict([
      ('Units', 'mm'),
      ('PupilCapture key', 'gaze.Xd.0 > base_data > circle_3d > center'),
    ])
    self._data_notes['eye-tracking-pupil']['circle3d_normal'] = OrderedDict([
      ('Units', 'mm'),
      ('PupilCapture key', 'gaze.Xd.0 > base_data > circle_3d > normal'),
    ])
    self._data_notes['eye-tracking-pupil']['circle3d_radius'] = OrderedDict([
      ('Units', 'mm'),
      ('PupilCapture key', 'gaze.Xd.0 > base_data > circle_3d > radius'),
    ])
    self._data_notes['eye-tracking-pupil']['diameter'] = OrderedDict([
      ('Units', 'pixels'),
      ('Notes', 'The estimated pupil diameter in image space, corresponding to the eye camera image'),
      ('PupilCapture key', 'gaze.Xd.0 > base_data > diameter'),
    ])
    self._data_notes['eye-tracking-pupil']['diameter3d'] = OrderedDict([
      ('Units', 'mm'),
      ('Notes', 'The estimated pupil diameter in 3D space'),
      ('PupilCapture key', 'gaze.Xd.0 > base_data > diameter_3d'),
    ])
    self._data_notes['eye-tracking-pupil']['polar_phi'] = OrderedDict([
      ('Notes', 'Pupil polar coordinate on 3D eye model. The model assumes a fixed eye ball size, so there is no radius key.'),
      ('See also', 'polar_theta is the other polar coordinate'),
      ('PupilCapture key', 'gaze.Xd.0 > base_data > phi'),
    ])
    self._data_notes['eye-tracking-pupil']['polar_theta'] = OrderedDict([
      ('Notes', 'Pupil polar coordinate on 3D eye model. The model assumes a fixed eye ball size, so there is no radius key.'),
      ('See also', 'polar_phi is the other polar coordinate'),
      ('PupilCapture key', 'gaze.Xd.0 > base_data > theta'),
    ])
    self._data_notes['eye-tracking-pupil']['position'] = OrderedDict([
      ('Description', 'The normalized pupil position in image space, corresponding to the eye camera image'),
      ('Units', 'normalized between [0, 1]'),
      ('Origin', 'bottom left'),
      (SensorStreamer.metadata_data_headings_key, ['x','y']),
      ('PupilCapture key', 'gaze.Xd.0 > base_data > norm_pos'),
    ])
    self._data_notes['eye-tracking-pupil']['projected_sphere_angle'] = OrderedDict([
      ('Description', 'Projection of the 3D eye ball sphere into image space corresponding to the eye camera image'),
      ('Units', 'degrees'),
      ('PupilCapture key', 'gaze.Xd.0 > base_data > projected_sphere > angle'),
    ])
    self._data_notes['eye-tracking-pupil']['projected_sphere_axes'] = OrderedDict([
      ('Description', 'Projection of the 3D eye ball sphere into image space corresponding to the eye camera image'),
      ('Units', 'pixels'),
      ('Origin', 'bottom left'),
      ('PupilCapture key', 'gaze.Xd.0 > base_data > projected_sphere > axes'),
    ])
    self._data_notes['eye-tracking-pupil']['projected_sphere_center'] = OrderedDict([
      ('Description', 'Projection of the 3D eye ball sphere into image space corresponding to the eye camera image'),
      ('Units', 'pixels'),
      ('Origin', 'bottom left'),
      (SensorStreamer.metadata_data_headings_key, ['x','y']),
      ('PupilCapture key', 'gaze.Xd.0 > base_data > projected_sphere > center'),
    ])
    self._data_notes['eye-tracking-pupil']['sphere_center'] = OrderedDict([
      ('Description', 'The 3D eye ball sphere'),
      ('Units', 'mm'),
      (SensorStreamer.metadata_data_headings_key, ['x','y','z']),
      ('PupilCapture key', 'gaze.Xd.0 > base_data > sphere > center'),
    ])
    self._data_notes['eye-tracking-pupil']['sphere_radius'] = OrderedDict([
      ('Description', 'The 3D eye ball sphere'),
      ('Units', 'mm'),
      ('PupilCapture key', 'gaze.Xd.0 > base_data > sphere > radius'),
    ])
    self._data_notes['eye-tracking-pupil']['timestamp'] = OrderedDict([
      ('Description', 'The timestamp recorded by the Pupil Capture software, '
                      'which should be more precise than the system time when the data was received (the time_s field).  '
                      'Note that Pupil Core time was synchronized with system time at the start of recording, accounting for communication delays.'),
      ('PupilCapture key', 'gaze.Xd.0 > base_data > timestamp'),
    ])
    
    # Time
    self._data_notes['eye-tracking-time']['pupilCore_time_s'] = OrderedDict([
      ('Description', 'The timestamp fetched from the Pupil Core service, which can be used for alignment to system time in time_s.  '
                      'As soon as system time time_s was recorded, a command was sent to Pupil Capture to get its time; '
                      'so a slight communication delay is included on the order of milliseconds.  '
                      'Note that Pupil Core time was synchronized with system time at the start of recording, accounting for communication delays.'),
    ])
    
    # Eye video
    self._data_notes['eye-tracking-video-eye']['frame_timestamp'] = OrderedDict([
      ('Description', 'The timestamp recorded by the Pupil Core service, '
                      'which should be more precise than the system time when the data was received (the time_s field).  '
                      'Note that Pupil Core time was synchronized with system time at the start of recording, accounting for communication delays.'),
    ])
    self._data_notes['eye-tracking-video-eye']['frame'] = OrderedDict([
      ('Format', 'Frames are in BGR format'),
    ])
    # World video
    self._data_notes['eye-tracking-video-world']['frame_timestamp'] = OrderedDict([
      ('Description', 'The timestamp recorded by the Pupil Core service, '
                      'which should be more precise than the system time when the data was received (the time_s field).  '
                      'Note that Pupil Core time was synchronized with system time at the start of recording, accounting for communication delays.'),
    ])
    self._data_notes['eye-tracking-video-world']['frame'] = OrderedDict([
      ('Format', 'Frames are in BGR format'),
    ])
    # World-gaze video
    self._data_notes['eye-tracking-video-worldGaze']['frame_timestamp'] = OrderedDict([
      ('Description', 'The timestamp recorded by the Pupil Core service, '
                      'which should be more precise than the system time when the data was received (the time_s field).  '
                      'Note that Pupil Core time was synchronized with system time at the start of recording, accounting for communication delays.'),
    ])
    self._data_notes['eye-tracking-video-worldGaze']['frame'] = OrderedDict([
      ('Format', 'Frames are in BGR format'),
      ('Description', 'The world video with a gaze estimate overlay.  '
                      'The estimate in eye-tracking-gaze > position was used.  '
                      'The gaze indicator is black if the gaze estimate is \'stale\','
                      'defined here as being predicted more than %gs before the video frame.' % self._gaze_estimate_stale_s),
    ])
    
    # Use the same notes for post-processed data, with slight adjustments.
    self._data_notes_postprocessed = self._data_notes.copy()
    for (device_name, device_notes) in self._data_notes_postprocessed.items():
      for (stream_name, stream_notes) in device_notes.items():
        if stream_name == 'frame_timestamp':
          stream_notes['Description'] = stream_notes['Description'].replace(' (the time_s field)', '')
          self._data_notes_postprocessed[device_name][stream_name] = stream_notes
      
    
#####################
###### TESTING ######
#####################
if __name__ == '__main__':
  print_status = True
  print_debug = True
  duration_s = 30
  test_clock_syncing = False
  test_external_recording = False
  test_external_data_merging = False
  test_streaming = True

  import h5py
  
  # Create an eye streamer.
  eye_streamer = EyeStreamer(print_status=print_status, print_debug=print_debug)
  
  if test_external_data_merging:
    # Final post-processed outputs
    data_dir_toUpdate = 'P:/MIT/Lab/Wearativity/data/2022-01-28 test eye video sizes/2022-01-28_18-59-55_notes-eye - Copy'
    hdf5_file_toUpdate = h5py.File(os.path.join(data_dir_toUpdate, '2022-01-28_19-00-01_streamLog_notes-eye.hdf5'), 'a')
    # Original streamed and external data
    data_dir_streamed = data_dir_toUpdate
    data_dir_external_original = 'C:/Users/jdelp/recordings/test_data/005 - Copy'
    # Archives for data no longer needed
    data_dir_archived = os.path.join(data_dir_toUpdate, '_archived_data_before_postprocessing')
    os.makedirs(data_dir_archived, exist_ok=True)
    hdf5_file_archived = h5py.File(os.path.join(data_dir_archived, 'archived.hdf5'), 'w')
    
    eye_streamer.merge_external_data_with_streamed_data(# Final post-processed outputs
        hdf5_file_toUpdate,
        data_dir_toUpdate,
        # Original streamed and external data
        data_dir_streamed,
        data_dir_external_original,
        # Archives for data no longer needed
        data_dir_archived,
        hdf5_file_archived)
  
  # Connect the streamer if it will be used any more.
  if test_clock_syncing or test_external_recording or test_streaming:
    eye_streamer.connect()

  # Test the clock offset/syncing.
  if test_clock_syncing:
    eye_streamer._print_status = True
    eye_streamer._print_debug = True
    eye_streamer._measure_pupil_clock_offset_s()
    eye_streamer._sync_pupil_time()
    eye_streamer._measure_pupil_clock_offset_s()
    eye_streamer._print_status = print_status
    eye_streamer._print_debug = print_debug

  # Test remote data recording.
  if test_external_recording:
    eye_streamer.start_external_data_recording('test/my_recording')
    time.sleep(3)
    eye_streamer.stop_external_data_recording()

  # Run the streamer to test data streaming.
  if test_streaming:
    eye_streamer.run()
    # eye_streamer._running = True
    # eye_streamer._run()
    time_start_s = time.time()
    time.sleep(duration_s)
    eye_streamer.stop()
    time_stop_s = time.time()
    duration_s = time_stop_s - time_start_s
    # Print some sample rates.
    print('Stream duration [s]: ', duration_s)
    for (device_name_key, stream_name) in [
          ('gaze', 'position'),
          ('video-world', 'frame'),
          ('video-eye', 'frame'),
          ('time', 'pupilCore_time_s'),
          ]:
      num_timesteps = eye_streamer.get_num_timesteps('eye-tracking-%s' % device_name_key, stream_name)
      print('Stream %s %s: N=%d, Fs=%0.2f' % (device_name_key, stream_name, num_timesteps, (num_timesteps/duration_s)))














