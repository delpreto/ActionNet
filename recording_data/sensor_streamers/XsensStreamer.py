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
from visualizers.XsensSkeletonVisualizer import XsensSkeletonVisualizer

import socket
import struct
import time
import numpy as np
import traceback
import h5py
import os
import copy
# For post-processing
import pandas
from bs4 import BeautifulSoup
import glob
from collections import OrderedDict

from utils.dict_utils import *
from utils.print_utils import *
from utils.time_utils import *
from utils.angle_utils import *


################################################
################################################
# A class to interface with the Xsens motion trackers.
# A full body suit and two optional gloves are supported.
# The following data will be streamed:
#   Euler pose and orientation data
#   Quaternion orientation data
#   Joint angles
#   Center of mass dynamics
#   Device timestamps for each timestep
# For the gloves, no joint angles are currently sent from the Xsens software.
# Note that device timestamps and frame counts will be treated as any other data stream,
#  so they will be timestamped with the Python system time.
#  This should facilitate alignment with other sensors, as well as among Xsens streams.
# Note that Xsens will send data for a single timestep as multiple messages.
#  The system timestamp of the first one will be used for all of them.
################################################
################################################
class XsensStreamer(SensorStreamer):

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

    self._log_source_tag = 'xsens'
    
    # Initialize counts of segments/joints/fingers.
    # These will be updated automatically later based on initial streaming data.
    self._num_segments = None # will be set to 23 for the full body
    self._num_fingers = None  # will be set to 0 or 40 depending on whether fingers are enabled
    self._num_joints = None # will be set to 28 = 22 regular joints + 6 ergonomic joints
    # Specify message types that might be received.
    self._xsens_msg_types = {
      'pose_euler':      1,
      'pose_quaternion': 2,
      # 'character_scale': 13,
      'joint_angle':     20,
      'center_of_mass':  24,
      'time_code_str':   25,
    }
    # Initialize a record of which ones are actually active.
    # This will be automatically set later based on iniital streaming data.
    self._xsens_is_streaming = dict([(msg_type, False) for msg_type in self._xsens_msg_types.values()])

    # Specify the Xsens streaming configuration.
    self._xsens_network_protocol = 'udp'
    self._xsens_network_ip = '127.0.0.1'
    self._xsens_network_port = 9763
    self._xsens_msg_start_code = b'MXTP'
    # Note that the buffer read size must be large enough to receive a full Xsens message.
    #  The longest message is currently from the stream "Position + Orientation (Quaternion)"
    #  which has a message length of 2040 when finger data is enabled.
    self._buffer_read_size = 2048
    self._buffer_max_size = self._buffer_read_size * 16
    
    # Post-processing configuration for merging Xsens recordings with streamed data.
    self._postprocessing_time_strategy = 'constant-rate' # 'interpolate-xsens', 'constant-rate', 'interpolate-system'
    
    # Initialize state.
    self._buffer = b''
    self._socket = None
    self._xsens_sample_index = None # The current Xsens timestep being processed (each timestep will send multiple messages)
    self._xsens_message_start_time_s = None    # When an Xsens message was first received
    self._xsens_timestep_receive_time_s = None # When the first Xsens message for an Xsens timestep was received

    # Update configuration based on existing data logs if desired.
    # Look in all HDF5 files to determine which streams were active
    #  and how many segments/fingers there were.
    if self._replaying_data_logs:
      self._num_segments = 23
      for file in os.listdir(self._log_player_options['log_dir']):
        if file.endswith('.hdf5'):
          hdf5_filepath = os.path.join(self._log_player_options['log_dir'], file)
          self._log_status('%s loading HDF5 file %s' % (type(self).__name__, hdf5_filepath))
          hdf5_file = h5py.File(hdf5_filepath, 'r')
          if 'xsens-COM' in hdf5_file:
            self._xsens_is_streaming[self._xsens_msg_types['center_of_mass']] = True
          if 'xsens-joints' in hdf5_file:
            self._xsens_is_streaming[self._xsens_msg_types['joint_angle']] = True
            self._num_joints = hdf5_file['xsens-joints']['child']['data'].shape[1]
          if 'xsens-time' in hdf5_file:
            self._xsens_is_streaming[self._xsens_msg_types['time_code_str']] = True
          if 'xsens-segments' in hdf5_file:
            if 'orientation_euler_deg' in hdf5_file['xsens-segments']:
              self._xsens_is_streaming[self._xsens_msg_types['pose_euler']] = True
              num_segments = hdf5_file['xsens-segments']['orientation_euler_deg']['data'].shape[1]
              self._num_fingers = num_segments - self._num_segments
            if 'orientation_quaternion' in hdf5_file['xsens-segments']:
              self._xsens_is_streaming[self._xsens_msg_types['pose_quaternion']] = True
              num_segments = hdf5_file['xsens-segments']['orientation_quaternion']['data'].shape[1]
              self._num_fingers = num_segments - self._num_segments
          hdf5_file.close()
      if self._print_debug:
        debug_msg  = 'Got Xsens state from past data logs:\n'
        debug_msg += '  num segments: %s\n' % self._num_segments
        debug_msg += '  num fingers : %s\n' % self._num_fingers
        debug_msg += '  num joints  : %s\n' % self._num_joints
        debug_msg += '  active streams: %s\n' % get_dict_str(self._xsens_is_streaming)
        self._log_debug(debug_msg.strip())


  # Set up streams associated with an Xsens message.
  # Streams will be set up when its message type is first receieved,
  #  so that segment/joint counts are known and so only streams
  #  that are currently active in Xsens will be created.
  def _setup_streams_for_xsens_message(self, message_type):
    
    # Define headings for each stream; will populate self._headings.
    self._define_data_headings()
    self._define_data_notes()
    
    # All streams will have the Xsens sample counter and time code added.
    extra_data_info = {
      'xsens_sample_number'     : {'data_type': 'int32',   'sample_size': [1]},
      'xsens_time_since_start_s': {'data_type': 'float32', 'sample_size': [1]}
      }
      
    # Segment positions and orientations
    if message_type == self._xsens_msg_types['pose_euler'] \
        or message_type == self._xsens_msg_types['pose_quaternion']:
      self.add_stream(device_name='xsens-segments',
                      stream_name='position_cm',
                      data_type='float32',
                      sample_size=(self._num_segments + self._num_fingers, 3),
                      sampling_rate_hz=None,
                      extra_data_info=extra_data_info,
                      data_notes=self._data_notes_stream['xsens-segments']['position_cm'])
    if message_type == self._xsens_msg_types['pose_euler']:
      self.add_stream(device_name='xsens-segments',
                      stream_name='orientation_euler_deg',
                      data_type='float32',
                      sample_size=(self._num_segments + self._num_fingers, 3),
                      sampling_rate_hz=None,
                      extra_data_info=extra_data_info,
                      data_notes=self._data_notes_stream['xsens-segments']['orientation_euler_deg'])
    if message_type == self._xsens_msg_types['pose_quaternion']:
      self.add_stream(device_name='xsens-segments',
                      stream_name='orientation_quaternion',
                      data_type='float32',
                      sample_size=(self._num_segments + self._num_fingers, 4),
                      sampling_rate_hz=None,
                      extra_data_info=extra_data_info,
                      data_notes=self._data_notes_stream['xsens-segments']['orientation_quaternion'])

    # Joint angles
    if message_type == self._xsens_msg_types['joint_angle']:
      self.add_stream(device_name='xsens-joints',
                      stream_name='rotation_deg',
                      data_type='float32',
                      sample_size=(self._num_joints, 3),
                      sampling_rate_hz=None,
                      extra_data_info=extra_data_info,
                      data_notes=self._data_notes_stream['xsens-joints']['rotation_deg'])
      self.add_stream(device_name='xsens-joints',
                      stream_name='parent',
                      data_type='float32',
                      sample_size=(self._num_joints),
                      sampling_rate_hz=None,
                      extra_data_info=extra_data_info,
                      data_notes=self._data_notes_stream['xsens-joints']['parent'])
      self.add_stream(device_name='xsens-joints',
                      stream_name='child',
                      data_type='float32',
                      sample_size=(self._num_joints),
                      sampling_rate_hz=None,
                      extra_data_info=extra_data_info,
                      data_notes=self._data_notes_stream['xsens-joints']['child'])

    # Center of mass dynamics
    if message_type == self._xsens_msg_types['center_of_mass']:
      self.add_stream(device_name='xsens-CoM',
                      stream_name='position_cm',
                      data_type='float32',
                      sample_size=(3),
                      sampling_rate_hz=None,
                      extra_data_info=extra_data_info,
                      data_notes=self._data_notes_stream['xsens-CoM']['position_cm'])
      self.add_stream(device_name='xsens-CoM',
                      stream_name='velocity_cm_s',
                      data_type='float32',
                      sample_size=(3),
                      sampling_rate_hz=None,
                      extra_data_info=extra_data_info,
                      data_notes=self._data_notes_stream['xsens-CoM']['velocity_cm_s'])
      self.add_stream(device_name='xsens-CoM',
                      stream_name='acceleration_cm_ss',
                      data_type='float32',
                      sample_size=(3),
                      sampling_rate_hz=None,
                      extra_data_info=extra_data_info,
                      data_notes=self._data_notes_stream['xsens-CoM']['acceleration_cm_ss'])

    # Time codes sent from the Xsens device
    if message_type == self._xsens_msg_types['time_code_str']:
      extra_data_info_time = extra_data_info.copy()
      extra_data_info_time['device_time_utc_str']  = {'data_type': 'S12', 'sample_size': [1]}
      extra_data_info_time['device_timestamp_str'] = {'data_type': 'S26', 'sample_size': [1]}
      self.add_stream(device_name='xsens-time',
                      stream_name='device_timestamp_s',
                      data_type='float64',
                      sample_size=(1),
                      sampling_rate_hz=None,
                      extra_data_info=extra_data_info_time,
                      data_notes=self._data_notes_stream['xsens-time']['device_timestamp_s'])

  # Connect to the Xsens network streams, and determine what streams are active.
  def _connect(self, timeout_s=10):
    # Open a socket to the Xsens network stream
    if 'tcp' == self._xsens_network_protocol.lower().strip():
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.listen(1) # number of waiting connections
        self._socket_connection, socket_address = self._socket.accept()
        self._socket_connection.setblocking(False)
        self._log_debug('New connection from %s:%d' % (self._socket_connection, socket_address))
        #self._socket_connection.settimeout(0.001)
    elif 'udp' == self._xsens_network_protocol.lower().strip():
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.settimeout(5) # timeout for all socket operations, such as receiving if the Xsens network stream is inactive
    self._socket.bind((self._xsens_network_ip, self._xsens_network_port))
    self._log_status('Started Xsens %s socket with IP %s port %d' % (self._xsens_network_protocol.upper(), self._xsens_network_ip, self._xsens_network_port))

    # Run it for a bit to learn what Xsens streams are active.
    self._log_status('Running for a few seconds to learn what Xsens streams are active')
    print_status = self._print_status
    print_debug = self._print_debug
    self._print_status = False
    self._print_debug = False
    self.run()
    time.sleep(3)
    self.stop()
    self.clear_data_all()
    self._print_status = print_status
    self._print_debug = print_debug
    self._log_debug('Found the following stream states (keys are Xsens message types):')
    self._log_debug(get_dict_str(self._xsens_is_streaming))
    self._log_status('Done connecting')

    return True

  #############################
  ###### MESSAGE PARSING ######
  #############################

  # Helper to read a specified number of bytes starting at a specified index
  # and to return an updated index counter.
  def _read_bytes(self, message, starting_index, num_bytes):
    if None in [message, starting_index, num_bytes]:
      res = None
      next_index = None
    elif starting_index + num_bytes <= len(message):
      res = message[starting_index : (starting_index + num_bytes)]
      next_index = starting_index + num_bytes
    else:
      res = None
      next_index = None
    return res, next_index

  # Parse an Xsens message to extract its information and data.
  def _process_xsens_message_from_buffer(self):
    # self._log_debug('\nProcessing buffer!')
    message = self._buffer

    # Use the starting code to determine the initial index
    next_index = message.find(self._xsens_msg_start_code)
    if next_index < 0:
      self._log_debug('No starting code')
      return None
    
    # Parse the header
    start_code,       next_index = self._read_bytes(message, next_index, len(self._xsens_msg_start_code))
    message_type,     next_index = self._read_bytes(message, next_index, 2)
    sample_counter,   next_index = self._read_bytes(message, next_index, 4)
    datagram_counter, next_index = self._read_bytes(message, next_index, 1)
    num_items,        next_index = self._read_bytes(message, next_index, 1)
    time_code,        next_index = self._read_bytes(message, next_index, 4)
    char_id,          next_index = self._read_bytes(message, next_index, 1)
    num_segments,     next_index = self._read_bytes(message, next_index, 1)
    num_props,        next_index = self._read_bytes(message, next_index, 1)
    num_fingers,      next_index = self._read_bytes(message, next_index, 1)
    reserved,         next_index = self._read_bytes(message, next_index, 2)
    payload_size,     next_index = self._read_bytes(message, next_index, 2)

    # Decode (and verify) header information
    try:
      message_type = int(message_type)
      payload_size = int.from_bytes(payload_size, byteorder='big', signed=False) # number of bytes in the actual data of the message
      sample_counter = int.from_bytes(sample_counter, byteorder='big', signed=False) # message index basically
      datagram_counter = int.from_bytes(datagram_counter, byteorder='big', signed=False) # index of datagram chunk
      time_code = int.from_bytes(time_code, byteorder='big', signed=False) # ms since the start of recording
      char_id = int.from_bytes(char_id, byteorder='big', signed=False) # id of the tracker person (if multiple)
      num_items = int.from_bytes(num_items, byteorder='big', signed=False) # number of points in this message
      num_segments = int.from_bytes(num_segments, byteorder='big', signed=False) # we always have 23 body segments
      num_fingers = int.from_bytes(num_fingers, byteorder='big', signed=False) # number of finger track segments
      num_props = int.from_bytes(num_props, byteorder='big', signed=False) # number of props (swords etc)
      reserved = int.from_bytes(reserved, byteorder='big', signed=False)
      assert datagram_counter == (1 << 7), 'Not a single last message' # We did not implement datagram splitting
      assert char_id == 0, 'We only support a single person (a single character).'
    except TypeError:
      # Not all message parts were received.
      # This will be dealt with below, after checking if some of it was at least received.
      pass

    # If the message type and sample counter are known,
    #  use this time as the best-guess timestamp for the data.
    if message_type is not None and sample_counter is not None:
      if self._xsens_sample_index != sample_counter:
        # self._log_debug('Updating timestamp receive time, and setting sample counter to %d' % sample_counter)
        self._xsens_timestep_receive_time_s = self._xsens_message_start_time_s
        self._xsens_sample_index = sample_counter

    # Check that all were present by checking the last one.
    if payload_size is None:
      self._log_debug('Incomplete message')
      return None

    # Organize the metadata so far
    metadata = {
            'message_type': message_type,
            'sample_counter': sample_counter,
            'datagram_counter': datagram_counter,
            'num_items': num_items,
            'time_since_start_s': time_code/1000.0,
            'char_id': char_id,
            'num_segments': num_segments,
            'num_props': num_props,
            'num_fingers': num_fingers,
            'reserved': reserved,
            'payload_size': payload_size,
        }
    # Validate that the number of segments/joints/fingers remains the same
    if self._num_segments is not None and self._num_segments != num_segments:
      self._log_error('ERROR: The number of Xsens segments changed from %d to %d' % (self._num_segments, num_segments))
    if self._num_fingers is not None and self._num_fingers != num_fingers and message_type in [self._xsens_msg_types['pose_euler'], self._xsens_msg_types['pose_quaternion']]:
      self._log_error('ERROR: The number of Xsens fingers changed from %d to %d' % (self._num_fingers, num_fingers))
    if self._num_joints is not None and self._num_joints != num_items and message_type == self._xsens_msg_types['joint_angle']:
      self._log_error('ERROR: The number of Xsens joints changed from %d to %d' % (self._num_joints, num_items))

    # Store the number of segments/joints/fingers if needed
    self._num_segments = num_segments # note that this field is correct even if the message is not segment-based
    if message_type in [self._xsens_msg_types['pose_euler'], self._xsens_msg_types['pose_quaternion']]:
      self._num_fingers = num_fingers
    if message_type == self._xsens_msg_types['joint_angle']:
      self._num_joints = num_items

    # Read the payload, and check that it is fully present
    payload, payload_end_index = self._read_bytes(message, next_index, payload_size)
    if payload is None:
      self._log_debug('No message payload yet')
      return None

    extra_data = {
      'xsens_sample_number'     : sample_counter,
      'xsens_time_since_start_s': time_code/1000.0,
    }
    
    if self._print_debug:
      self._log_debug(get_dict_str(metadata))
      self._log_debug(get_dict_str(extra_data))
      self._log_debug(self._xsens_timestep_receive_time_s)

    # Set up the streams for this message if needed
    if not self._xsens_is_streaming[message_type]:
      self._log_debug('Setting up Xsens streams for message type %d' % message_type)
      self._setup_streams_for_xsens_message(message_type)
      self._xsens_is_streaming[message_type] = True

    # Euler and Quaternion messages are very similar,
    #   so parse them with the same code.
    # Euler-focused data contains:
    #   Segment positions (x/y/z) in cm
    #   Segment rotation (x/y/z) using a Y-Up and right-handed coordinate system
    # Quaternion-focused data contains:
    #   Segment positions (x/y/z) in cm
    #   Segment rotation (re/i/j/k) using a Z-Up and right-handed coordinate system
    # Note that the position data from the Euler stream and the Quaternion
    #   streams should be redundant.
    #   So if Euler is streaming, ignore the position from the quaternion.
    is_euler = message_type == self._xsens_msg_types['pose_euler']
    is_quaternion = message_type == self._xsens_msg_types['pose_quaternion']
    if is_euler or is_quaternion:
      if is_euler:
        num_rotation_elements = 3
        rotation_stream_name = 'orientation_euler_deg'
      else:
        num_rotation_elements = 4
        rotation_stream_name = 'orientation_quaternion'
      segment_positions_cm = np.zeros((num_segments + num_fingers, 3), dtype=np.float32)
      segment_rotations = np.zeros((num_segments + num_fingers, num_rotation_elements), dtype=np.float32)
      # Read the position and rotation of each segment
      for segment_index in range(num_segments + num_fingers):
        segment_id, next_index = self._read_bytes(message, next_index, 4)
        segment_position_cm, next_index = self._read_bytes(message, next_index, 3*4) # read x/y/z at once - each is 4 bytes
        segment_rotation, next_index = self._read_bytes(message, next_index, num_rotation_elements*4) # read x/y/z at once - each is 4 bytes

        segment_id = int.from_bytes(segment_id, byteorder='big', signed=False)
        segment_position_cm = np.array(struct.unpack('!3f', segment_position_cm), np.float32)
        segment_rotation = np.array(struct.unpack('!%df' % num_rotation_elements, segment_rotation), np.float32)
        
        # If using the Euler stream, received data is YZX so swap order to get XYZ.
        #  (Note that positions from the quaternion stream are already XYZ.)
        if is_euler:
          segment_position_cm = segment_position_cm[[2,0,1]]
          segment_rotation = segment_rotation[[2,0,1]]
        # If using the Quaternion stream, received data is in m so convert to cm.
        #  (Note that positions from the Euler stream are already in cm.)
        if is_quaternion:
          segment_position_cm = 100.0*segment_position_cm
        
        # Note that segment IDs from Xsens are 1-based,
        #  but otherwise should be usable as the matrix index.
        segment_positions_cm[segment_id-1, :] = segment_position_cm
        segment_rotations[segment_id-1, :] = segment_rotation

      # Store the data
      self.append_data('xsens-segments', rotation_stream_name,
                        self._xsens_timestep_receive_time_s, segment_rotations, extra_data=extra_data)
      if is_euler or not self._xsens_is_streaming[self._xsens_msg_types['pose_euler']]:
        self.append_data('xsens-segments', 'position_cm',
                          self._xsens_timestep_receive_time_s, segment_positions_cm, extra_data=extra_data)
      if self._print_debug:
        self._log_debug(segment_positions_cm)
        self._log_debug(segment_rotations)

    # Joint angle data contains:
    #  The parent and child segments of the joint.
    #   These are represented as a single integer: 256*segment_id + point_id
    #  Rotation around the x/y/z axes in degrees
    elif message_type == self._xsens_msg_types['joint_angle']:
      joint_parents = np.zeros((num_items), dtype=np.float32)
      joint_childs = np.zeros((num_items), dtype=np.float32)
      joint_rotations_deg = np.zeros((num_items, 3), dtype=np.float32)
      # Read the ids and rotations of each joint
      for joint_index in range(num_items):
          joint_parent, next_index = self._read_bytes(message, next_index, 4)
          joint_child, next_index = self._read_bytes(message, next_index, 4)
          joint_rotation_deg, next_index = self._read_bytes(message, next_index, 3*4) # read x/y/z at once - each is 4 bytes

          joint_parent = int.from_bytes(joint_parent, byteorder='big', signed=False)
          joint_child = int.from_bytes(joint_child, byteorder='big', signed=False)
          joint_rotation_deg = np.array(struct.unpack('!3f', joint_rotation_deg), np.float32)

          # Convert IDs from segmentID*256+localPointID to segmentID.localPointID
          joint_parent_segment = int(joint_parent/256)
          joint_parent_point = joint_parent - joint_parent_segment*256
          joint_child_segment = int(joint_child/256)
          joint_child_point = joint_child - joint_child_segment*256
          joint_parent = joint_parent_segment + joint_parent_point/1000.0
          joint_child = joint_child_segment + joint_child_point/1000.0

          # Record the joint data
          joint_parents[joint_index] = joint_parent
          joint_childs[joint_index] = joint_child
          joint_rotations_deg[joint_index, :] = joint_rotation_deg

      # Store the data
      self.append_data('xsens-joints', 'rotation_deg',
                        self._xsens_timestep_receive_time_s, joint_rotations_deg, extra_data=extra_data)
      self.append_data('xsens-joints', 'parent',
                        self._xsens_timestep_receive_time_s, joint_parents, extra_data=extra_data)
      self.append_data('xsens-joints', 'child',
                        self._xsens_timestep_receive_time_s, joint_childs, extra_data=extra_data)
      if self._print_debug:
        self._log_debug(joint_rotations_deg)
        self._log_debug(joint_parents)
        self._log_debug(joint_childs)

    # Center of mass data contains:
    #  x/y/z position in cm
    #  x/y/z velocity in cm/s
    #  x/y/z acceleration in cm/s/s
    elif message_type == self._xsens_msg_types['center_of_mass']:
      com_position_m, next_index = self._read_bytes(message, next_index, 3*4) # read x/y/z at once - each is 4 bytes
      com_velocity_m_s, next_index = self._read_bytes(message, next_index, 3*4) # read x/y/z at once - each is 4 bytes
      com_acceleration_m_ss, next_index = self._read_bytes(message, next_index, 3*4) # read x/y/z at once - each is 4 bytes

      com_position_m = np.array(struct.unpack('!3f', com_position_m), np.float32)
      com_velocity_m_s = np.array(struct.unpack('!3f', com_velocity_m_s), np.float32)
      com_acceleration_m_ss = np.array(struct.unpack('!3f', com_acceleration_m_ss), np.float32)
      
      # Store the data
      self.append_data('xsens-CoM', 'position_cm',
                        self._xsens_timestep_receive_time_s, 100.0*com_position_m, extra_data=extra_data)
      self.append_data('xsens-CoM', 'velocity_cm_s',
                        self._xsens_timestep_receive_time_s, 100.0*com_velocity_m_s, extra_data=extra_data)
      self.append_data('xsens-CoM', 'acceleration_cm_ss',
                        self._xsens_timestep_receive_time_s, 100.0*com_acceleration_m_ss, extra_data=extra_data)
      if self._print_debug:
        self._log_debug(100.0*com_position_m)
        self._log_debug(100.0*com_velocity_m_s)
        self._log_debug(100.0*com_acceleration_m_ss)

    # Time data contains:
    #  A string for the sample timestamp formatted as HH:MM:SS.mmm
    elif message_type == self._xsens_msg_types['time_code_str']:

      str_length, next_index = self._read_bytes(message, next_index, 4)
      str_length = int.from_bytes(str_length, byteorder='big', signed=True)
      assert str_length == 12, 'Unexpected number of bytes in the time code string: %d instead of 12' % str_length

      time_code_str, next_index = self._read_bytes(message, next_index, str_length)
      time_code_str = time_code_str.decode('utf-8')
      
      # The received string is a time in UTC without a date.
      # Convert this to local time with the current date, then to seconds since epoch.
      time_code_s = get_time_s_from_utc_timeNoDate_str(time_code_str, input_time_format='%H:%M:%S.%f')
      
      # Store the data
      extra_data['device_time_utc_str'] = time_code_str
      extra_data['device_timestamp_str'] = get_time_str(time_code_s, '%Y-%m-%d %H:%M:%S.%f')
      self.append_data('xsens-time', 'device_timestamp_s',
                        self._xsens_timestep_receive_time_s, time_code_s, extra_data=extra_data)
      # self._log_debug(time_code_str)

    # The message had a type that is not currently being processed/recorded.
    # No processing is required, but the pointer should still be advanced to ignore the message.
    else:
      self._log_debug('Unknown message type:', message_type)
      return payload_end_index-1

    # Check that the entire message was parsed.
    assert payload_end_index == next_index, 'The Xsens payload should end at byte %d, but the last byte processed was %d' % (payload_end_index, next_index-1)

    # The message was successfully parsed.
    # Return the last index of the message that was used.
    return next_index-1



  #####################
  ###### RUNNING ######
  #####################

  # Loop until self._running is False
  def _run(self):

    try:
      # Run for a few seconds to clear any frames in the input buffer.
      # Often this contains a few frames before the Xsens software reset the frame index.
      run_start_time_s = time.time()
      save_data_start_time_s = run_start_time_s + 0.1
      while self._running:
        # Receive more data
        try:
          if 'tcp' == self._xsens_network_protocol.lower().strip():
            data = self._socket_connection.recv(self._buffer_read_size)
          else: #'udp' == self._xsens_network_protocol.lower().strip():
            data = self._socket.recv(self._buffer_read_size)
          if len(self._buffer)+len(data) <= self._buffer_max_size:
            self._buffer += data
          else:
            # Remove old data if the buffer is overflowing
            self._buffer = data
        except:
          # Xsens stopped running / needs recalibration?
          self._log_warn('WARNING: Did not receive data from the Xsens. Attempting to reconnect in 5 seconds.')
          time.sleep(5)
          self._buffer = b''
          continue

        # Record this as the message arrival time if it's the first time
        #  seeing this message start code in the buffer.
        message_start_index = self._buffer.find(self._xsens_msg_start_code)
        if message_start_index >= 0 and self._xsens_message_start_time_s is None:
          # self._log_debug('Recording xsens message start time')
          self._xsens_message_start_time_s = time.time()

        # Try to process the message
        message_end_index = self._process_xsens_message_from_buffer()
        # If the message was complete, remove it from the buffer
        #  and note that we're waiting for a new start code.
        if message_end_index is not None:
          # self._log_debug('Clearing xsens message start time')
          self._buffer = self._buffer[message_end_index+1:]
          self._xsens_message_start_time_s = None
        
        # Clear the data if this is during the initial flush period.
        if time.time() < save_data_start_time_s:
          self.clear_data_all()
    except KeyboardInterrupt: # The program was likely terminated
      pass
    except:
      self._log_error('\n\n***ERROR RUNNING XsensStreamer:\n%s\n' % traceback.format_exc())
    finally:
      pass

  # Clean up and quit
  def quit(self):
    self._log_debug('XsensStreamer quitting')
    SensorStreamer.quit(self)


  ###########################
  ###### VISUALIZATION ######
  ###########################

  # Specify how the stream should be visualized.
  # No visualization_options are currently supported.
  def get_default_visualization_options(self, visualization_options=None):
    # Visualize the segments position stream by drawing a skeleton.
    options = {}
    options['xsens-segments'] = {}
    options['xsens-segments']['position_cm'] = {
      'class': XsensSkeletonVisualizer,
    }

    # Don't visualize the other devices/streams.
    for (device_name, device_info) in self._streams_info.items():
      options.setdefault(device_name, {})
      for (stream_name, stream_info) in device_info.items():
        options[device_name].setdefault(stream_name, {'class': None})

    return options

  #####################################
  ###### EXTERNAL DATA RECORDING ######
  #####################################

  # Whether recording via the sensor's dedicated software will require user action.
  def external_data_recording_requires_user(self):
    return True

  # Tell the user to start recording via the Xsens software.
  def start_external_data_recording(self, recording_dir):
    recording_dir = os.path.realpath(os.path.join(recording_dir, 'xsens'))
    os.makedirs(recording_dir, exist_ok=True)
    msg = '\n\n--------------------\n'
    msg += 'Start an Xsens recording to the following directory:'
    msg += '\n%s' % recording_dir
    msg+= '\n> Waiting for a new mvn file to appear in that directory... '
    self._log_userAction(msg)
    try:
      import pyperclip
      pyperclip.copy(recording_dir)
    except ModuleNotFoundError:
      pass
    mvn_files = glob.glob(os.path.join(recording_dir, '*.mvn'))
    new_mvn_filename = None
    while new_mvn_filename is None:
      new_mvn_files = [file for file in glob.glob(os.path.join(recording_dir, '*.mvn'))
                       if file not in mvn_files]
      if len(new_mvn_files) > 0:
        new_mvn_filename = new_mvn_files[0]
      else:
        time.sleep(0.2)
    self._external_recording_mvn_filepath = os.path.join(recording_dir, new_mvn_filename)
    self._log_userAction('--------------------\n')

  # Tell the user to stop recording via the Xsens software.
  def stop_external_data_recording(self):
    self._log_userAction('\n\n--------------------')
    self._log_userAction('Stop the Xsens recording')
    time.sleep(3) # wait at least a little, since the below may do nothing if Xsens happens to flush some data to the file
    try:
      timeout_s = 10
      self._log_userAction('\n> Waiting for the mvn file to increase in size (or for %ds to elapse)... ' % timeout_s)
      start_time_s = time.time()
      mvn_size_bytes_original = os.path.getsize(self._external_recording_mvn_filepath)
      while (os.path.getsize(self._external_recording_mvn_filepath) <= mvn_size_bytes_original) \
              and (time.time() - start_time_s < timeout_s):
        time.sleep(0.2)
    except FileNotFoundError:
      pass
    self._log_userAction('')
    self._log_userAction('Note that to later merge the data with the streamed log, use')
    self._log_userAction(' the Xsens software to export an Excel file in the same folder')
    self._log_userAction(' as the recording.  HD reprocessing can be done too if desired.')
    self._log_userAction('--------------------\n')
  
  # Update a streamed data log with data recorded from the Xsens software.
  # An exported Excel or MVNX file of the recording should be in data_dir_external_original/xsens.
  # The MVNX file will be preferred if it is available, since it contains frame timestamps.
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
  
    self._log_status('XsensStreamer merging streamed data with Xsens data')
    
    # Check if a file with a desired extension has been exported in the recording directory.
    def get_filepath_forExportType(dir_to_check, extension):
      filepaths = glob.glob(os.path.join(dir_to_check, '*.%s' % extension))
      filepaths = [x for x in filepaths if '~$' not in x.lower()]
      # See if there is one labeled 'HD' and if so use it (prefer the HD-reprocessed data).
      filepaths_hd = [x for x in filepaths if '_hd.%s' % extension in x.lower()]
      if len(filepaths_hd) > 0:
        filepaths = filepaths_hd
      # Check that a single file was found.
      if len(filepaths) == 0:
        error_msg = 'No exported %s file found in %s' % (extension.upper(), dir_to_check)
        return (None, error_msg)
      if len(filepaths) > 1:
        error_msg = 'Multiple exported %s files found in %s' % (extension.upper(), dir_to_check)
        return (None, error_msg)
      return (filepaths[0], '')
    
    # Look for an MVNX and/or Excel export.
    data_dir_external_original = os.path.join(data_dir_external_original, 'xsens')
    (excel_filepath, error_msg_excel) = get_filepath_forExportType(data_dir_external_original, 'xlsx')
    (mvnx_filepath, error_msg_mvnx) = get_filepath_forExportType(data_dir_external_original, 'mvnx')
    
    # Call the appropriate merging function.
    if mvnx_filepath is not None:
      self._merge_mvnx_data_with_streamed_data(mvnx_filepath, hdf5_file_toUpdate, hdf5_file_archived)
    elif excel_filepath is not None:
      self._merge_excel_data_with_streamed_data(excel_filepath, hdf5_file_toUpdate, hdf5_file_archived)
    else:
      self._log_error('\n\nAborting data merge for Xsens!')
      self._log_error('\n  ' + error_msg_excel)
      self._log_error('\n  ' + error_msg_mvnx)

  # A merging helper function to estimate what time every Xsens frame would
  #  have been received by the system, interpolating between frames that
  #  were missed during the actual streaming.
  def _interpolate_system_time(self, hdf5_file_streamed, all_frame_numbers):
    # Get the system time that was associated with frame numbers during streaming,
    #  and the 'actual' frame time recorded by the Xsens device.
    self._log_debug('Loading and interpolating streamed timestamps')
    if 'xsens-time' in hdf5_file_streamed:
      stream_group = hdf5_file_streamed['xsens-time']['device_timestamp_s']
      streamed_xsens_time_s = np.squeeze(stream_group['data'])
    else:
      # If the Xsens time was not streamed, just get system time from the first device/stream.
      device_group = hdf5_file_streamed[list(hdf5_file_streamed.keys())[0]]
      stream_group = device_group[list(device_group.keys())[0]]
      streamed_xsens_time_s = None
    streamed_system_times_s = np.squeeze(stream_group['time_s'])
    streamed_frame_numbers = np.squeeze(stream_group['xsens_sample_number'])
    streamed_xsens_times_since_start_s = np.squeeze(stream_group['xsens_time_since_start_s'])
  
    # Trim the datasets to ignore times before Xsens started recording,
    #  at which point it resets frame numbers to 0.
    streamed_frame_numbers_diff = np.diff(streamed_frame_numbers)
    if np.any(streamed_frame_numbers_diff < 0):
      # Start after the last time the frame numbers reset.
      stream_start_index = np.where(streamed_frame_numbers_diff < 0)[0][-1]+1
    else:
      stream_start_index = 0
    # Trim the datasets to remove trailing 0s, in case the data logging ended unexpectedly
    #  and did not successfully resize the dataset to fit the data.
    if np.any(streamed_system_times_s == 0):
      stream_end_index = np.where(streamed_system_times_s == 0)[0][0]-1
    else:
      stream_end_index = len(streamed_system_times_s)-1
    # It seems to take about 10-30 frames or so for the stream to settle,
    #  so remove the initial frames from the datasets.
    #  (often these initial frames will be received super close together rather than
    #   at a reasonable sampling rate).
    #  Update: XsensStreamer now discards the first 0.1 seconds of data, so this
    #   should address the above issue, but still trim the dataset just in case.
    # A few ending frames also sometimes seem strange, so remove those too.
    settling_frame_count = 50
    if (stream_end_index - stream_start_index) > settling_frame_count:
      stream_start_index = stream_start_index + settling_frame_count
    if (stream_end_index - stream_start_index) > settling_frame_count:
      stream_end_index = stream_end_index - settling_frame_count
    # Apply the desired trimming.
    streamed_system_times_s = streamed_system_times_s[stream_start_index:stream_end_index+1]
    streamed_frame_numbers = streamed_frame_numbers[stream_start_index:stream_end_index+1]
    streamed_xsens_times_since_start_s = streamed_xsens_times_since_start_s[stream_start_index:stream_end_index+1]
    if streamed_xsens_time_s is not None:
      streamed_xsens_time_s = streamed_xsens_time_s[stream_start_index:stream_end_index+1]
  
    # Interpolate between missed frames to generate a system time for every Excel frame
    #  that estimates when it would have arrived during live streaming.
    # The streaming may have skipped frames due to network stream limitations,
    #  but a piecewise linear interpolation can be used between known frames.
    # For frames before streaming started or after streaming ended, fit a
    #  linear line to all known points and use it to extrapolate.
    all_frame_numbers = np.array(all_frame_numbers)
    frame_numbers_preStream = all_frame_numbers[all_frame_numbers < min(streamed_frame_numbers)]
    frame_numbers_inStream = all_frame_numbers[(all_frame_numbers >= min(streamed_frame_numbers)) & (all_frame_numbers <= max(streamed_frame_numbers))]
    frame_numbers_postStream = all_frame_numbers[all_frame_numbers > max(streamed_frame_numbers)]
    # Interpolate/extrapolate for the system time.
    (number_to_time_m, number_to_time_b) = np.polyfit(streamed_frame_numbers, streamed_system_times_s, deg=1)
    frame_times_s_preStream = frame_numbers_preStream * number_to_time_m + number_to_time_b
    frame_times_s_postStream = frame_numbers_postStream * number_to_time_m + number_to_time_b
    frame_times_s_inStream = np.interp(frame_numbers_inStream, streamed_frame_numbers, streamed_system_times_s)
    all_system_times_s = np.concatenate((frame_times_s_preStream,
                                         frame_times_s_inStream,
                                         frame_times_s_postStream))
    # Return the results and some useful intermediaries.
    return (all_system_times_s,
            streamed_system_times_s, streamed_frame_numbers,
            streamed_xsens_time_s, streamed_xsens_times_since_start_s,
            frame_numbers_preStream, frame_numbers_postStream, frame_numbers_inStream)
  
  # Update a streamed data log with data recorded from the Xsens software and exported to Excel.
  def _merge_excel_data_with_streamed_data(self, excel_filepath,
                                           hdf5_file_toUpdate, hdf5_file_archived):
  
    self._log_status('XsensStreamer merging streamed data with Xsens Excel data')
    
    # Load the Excel data.
    # Will be a dictionary mapping sheet names to dataframes.
    self._log_debug('Loading exported Xsens data from %s' % excel_filepath)
    excel_dataframes = pandas.read_excel(excel_filepath, sheet_name=None)
    
    # Get a list of frame numbers recorded in the Xsens data.
    # This will be the same for all sheets with real data, but sheets
    #  such as metadata and 'Markers' should be skipped.
    all_frame_numbers = None
    num_frames = 0
    for (sheet_name, dataframe) in excel_dataframes.items():
      if 'Frame' in dataframe:
        sheet_frame_numbers = np.array(dataframe.Frame)
        if len(sheet_frame_numbers) > num_frames:
          all_frame_numbers = sheet_frame_numbers
          num_frames = len(all_frame_numbers)
    assert num_frames > 0

    # Interpolate between missed frames to generate a system time for every Excel frame
    #  that estimates when it would have arrived during live streaming.
    (all_system_times_s,
     streamed_system_times_s, streamed_frame_numbers,
     streamed_xsens_time_s, streamed_xsens_times_since_start_s,
     frame_numbers_preStream, frame_numbers_postStream, frame_numbers_inStream) \
      = self._interpolate_system_time(hdf5_file_toUpdate, all_frame_numbers)
      
    # Record a timestamp for every frame in the Excel file that estimates when the data was recorded.
    # This could be done by
    #   1) estimating the start time then assuming Xsens sampled perfectly at a fixed rate,
    #   2) interpolating between timestamps recorded by the Xsens device for streamed frames,
    #   3) interpolating between the times at which data was received by Python for streamed frames.
    # See above comments/code regarding the system-time interpolation approach.
    #   Note that system times at which data was received include network/processing delays.
    # Assuming a constant rate is usually a bit precarious for hardware systems,
    #   but some testing indicates that it does yield times that line up very nicely
    #   with the streamed times at which data was received (with the streamed times
    #   being a bit erratic around the straight constant-fps line, as would be
    #   expected for somewhat stochastic network delays).
    #  Also, exporting data as MVNX and importing into Excel as an XML file to inspect
    #   the device timestamps indicates that assuming a constant rate would yield errors
    #   between -0.5ms and 0.75ms (with a few outliers at -1.3ms),
    #   and this error bound is constant even after ~20s.
    # Interpolating Xsens device timestamps seems promising, but how well the extrapolation
    #  works to data before the stream start and after the stream end has not been tested.
    # So for now, the 'constant-rate' assumption is preferred.
    if self._postprocessing_time_strategy == 'constant-rate':
      # Get a sequence of frame times assuming a constant rate and starting at 0.
      xsens_Fs = 60.0
      xsens_Ts = 1/xsens_Fs
      all_times_s = all_frame_numbers * xsens_Ts
      all_xsens_times_since_start_s = all_times_s - min(all_times_s) # min(all_times_s) should be 0 since Xsens always starts recording frame indexes at 0, but include just in case
      # Shift the Excel times so they represent 'real' time instead of starting at 0.
      # Compute the average offset between the Excel time and the real time
      #  for each streamed frame, to average over network/processing delays
      #  instead of just using one frame to compute the Excel start time.
      offsets = []
      for index_streamed in range(len(streamed_frame_numbers)):
        frame_streamed = streamed_frame_numbers[index_streamed]
        if streamed_xsens_time_s is not None:
          t_streamed = streamed_xsens_time_s[index_streamed]
        else:
          t_streamed = streamed_system_times_s[index_streamed]
        index_excel = np.where(all_frame_numbers == frame_streamed)[0][0]
        t_excel = all_times_s[index_excel]
        offsets.append(t_streamed - t_excel)
      all_times_s = all_times_s + np.mean(offsets)
    elif self._postprocessing_time_strategy == 'interpolate-system':
      # Use the interpolations computed above
      all_times_s = all_system_times_s
      # Interpolate/extrapolate for the Xsens time since start.
      (number_to_time_m, number_to_time_b) = np.polyfit(streamed_frame_numbers, streamed_xsens_times_since_start_s, deg=1)
      frame_times_since_start_s_preStream = frame_numbers_preStream * number_to_time_m + number_to_time_b
      frame_times_since_start_s_postStream = frame_numbers_postStream * number_to_time_m + number_to_time_b
      frame_times_since_start_s_inStream = np.interp(frame_numbers_inStream, streamed_frame_numbers, streamed_xsens_times_since_start_s)
      all_xsens_times_since_start_s = np.concatenate((frame_times_since_start_s_preStream,
                                                      frame_times_since_start_s_inStream,
                                                      frame_times_since_start_s_postStream))
    elif self._postprocessing_time_strategy == 'interpolate-xsens' and streamed_xsens_time_s is not None:
      # Interpolate/extrapolate the Xsens timestamps.
      (number_to_time_m, number_to_time_b) = np.polyfit(streamed_frame_numbers, streamed_xsens_time_s, deg=1)
      frame_times_s_preStream = frame_numbers_preStream * number_to_time_m + number_to_time_b
      frame_times_s_postStream = frame_numbers_postStream * number_to_time_m + number_to_time_b
      frame_times_s_inStream = np.interp(frame_numbers_inStream, streamed_frame_numbers, streamed_xsens_time_s)
      all_times_s = np.concatenate((frame_times_s_preStream,
                                      frame_times_s_inStream,
                                      frame_times_s_postStream))
      all_xsens_times_since_start_s = all_times_s - min(all_times_s)
    else:
      raise AssertionError('Invalid post-processing time strategy: ' + self._postprocessing_time_strategy)
      
    # Compute time strings for each timestamp.
    all_times_str = [get_time_str(t, '%Y-%m-%d %H:%M:%S.%f') for t in all_times_s]
    
    # Move all streamed data to the archive HDF5 file.
    self._log_debug('Moving streamed data to the archive HDF5 file')
    for (device_name, device_group) in hdf5_file_toUpdate.items():
      if 'xsens-' in device_name:
        hdf5_file_toUpdate.copy(device_group, hdf5_file_archived,
                                name=None, shallow=False,
                                expand_soft=True, expand_external=True, expand_refs=True,
                                without_attrs=False)
        device_group_metadata = dict(device_group.attrs.items())
        hdf5_file_archived[device_name].attrs.update(device_group_metadata)
        del hdf5_file_toUpdate[device_name]
    
    # Import Excel data into the HDF5 file.
    self._log_debug('Importing data from Excel to the new HDF5 file')
    
    def add_hdf5_data(device_name, stream_name,
                      excel_sheet_name,
                      target_data_shape_per_frame,
                      data=None, # if given, will be used instead of Excel data (sheet name and target shape arguments are ignored)
                      data_processing_fn=None,
                      stream_group_metadata=None):
      # Extract the desired sheet data.
      if data is None:
        dataframe = excel_dataframes[excel_sheet_name]
        data = dataframe.to_numpy()
        # Remove the 'Frame' column.
        data = data[:, 1:]
        # Reshape so that data for each frame has the desired shape.
        data = data.reshape((data.shape[0], *target_data_shape_per_frame))
        assert data.shape[0] == num_frames
        # Process the data if desired.
        if callable(data_processing_fn):
          data = data_processing_fn(data)
      
      # Add the data to the HDF5 file.
      if device_name not in hdf5_file_toUpdate:
        hdf5_file_toUpdate.create_group(device_name)
      if stream_name in hdf5_file_toUpdate[device_name]:
        del hdf5_file_toUpdate[device_name][stream_name]
      hdf5_file_toUpdate[device_name].create_group(stream_name)
      stream_group = hdf5_file_toUpdate[device_name][stream_name]
      stream_group.create_dataset('data', data=data)
      stream_group.create_dataset('xsens_sample_number', [num_frames, 1],
                                  data=all_frame_numbers)
      stream_group.create_dataset('xsens_time_since_start_s', [num_frames, 1],
                                  data=all_xsens_times_since_start_s)
      stream_group.create_dataset('time_s', [num_frames, 1], dtype='float64',
                                  data=all_times_s)
      stream_group.create_dataset('time_str', [num_frames, 1], dtype='S26',
                                  data=all_times_str)
      # Copy the original device-level and stream-level metadata.
      if device_name in hdf5_file_archived:
        archived_device_group = hdf5_file_archived[device_name]
        device_group_metadata_original = dict(archived_device_group.attrs.items())
        hdf5_file_toUpdate[device_name].attrs.update(device_group_metadata_original)
        if stream_name in archived_device_group:
          archived_stream_group = archived_device_group[stream_name]
          stream_group_metadata_original = dict(archived_stream_group.attrs.items())
          hdf5_file_toUpdate[device_name][stream_name].attrs.update(stream_group_metadata_original)
      else:
        # Create a basic device-level metadata if there was none to copy from the original file.
        device_group_metadata = {SensorStreamer.metadata_class_name_key: type(self).__name__}
        hdf5_file_toUpdate[device_name].attrs.update(device_group_metadata)
      # Override with provided stream-level metadata if desired.
      if stream_group_metadata is not None:
        hdf5_file_toUpdate[device_name][stream_name].attrs.update(
            convert_dict_values_to_str(stream_group_metadata, preserve_nested_dicts=False))
    
    # Define the data notes to use for each stream.
    self._define_data_notes()
    
    # Segment orientation data (quaternion)
    add_hdf5_data(device_name='xsens-segments',
                  stream_name='orientation_quaternion',
                  excel_sheet_name='Segment Orientation - Quat',
                  target_data_shape_per_frame=[-1, 4], # 4-element quaternion per segment per frame
                  data_processing_fn=None,
                  stream_group_metadata=self._data_notes_excel['xsens-segments']['orientation_quaternion'])
    # Segment orientation data (Euler)
    add_hdf5_data(device_name='xsens-segments',
                  stream_name='orientation_euler_deg',
                  excel_sheet_name='Segment Orientation - Euler',
                  target_data_shape_per_frame=[-1, 3], # 3-element Euler vector per segment per frame
                  data_processing_fn=None,
                  stream_group_metadata=self._data_notes_excel['xsens-segments']['orientation_euler_deg'])
    
    # Segment position data.
    add_hdf5_data(device_name='xsens-segments',
                  stream_name='position_cm',
                  excel_sheet_name='Segment Position',
                  target_data_shape_per_frame=[-1, 3], # 3-element x/y/z vector per segment per frame
                  data_processing_fn=lambda data: data*100.0, # convert from m to cm
                  stream_group_metadata=self._data_notes_excel['xsens-segments']['position_cm'])
    # Segment velocity data.
    add_hdf5_data(device_name='xsens-segments',
                  stream_name='velocity_cm_s',
                  excel_sheet_name='Segment Velocity',
                  target_data_shape_per_frame=[-1, 3], # 3-element x/y/z vector per segment per frame
                  data_processing_fn=lambda data: data*100.0, # convert from m to cm
                  stream_group_metadata=self._data_notes_excel['xsens-segments']['velocity_cm_s'])
    # Segment acceleration data.
    add_hdf5_data(device_name='xsens-segments',
                  stream_name='acceleration_cm_ss',
                  excel_sheet_name='Segment Acceleration',
                  target_data_shape_per_frame=[-1, 3], # 3-element x/y/z vector per segment per frame
                  data_processing_fn=lambda data: data*100.0, # convert from m to cm
                  stream_group_metadata=self._data_notes_excel['xsens-segments']['acceleration_cm_ss'])
    # Segment angular velocity.
    add_hdf5_data(device_name='xsens-segments',
                  stream_name='angular_velocity_deg_s',
                  excel_sheet_name='Segment Angular Velocity',
                  target_data_shape_per_frame=[-1, 3], # 3-element x/y/z vector per segment per frame
                  data_processing_fn=lambda data: data*180.0/np.pi, # convert from rad to deg
                  stream_group_metadata=self._data_notes_excel['xsens-segments']['angular_velocity_deg_s'])
    # Segment angular acceleration.
    add_hdf5_data(device_name='xsens-segments',
                  stream_name='angular_acceleration_deg_ss',
                  excel_sheet_name='Segment Angular Acceleration',
                  target_data_shape_per_frame=[-1, 3], # 3-element x/y/z vector per segment per frame
                  data_processing_fn=lambda data: data*180.0/np.pi, # convert from rad to deg
                  stream_group_metadata=self._data_notes_excel['xsens-segments']['angular_acceleration_deg_ss'])

    # Joint angles ZXY.
    add_hdf5_data(device_name='xsens-joints',
                  stream_name='rotation_zxy_deg',
                  excel_sheet_name='Joint Angles ZXY',
                  target_data_shape_per_frame=[-1, 3], # 3-element vector per segment per frame
                  data_processing_fn=None,
                  stream_group_metadata=self._data_notes_excel['xsens-joints']['rotation_zxy_deg'])
    # Joint angles XZY.
    add_hdf5_data(device_name='xsens-joints',
                  stream_name='rotation_xzy_deg',
                  excel_sheet_name='Joint Angles XZY',
                  target_data_shape_per_frame=[-1, 3], # 3-element vector per segment per frame
                  data_processing_fn=None,
                  stream_group_metadata=self._data_notes_excel['xsens-joints']['rotation_xzy_deg'])
    # Ergonomic joint angles ZXY.
    add_hdf5_data(device_name='xsens-ergonomic-joints',
                  stream_name='rotation_zxy_deg',
                  excel_sheet_name='Ergonomic Joint Angles ZXY',
                  target_data_shape_per_frame=[-1, 3], # 3-element vector per segment per frame
                  data_processing_fn=None,
                  stream_group_metadata=self._data_notes_excel['xsens-ergonomic-joints']['rotation_zxy_deg'])
    # Ergonomic joint angles XZY.
    add_hdf5_data(device_name='xsens-ergonomic-joints',
                  stream_name='rotation_xzy_deg',
                  excel_sheet_name='Ergonomic Joint Angles XZY',
                  target_data_shape_per_frame=[-1, 3], # 3-element vector per segment per frame
                  data_processing_fn=None,
                  stream_group_metadata=self._data_notes_excel['xsens-ergonomic-joints']['rotation_xzy_deg'])

    # Center of mass position.
    add_hdf5_data(device_name='xsens-CoM',
                  stream_name='position_cm',
                  excel_sheet_name='Center of Mass',
                  target_data_shape_per_frame=[9], # 9-element vector per segment per frame (x/y/z for position/velocity/acceleration)
                  data_processing_fn=lambda data: data[:, 0:3]*100.0, # convert from m to cm, and select the 3 position columns
                  stream_group_metadata=self._data_notes_excel['xsens-CoM']['position_cm'])
    # Center of mass velocity.
    add_hdf5_data(device_name='xsens-CoM',
                  stream_name='velocity_cm_s',
                  excel_sheet_name='Center of Mass',
                  target_data_shape_per_frame=[9], # 9-element vector per segment per frame (x/y/z for position/velocity/acceleration)
                  data_processing_fn=lambda data: data[:, 3:6]*100.0, # convert from m to cm, and select the 3 velocity columns
                  stream_group_metadata=self._data_notes_excel['xsens-CoM']['velocity_cm_s'])
    # Center of mass acceleration.
    add_hdf5_data(device_name='xsens-CoM',
                  stream_name='acceleration_cm_ss',
                  excel_sheet_name='Center of Mass',
                  target_data_shape_per_frame=[9], # 9-element vector per segment per frame (x/y/z for position/velocity/acceleration)
                  data_processing_fn=lambda data: data[:, 6:9]*100.0, # convert from m to cm, and select the 3 acceleration columns
                  stream_group_metadata=self._data_notes_excel['xsens-CoM']['acceleration_cm_ss'])
    
    # Sensor data - acceleration
    add_hdf5_data(device_name='xsens-sensors',
                  stream_name='free_acceleration_cm_ss',
                  excel_sheet_name='Sensor Free Acceleration',
                  target_data_shape_per_frame=[-1, 3], # 3-element x/y/z vector per segment per frame
                  data_processing_fn=lambda data: data*100.0, # convert from m to cm
                  stream_group_metadata=self._data_notes_excel['xsens-sensors']['free_acceleration_cm_ss'])
    # Sensor data - magnetic field
    add_hdf5_data(device_name='xsens-sensors',
                  stream_name='magnetic_field',
                  excel_sheet_name='Sensor Magnetic Field',
                  target_data_shape_per_frame=[-1, 3], # 3-element x/y/z vector per segment per frame
                  data_processing_fn=None,
                  stream_group_metadata=self._data_notes_excel['xsens-sensors']['magnetic_field'])
    # Sensor data - orientation
    add_hdf5_data(device_name='xsens-sensors',
                  stream_name='orientation_quaternion',
                  excel_sheet_name='Sensor Orientation - Quat',
                  target_data_shape_per_frame=[-1, 4], # 4-element quaternion per segment per frame
                  data_processing_fn=None,
                  stream_group_metadata=self._data_notes_excel['xsens-sensors']['orientation_quaternion'])
    # Sensor data - orientation
    add_hdf5_data(device_name='xsens-sensors',
                  stream_name='orientation_euler_deg',
                  excel_sheet_name='Sensor Orientation - Euler',
                  target_data_shape_per_frame=[-1, 3], # 3-element x/y/z vector per segment per frame
                  data_processing_fn=None,
                  stream_group_metadata=self._data_notes_excel['xsens-sensors']['orientation_euler_deg'])
    
    # Sytem time
    add_hdf5_data(device_name='xsens-time',
                  stream_name='stream_receive_time_s',
                  excel_sheet_name=None,
                  target_data_shape_per_frame=None,
                  data=all_system_times_s,
                  stream_group_metadata=self._data_notes_excel['xsens-time']['stream_receive_time_s'])

  
  
  
  
  
  
  def _get_mvnx_metadata(self, mvnx_xml):
    # Extract metadata about MVN version.
    mvnx_version = mvnx_xml.find('mvnx').get('version')
    mvn_version = mvnx_xml.find('mvn').get('build')
    if mvnx_version != '4':
      raise AssertionError('The MVNX import script currently assumes MVNX version 4, but version %s is being used instead. The script should be checked for compatibility.' % mvnx_version)
    # Extract the sensor names.
    sensor_labels = [sensor_xml.get('label') for sensor_xml in mvnx_xml.find_all('sensor')]
    
    # Extract segment metadata for the main skeleton.
    body_segment_mesh_points_xyz_cm = OrderedDict()
    for segment_xml in mvnx_xml.find('segments').find_all('segment'):
      segment_mesh_points_xyz_cm_forSegment = OrderedDict()
      for segment_point_xml in segment_xml.find_all('point'):
        point_position_xyz_cm = [round(100*float(d), 6) for d in segment_point_xml.find('pos_b').contents[0].split()]
        segment_mesh_points_xyz_cm_forSegment[segment_point_xml.get('label')] = point_position_xyz_cm
      body_segment_mesh_points_xyz_cm[segment_xml.get('label')] = segment_mesh_points_xyz_cm_forSegment
    body_segment_labels = list(body_segment_mesh_points_xyz_cm.keys())
    # Extract finger segment metadata for the left hand if it is available.
    fingerLeft_segment_mesh_points_xyz_cm = OrderedDict()
    if mvnx_xml.find('fingerTrackingSegmentsLeft') is not None:
      for segment_xml in mvnx_xml.find('fingerTrackingSegmentsLeft').find_all('segment'):
        segment_mesh_points_xyz_cm_forSegment = OrderedDict()
        for segment_point_xml in segment_xml.find_all('point'):
          point_position_xyz_cm = [round(100*float(d), 6) for d in segment_point_xml.find('pos_b').contents[0].split()]
          segment_mesh_points_xyz_cm_forSegment[segment_point_xml.get('label')] = point_position_xyz_cm
        fingerLeft_segment_mesh_points_xyz_cm[segment_xml.get('label')] = segment_mesh_points_xyz_cm_forSegment
    fingerLeft_segment_labels = list(fingerLeft_segment_mesh_points_xyz_cm.keys())
    # Extract finger segment metadata for the right hand if it is available.
    fingerRight_segment_mesh_points_xyz_cm = OrderedDict()
    if mvnx_xml.find('fingerTrackingSegmentsRight') is not None:
      for segment_xml in mvnx_xml.find('fingerTrackingSegmentsRight').find_all('segment'):
        segment_mesh_points_xyz_cm_forSegment = OrderedDict()
        for segment_point_xml in segment_xml.find_all('point'):
          point_position_xyz_cm = [round(100*float(d), 6) for d in segment_point_xml.find('pos_b').contents[0].split()]
          segment_mesh_points_xyz_cm_forSegment[segment_point_xml.get('label')] = point_position_xyz_cm
        fingerRight_segment_mesh_points_xyz_cm[segment_xml.get('label')] = segment_mesh_points_xyz_cm_forSegment
    fingerRight_segment_labels = list(fingerRight_segment_mesh_points_xyz_cm.keys())
    
    # Extract joint metadata for the main skeleton.
    body_joint_segment_connections = OrderedDict()
    for joint_xml in mvnx_xml.find('joints').find_all('joint'):
      body_joint_segment_connections[joint_xml.get('label')[1:]] = [ # trim the initial 'j' from the label
        joint_xml.find('connector1').contents[0],
        joint_xml.find('connector2').contents[0],
        ]
    body_joint_labels = list(body_joint_segment_connections.keys())
    # Extract finger joint metadata for the left hand if it is available.
    fingerLeft_joint_segment_connections = OrderedDict()
    if mvnx_xml.find('fingerTrackingJointsLeft') is not None:
      for joint_xml in mvnx_xml.find('fingerTrackingJointsLeft').find_all('joint'):
        fingerLeft_joint_segment_connections[joint_xml.get('label')[1:]] = [ # trim the initial 'j' from the label
          joint_xml.find('connector1').contents[0],
          joint_xml.find('connector2').contents[0],
          ]
    fingerLeft_joint_labels = list(fingerLeft_joint_segment_connections.keys())
    # Extract finger joint metadata for the right hand if it is available.
    fingerRight_joint_segment_connections = OrderedDict()
    if mvnx_xml.find('fingerTrackingJointsRight') is not None:
      for joint_xml in mvnx_xml.find('fingerTrackingJointsRight').find_all('joint'):
        fingerRight_joint_segment_connections[joint_xml.get('label')[1:]] = [ # trim the initial 'j' from the label
          joint_xml.find('connector1').contents[0],
          joint_xml.find('connector2').contents[0],
          ]
    fingerRight_joint_labels = list(fingerRight_joint_segment_connections.keys())
    
    # Extract metadata about the ergonomic joint angles.
    ergonomic_joint_segment_connections = OrderedDict()
    for joint_xml in mvnx_xml.find_all('ergonomicJointAngle'):
      ergonomic_joint_segment_connections[joint_xml.get('label')] = [
        joint_xml.get('parentSegment'),
        joint_xml.get('childSegment'),
        ]
    ergonomic_joint_labels = list(ergonomic_joint_segment_connections.keys())
    
    # Extract the foot contact names.
    foot_contact_labels = [tag.get('label') for tag in mvnx_xml.find_all('contactDefinition')]
    
    # Define joint rotation names for each entry.
    body_joint_rotation_type_ordering_raw = OrderedDict([
      ('L5S1',   ('Lateral Bending', 'Axial Bending',  'Flexion/Extension')),
      ('L4L3',   ('Lateral Bending', 'Axial Rotation', 'Flexion/Extension')),
      ('L1T12',  ('Lateral Bending', 'Axial Rotation', 'Flexion/Extension')),
      ('T9T8',   ('Lateral Bending', 'Axial Rotation', 'Flexion/Extension')),
      ('T1C7',   ('Lateral Bending', 'Axial Rotation', 'Flexion/Extension')),
      ('C1Head', ('Lateral Bending', 'Axial Rotation', 'Flexion/Extension')),
      ('RightT4Shoulder', ('Abduction/Adduction',              'Internal/External Rotation', 'Flexion/Extension')),
      ('RightShoulder',   ('Abduction/Adduction',              'Internal/External Rotation', 'Flexion/Extension')),
      ('RightElbow',      ('Ulnar Deviation/Radial Deviation', 'Pronation/Supination',       'Flexion/Extension')),
      ('RightWrist',      ('Ulnar Deviation/Radial Deviation', 'Pronation/Supination',       'Flexion/Extension')),
      ('LeftT4Shoulder',  ('Abduction/Adduction',              'Internal/External Rotation', 'Flexion/Extension')),
      ('LeftShoulder',    ('Abduction/Adduction',              'Internal/External Rotation', 'Flexion/Extension')),
      ('LeftElbow',       ('Ulnar Deviation/Radial Deviation', 'Pronation/Supination',       'Flexion/Extension')),
      ('LeftWrist',       ('Ulnar Deviation/Radial Deviation', 'Pronation/Supination',       'Flexion/Extension')),
      ('RightHip',        ('Abduction/Adduction',              'Internal/External Rotation', 'Flexion/Extension')),
      ('RightKnee',       ('Abduction/Adduction',              'Internal/External Rotation', 'Flexion/Extension')),
      ('RightAnkle',      ('Abduction/Adduction',              'Internal/External Rotation', 'Dorsiflexion/Plantarflexion')),
      ('RightBallFoot',   ('Abduction/Adduction',              'Internal/External Rotation', 'Flexion/Extension')),
      ('LeftHip',         ('Abduction/Adduction',              'Internal/External Rotation', 'Flexion/Extension')),
      ('LeftKnee',        ('Abduction/Adduction',              'Internal/External Rotation', 'Flexion/Extension')),
      ('LeftAnkle',       ('Abduction/Adduction',              'Internal/External Rotation', 'Dorsiflexion/Plantarflexion')),
      ('LeftBallFoot',    ('Abduction/Adduction',              'Internal/External Rotation', 'Flexion/Extension')),
    ])
    body_joint_rotation_type_ordering = [(key, body_joint_rotation_type_ordering_raw[key]) for key in body_joint_labels]
    fingerLeft_joint_rotation_type_ordering = OrderedDict([
      (key, ('Abduction/Adduction', 'Internal/External Rotation', 'Flexion/Extension'))
       for key in fingerLeft_joint_labels])
    fingerRight_joint_rotation_type_ordering = OrderedDict([
      (key, ('Abduction/Adduction', 'Internal/External Rotation', 'Flexion/Extension'))
       for key in fingerRight_joint_labels])
    
    # Compile the metadata.
    metadata = OrderedDict([
      ('num_segments_body', len(body_segment_labels)),
      ('num_segments_fingers_left', len(fingerLeft_segment_labels)),
      ('num_segments_fingers_right', len(fingerRight_segment_labels)),
      ('num_joints_body', len(body_joint_labels)),
      ('num_joints_fingers_left', len(fingerLeft_joint_labels)),
      ('num_joints_fingers_right', len(fingerRight_joint_labels)),
      ('num_joints_ergonomic', len(ergonomic_joint_labels)),
      ('num_sensors', len(sensor_labels)),
      ('segment_names_body', body_segment_labels),
      ('segment_names_fingers_left', fingerLeft_segment_labels),
      ('segment_names_fingers_right', fingerRight_segment_labels),
      ('joint_names_body', body_joint_labels),
      ('joint_names_fingers_left', fingerLeft_joint_labels),
      ('joint_names_fingers_right', fingerRight_joint_labels),
      ('joint_rotation_order_body', body_joint_rotation_type_ordering),
      ('joint_rotation_order_fingers_left', fingerLeft_joint_rotation_type_ordering),
      ('joint_rotation_order_fingers_right', fingerRight_joint_rotation_type_ordering),
      ('ergonomic_joint_names', ergonomic_joint_labels),
      ('sensor_names', sensor_labels),
      ('foot_contact_names', foot_contact_labels),
      ('joint_segment_connections_body', body_joint_segment_connections),
      ('joint_segment_connections_fingers_left', fingerLeft_joint_segment_connections),
      ('joint_segment_connections_fingers_right', fingerRight_joint_segment_connections),
      ('ergonomic_joint_segment_connections', ergonomic_joint_segment_connections),
      ('segment_mesh_points_body_xyz_cm', body_segment_mesh_points_xyz_cm),
      ('segment_mesh_points_fingers_left_xyz_cm', fingerLeft_segment_mesh_points_xyz_cm),
      ('segment_mesh_points_fingers_right_xyz_cm', fingerRight_segment_mesh_points_xyz_cm),
      ('mvn_version', mvn_version),
      ('mvnx_version', mvnx_version),
    ])
    if len(fingerLeft_segment_labels) == 0:
      for key in list(metadata.keys()):
        if 'fingers_left' in key:
          del metadata[key]
    if len(fingerRight_segment_labels) == 0:
      for key in list(metadata.keys()):
        if 'fingers_right' in key:
          del metadata[key]
    return metadata
  
  # Update a streamed data log with data recorded from the Xsens software exported to MVNX.
  def _merge_mvnx_data_with_streamed_data(self, mvnx_filepath,
                                          hdf5_file_toUpdate, hdf5_file_archived):
  
    self._log_status('XsensStreamer merging streamed data with Xsens MVNX data')
    
    # Read and parse the MVNX file as an XML file.
    self._log_debug('Reading and parsing the MVNX file')
    with open(mvnx_filepath, 'r') as fin:
      mvnx_contents = fin.read()
    mvnx_xml = BeautifulSoup(mvnx_contents, 'xml')
  
    # Get metadata.
    metadata = self._get_mvnx_metadata(mvnx_xml)
    metadata_segments = dict([(key, value) for (key, value) in metadata.items() if 'joint' not in key and 'sensor' not in key and 'contact' not in key])
    metadata_joints = dict([(key, value) for (key, value) in metadata.items() if 'segment' not in key and 'sensor' not in key and 'contact' not in key])
    metadata_com = dict([(key, value) for (key, value) in metadata.items() if 'segment' not in key and 'joint' not in key and 'sensor' not in key and 'contact' not in key])
    metadata_sensors = dict([(key, value) for (key, value) in metadata.items() if 'segment' not in key and 'joint' not in key and 'contact' not in key])
    metadata_footContacts = dict([(key, value) for (key, value) in metadata.items() if 'segment' not in key and 'joint' not in key and 'sensor' not in key])
    
    have_fingers_left = 'segment_names_fingers_left' in metadata
    have_fingers_right = 'segment_names_fingers_right' in metadata
    # print_var(metadata, 'metadata')

    # Extract all frames
    frames = mvnx_xml.find_all('frame')

    # Extract the calibration frames.
    calibration_types = ['identity', 'Tpose', 'Tpose_ISB'] # should match MVNX types if make lowercase and replace underscores with hyphones
    calibration_data = dict([(key, None) for key in calibration_types])
    for (frame_index, frame) in enumerate(frames[0:5]): # should be the first 3 frames, but increase a bit just to be safe
      for calibration_type in calibration_types:
        if frame.get('type') == calibration_type.lower().replace('_', '-'):
          calibration_data[calibration_type] = {
            'segment_orientations_body_quaternion': np.atleast_2d([round(float(q), 6) for q in frame.find('orientation').contents[0].split()]),
            'segment_positions_body_m': np.atleast_2d([round(float(q), 6) for q in frame.find('position').contents[0].split()]),
          }

    # Extract data from all non-calibration frames.
    frames = [frame for (i, frame) in enumerate(frames) if frame.get('type') == 'normal']
    frame_indexes = [int(frame.get('index')) for frame in frames]
    num_frames = len(frames)

    # Get time information.
    times_since_start_s = [round(float(frame.get('time')))/1000.0 for frame in frames]
    times_utc_str = [frame.get('tc') for frame in frames]
    times_s = [float(frame.get('ms'))/1000.0 for frame in frames]
    times_str = [get_time_str(time_s, '%Y-%m-%d %H:%M:%S.%f') for time_s in times_s]

    # Check that the timestamps monotonically increase.
    if np.any(np.diff(times_s) < 0):
      print_var(times_str, 'times_str')
      msg = '\n'*2
      msg += 'x'*75
      msg += 'XsensStreamer aborting merge due to incorrect timestamps in the MVNX file (they are not monotonically increasing)'
      msg += 'x'*75
      msg = '\n'*2
      print(msg)
      return

    # A helper to get a matrix of data from all frames for a given tag, such as 'position'.
    def get_tagged_data(tag):
      datas = [frame.find(tag) for frame in frames]
      data = np.array([[round(float(x), 6) for x in data.contents[0].split()] for data in datas])
      return data

    # Extract the data!
    segment_orientations_body_quaternion_wijk = get_tagged_data('orientation')
    segment_positions_body_m                  = get_tagged_data('position')
    segment_velocities_body_m_s               = get_tagged_data('velocity')
    segment_accelerations_body_m_ss           = get_tagged_data('acceleration')
    segment_angular_velocities_body_rad_s     = get_tagged_data('angularVelocity')
    segment_angular_accelerations_body_rad_ss = get_tagged_data('angularAcceleration')
    foot_contacts                             = get_tagged_data('footContacts')
    sensor_freeAccelerations_m_ss             = get_tagged_data('sensorFreeAcceleration')
    sensor_magnetic_fields_au                 = get_tagged_data('sensorMagneticField')
    sensor_orientations_quaternion_wijk       = get_tagged_data('sensorOrientation')
    joint_angles_body_eulerZXY_xyz_rad        = get_tagged_data('jointAngle')*np.pi/180.0 # convert from degrees to radians
    joint_angles_body_eulerXZY_xyz_rad        = get_tagged_data('jointAngleXZY')*np.pi/180.0 # convert from degrees to radians
    joint_angles_ergonomic_eulerZXY_xyz_rad   = get_tagged_data('jointAngleErgo')*np.pi/180.0 # convert from degrees to radians
    joint_angles_ergonomic_eulerXZY_xyz_rad   = get_tagged_data('jointAngleErgoXZY')*np.pi/180.0 # convert from degrees to radians
    center_of_mass                            = get_tagged_data('centerOfMass')
    center_of_mass_positions_m                = center_of_mass[:, 0:3]
    center_of_mass_velocities_m_s             = center_of_mass[:, 3:6]
    center_of_mass_accelerations_m_ss         = center_of_mass[:, 6:9]
    # Extract data for the left hand if it is available.
    if have_fingers_left:
      segment_orientations_fingersLeft_quaternion_wijk = get_tagged_data('orientationFingersLeft')
      segment_positions_fingersLeft_m                  = get_tagged_data('positionFingersLeft')
      joint_angles_fingersLeft_eulerZXY_xyz_rad        = get_tagged_data('jointAngleFingersLeft')*np.pi/180.0 # convert from degrees to radians
      joint_angles_fingersLeft_eulerXZY_xyz_rad        = get_tagged_data('jointAngleFingersLeftXZY')*np.pi/180.0 # convert from degrees to radians
    else:
      segment_orientations_fingersLeft_quaternion_wijk = None
      segment_positions_fingersLeft_m = None
      joint_angles_fingersLeft_eulerZXY_xyz_rad = None
      joint_angles_fingersLeft_eulerXZY_xyz_rad = None
    # Extract data for the right hand if it is available.
    if have_fingers_right:
      segment_orientations_fingersRight_quaternion_wijk = get_tagged_data('orientationFingersRight')
      segment_positions_fingersRight_m                  = get_tagged_data('positionFingersRight')
      joint_angles_fingersRight_eulerZXY_xyz_rad        = get_tagged_data('jointAngleFingersRight')*np.pi/180.0 # convert from degrees to radians
      joint_angles_fingersRight_eulerXZY_xyz_rad        = get_tagged_data('jointAngleFingersRightXZY')*np.pi/180.0 # convert from degrees to radians
    else:
      segment_orientations_fingersRight_quaternion_wijk = None
      segment_positions_fingersRight_m = None
      joint_angles_fingersRight_eulerZXY_xyz_rad = None
      joint_angles_fingersRight_eulerXZY_xyz_rad = None
    
    # Check a few dimensions.
    assert metadata['num_segments_body'] == segment_orientations_body_quaternion_wijk.shape[1]/4
    if have_fingers_left:
      assert metadata['num_segments_fingers_left'] == segment_orientations_fingersLeft_quaternion_wijk.shape[1]/4
    if have_fingers_right:
      assert metadata['num_segments_fingers_right'] == segment_orientations_fingersRight_quaternion_wijk.shape[1]/4
    
    # Create Euler orientations from quaternion orientations.
    segment_orientations_body_eulerZXY_xyz_rad = np.empty([num_frames, 3*metadata['num_segments_body']])
    for segment_index in range(metadata['num_segments_body']):
      for frame_index in range(num_frames):
        quat_wijk = segment_orientations_body_quaternion_wijk[frame_index, (segment_index*4):(segment_index*4+4)]
        eulers_rad = euler_from_quaternion(w=quat_wijk[0], x=quat_wijk[1], y=quat_wijk[2], z=quat_wijk[3], euler_sequence='ZXY', degrees=False)
        segment_orientations_body_eulerZXY_xyz_rad[frame_index, (segment_index*3):(segment_index*3+3)] = eulers_rad
    if have_fingers_left:
      segment_orientations_fingersLeft_eulerZXY_xyz_rad = np.empty([num_frames, 3*metadata['num_segments_fingers_left']])
      for segment_index in range(metadata['num_segments_fingers_left']):
        for frame_index in range(num_frames):
          quat_wijk = segment_orientations_fingersLeft_quaternion_wijk[frame_index, (segment_index*4):(segment_index*4+4)]
          eulers_rad = euler_from_quaternion(w=quat_wijk[0], x=quat_wijk[1], y=quat_wijk[2], z=quat_wijk[3], euler_sequence='ZXY', degrees=False)
          segment_orientations_fingersLeft_eulerZXY_xyz_rad[frame_index, (segment_index*3):(segment_index*3+3)] = eulers_rad
    if have_fingers_right:
      segment_orientations_fingersRight_eulerZXY_xyz_rad = np.empty([num_frames, 3*metadata['num_segments_fingers_right']])
      for segment_index in range(metadata['num_segments_fingers_right']):
        for frame_index in range(num_frames):
          quat_wijk = segment_orientations_fingersRight_quaternion_wijk[frame_index, (segment_index*4):(segment_index*4+4)]
          eulers_rad = euler_from_quaternion(w=quat_wijk[0], x=quat_wijk[1], y=quat_wijk[2], z=quat_wijk[3], euler_sequence='ZXY', degrees=False)
          segment_orientations_fingersRight_eulerZXY_xyz_rad[frame_index, (segment_index*3):(segment_index*3+3)] = eulers_rad
    sensor_orientations_eulerZXY_xyz_rad = np.empty([num_frames, 3*metadata['num_sensors']])
    for sensor_index in range(metadata['num_sensors']):
      for frame_index in range(num_frames):
        quat_wijk = sensor_orientations_quaternion_wijk[frame_index, (sensor_index*4):(sensor_index*4+4)]
        eulers_rad = euler_from_quaternion(w=quat_wijk[0], x=quat_wijk[1], y=quat_wijk[2], z=quat_wijk[3], euler_sequence='ZXY', degrees=False)
        sensor_orientations_eulerZXY_xyz_rad[frame_index, (sensor_index*3):(sensor_index*3+3)] = eulers_rad
    
    
    
    
    
    
    
    # Move all streamed data to the archive HDF5 file.
    self._log_debug('Moving streamed data to the archive HDF5 file')
    for (device_name, device_group) in hdf5_file_toUpdate.items():
      if 'xsens-' in device_name:
        hdf5_file_toUpdate.copy(device_group, hdf5_file_archived,
                                name=None, shallow=False,
                                expand_soft=True, expand_external=True, expand_refs=True,
                                without_attrs=False)
        device_group_metadata = dict(device_group.attrs.items())
        hdf5_file_archived[device_name].attrs.update(device_group_metadata)
        del hdf5_file_toUpdate[device_name]
    
    # TODO add this back
    # TODO add this back
    # TODO add this back
    # TODO add this back
    # TODO add this back
    # # Interpolate between missed frames to generate a system time for every Excel frame
    # #  that estimates when it would have arrived during live streaming.
    # interpolation_results = self._interpolate_system_time(hdf5_file_archived, frame_indexes)
    # system_times_s = interpolation_results[0]
    
    # Helper to import the MVNX data into the HDF5 file.
    def add_hdf5_data(device_name, stream_name,
                      data, target_data_shape_per_frame,
                      stream_group_metadata=None):
      # Reshape so that data for each frame has the desired shape.
      data = data.reshape((data.shape[0], *target_data_shape_per_frame))
      assert data.shape[0] == num_frames
      # Create the stream group.
      if device_name not in hdf5_file_toUpdate:
        hdf5_file_toUpdate.create_group(device_name)
      if stream_name in hdf5_file_toUpdate[device_name]:
        del hdf5_file_toUpdate[device_name][stream_name]
      hdf5_file_toUpdate[device_name].create_group(stream_name)
      stream_group = hdf5_file_toUpdate[device_name][stream_name]
      # Create the datasets.
      stream_group.create_dataset('data', data=data)
      stream_group.create_dataset('xsens_sample_number', [num_frames, 1],
                                  data=frame_indexes)
      stream_group.create_dataset('xsens_time_since_start_s', [num_frames, 1],
                                  data=times_since_start_s)
      stream_group.create_dataset('time_s', [num_frames, 1], dtype='float64',
                                  data=times_s)
      stream_group.create_dataset('time_str', [num_frames, 1], dtype='S26',
                                  data=times_str)
      # Copy the original device-level and stream-level metadata.
      if device_name in hdf5_file_archived:
        archived_device_group = hdf5_file_archived[device_name]
        device_group_metadata_original = dict(archived_device_group.attrs.items())
        hdf5_file_toUpdate[device_name].attrs.update(device_group_metadata_original)
        if stream_name in archived_device_group:
          archived_stream_group = archived_device_group[stream_name]
          stream_group_metadata_original = dict(archived_stream_group.attrs.items())
          hdf5_file_toUpdate[device_name][stream_name].attrs.update(stream_group_metadata_original)
      else:
        # Create a basic device-level metadata if there was none to copy from the original file.
        device_group_metadata = {SensorStreamer.metadata_class_name_key: type(self).__name__}
        hdf5_file_toUpdate[device_name].attrs.update(device_group_metadata)
      # Override with provided stream-level metadata if desired.
      if stream_group_metadata is not None:
        hdf5_file_toUpdate[device_name][stream_name].attrs.update(
            convert_dict_values_to_str(stream_group_metadata, preserve_nested_dicts=False))
    
    # TODO merge previous and current metadata approaches.
    # # Define the data notes to use for each stream.
    # self._define_data_notes()
    
    # Import the data!
    self._log_debug('Importing data to the new HDF5 file')
    
    # Segment orientation data (quaternion)
    add_hdf5_data(device_name='xsens-segments',
                  stream_name='body_orientation_quaternion_wijk',
                  data=segment_orientations_body_quaternion_wijk,
                  target_data_shape_per_frame=[-1, 4], # 4-element quaternion per segment per frame
                  stream_group_metadata=metadata_segments)
    if have_fingers_left:
      add_hdf5_data(device_name='xsens-segments',
                    stream_name='fingersLeft_orientation_quaternion_wijk',
                    data=segment_orientations_fingersLeft_quaternion_wijk,
                    target_data_shape_per_frame=[-1, 4], # 4-element quaternion per segment per frame
                    stream_group_metadata=metadata_segments)
    if have_fingers_right:
      add_hdf5_data(device_name='xsens-segments',
                    stream_name='fingersRight_orientation_quaternion_wijk',
                    data=segment_orientations_fingersRight_quaternion_wijk,
                    target_data_shape_per_frame=[-1, 4], # 4-element quaternion per segment per frame
                    stream_group_metadata=metadata_segments)
    # Segment orientation data (Euler)
    add_hdf5_data(device_name='xsens-segments',
                  stream_name='body_orientation_eulerZXY_xyz_rad',
                  data=segment_orientations_body_eulerZXY_xyz_rad,
                  target_data_shape_per_frame=[-1, 3], # 3-element Euler vector per segment per frame
                  stream_group_metadata=metadata_segments)
    if have_fingers_left:
      add_hdf5_data(device_name='xsens-segments',
                    stream_name='fingersLeft_orientation_eulerZXY_xyz_rad',
                    data=segment_orientations_fingersLeft_eulerZXY_xyz_rad,
                    target_data_shape_per_frame=[-1, 3], # 3-element Euler vector per segment per frame
                    stream_group_metadata=metadata_segments)
    if have_fingers_right:
      add_hdf5_data(device_name='xsens-segments',
                    stream_name='fingersRight_orientation_eulerZXY_xyz_rad',
                    data=segment_orientations_fingersRight_eulerZXY_xyz_rad,
                    target_data_shape_per_frame=[-1, 3], # 3-element Euler vector per segment per frame
                    stream_group_metadata=metadata_segments)

    # Segment position data.
    add_hdf5_data(device_name='xsens-segments',
                  stream_name='body_position_xyz_m',
                  data=segment_positions_body_m,
                  target_data_shape_per_frame=[-1, 3], # 3-element x/y/z vector per segment per frame
                  stream_group_metadata=metadata_segments)
    if have_fingers_left:
      add_hdf5_data(device_name='xsens-segments',
                    stream_name='fingersLeft_position_xyz_m',
                    data=segment_positions_fingersLeft_m,
                    target_data_shape_per_frame=[-1, 3], # 3-element x/y/z vector per segment per frame
                    stream_group_metadata=metadata_segments)
    if have_fingers_right:
      add_hdf5_data(device_name='xsens-segments',
                    stream_name='fingersRight_position_xyz_m',
                    data=segment_positions_fingersRight_m,
                    target_data_shape_per_frame=[-1, 3], # 3-element x/y/z vector per segment per frame
                    stream_group_metadata=metadata_segments)
    # Segment velocity data.
    add_hdf5_data(device_name='xsens-segments',
                  stream_name='body_velocity_xyz_m_s',
                  data=segment_velocities_body_m_s,
                  target_data_shape_per_frame=[-1, 3], # 3-element x/y/z vector per segment per frame
                  stream_group_metadata=metadata_segments)
    # Segment acceleration data.
    add_hdf5_data(device_name='xsens-segments',
                  stream_name='body_acceleration_xyz_m_ss',
                  data=segment_accelerations_body_m_ss,
                  target_data_shape_per_frame=[-1, 3], # 3-element x/y/z vector per segment per frame
                  stream_group_metadata=metadata_segments)
    # Segment angular velocity.
    add_hdf5_data(device_name='xsens-segments',
                  stream_name='body_angular_velocity_xyz_rad_s',
                  data=segment_angular_velocities_body_rad_s,
                  target_data_shape_per_frame=[-1, 3], # 3-element x/y/z vector per segment per frame
                  stream_group_metadata=metadata_segments)
    # Segment angular acceleration.
    add_hdf5_data(device_name='xsens-segments',
                  stream_name='body_angular_acceleration_xyz_rad_ss',
                  data=segment_angular_accelerations_body_rad_ss,
                  target_data_shape_per_frame=[-1, 3], # 3-element x/y/z vector per segment per frame
                  stream_group_metadata=metadata_segments)

    # Joint angles ZXY.
    add_hdf5_data(device_name='xsens-joints',
                  stream_name='body_joint_angles_eulerZXY_xyz_rad',
                  data=joint_angles_body_eulerZXY_xyz_rad,
                  target_data_shape_per_frame=[-1, 3], # 3-element vector per segment per frame
                  stream_group_metadata=metadata_joints)
    if have_fingers_left:
      add_hdf5_data(device_name='xsens-joints',
                    stream_name='fingersLeft_joint_angles_eulerZXY_xyz_rad',
                    data=joint_angles_fingersLeft_eulerZXY_xyz_rad,
                    target_data_shape_per_frame=[-1, 3], # 3-element vector per segment per frame
                    stream_group_metadata=metadata_joints)
    if have_fingers_right:
      add_hdf5_data(device_name='xsens-joints',
                    stream_name='fingersRight_joint_angles_eulerZXY_xyz_rad',
                    data=joint_angles_fingersRight_eulerZXY_xyz_rad,
                    target_data_shape_per_frame=[-1, 3], # 3-element vector per segment per frame
                    stream_group_metadata=metadata_joints)
    # Joint angles XZY.
    add_hdf5_data(device_name='xsens-joints',
                  stream_name='body_joint_angles_eulerXZY_xyz_rad',
                  data=joint_angles_body_eulerXZY_xyz_rad,
                  target_data_shape_per_frame=[-1, 3], # 3-element vector per segment per frame
                  stream_group_metadata=metadata_joints)
    if have_fingers_left:
      add_hdf5_data(device_name='xsens-joints',
                  stream_name='fingersLeft_joint_angles_eulerXZY_xyz_rad',
                  data=joint_angles_fingersLeft_eulerXZY_xyz_rad,
                  target_data_shape_per_frame=[-1, 3], # 3-element vector per segment per frame
                  stream_group_metadata=metadata_joints)
    if have_fingers_right:
      add_hdf5_data(device_name='xsens-joints',
                  stream_name='fingersRight_joint_angles_eulerXZY_xyz_rad',
                  data=joint_angles_fingersRight_eulerXZY_xyz_rad,
                  target_data_shape_per_frame=[-1, 3], # 3-element vector per segment per frame
                  stream_group_metadata=metadata_joints)
    # Ergonomic joint angles ZXY.
    add_hdf5_data(device_name='xsens-ergonomic-joints',
                  stream_name='ergonomic_joint_angles_eulerZXY_xyz_rad',
                  data=joint_angles_ergonomic_eulerZXY_xyz_rad,
                  target_data_shape_per_frame=[-1, 3], # 3-element vector per segment per frame
                  stream_group_metadata=metadata_joints)
    # Ergonomic joint angles XZY.
    add_hdf5_data(device_name='xsens-ergonomic-joints',
                  stream_name='ergonomic_joint_angles_eulerXZY_xyz_rad',
                  data=joint_angles_ergonomic_eulerXZY_xyz_rad,
                  target_data_shape_per_frame=[-1, 3], # 3-element vector per segment per frame
                  stream_group_metadata=metadata_joints)

    # Center of mass position.
    add_hdf5_data(device_name='xsens-CoM',
                  stream_name='position_xyz_m',
                  data=center_of_mass_positions_m,
                  target_data_shape_per_frame=[3], # 3-element x/y/z vector per frame
                  stream_group_metadata=metadata_com)
    # Center of mass velocity.
    add_hdf5_data(device_name='xsens-CoM',
                  stream_name='velocity_xyz_m_s',
                  data=center_of_mass_velocities_m_s,
                  target_data_shape_per_frame=[3], # 3-element x/y/z vector per frame
                  stream_group_metadata=metadata_com)
    # Center of mass acceleration.
    add_hdf5_data(device_name='xsens-CoM',
                  stream_name='acceleration_xyz_m_ss',
                  data=center_of_mass_accelerations_m_ss,
                  target_data_shape_per_frame=[3], # 3-element x/y/z vector per frame
                  stream_group_metadata=metadata_com)

    # Sensor data - acceleration
    add_hdf5_data(device_name='xsens-sensors',
                  stream_name='free_acceleration_xyz_m_ss',
                  data=sensor_freeAccelerations_m_ss,
                  target_data_shape_per_frame=[-1, 3], # 3-element x/y/z vector per segment per frame
                  stream_group_metadata=metadata_sensors)
    # Sensor data - magnetic field
    add_hdf5_data(device_name='xsens-sensors',
                  stream_name='magnetic_field_xyz_au',
                  data=sensor_magnetic_fields_au,
                  target_data_shape_per_frame=[-1, 3], # 3-element x/y/z vector per segment per frame
                  stream_group_metadata=metadata_sensors)
    # Sensor data - orientation
    add_hdf5_data(device_name='xsens-sensors',
                  stream_name='sensor_orientation_quaternion_wijk',
                  data=sensor_orientations_quaternion_wijk,
                  target_data_shape_per_frame=[-1, 4], # 4-element quaternion vector per segment per frame
                  stream_group_metadata=metadata_sensors)
    # Sensor data - orientation
    add_hdf5_data(device_name='xsens-sensors',
                  stream_name='sensor_orientation_eulerZXY_xyz_rad',
                  data=sensor_orientations_eulerZXY_xyz_rad,
                  target_data_shape_per_frame=[-1, 3], # 3-element x/y/z vector per segment per frame
                  stream_group_metadata=metadata_sensors)

    # Foot contacts
    add_hdf5_data(device_name='xsens-foot-contacts',
                  stream_name='is_contacting_ground',
                  data=foot_contacts,
                  target_data_shape_per_frame=[4], # 4-element contact vector per segment per frame
                  stream_group_metadata=metadata_footContacts)
    
    # Calibration poses.
    device_group = hdf5_file_toUpdate.create_group('xsens-segments-tpose')
    device_group.create_dataset('body_orientation_identity_quaternion_wijk',
                                data=calibration_data['identity']['segment_orientations_body_quaternion'].reshape((-1, 4)))
    device_group.create_dataset('body_orientation_Tpose_quaternion_wijk',
                                data=calibration_data['Tpose']['segment_orientations_body_quaternion'].reshape((-1, 4)))
    device_group.create_dataset('body_orientation_TposeISB_quaternion_wijk',
                                data=calibration_data['Tpose_ISB']['segment_orientations_body_quaternion'].reshape((-1, 4)))
    device_group.create_dataset('body_position_identity_xyz_m',
                                data=calibration_data['identity']['segment_positions_body_m'].reshape((-1, 3)))
    device_group.create_dataset('body_position_Tpose_xyz_m',
                                data=calibration_data['Tpose']['segment_positions_body_m'].reshape((-1, 3)))
    device_group.create_dataset('body_position_TposeISB_xyz_m',
                                data=calibration_data['Tpose_ISB']['segment_positions_body_m'].reshape((-1, 3)))
  
    # # System time
    # add_hdf5_data(device_name='xsens-time',
    #               stream_name='stream_receive_time_s',
    #               data=system_times_s,
    #               target_data_shape_per_frame=[1],
    #               stream_group_metadata=self._data_notes_mvnx['xsens-time']['stream_receive_time_s'])
  
  
  
  
  
  #####################################
  ###### DATA NOTES AND HEADINGS ######
  #####################################
  
  def _define_data_notes(self):
    self._define_data_headings()
    
    ######
    # Notes for streaming data.
    ######
    
    self._data_notes_stream = {}
    self._data_notes_stream.setdefault('xsens-segments', {})
    self._data_notes_stream.setdefault('xsens-joints', {})
    self._data_notes_stream.setdefault('xsens-CoM', {})
    self._data_notes_stream.setdefault('xsens-time', {})
    
    # Segments
    self._data_notes_stream['xsens-segments']['position_cm'] = OrderedDict([
      ('Units', 'cm'),
      ('Coordinate frame', 'A Y-up right-handed frame if Euler data is streamed, otherwise a Z-up right-handed frame'),
      ('Matrix ordering', 'To align with data headings, unwrap a frame\'s matrix as data[frame_index][0][0], data[frame_index][0][1], data[frame_index][0][2], data[frame_index][1][0], ...' \
       + '   | And if no fingers were included in the data, only use the first 69 data headings (the first 23 segments)'),
      (SensorStreamer.metadata_data_headings_key, self._headings['xsens-segments']['position_cm'])
    ])
    self._data_notes_stream['xsens-segments']['orientation_euler_deg'] = OrderedDict([
      ('Units', 'degrees'),
      ('Coordinate frame', 'A Y-Up, right-handed coordinate system'),
      ('Matrix ordering', 'To align with data headings, unwrap a frame\'s matrix as data[frame_index][0][0], data[frame_index][0][1], data[frame_index][0][2], data[frame_index][1][0], ...' \
       + '   | And if no fingers were included in the data, only use the first 69 data headings (the first 23 segments)'),
      (SensorStreamer.metadata_data_headings_key, self._headings['xsens-segments']['orientation_euler_deg']),
      # ('Developer note', 'Streamed data did not seem to match Excel data exported from Xsens; on recent tests it was close, while on older tests it seemed very different.'),
    ])
    self._data_notes_stream['xsens-segments']['orientation_quaternion'] = OrderedDict([
      ('Coordinate frame', 'A Z-Up, right-handed coordinate system'),
      ('Normalization', 'Normalized but not necessarily positive-definite'),
      ('Matrix ordering', 'To align with data headings, unwrap a frame\'s matrix as data[frame_index][0][0], data[frame_index][0][1], data[frame_index][0][2], data[frame_index][0][3], data[frame_index][1][0], ...' \
       + '   | And if no fingers were included in the data, only use the first 92 data headings (the first 23 segments)'),
      (SensorStreamer.metadata_data_headings_key, self._headings['xsens-segments']['orientation_quaternion'])
    ])
    # Joints
    self._data_notes_stream['xsens-joints']['rotation_deg'] = OrderedDict([
      ('Units', 'degrees'),
      ('Coordinate frame', 'A Z-Up, right-handed coordinate system'),
      ('Matrix ordering', 'To align with data headings, unwrap a frame\'s matrix as data[frame_index][0][0], data[frame_index][0][1], data[frame_index][0][2], data[frame_index][1][0], ...'),
      ('Joint parents - segment IDs',    self._headings['xsens-joints']['joint_rotation_streamed_parents_segmentIDs']['streamed']),
      ('Joint parents - segment Names',  self._headings['xsens-joints']['joint_rotation_streamed_parents_segmentNames']['streamed']),
      ('Joint parents - point IDs',      self._headings['xsens-joints']['joint_rotation_streamed_parents_pointIDs']['streamed']),
      ('Joint children - segment IDs',   self._headings['xsens-joints']['joint_rotation_streamed_children_segmentIDs']['streamed']),
      ('Joint children - segment Names', self._headings['xsens-joints']['joint_rotation_streamed_children_segmentNames']['streamed']),
      ('Joint children - point IDs',     self._headings['xsens-joints']['joint_rotation_streamed_children_pointIDs']['streamed']),
      ('Segment ID to Name mapping', self._headings['xsens-joints']['segmentIDsToNames']),
      (SensorStreamer.metadata_data_headings_key, self._headings['xsens-joints']['joint_rotation_names_streamed'])
    ])
    self._data_notes_stream['xsens-joints']['parent'] = OrderedDict([
      ('Format', 'segmentID.pointID'),
      ('Segment ID to Name mapping', self._headings['xsens-joints']['segmentIDsToNames']),
      (SensorStreamer.metadata_data_headings_key, self._headings['xsens-joints']['joint_names_streamed'])
    ])
    self._data_notes_stream['xsens-joints']['child'] = OrderedDict([
      ('Format', 'segmentID.pointID'),
      ('Segment ID to Name mapping', self._headings['xsens-joints']['segmentIDsToNames']),
      (SensorStreamer.metadata_data_headings_key, self._headings['xsens-joints']['joint_names_streamed'])
    ])
    # Center of mass
    self._data_notes_stream['xsens-CoM']['position_cm'] = OrderedDict([
      ('Units', 'cm'),
      ('Coordinate frame', 'A Z-up, right-handed coordinate system'),
      (SensorStreamer.metadata_data_headings_key, self._headings['xsens-CoM']['position_cm'])
    ])
    self._data_notes_stream['xsens-CoM']['velocity_cm_s'] = OrderedDict([
      ('Units', 'cm/s'),
      ('Coordinate frame', 'A Z-up, right-handed coordinate system'),
      (SensorStreamer.metadata_data_headings_key, self._headings['xsens-CoM']['velocity_cm_s'])
    ])
    self._data_notes_stream['xsens-CoM']['acceleration_cm_ss'] = OrderedDict([
      ('Units', 'cm/s/s'),
      ('Coordinate frame', 'A Z-up, right-handed coordinate system'),
      (SensorStreamer.metadata_data_headings_key, self._headings['xsens-CoM']['acceleration_cm_ss'])
    ])
    # Time
    self._data_notes_stream['xsens-time']['device_timestamp_s'] = OrderedDict([
      ('Description', 'The timestamp recorded by the Xsens device, which is more precise than the system time when the data was received (the time_s field)'),
    ])

    ######
    # Notes for data imported from Excel exports.
    ######

    self._data_notes_excel = {}
    self._data_notes_excel.setdefault('xsens-segments', {})
    self._data_notes_excel.setdefault('xsens-joints', {})
    self._data_notes_excel.setdefault('xsens-ergonomic-joints', {})
    self._data_notes_excel.setdefault('xsens-CoM', {})
    self._data_notes_excel.setdefault('xsens-sensors', {})
    self._data_notes_excel.setdefault('xsens-time', {})

    # Segments
    self._data_notes_excel['xsens-segments']['position_cm'] = self._data_notes_stream['xsens-segments']['position_cm'].copy()
    self._data_notes_excel['xsens-segments']['position_cm']['Coordinate frame'] = 'A Z-up right-handed frame'
    
    self._data_notes_excel['xsens-segments']['velocity_cm_s'] = self._data_notes_excel['xsens-segments']['position_cm'].copy()
    self._data_notes_excel['xsens-segments']['velocity_cm_s']['Units'] = 'cm/s'
    self._data_notes_excel['xsens-segments']['velocity_cm_s']['Matrix ordering'] = 'To align with data headings, unwrap a frame\'s matrix as data[frame_index][0][0], data[frame_index][0][1], data[frame_index][0][2], data[frame_index][1][0], ...' \
     + '   | And only use the first 69 data headings (the first 23 segments)'
    
    self._data_notes_excel['xsens-segments']['acceleration_cm_ss'] = self._data_notes_excel['xsens-segments']['velocity_cm_s'].copy()
    self._data_notes_excel['xsens-segments']['acceleration_cm_ss']['Units'] = 'cm/s/s'
    
    self._data_notes_excel['xsens-segments']['angular_velocity_deg_s'] = self._data_notes_excel['xsens-segments']['velocity_cm_s'].copy()
    self._data_notes_excel['xsens-segments']['angular_velocity_deg_s']['Units'] = 'degrees/s'
    
    self._data_notes_excel['xsens-segments']['angular_acceleration_deg_ss'] = self._data_notes_excel['xsens-segments']['velocity_cm_s'].copy()
    self._data_notes_excel['xsens-segments']['angular_acceleration_deg_ss']['Units'] = 'degrees/s/s'

    self._data_notes_excel['xsens-segments']['orientation_euler_deg'] = self._data_notes_stream['xsens-segments']['orientation_euler_deg'].copy()
    self._data_notes_excel['xsens-segments']['orientation_quaternion'] = self._data_notes_stream['xsens-segments']['orientation_quaternion'].copy()
    
    # Joints
    self._data_notes_excel['xsens-joints']['rotation_zxy_deg'] = self._data_notes_stream['xsens-joints']['rotation_deg'].copy()
    self._data_notes_excel['xsens-joints']['rotation_zxy_deg'][SensorStreamer.metadata_data_headings_key] = self._headings['xsens-joints']['joint_rotation_names_bodyFingers']
    self._data_notes_excel['xsens-joints']['rotation_zxy_deg']['Matrix ordering'] = 'To align with data headings, unwrap a frame\'s matrix as data[frame_index][0][0], data[frame_index][0][1], data[frame_index][0][2], data[frame_index][1][0], ...' \
            + '   | And if no fingers were included in the data, only use the first 66 data headings (the first 22 joints)'
    self._data_notes_excel['xsens-joints']['rotation_zxy_deg']['Joint parents - segment IDs']    = self._headings['xsens-joints']['joint_rotation_streamed_parents_segmentIDs']['body']
    self._data_notes_excel['xsens-joints']['rotation_zxy_deg']['Joint parents - segment Names']  = self._headings['xsens-joints']['joint_rotation_streamed_parents_segmentNames']['body']
    self._data_notes_excel['xsens-joints']['rotation_zxy_deg']['Joint parents - point IDs']      = self._headings['xsens-joints']['joint_rotation_streamed_parents_pointIDs']['body']
    self._data_notes_excel['xsens-joints']['rotation_zxy_deg']['Joint children - segment IDs']   = self._headings['xsens-joints']['joint_rotation_streamed_children_segmentIDs']['body']
    self._data_notes_excel['xsens-joints']['rotation_zxy_deg']['Joint children - segment Names'] = self._headings['xsens-joints']['joint_rotation_streamed_children_segmentNames']['body']
    self._data_notes_excel['xsens-joints']['rotation_zxy_deg']['Joint children - point IDs']     = self._headings['xsens-joints']['joint_rotation_streamed_children_pointIDs']['body']

    self._data_notes_excel['xsens-joints']['rotation_xzy_deg'] = self._data_notes_excel['xsens-joints']['rotation_zxy_deg'].copy()
    
    self._data_notes_excel['xsens-ergonomic-joints']['rotation_zxy_deg'] = self._data_notes_excel['xsens-joints']['rotation_zxy_deg'].copy()
    self._data_notes_excel['xsens-ergonomic-joints']['rotation_zxy_deg']['Matrix ordering'] = 'To align with data headings, unwrap a frame\'s matrix as data[frame_index][0][0], data[frame_index][0][1], data[frame_index][0][2], data[frame_index][1][0], ...'
    self._data_notes_excel['xsens-ergonomic-joints']['rotation_zxy_deg'][SensorStreamer.metadata_data_headings_key] = self._headings['xsens-joints']['joint_rotation_names_ergonomic']
    self._data_notes_excel['xsens-ergonomic-joints']['rotation_zxy_deg']['Joint parents - segment IDs']    = self._headings['xsens-joints']['joint_rotation_streamed_parents_segmentIDs']['ergonomic']
    self._data_notes_excel['xsens-ergonomic-joints']['rotation_zxy_deg']['Joint parents - segment Names']  = self._headings['xsens-joints']['joint_rotation_streamed_parents_segmentNames']['ergonomic']
    self._data_notes_excel['xsens-ergonomic-joints']['rotation_zxy_deg']['Joint parents - point IDs']      = self._headings['xsens-joints']['joint_rotation_streamed_parents_pointIDs']['ergonomic']
    self._data_notes_excel['xsens-ergonomic-joints']['rotation_zxy_deg']['Joint children - segment IDs']   = self._headings['xsens-joints']['joint_rotation_streamed_children_segmentIDs']['ergonomic']
    self._data_notes_excel['xsens-ergonomic-joints']['rotation_zxy_deg']['Joint children - segment Names'] = self._headings['xsens-joints']['joint_rotation_streamed_children_segmentNames']['ergonomic']
    self._data_notes_excel['xsens-ergonomic-joints']['rotation_zxy_deg']['Joint children - point IDs']     = self._headings['xsens-joints']['joint_rotation_streamed_children_pointIDs']['ergonomic']
    
    self._data_notes_excel['xsens-ergonomic-joints']['rotation_xzy_deg'] = self._data_notes_excel['xsens-ergonomic-joints']['rotation_zxy_deg'].copy()
    
    # Center of mass
    self._data_notes_excel['xsens-CoM']['position_cm'] = self._data_notes_stream['xsens-CoM']['position_cm'].copy()
    self._data_notes_excel['xsens-CoM']['velocity_cm_s'] = self._data_notes_stream['xsens-CoM']['velocity_cm_s'].copy()
    self._data_notes_excel['xsens-CoM']['acceleration_cm_ss'] = self._data_notes_stream['xsens-CoM']['acceleration_cm_ss'].copy()
    
    # Sensors
    self._data_notes_excel['xsens-sensors']['free_acceleration_cm_ss'] = self._data_notes_stream['xsens-segments']['position_cm'].copy()
    self._data_notes_excel['xsens-sensors']['free_acceleration_cm_ss']['Matrix ordering'] = 'To align with data headings, unwrap a frame\'s matrix as data[frame_index][0][0], data[frame_index][0][1], data[frame_index][0][2], data[frame_index][1][0], ...' \
        + '   | And only use data headings for which the data is not all 0 or all NaN'
    self._data_notes_excel['xsens-sensors']['free_acceleration_cm_ss']['Units'] = 'cm/s/s'
    del self._data_notes_excel['xsens-sensors']['free_acceleration_cm_ss']['Coordinate frame']
    self._data_notes_excel['xsens-sensors']['magnetic_field'] = self._data_notes_excel['xsens-sensors']['free_acceleration_cm_ss'].copy()
    self._data_notes_excel['xsens-sensors']['magnetic_field']['Units'] = 'a.u. according to the manual, but more likely gauss based on the magnitudes'
    
    self._data_notes_excel['xsens-sensors']['orientation_quaternion'] = self._data_notes_stream['xsens-segments']['orientation_quaternion'].copy()
    self._data_notes_excel['xsens-sensors']['orientation_quaternion']['Matrix ordering'] = 'To align with data headings, unwrap a frame\'s matrix as data[frame_index][0][0], data[frame_index][0][1], data[frame_index][0][2], data[frame_index][0][3], data[frame_index][1][0], ...' \
        + '   | And only use data headings for which the data is not all 0 or all NaN'
    del self._data_notes_excel['xsens-sensors']['orientation_quaternion']['Coordinate frame']
    
    self._data_notes_excel['xsens-sensors']['orientation_euler_deg'] = self._data_notes_stream['xsens-segments']['orientation_euler_deg'].copy()
    self._data_notes_excel['xsens-sensors']['orientation_euler_deg']['Matrix ordering'] = 'To align with data headings, unwrap a frame\'s matrix as data[frame_index][0][0], data[frame_index][0][1], data[frame_index][0][2], data[frame_index][1][0], ...' \
        + '   | And only use data headings for which the data is not all 0 or all NaN'
    del self._data_notes_excel['xsens-sensors']['orientation_euler_deg']['Coordinate frame']
    
    # Time
    self._data_notes_excel['xsens-time']['stream_receive_time_s'] = OrderedDict([
      ('Description', 'The estimated system time at which each frame was received by Python during live streaming'),
    ])

    ######
    # Notes for data imported from MVNX exports.
    ######
    self._data_notes_mvnx = copy.deepcopy(self._data_notes_excel)
    
    # Update the data headings for the sensors.
    #  The Excel file contains all segment names and has hidden columns of 0 for ones that don't have sensors,
    #  while the MVNX only lists actual sensor locations.
    for sensors_key in self._data_notes_mvnx['xsens-sensors'].keys():
      if 'quaternion' in sensors_key:
        self._data_notes_mvnx['xsens-sensors'][sensors_key][SensorStreamer.metadata_data_headings_key] = self._headings['xsens-sensors']['sensors-quaternion']
        self._data_notes_mvnx['xsens-sensors'][sensors_key]['Matrix ordering'] = 'To align with data headings, unwrap a frame\'s matrix as data[frame_index][0][0], data[frame_index][0][1], data[frame_index][0][2], data[frame_index][0][3], data[frame_index][1][0], ...'
      else:
        self._data_notes_mvnx['xsens-sensors'][sensors_key][SensorStreamer.metadata_data_headings_key] = self._headings['xsens-sensors']['sensors-xyz']
        self._data_notes_mvnx['xsens-sensors'][sensors_key]['Matrix ordering'] = 'To align with data headings, unwrap a frame\'s matrix as data[frame_index][0][0], data[frame_index][0][1], data[frame_index][0][2], data[frame_index][1][0], ...'

    # Foot contacts
    self._data_notes_mvnx.setdefault('xsens-foot-contacts', {})
    self._data_notes_mvnx['xsens-foot-contacts']['foot-contacts'] = OrderedDict([
      ('Description', 'Which points of the foot are estimated to be in contact with the ground'),
      (SensorStreamer.metadata_data_headings_key, self._headings['xsens-foot-contacts']['foot-contacts']),
    ])
    
  def _define_data_headings(self):
    segment_names_body = [
      # Main-body segments
      'Pelvis', 'L5', 'L3', 'T12', 'T8', 'Neck', 'Head',
      'Right Shoulder',  'Right Upper Arm', 'Right Forearm', 'Right Hand',
      'Left Shoulder',   'Left Upper Arm',  'Left Forearm',  'Left Hand',
      'Right Upper Leg', 'Right Lower Leg', 'Right Foot',    'Right Toe',
      'Left Upper Leg',  'Left Lower Leg',  'Left Foot',     'Left Toe',
      ]
      # Note: props 1-4 would be between body and fingers here if there are any
    segment_names_fingers = [
      # Fingers of left hand
      'Left Carpus',            'Left First Metacarpal',         'Left First Proximal Phalange', 'Left First Distal Phalange',
      'Left Second Metacarpal', 'Left Second Proximal Phalange', 'Left Second Middle Phalange',  'Left Second Distal Phalange',
      'Left Third Metacarpal',  'Left Third Proximal Phalange',  'Left Third Middle Phalange',   'Left Third Distal Phalange',
      'Left Fourth Metacarpal', 'Left Fourth Proximal Phalange', 'Left Fourth Middle Phalange',  'Left Fourth Distal Phalange',
      'Left Fifth Metacarpal',  'Left Fifth Proximal Phalange',  'Left Fifth Middle Phalange',   'Left Fifth Distal Phalange',
      # Fingers of right hand
      'Right Carpus',            'Right First Metacarpal',         'Right First Proximal Phalange', 'Right First Distal Phalange',
      'Right Second Metacarpal', 'Right Second Proximal Phalange', 'Right Second Middle Phalange',  'Right Second Distal Phalange',
      'Right Third Metacarpal',  'Right Third Proximal Phalange',  'Right Third Middle Phalange',   'Right Third Distal Phalange',
      'Right Fourth Metacarpal', 'Right Fourth Proximal Phalange', 'Right Fourth Middle Phalange',  'Right Fourth Distal Phalange',
      'Right Fifth Metacarpal',  'Right Fifth Proximal Phalange',  'Right Fifth Middle Phalange',   'Right Fifth Distal Phalange',
    ]
    sensor_names = segment_names_body[0:1] + segment_names_body[4:5] + segment_names_body[6:18] + segment_names_body[19:22]
    joint_rotation_names_body = [
      'L5S1 Lateral Bending',    'L5S1 Axial Bending',     'L5S1 Flexion/Extension',
      'L4L3 Lateral Bending',    'L4L3 Axial Rotation',    'L4L3 Flexion/Extension',
      'L1T12 Lateral Bending',   'L1T12 Axial Rotation',   'L1T12 Flexion/Extension',
      'T9T8 Lateral Bending',    'T9T8 Axial Rotation',    'T9T8 Flexion/Extension',
      'T1C7 Lateral Bending',    'T1C7 Axial Rotation',    'T1C7 Flexion/Extension',
      'C1 Head Lateral Bending', 'C1 Head Axial Rotation', 'C1 Head Flexion/Extension',
      'Right T4 Shoulder Abduction/Adduction', 'Right T4 Shoulder Internal/External Rotation', 'Right T4 Shoulder Flexion/Extension',
      'Right Shoulder Abduction/Adduction',    'Right Shoulder Internal/External Rotation',    'Right Shoulder Flexion/Extension',
      'Right Elbow Ulnar Deviation/Radial Deviation', 'Right Elbow Pronation/Supination', 'Right Elbow Flexion/Extension',
      'Right Wrist Ulnar Deviation/Radial Deviation', 'Right Wrist Pronation/Supination', 'Right Wrist Flexion/Extension',
      'Left T4 Shoulder Abduction/Adduction', 'Left T4 Shoulder Internal/External Rotation', 'Left T4 Shoulder Flexion/Extension',
      'Left Shoulder Abduction/Adduction',    'Left Shoulder Internal/External Rotation',    'Left Shoulder Flexion/Extension',
      'Left Elbow Ulnar Deviation/Radial Deviation', 'Left Elbow Pronation/Supination', 'Left Elbow Flexion/Extension',
      'Left Wrist Ulnar Deviation/Radial Deviation', 'Left Wrist Pronation/Supination', 'Left Wrist Flexion/Extension',
      'Right Hip Abduction/Adduction',       'Right Hip Internal/External Rotation',       'Right Hip Flexion/Extension',
      'Right Knee Abduction/Adduction',      'Right Knee Internal/External Rotation',      'Right Knee Flexion/Extension',
      'Right Ankle Abduction/Adduction',     'Right Ankle Internal/External Rotation',     'Right Ankle Dorsiflexion/Plantarflexion',
      'Right Ball Foot Abduction/Adduction', 'Right Ball Foot Internal/External Rotation', 'Right Ball Foot Flexion/Extension',
      'Left Hip Abduction/Adduction',        'Left Hip Internal/External Rotation',        'Left Hip Flexion/Extension',
      'Left Knee Abduction/Adduction',       'Left Knee Internal/External Rotation',       'Left Knee Flexion/Extension',
      'Left Ankle Abduction/Adduction',      'Left Ankle Internal/External Rotation',      'Left Ankle Dorsiflexion/Plantarflexion',
      'Left Ball Foot Abduction/Adduction',  'Left Ball Foot Internal/External Rotation',  'Left Ball Foot Flexion/Extension',
      ]
    joint_rotation_names_fingers = [
      'Left First CMC Abduction/Adduction',  'Left First CMC Internal/External Rotation',  'Left First CMC Flexion/Extension',
      'Left First MCP Abduction/Adduction',  'Left First MCP Internal/External Rotation',  'Left First MCP Flexion/Extension',
      'Left IP Abduction/Adduction', 'Left IP Internal/External Rotation', 'Left IP Flexion/Extension',
      'Left Second CMC Abduction/Adduction', 'Left Second CMC Internal/External Rotation', 'Left Second CMC Flexion/Extension',
      'Left Second MCP Abduction/Adduction', 'Left Second MCP Internal/External Rotation', 'Left Second MCP Flexion/Extension',
      'Left Second PIP Abduction/Adduction', 'Left Second PIP Internal/External Rotation', 'Left Second PIP Flexion/Extension',
      'Left Second DIP Abduction/Adduction', 'Left Second DIP Internal/External Rotation', 'Left Second DIP Flexion/Extension',
      'Left Third CMC Abduction/Adduction',  'Left Third CMC Internal/External Rotation',  'Left Third CMC Flexion/Extension',
      'Left Third MCP Abduction/Adduction',  'Left Third MCP Internal/External Rotation',  'Left Third MCP Flexion/Extension',
      'Left Third PIP Abduction/Adduction',  'Left Third PIP Internal/External Rotation',  'Left Third PIP Flexion/Extension',
      'Left Third DIP Abduction/Adduction',  'Left Third DIP Internal/External Rotation',  'Left Third DIP Flexion/Extension',
      'Left Fourth CMC Abduction/Adduction', 'Left Fourth CMC Internal/External Rotation', 'Left Fourth CMC Flexion/Extension',
      'Left Fourth MCP Abduction/Adduction', 'Left Fourth MCP Internal/External Rotation', 'Left Fourth MCP Flexion/Extension',
      'Left Fourth PIP Abduction/Adduction', 'Left Fourth PIP Internal/External Rotation', 'Left Fourth PIP Flexion/Extension',
      'Left Fourth DIP Abduction/Adduction', 'Left Fourth DIP Internal/External Rotation', 'Left Fourth DIP Flexion/Extension',
      'Left Fifth CMC Abduction/Adduction',  'Left Fifth CMC Internal/External Rotation',  'Left Fifth CMC Flexion/Extension',
      'Left Fifth MCP Abduction/Adduction',  'Left Fifth MCP Internal/External Rotation',  'Left Fifth MCP Flexion/Extension',
      'Left Fifth PIP Abduction/Adduction',  'Left Fifth PIP Internal/External Rotation',  'Left Fifth PIP Flexion/Extension',
      'Left Fifth DIP Abduction/Adduction',  'Left Fifth DIP Internal/External Rotation',  'Left Fifth DIP Flexion/Extension',
      'Right First CMC Abduction/Adduction', 'Right First CMC Internal/External Rotation', 'Right First CMC Flexion/Extension',
      'Right First MCP Abduction/Adduction', 'Right First MCP Internal/External Rotation', 'Right First MCP Flexion/Extension',
      'Right IP Abduction/Adduction',         'Right IP Internal/External Rotation',         'Right IP Flexion/Extension',
      'Right Second CMC Abduction/Adduction', 'Right Second CMC Internal/External Rotation', 'Right Second CMC Flexion/Extension',
      'Right Second MCP Abduction/Adduction', 'Right Second MCP Internal/External Rotation', 'Right Second MCP Flexion/Extension',
      'Right Second PIP Abduction/Adduction', 'Right Second PIP Internal/External Rotation', 'Right Second PIP Flexion/Extension',
      'Right Second DIP Abduction/Adduction', 'Right Second DIP Internal/External Rotation', 'Right Second DIP Flexion/Extension',
      'Right Third CMC Abduction/Adduction',  'Right Third CMC Internal/External Rotation',  'Right Third CMC Flexion/Extension',
      'Right Third MCP Abduction/Adduction',  'Right Third MCP Internal/External Rotation',  'Right Third MCP Flexion/Extension',
      'Right Third PIP Abduction/Adduction',  'Right Third PIP Internal/External Rotation',  'Right Third PIP Flexion/Extension',
      'Right Third DIP Abduction/Adduction',  'Right Third DIP Internal/External Rotation',  'Right Third DIP Flexion/Extension',
      'Right Fourth CMC Abduction/Adduction', 'Right Fourth CMC Internal/External Rotation', 'Right Fourth CMC Flexion/Extension',
      'Right Fourth MCP Abduction/Adduction', 'Right Fourth MCP Internal/External Rotation', 'Right Fourth MCP Flexion/Extension',
      'Right Fourth PIP Abduction/Adduction', 'Right Fourth PIP Internal/External Rotation', 'Right Fourth PIP Flexion/Extension',
      'Right Fourth DIP Abduction/Adduction', 'Right Fourth DIP Internal/External Rotation', 'Right Fourth DIP Flexion/Extension',
      'Right Fifth CMC Abduction/Adduction',  'Right Fifth CMC Internal/External Rotation',  'Right Fifth CMC Flexion/Extension',
      'Right Fifth MCP Abduction/Adduction',  'Right Fifth MCP Internal/External Rotation',  'Right Fifth MCP Flexion/Extension',
      'Right Fifth PIP Abduction/Adduction',  'Right Fifth PIP Internal/External Rotation',  'Right Fifth PIP Flexion/Extension',
      'Right Fifth DIP Abduction/Adduction',  'Right Fifth DIP Internal/External Rotation',  'Right Fifth DIP Flexion/Extension',
    ]
    joint_rotation_names_ergonomic = [
      'T8_Head Lateral Bending',          'T8_Head Axial Bending',          'T8_Head Flexion/Extension',
      'T8_LeftUpperArm Lateral Bending',  'T8_LeftUpperArm Axial Bending',  'T8_LeftUpperArm Flexion/Extension',
      'T8_RightUpperArm Lateral Bending', 'T8_RightUpperArm Axial Bending', 'T8_RightUpperArm Flexion/Extension',
      'Pelvis_T8 Lateral Bending',        'Pelvis_T8 Axial Bending',        'Pelvis_T8 Flexion/Extension',
      'Vertical_Pelvis Lateral Bending',  'Vertical_Pelvis Axial Bending',  'Vertical_Pelvis Flexion/Extension',
      'Vertical_T8 Lateral Bending',      'Vertical_T8 Axial Bending',      'Vertical_T8 Flexion/Extension',
    ]
    joint_names_body = [
      'L5S1', 'L4L3', 'L1T12', 'T9T8', 'T1C7', 'C1 Head',
      'Right T4 Shoulder', 'Right Shoulder', 'Right Elbow', 'Right Wrist',
      'Left T4 Shoulder',  'Left Shoulder',  'Left Elbow',  'Left Wrist',
      'Right Hip', 'Right Knee', 'Right Ankle', 'Right Ball Foot',
      'Left Hip',  'Left Knee',  'Left Ankle',  'Left Ball Foot',
      ]
    joint_names_fingers = [
      'Left First CMC',   'Left First MCP',   'Left IP',
      'Left Second CMC',  'Left Second MCP',  'Left Second PIP',  'Left Second DIP',
      'Left Third CMC',   'Left Third MCP',   'Left Third PIP',   'Left Third DIP',
      'Left Fourth CMC',  'Left Fourth MCP',  'Left Fourth PIP',  'Left Fourth DIP',
      'Left Fifth CMC',   'Left Fifth MCP',   'Left Fifth PIP',   'Left Fifth DIP',
      'Right First CMC',  'Right First MCP',  'Right IP',
      'Right Second CMC', 'Right Second MCP', 'Right Second PIP', 'Right Second DIP',
      'Right Third CMC',  'Right Third MCP',  'Right Third PIP',  'Right Third DIP',
      'Right Fourth CMC', 'Right Fourth MCP', 'Right Fourth PIP', 'Right Fourth DIP',
      'Right Fifth CMC',  'Right Fifth MCP',  'Right Fifth PIP',  'Right Fifth DIP',
    ]
    joint_names_ergonomic = [
      'T8_Head',
      'T8_LeftUpperArm',
      'T8_RightUpperArm',
      'Pelvis_T8',
      'Vertical_Pelvis',
      'Vertical_T8',
    ]
    # Record the parent/child segment and point for each streamed joint.
    # The long lists were copied from a test data stream.
    joint_parents_segmentIDsPointIDs = [1.002, 2.002, 3.002, 4.002, 5.002, 6.002, 5.003, 8.002, 9.002, 10.002, 5.004, 12.002, 13.002, 14.002, 1.003, 16.002, 17.002, 18.002, 1.004, 20.002, 21.002, 22.002, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0]
    joint_parents_segmentIDs = [int(x) for x in joint_parents_segmentIDsPointIDs]
    joint_parents_pointIDs = [round((x - int(x))*1000) for x in joint_parents_segmentIDsPointIDs]
    joint_children_segmentIDsPointIDs = [2.001, 3.001, 4.001, 5.001, 6.001, 7.001, 8.001, 9.001, 10.001, 11.001, 12.001, 13.001, 14.001, 15.001, 16.001, 17.001, 18.001, 19.001, 20.001, 21.001, 22.001, 23.001, 7.0, 13.0, 9.0, 5.0, 1.0, 5.0]
    joint_children_segmentIDs = [int(x) for x in joint_children_segmentIDsPointIDs]
    joint_children_pointIDs = [round((x - int(x))*1000) for x in joint_children_segmentIDsPointIDs]
    # Convert to dictionaries mapping joint names to segment names and point IDs.
    #  to avoid dealing with orderings and indexes.
    # Note that the segment IDs are 1-indexed.
    joint_parents_segmentIDs = OrderedDict(
        [(joint_names_body[i], joint_parent_segmentID)
          for (i, joint_parent_segmentID) in enumerate(joint_parents_segmentIDs[0:22])]
      + [(joint_names_ergonomic[i], joint_parent_segmentID)
          for (i, joint_parent_segmentID) in enumerate(joint_parents_segmentIDs[22:])]
    )
    joint_parents_segmentNames = OrderedDict(
        [(joint_name, segment_names_body[segmentID-1])
          for (joint_name, segmentID) in joint_parents_segmentIDs.items()]
    )
    joint_parents_pointIDs = OrderedDict(
        [(joint_names_body[i], joint_parent_pointID)
          for (i, joint_parent_pointID) in enumerate(joint_parents_pointIDs[0:22])]
      + [(joint_names_ergonomic[i], joint_parent_pointID)
          for (i, joint_parent_pointID) in enumerate(joint_parents_pointIDs[22:])]
    )
    joint_children_segmentIDs = OrderedDict(
        [(joint_names_body[i], joint_child_segmentID)
          for (i, joint_child_segmentID) in enumerate(joint_children_segmentIDs[0:22])]
      + [(joint_names_ergonomic[i], joint_child_segmentID)
          for (i, joint_child_segmentID) in enumerate(joint_children_segmentIDs[22:])]
    )
    joint_children_segmentNames = OrderedDict(
        [(joint_name, segment_names_body[segmentID-1])
          for (joint_name, segmentID) in joint_children_segmentIDs.items()]
    )
    joint_children_pointIDs = OrderedDict(
        [(joint_names_body[i], joint_child_pointID)
          for (i, joint_child_pointID) in enumerate(joint_children_pointIDs[0:22])]
      + [(joint_names_ergonomic[i], joint_child_pointID)
          for (i, joint_child_pointID) in enumerate(joint_children_pointIDs[22:])]
    )
    # Foot contact points
    foot_contact_names = ['LeftFoot_Heel', 'LeftFoot_Toe', 'RightFoot_Heel', 'RightFoot_Toe']

    self._headings = {}
    # Center of Mass
    self._headings.setdefault('xsens-CoM', {})
    self._headings['xsens-CoM']['position_cm'] = ['x', 'y', 'z']
    self._headings['xsens-CoM']['velocity_cm_s'] = ['x', 'y', 'z']
    self._headings['xsens-CoM']['acceleration_cm_ss'] = ['x', 'y', 'z']
    # Segment orientation - quaternion
    self._headings.setdefault('xsens-segments', {})
    quaternion_elements = ['q0_re', 'q1_i', 'q2_j', 'q3_k']
    self._headings['xsens-segments']['orientation_quaternion'] = \
      ['%s (%s)' % (name, element) for name in (segment_names_body + segment_names_fingers)
                                   for element in quaternion_elements]
    # Segment orientation - Euler
    self._headings.setdefault('xsens-segments', {})
    euler_elements = ['x', 'y', 'z']
    self._headings['xsens-segments']['orientation_euler_deg'] = \
      ['%s (%s)' % (name, element) for name in (segment_names_body + segment_names_fingers)
                                   for element in euler_elements]
    # Segment positions
    self._headings.setdefault('xsens-segments', {})
    position_elements = ['x', 'y', 'z']
    self._headings['xsens-segments']['position_cm'] = \
      ['%s (%s)' % (name, element) for name in (segment_names_body + segment_names_fingers)
                                   for element in position_elements]
    # Sensors
    self._headings.setdefault('xsens-sensors', {})
    sensor_elements = ['x', 'y', 'z']
    self._headings['xsens-sensors']['sensors-xyz'] = \
      ['%s (%s)' % (name, element) for name in (sensor_names)
                                   for element in sensor_elements]
    sensor_elements = ['q0_re', 'q1_i', 'q2_j', 'q3_k']
    self._headings['xsens-sensors']['sensors-quaternion'] = \
      ['%s (%s)' % (name, element) for name in (sensor_names)
                                   for element in sensor_elements]
    # Joint rotation names
    self._headings.setdefault('xsens-joints', {})
    self._headings['xsens-joints']['joint_rotation_names_body'] = joint_rotation_names_body
    self._headings['xsens-joints']['joint_rotation_names_fingers'] = joint_rotation_names_fingers
    self._headings['xsens-joints']['joint_rotation_names_ergonomic'] = joint_rotation_names_ergonomic
    self._headings['xsens-joints']['joint_rotation_names_streamed'] = joint_rotation_names_body + joint_rotation_names_ergonomic
    self._headings['xsens-joints']['joint_rotation_names_bodyFingers'] = joint_rotation_names_body + joint_rotation_names_fingers
    # Joint names (used within the rotation names above)
    self._headings['xsens-joints']['joint_names_body'] = joint_names_body
    self._headings['xsens-joints']['joint_names_fingers'] = joint_names_fingers
    self._headings['xsens-joints']['joint_names_ergonomic'] = joint_names_ergonomic
    self._headings['xsens-joints']['joint_names_streamed'] = joint_names_body + joint_names_ergonomic
    # Joint parent/child segment/point IDs/names
    self._headings.setdefault('xsens-joints', {})
    self._headings['xsens-joints']['joint_rotation_streamed_parents_segmentIDs'] = {
      'streamed' : joint_parents_segmentIDs,
      'body'     : OrderedDict(list(joint_parents_segmentIDs.items())[0:22]),
      'ergonomic': OrderedDict(list(joint_parents_segmentIDs.items())[22:])}
    self._headings['xsens-joints']['joint_rotation_streamed_parents_segmentNames'] = {
      'streamed' : joint_parents_segmentNames,
      'body'     : OrderedDict(list(joint_parents_segmentNames.items())[0:22]),
      'ergonomic': OrderedDict(list(joint_parents_segmentNames.items())[22:])}
    self._headings['xsens-joints']['joint_rotation_streamed_parents_pointIDs'] = {
      'streamed' : joint_parents_pointIDs,
      'body'     : OrderedDict(list(joint_parents_pointIDs.items())[0:22]),
      'ergonomic': OrderedDict(list(joint_parents_pointIDs.items())[22:])}
    self._headings['xsens-joints']['joint_rotation_streamed_children_segmentIDs'] = {
      'streamed' : joint_children_segmentIDs,
      'body'     : OrderedDict(list(joint_children_segmentIDs.items())[0:22]),
      'ergonomic': OrderedDict(list(joint_children_segmentIDs.items())[22:])}
    self._headings['xsens-joints']['joint_rotation_streamed_children_segmentNames'] = {
      'streamed' : joint_children_segmentNames,
      'body'     : OrderedDict(list(joint_children_segmentNames.items())[0:22]),
      'ergonomic': OrderedDict(list(joint_children_segmentNames.items())[22:])}
    self._headings['xsens-joints']['joint_rotation_streamed_children_pointIDs'] = {
      'streamed' : joint_children_pointIDs,
      'body'     : OrderedDict(list(joint_children_pointIDs.items())[0:22]),
      'ergonomic': OrderedDict(list(joint_children_pointIDs.items())[22:])}
    self._headings['xsens-joints']['segmentIDsToNames'] = OrderedDict([(i+1, name) for (i, name) in enumerate(segment_names_body + segment_names_fingers)])
    # Foot contacts
    self._headings.setdefault('xsens-foot-contacts', {})
    self._headings['xsens-foot-contacts']['foot-contacts'] = foot_contact_names
    

#####################
###### TESTING ######
#####################
if __name__ == '__main__':
  duration_s = 60
  xsens_streamer = XsensStreamer(print_status=True, print_debug=False)
  xsens_streamer.connect()
  xsens_streamer.run()
  start_time_s = time.time()
  try:
    while time.time() - start_time_s < duration_s:
      time.sleep(2)
      num_timesteps = xsens_streamer.get_num_timesteps('xsens-time', 'device_timestamp_s')
      print(' Duration: %6.2fs | Timesteps: %4d | FPS: %6.2f' % (time.time() - start_time_s, num_timesteps, ((num_timesteps-1)/(time.time() - start_time_s))))
  except:
    pass
  xsens_streamer.stop()









