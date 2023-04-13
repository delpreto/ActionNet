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
import itertools

from sensor_streamers.SensorStreamer import SensorStreamer
from visualizers.PnsSkeletonVisualizer import PnsSkeletonVisualizer
from visualizers.LinePlotVisualizer import LinePlotVisualizer
from visualizers.HeatmapVisualizer import HeatmapVisualizer

import socket
import re
import numpy as np
import time
from collections import OrderedDict
import traceback
import time
import math as m

################################################
################################################
# A template class for implementing a new sensor.
################################################
################################################
class PnsStreamer(SensorStreamer):

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
                 log_player_options=None, visualization_options=None,
                 print_status=True, print_debug=False, log_history_filepath=None):
        SensorStreamer.__init__(self, streams_info=None,
                                visualization_options=visualization_options,
                                log_player_options=log_player_options,
                                print_status=print_status, print_debug=print_debug,
                                log_history_filepath=log_history_filepath)

        ## TODO: Add a tag here for your sensor that can be used in log messages.
        #        Try to keep it under 10 characters long.
        #        For example, 'myo' or 'scale'.
        self._log_source_tag = 'NeuronStudio'

        ## TODO: Initialize any state that your sensor needs.
        # Initialize counts
        self._num_segments = None  # Neuron Studio sensor에서 사용하려는 segment 개수 정의함

        # Initialize state
        self._buffer = b''
        self._buffer_read_size = 10000
        self._socket = None
        self._pns_sample_index = None  # The current NeuronStudio timestep being processed (each timestep will send multiple messages)
        self._pns_message_start_time_s = None  # When a NeuronStudio message was first received
        self._pns_timestep_receive_time_s = None  # When the first NeuronStudio message for a NeuronStudio timestep was received

        # Specify the NeuronStudio streaming configuration.
        self._pns_network_protocol = 'udp'
        self._pns_network_ip = '127.0.0.5'
        self._pns_network_port = 5005

        ## TODO: Add devices and streams to organize data from your sensor.
        #        Data is organized as devices and then streams.
        #        For example, a Myo device may have streams for EMG and Acceleration.
        #        If desired, this could also be done in the connect() method instead.
        self.add_stream(device_name='pns-joint-quaternion',
                        stream_name='angle-values',
                        data_type='float32',
                        sample_size=[84],
                        # the size of data saved for each timestep; here, we expect a 2-element vector per timestep
                        sampling_rate_hz=240,  # the expected sampling rate for the stream
                        extra_data_info={},
                        # can add extra information beyond the data and the timestamp if needed (probably not needed, but see MyoStreamer for an example if desired)
                        # Notes can add metadata about the stream,
                        #  such as an overall description, data units, how to interpret the data, etc.
                        # The SensorStreamer.metadata_data_headings_key is special, and is used to
                        #  describe the headings for each entry in a timestep's data.
                        #  For example - if the data was saved in a spreadsheet with a row per timestep, what should the column headings be.
                        data_notes=OrderedDict([
                            ('Description', 'Joint Pos/Angle data from the hip joint.'
                             ),
                            ('Units', ''),
                            (SensorStreamer.metadata_data_headings_key,
                             ['Hip-LocalRot-X', 'Hip-LocalRot-Y', 'Hip-LocalRot-Z', 'Hip-LocalRot-W',
                              'RightUpLeg-LocalRot-X', 'RightUpLeg-LocalRot-Y', 'RightUpLeg-LocalRot-Z',
                              'RightUpLeg-LocalRot-W',
                              'RightLeg-LocalRot-X', 'RightLeg-LocalRot-Y', 'RightLeg-LocalRot-Z',
                              'RightLeg-LocalRot-W',
                              'RightFoot-LocalRot-X', 'RightFoot-LocalRot-Y', 'RightFoot-LocalRot-Z',
                              'RightFoot-LocalRot-W',
                              'LeftUpLeg-LocalRot-X', 'LeftUpLeg-LocalRot-Y', 'LeftUpLeg-LocalRot-Z',
                              'LeftUpLeg-LocalRot-W',
                              'LeftLeg-LocalRot-X', 'LeftLeg-LocalRot-Y', 'LeftLeg-LocalRot-Z', 'LeftLeg-LocalRot-W',
                              'LeftFoot-LocalRot-X', 'LeftFoot-LocalRot-Y', 'LeftFoot-LocalRot-Z',
                              'LeftFoot-LocalRot-W',
                              'Spine-LocalRot-X', 'Spine-LocalRot-Y', 'Spine-LocalRot-Z', 'Spine-LocalRot-W',
                              'Spine1-LocalRot-X', 'Spine1-LocalRot-Y', 'Spine1-LocalRot-Z', 'Spine1-LocalRot-W',
                              'Spine2-LocalRot-X', 'Spine2-LocalRot-Y', 'Spine2-LocalRot-Z', 'Spine2-LocalRot-W',
                              'Neck-LocalRot-X', 'Neck-LocalRot-Y', 'Neck-LocalRot-Z', 'Neck-LocalRot-W',
                              'Neck1-LocalRot-X', 'Neck1-LocalRot-Y', 'Neck1-LocalRot-Z', 'Neck1-LocalRot-W',
                              'Head-LocalRot-X', 'Head-LocalRot-Y', 'Head-LocalRot-Z', 'Head-LocalRot-W',
                              'RightShoulder-LocalRot-X', 'RightShoulder-LocalRot-Y', 'RightShoulder-LocalRot-Z',
                              'RightShoulder-LocalRot-W',
                              'RightArm-LocalRot-X', 'RightArm-LocalRot-Y', 'RightArm-LocalRot-Z',
                              'RightArm-LocalRot-W',
                              'RightForeArm-LocalRot-X', 'RightForeArm-LocalRot-Y', 'RightForeArm-LocalRot-Z',
                              'RightForeArm-LocalRot-W',
                              'RightHand-LocalRot-X', 'RightHand-LocalRot-Y', 'RightHand-LocalRot-Z',
                              'RightHand-LocalRot-W',
                              'LeftShoulder-LocalRot-X', 'LeftShoulder-LocalRot-Y', 'LeftShoulder-LocalRot-Z',
                              'LeftShoulder-LocalRot-W',
                              'LeftArm-LocalRot-X', 'LeftArm-LocalRot-Y', 'LeftArm-LocalRot-Z', 'LeftArm-LocalRot-W',
                              'LeftForeArm-LocalRot-X', 'LeftForeArm-LocalRot-Y', 'LeftForeArm-LocalRot-Z',
                              'LeftForeArm-LocalRot-W',
                              'LeftHand-LocalRot-X', 'LeftHand-LocalRot-Y', 'LeftHand-LocalRot-Z',
                              'LeftHand-LocalRot-W',
                              ]),
                        ]))
        self.add_stream(device_name='pns-joint-euler',
                        stream_name='angle-values',
                        data_type='float32',
                        sample_size=[63],
                        # the size of data saved for each timestep; here, we expect a 2-element vector per timestep
                        sampling_rate_hz=240,  # the expected sampling rate for the stream
                        extra_data_info={},
                        # can add extra information beyond the data and the timestamp if needed (probably not needed, but see MyoStreamer for an example if desired)
                        # Notes can add metadata about the stream,
                        #  such as an overall description, data units, how to interpret the data, etc.
                        # The SensorStreamer.metadata_data_headings_key is special, and is used to
                        #  describe the headings for each entry in a timestep's data.
                        #  For example - if the data was saved in a spreadsheet with a row per timestep, what should the column headings be.
                        data_notes=OrderedDict([
                            ('Description', 'Joint Pos/Angle data from the hip joint.'
                             ),
                            ('Units', ''),
                            (SensorStreamer.metadata_data_headings_key,
                             ['Hip-Euler-X', 'Hip-Euler-Y', 'Hip-Euler-Z',
                              'RightUpLeg-Euler-X', 'RightUpLeg-Euler-Y', 'RightUpLeg-Euler-Z',
                              'RightLeg-Euler-X', 'RightLeg-Euler-Y', 'RightLeg-Euler-Z',
                              'RightFoot-Euler-X', 'RightFoot-Euler-Y', 'RightFoot-Euler-Z',
                              'LeftUpLeg-Euler-X', 'LeftUpLeg-Euler-Y', 'LeftUpLeg-Euler-Z',
                              'LeftLeg-Euler-X', 'LeftLeg-Euler-Y', 'LeftLeg-Euler-Z',
                              'LeftFoot-Euler-X', 'LeftFoot-Euler-Y', 'LeftFoot-Euler-Z',
                              'Spine-Euler-X', 'Spine-Euler-Y', 'Spine-Euler-Z',
                              'Spine1-Euler-X', 'Spine1-Euler-Y', 'Spine1-Euler-Z',
                              'Spine2-Euler-X', 'Spine2-Euler-Y', 'Spine2-Euler-Z',
                              'Neck-Euler-X', 'Neck-Euler-Y', 'Neck-Euler-Z',
                              'Neck1-Euler-X', 'Neck1-Euler-Y', 'Neck1-Euler-Z',
                              'Head-Euler-X', 'Head-Euler-Y', 'Head-Euler-Z',
                              'RightShoulder-Euler-X', 'RightShoulder-Euler-Y', 'RightShoulder-Euler-Z',
                              'RightArm-Euler-X', 'RightArm-Euler-Y', 'RightArm-Euler-Z',
                              'RightForeArm-Euler-X', 'RightForeArm-Euler-Y', 'RightForeArm-Euler-Z',
                              'RightHand-Euler-X', 'RightHand-Euler-Y', 'RightHand-Euler-Z',
                              'LeftShoulder-Euler-X', 'LeftShoulder-Euler-Y', 'LeftShoulder-Euler-Z',
                              'LeftArm-Euler-X', 'LeftArm-Euler-Y', 'LeftArm-Euler-Z',
                              'LeftForeArm-Euler-X', 'LeftForeArm-Euler-Y', 'LeftForeArm-Euler-Z',
                              'LeftHand-Euler-X', 'LeftHand-Euler-Y', 'LeftHand-Euler-Z',
                              ]),
                        ]))
        self.add_stream(device_name='pns-joint-position',
                        stream_name='cm-values',
                        data_type='float32',
                        sample_size=[63],
                        # the size of data saved for each timestep; here, we expect a 2-element vector per timestep
                        sampling_rate_hz=240,  # the expected sampling rate for the stream
                        extra_data_info={},
                        # can add extra information beyond the data and the timestamp if needed (probably not needed, but see MyoStreamer for an example if desired)
                        # Notes can add metadata about the stream,
                        #  such as an overall description, data units, how to interpret the data, etc.
                        # The SensorStreamer.metadata_data_headings_key is special, and is used to
                        #  describe the headings for each entry in a timestep's data.
                        #  For example - if the data was saved in a spreadsheet with a row per timestep, what should the column headings be.
                        data_notes=OrderedDict([
                            ('Description', 'Joint Pos/Angle data from the hip joint.'
                             ),
                            ('Units', ''),
                            (SensorStreamer.metadata_data_headings_key,
                             ['Hip-Position-X', 'Hip-Position-Y', 'Hip-Position-Z',
                              'RightUpLeg-Position-X', 'RightUpLeg-Position-Y', 'RightUpLeg-Position-Z',
                              'RightLeg-Position-X', 'RightLeg-Position-Y', 'RightLeg-Position-Z',
                              'RightFoot-Position-X', 'RightFoot-Position-Y', 'RightFoot-Position-Z',
                              'LeftUpLeg-Position-X', 'LeftUpLeg-Position-Y', 'LeftUpLeg-Position-Z',
                              'LeftLeg-Position-X', 'LeftLeg-Position-Y', 'LeftLeg-Position-Z',
                              'LeftFoot-Position-X', 'LeftFoot-Position-Y', 'LeftFoot-Position-Z',
                              'Spine-Position-X', 'Spine-Position-Y', 'Spine-Position-Z',
                              'Spine1-Position-X', 'Spine1-Position-Y', 'Spine1-Position-Z',
                              'Spine2-Position-X', 'Spine2-Position-Y', 'Spine2-Position-Z',
                              'Neck-Position-X', 'Neck-Position-Y', 'Neck-Position-Z',
                              'Neck1-Position-X', 'Neck1-Position-Y', 'Neck1-Position-Z',
                              'Head-Position-X', 'Head-Position-Y', 'Head-Position-Z',
                              'RightShoulder-Position-X', 'RightShoulder-Position-Y', 'RightShoulder-Position-Z',
                              'RightArm-Position-X', 'RightArm-Position-Y', 'RightArm-Position-Z',
                              'RightForeArm-Position-X', 'RightForeArm-Position-Y', 'RightForeArm-Position-Z',
                              'RightHand-Position-X', 'RightHand-Position-Y', 'RightHand-Position-Z',
                              'LeftShoulder-Position-X', 'LeftShoulder-Position-Y', 'LeftShoulder-Position-Z',
                              'LeftArm-Position-X', 'LeftArm-Position-Y', 'LeftArm-Position-Z',
                              'LeftForeArm-Position-X', 'LeftForeArm-Position-Y', 'LeftForeArm-Position-Z',
                              'LeftHand-Position-X', 'LeftHand-Position-Y', 'LeftHand-Position-Z',
                              ]),
                        ]))

        self.add_stream(device_name='pns-joint-local-position',
                        stream_name='cm-values',
                        data_type='float32',
                        sample_size=[63],
                        # the size of data saved for each timestep; here, we expect a 2-element vector per timestep
                        sampling_rate_hz=240,  # the expected sampling rate for the stream
                        extra_data_info={},
                        # can add extra information beyond the data and the timestamp if needed (probably not needed, but see MyoStreamer for an example if desired)
                        # Notes can add metadata about the stream,
                        #  such as an overall description, data units, how to interpret the data, etc.
                        # The SensorStreamer.metadata_data_headings_key is special, and is used to
                        #  describe the headings for each entry in a timestep's data.
                        #  For example - if the data was saved in a spreadsheet with a row per timestep, what should the column headings be.
                        data_notes=OrderedDict([
                            ('Description', 'Joint Pos/Angle data from the hip joint.'
                             ),
                            ('Units', ''),
                            (SensorStreamer.metadata_data_headings_key,
                             ['Hip-Position-X', 'Hip-Position-Y', 'Hip-Position-Z',
                              'RightUpLeg-Position-X', 'RightUpLeg-Position-Y', 'RightUpLeg-Position-Z',
                              'RightLeg-Position-X', 'RightLeg-Position-Y', 'RightLeg-Position-Z',
                              'RightFoot-Position-X', 'RightFoot-Position-Y', 'RightFoot-Position-Z',
                              'LeftUpLeg-Position-X', 'LeftUpLeg-Position-Y', 'LeftUpLeg-Position-Z',
                              'LeftLeg-Position-X', 'LeftLeg-Position-Y', 'LeftLeg-Position-Z',
                              'LeftFoot-Position-X', 'LeftFoot-Position-Y', 'LeftFoot-Position-Z',
                              'Spine-Position-X', 'Spine-Position-Y', 'Spine-Position-Z',
                              'Spine1-Position-X', 'Spine1-Position-Y', 'Spine1-Position-Z',
                              'Spine2-Position-X', 'Spine2-Position-Y', 'Spine2-Position-Z',
                              'Neck-Position-X', 'Neck-Position-Y', 'Neck-Position-Z',
                              'Neck1-Position-X', 'Neck1-Position-Y', 'Neck1-Position-Z',
                              'Head-Position-X', 'Head-Position-Y', 'Head-Position-Z',
                              'RightShoulder-Position-X', 'RightShoulder-Position-Y', 'RightShoulder-Position-Z',
                              'RightArm-Position-X', 'RightArm-Position-Y', 'RightArm-Position-Z',
                              'RightForeArm-Position-X', 'RightForeArm-Position-Y', 'RightForeArm-Position-Z',
                              'RightHand-Position-X', 'RightHand-Position-Y', 'RightHand-Position-Z',
                              'LeftShoulder-Position-X', 'LeftShoulder-Position-Y', 'LeftShoulder-Position-Z',
                              'LeftArm-Position-X', 'LeftArm-Position-Y', 'LeftArm-Position-Z',
                              'LeftForeArm-Position-X', 'LeftForeArm-Position-Y', 'LeftForeArm-Position-Z',
                              'LeftHand-Position-X', 'LeftHand-Position-Y', 'LeftHand-Position-Z',
                              ]),
                        ]))

        self.add_stream(device_name='pns-joint-angular-velocity',
                        stream_name='velocity-values',
                        data_type='float32',
                        sample_size=[63],
                        # the size of data saved for each timestep; here, we expect a 2-element vector per timestep
                        sampling_rate_hz=240,  # the expected sampling rate for the stream
                        extra_data_info={},
                        # can add extra information beyond the data and the timestamp if needed (probably not needed, but see MyoStreamer for an example if desired)
                        # Notes can add metadata about the stream,
                        #  such as an overall description, data units, how to interpret the data, etc.
                        # The SensorStreamer.metadata_data_headings_key is special, and is used to
                        #  describe the headings for each entry in a timestep's data.
                        #  For example - if the data was saved in a spreadsheet with a row per timestep, what should the column headings be.
                        data_notes=OrderedDict([
                            ('Description', 'Joint Pos/Angle data from the hip joint.'
                             ),
                            ('Units', ''),
                            (SensorStreamer.metadata_data_headings_key,
                             ['Hip-Position-X', 'Hip-Position-Y', 'Hip-Position-Z',
                              'RightUpLeg-Position-X', 'RightUpLeg-Position-Y', 'RightUpLeg-Position-Z',
                              'RightLeg-Position-X', 'RightLeg-Position-Y', 'RightLeg-Position-Z',
                              'RightFoot-Position-X', 'RightFoot-Position-Y', 'RightFoot-Position-Z',
                              'LeftUpLeg-Position-X', 'LeftUpLeg-Position-Y', 'LeftUpLeg-Position-Z',
                              'LeftLeg-Position-X', 'LeftLeg-Position-Y', 'LeftLeg-Position-Z',
                              'LeftFoot-Position-X', 'LeftFoot-Position-Y', 'LeftFoot-Position-Z',
                              'Spine-Position-X', 'Spine-Position-Y', 'Spine-Position-Z',
                              'Spine1-Position-X', 'Spine1-Position-Y', 'Spine1-Position-Z',
                              'Spine2-Position-X', 'Spine2-Position-Y', 'Spine2-Position-Z',
                              'Neck-Position-X', 'Neck-Position-Y', 'Neck-Position-Z',
                              'Neck1-Position-X', 'Neck1-Position-Y', 'Neck1-Position-Z',
                              'Head-Position-X', 'Head-Position-Y', 'Head-Position-Z',
                              'RightShoulder-Position-X', 'RightShoulder-Position-Y', 'RightShoulder-Position-Z',
                              'RightArm-Position-X', 'RightArm-Position-Y', 'RightArm-Position-Z',
                              'RightForeArm-Position-X', 'RightForeArm-Position-Y', 'RightForeArm-Position-Z',
                              'RightHand-Position-X', 'RightHand-Position-Y', 'RightHand-Position-Z',
                              'LeftShoulder-Position-X', 'LeftShoulder-Position-Y', 'LeftShoulder-Position-Z',
                              'LeftArm-Position-X', 'LeftArm-Position-Y', 'LeftArm-Position-Z',
                              'LeftForeArm-Position-X', 'LeftForeArm-Position-Y', 'LeftForeArm-Position-Z',
                              'LeftHand-Position-X', 'LeftHand-Position-Y', 'LeftHand-Position-Z',
                              ]),
                        ]))

    #######################################
    # Connect to the sensor.
    # @param timeout_s How long to wait for the sensor to respond.
    def _connect(self, timeout_s=10):
        # Open a socket to the NeuronStudio network stream
        ## TODO: Add code for connecting to your sensor.
        #        Then return True or False to indicate whether connection was successful.

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self._socket.settimeout(
            5)  # timeout for all socket operations, such as receiving if the NeuronStuido network stream is inactive
        self._socket.bind((self._pns_network_ip, self._pns_network_port))
        self._log_status('Successfully connected to the NueronStudio streamer.')
        return True


    #######################################
    ###### INTERFACE WITH THE SENSOR ######
    #######################################

    ## TODO: Add functions to control your sensor and acquire data.
    #        [Optional but probably useful]

    # A function to read a timestep of data for the first stream.
    def _read_data(self):
        # For example, may want to return the data for the timestep
        #  and the time at which it was received.
        try:
            bytesAddressPair = self._socket.recvfrom(self._buffer_read_size)
        except:
            self._log_error('\n\n***ERROR reading from PnsStreamer:\n%s\n' % traceback.format_exc())
            time.sleep(1)
            return (None, None, None, None, None)

        def _rotation_X(theta):
            return np.matrix([[1, 0, 0],
                              [0, m.cos(m.pi*(theta/180)), -m.sin(m.pi*(theta/180))],
                              [0, m.sin(m.pi*(theta/180)), m.cos(m.pi*(theta/180))]])

        def _rotation_Y(theta):
            return np.matrix([[m.cos(m.pi*(theta/180)), 0, m.sin(m.pi*(theta/180))],
                              [0, 1, 0],
                              [-m.sin(m.pi*(theta/180)), 0, m.cos(m.pi*(theta/180))]])

        def _rotation_Z(theta):
            return np.matrix([[m.cos(m.pi*(theta/180)), -m.sin(m.pi*(theta/180)), 0],
                              [m.sin(m.pi*(theta/180)), m.cos(m.pi*(theta/180)), 0],
                              [0, 0, 1]])

        def quaternion_rotation_matrix(Q):
            """
            Covert a quaternion into a full three-dimensional rotation matrix.

            Input
            :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)

            Output
            :return: A 3x3 element matrix representing the full 3D rotation matrix.
                     This rotation matrix converts a point in the local reference
                     frame to a point in the global reference frame.
            """
            # Extract the values from Q
            q0 = Q[0]
            q1 = Q[1]
            q2 = Q[2]
            q3 = Q[3]

            # First row of the rotation matrix
            r00 = 2 * (q0 * q0 + q1 * q1) - 1
            r01 = 2 * (q1 * q2 - q0 * q3)
            r02 = 2 * (q1 * q3 + q0 * q2)

            # Second row of the rotation matrix
            r10 = 2 * (q1 * q2 + q0 * q3)
            r11 = 2 * (q0 * q0 + q2 * q2) - 1
            r12 = 2 * (q2 * q3 - q0 * q1)

            # Third row of the rotation matrix
            r20 = 2 * (q1 * q3 - q0 * q2)
            r21 = 2 * (q2 * q3 + q0 * q1)
            r22 = 2 * (q0 * q0 + q3 * q3) - 1

            # 3x3 rotation matrix
            rot_matrix = np.array([[r00, r01, r02],
                                   [r10, r11, r12],
                                   [r20, r21, r22]])

            return rot_matrix

        message = bytesAddressPair[0].decode("utf-8")

        # address = bytesAddressPair[1]
        clientMsg = "{}".format(message)
        # clientIP = "Client IP Address:{}".format(address)

        data = clientMsg.split()
        # data = data.split(',')

        # Extract the device timestamp.
        time_s = re.sub(",|'" , "", data[0])
        time_s = float(time_s.replace('[', ''))
        time_s_save = time_s


        jointList = []

        for i in range(21):
            jointList.append(data[11*i + 1])
            jointList.append(float(data[11 * i + 2].split('(')[1].split(',')[0]))
            jointList.append(float(data[11 * i + 3].split(',')[0]))
            jointList.append(float(data[11 * i + 4].split(',')[0]))
            jointList.append(float(data[11 * i + 5].split(')')[0]))
            jointList.append(float(data[11 * i + 6].split('(')[1].split(',')[0]))
            jointList.append(float(data[11 * i + 7].split(',')[0]))
            jointList.append(float(data[11 * i + 8].split(')')[0]))
            jointList.append(float(data[11 * i + 9].split('(')[1].split(',')[0]))
            jointList.append(float(data[11 * i + 10].split(',')[0]))
            jointList.append(float(data[11 * i + 11].split(')')[0]))

        # print(jointList)

        # Joint Data
        data_quaternion = []
        data_euler = []
        data_position = []

        for i in range(len(jointList)):
            if i % 11 == 0:
                continue
            elif (i % 11 == 1) or (i % 11 == 2) or (i % 11 == 3) or (i % 11 == 4):
                data_quaternion.append(jointList[i])
            elif (i % 11 == 5) or (i % 11 == 6) or (i % 11 == 7):
                data_euler.append(jointList[i])
            elif (i % 11 == 8) or (i % 11 == 9) or (i % 11 == 10):
                data_position.append(jointList[i])



        # Before Tuning
        data_position_tun = []


        #Hip
        hip_posi = data_position[0:3]
        hip_euler = data_euler[0:3]
        hip_quater = data_quaternion[0:4]
        data_position_tun.append(hip_posi)

        #RightUpLeg
        RightUpLeg_posi = data_position[3:6]
        RightUpLeg_euler = data_euler[3:6]
        RightUpLeg_quater = data_quaternion[4:8]

        # RightLeg
        RightLeg_posi = data_position[6:9]
        RightLeg_euler = data_euler[6:9]
        RightLeg_quater = data_quaternion[8:12]

        # RightFoot
        RightFoot_posi = data_position[9:12]
        RightFoot_euler = data_euler[9:12]
        RightFoot_quater = data_quaternion[12:16]

        # LeftUpLeg
        LeftUpLeg_posi = data_position[12:15]
        LeftUpLeg_euler = data_euler[12:15]
        LeftUpLeg_quater = data_quaternion[16:20]

        # LeftLeg
        LeftLeg_posi = data_position[15:18]
        LeftLeg_euler = data_euler[15:18]
        LeftLeg_quater = data_quaternion[20:24]

        # LeftFoot
        LeftFoot_posi = data_position[18:21]
        LeftFoot_euler = data_euler[18:21]
        LeftFoot_quater = data_quaternion[24:28]

        # Spine
        Spine_posi = data_position[21:24]
        Spine_euler = data_euler[21:24]
        Spine_quater = data_quaternion[28:32]

        # Spine1
        Spine1_posi = data_position[24:27]
        Spine1_euler = data_euler[24:27]
        Spine1_quater = data_quaternion[32:36]

        # Spine2
        Spine2_posi = data_position[27:30]
        Spine2_euler = data_euler[27:30]
        Spine2_quater = data_quaternion[36:40]

        # Neck
        Neck_posi = data_position[30:33]
        Neck_euler = data_euler[30:33]
        Neck_quater = data_quaternion[40:44]

        # Neck1
        Neck1_posi = data_position[33:36]
        Neck1_euler = data_euler[33:36]
        Neck1_quater = data_quaternion[44:48]

        # Head
        Head_posi = data_position[36:39]
        Head_euler = data_euler[36:39]
        Head_quater = data_quaternion[48:52]

        # RightShoulder
        RightShoulder_posi = data_position[39:42]
        RightShoulder_euler = data_euler[39:42]
        RightShoulder_quater = data_quaternion[52:56]

        # RightArm
        RightArm_posi = data_position[42:45]
        RightArm_euler = data_euler[42:45]
        RightArm_quater = data_quaternion[56:60]

        # RightForeArm
        RightForeArm_posi = data_position[45:48]
        RightForeArm_euler = data_euler[45:48]
        RightForeArm_quater = data_quaternion[60:64]

        # RightHand
        RightHand_posi = data_position[48:51]
        RightHand_euler = data_euler[48:51]
        RightHand_quater = data_quaternion[64:68]

        # LeftShoulder
        LeftShoulder_posi = data_position[51:54]
        LeftShoulder_euler = data_euler[51:54]
        LeftShoulder_quater = data_quaternion[68:72]

        # LeftArm
        LeftArm_posi = data_position[54:57]
        LeftArm_euler = data_euler[54:57]
        LeftArm_quater = data_quaternion[72:76]

        # LeftForeArm
        LeftForeArm_posi = data_position[57:60]
        LeftForeArm_euler = data_euler[57:60]
        LeftForeArm_quater = data_quaternion[76:80]

        # LeftHand
        LeftHand_posi = data_position[60:63]
        LeftHand_euler = data_euler[60:63]
        LeftHand_quater = data_quaternion[80:84]

        # After Tuning
        # RightUpLeg
        # RightUpLeg_posi_tun = _rotation_Y(hip_euler[1]) @ _rotation_Z(hip_euler[2]) @ _rotation_X(
        #     hip_euler[0]) @ np.array(RightUpLeg_posi)
        RightUpLeg_posi_tun = quaternion_rotation_matrix(hip_quater) @ np.array(RightUpLeg_posi)
        RightUpLeg_posi_tun_ = [x+y for x, y in zip(hip_posi, np.ravel(RightUpLeg_posi_tun))]
        data_position_tun.append(RightUpLeg_posi_tun_)
        # print(RightUpLeg_posi_tun_)

        # RightLeg
        # RightLeg_posi_tun = _rotation_Y(RightUpLeg_euler[1]) @ _rotation_Z(RightUpLeg_euler[2]) @ _rotation_X(
        #     RightUpLeg_euler[0]) @ np.array(RightLeg_posi)
        RightLeg_posi_tun = quaternion_rotation_matrix(hip_quater) @ quaternion_rotation_matrix(RightUpLeg_quater) @ np.array(RightLeg_posi)
        RightLeg_posi_tun_ = [x+y for x,y in zip(np.ravel(RightUpLeg_posi_tun_), np.ravel(RightLeg_posi_tun))]
        data_position_tun.append(RightLeg_posi_tun_)

        # RightFoot
        # RightFoot_posi_tun = _rotation_Y(RightLeg_euler[1]) @ _rotation_Z(RightLeg_euler[2]) @ _rotation_X(
        #     RightLeg_euler[0]) @ np.array(RightFoot_posi)
        RightFoot_posi_tun = quaternion_rotation_matrix(hip_quater) @ quaternion_rotation_matrix(RightUpLeg_quater) @ quaternion_rotation_matrix(RightLeg_quater) @ np.array(RightFoot_posi)
        RightFoot_posi_tun_ = [x+y for x,y in zip(np.ravel(RightLeg_posi_tun_), list(np.ravel(RightFoot_posi_tun)))]
        data_position_tun.append(RightFoot_posi_tun_)


        # LeftUpLeg
        # LeftUpLeg_posi_tun = _rotation_Y(hip_euler[1]) @ _rotation_Z(hip_euler[2]) @ _rotation_X(
        #     hip_euler[0]) @ np.array(LeftUpLeg_posi)
        LeftUpLeg_posi_tun = quaternion_rotation_matrix(hip_quater) @ np.array(LeftUpLeg_posi)
        LeftUpLeg_posi_tun_ = [x + y for x, y in zip(hip_posi, list(np.ravel(LeftUpLeg_posi_tun)))]
        data_position_tun.append(LeftUpLeg_posi_tun_)


        # LeftLeg
        # LeftLeg_posi_tun = _rotation_Y(LeftUpLeg_euler[1]) @ _rotation_Z(LeftUpLeg_euler[2]) @ _rotation_X(
        #     LeftUpLeg_euler[0]) @ np.array(LeftLeg_posi)
        LeftLeg_posi_tun = quaternion_rotation_matrix(hip_quater) @ quaternion_rotation_matrix(LeftUpLeg_quater) @ np.array(LeftLeg_posi)
        LeftLeg_posi_tun_ = [x + y for x, y in zip(np.ravel(LeftUpLeg_posi_tun_), list(np.ravel(LeftLeg_posi_tun)))]
        data_position_tun.append(LeftLeg_posi_tun_)

        # LeftFoot
        # LeftFoot_posi_tun = _rotation_Y(LeftLeg_euler[1]) @ _rotation_Z(LeftLeg_euler[2]) @ _rotation_X(
        #     LeftLeg_euler[0]) @ np.array(LeftFoot_posi)
        LeftFoot_posi_tun = quaternion_rotation_matrix(hip_quater) @ quaternion_rotation_matrix(
            LeftUpLeg_quater) @ quaternion_rotation_matrix(
            LeftLeg_quater) @ np.array(LeftFoot_posi)
        LeftFoot_posi_tun_ = [x + y for x, y in zip(np.ravel(LeftLeg_posi_tun_), list(np.ravel(LeftFoot_posi_tun)))]
        data_position_tun.append(LeftFoot_posi_tun_)


        # Spine
        # Spine_posi_tun = _rotation_Y(hip_euler[1]) @ _rotation_Z(hip_euler[2]) @ _rotation_X(
        #     hip_euler[0]) @ np.array(Spine_posi)
        Spine_posi_tun = quaternion_rotation_matrix(hip_quater) @ np.array(Spine_posi)
        Spine_posi_tun_ = [x + y for x, y in zip(hip_posi, list(np.ravel(Spine_posi_tun)))]
        data_position_tun.append(Spine_posi_tun_)


        # Spine1
        # Spine1_posi_tun = _rotation_Y(Spine_euler[1]) @ _rotation_Z(Spine_euler[2]) @ _rotation_X(
        #     Spine_euler[0]) @ np.array(Spine1_posi)
        Spine1_posi_tun = quaternion_rotation_matrix(hip_quater) @ \
                          quaternion_rotation_matrix(Spine_quater) @ np.array(Spine1_posi)
        Spine1_posi_tun_ = [x + y for x, y in zip(np.ravel(Spine_posi_tun_), list(np.ravel(Spine1_posi_tun)))]
        data_position_tun.append(Spine1_posi_tun_)


        # Spine2
        # Spine2_posi_tun = _rotation_Y(Spine1_euler[1]) @ _rotation_Z(Spine1_euler[2]) @ _rotation_X(
        #     Spine1_euler[0]) @ np.array(Spine2_posi)
        Spine2_posi_tun = quaternion_rotation_matrix(hip_quater) @ \
                          quaternion_rotation_matrix(Spine_quater) @\
                          quaternion_rotation_matrix(Spine1_quater) @ np.array(Spine2_posi)
        Spine2_posi_tun_ = [x + y for x, y in zip(np.ravel(Spine1_posi_tun_), list(np.ravel(Spine2_posi_tun)))]
        data_position_tun.append(Spine2_posi_tun_)


        # Neck
        # Neck_posi_tun = _rotation_Y(Spine2_euler[1]) @ _rotation_Z(Spine2_euler[2]) @ _rotation_X(
        #     Spine2_euler[0]) @ np.array(Neck_posi)
        Neck_posi_tun = quaternion_rotation_matrix(hip_quater) @ \
                          quaternion_rotation_matrix(Spine_quater) @ \
                          quaternion_rotation_matrix(Spine1_quater) @\
                          quaternion_rotation_matrix(Spine2_quater) @ np.array(Neck_posi)
        Neck_posi_tun_ = [x + y for x, y in zip(np.ravel(Spine2_posi_tun_), list(np.ravel(Neck_posi_tun)))]
        data_position_tun.append(Neck_posi_tun_)


        # Neck1
        # Neck1_posi_tun = _rotation_Y(Neck_euler[1]) @ _rotation_Z(Neck_euler[2]) @ _rotation_X(
        #     Neck_euler[0]) @ np.array(Neck1_posi)
        Neck1_posi_tun = quaternion_rotation_matrix(hip_quater) @ \
                        quaternion_rotation_matrix(Spine_quater) @ \
                        quaternion_rotation_matrix(Spine1_quater) @ \
                        quaternion_rotation_matrix(Spine2_quater) @ \
                        quaternion_rotation_matrix(Neck_quater) @ np.array(Neck1_posi)
        Neck1_posi_tun_ = [x + y for x, y in zip(np.ravel(Neck_posi_tun_), list(np.ravel(Neck1_posi_tun)))]
        data_position_tun.append(Neck1_posi_tun_)


        # Head
        # Head_posi_tun = _rotation_Y(Neck1_euler[1]) @ _rotation_Z(Neck1_euler[2]) @ _rotation_X(
        #     Neck1_euler[0]) @ np.array(Head_posi)
        Head_posi_tun = quaternion_rotation_matrix(hip_quater) @ \
                         quaternion_rotation_matrix(Spine_quater) @ \
                         quaternion_rotation_matrix(Spine1_quater) @ \
                         quaternion_rotation_matrix(Spine2_quater) @ \
                         quaternion_rotation_matrix(Neck_quater) @\
                         quaternion_rotation_matrix(Neck1_quater) @ np.array(Head_posi)
        Head_posi_tun_ = [x + y for x, y in zip(np.ravel(Neck1_posi_tun_), list(np.ravel(Head_posi_tun)))]
        data_position_tun.append(Head_posi_tun_)


        # RightShoulder
        # RightShoulder_posi_tun = _rotation_Y(Spine2_euler[1]) @ _rotation_Z(Spine2_euler[2]) @ _rotation_X(
        #     Spine2_euler[0]) @ np.array(RightShoulder_posi)
        RightShoulder_posi_tun = quaternion_rotation_matrix(hip_quater) @ \
                        quaternion_rotation_matrix(Spine_quater) @ \
                        quaternion_rotation_matrix(Spine1_quater) @ \
                        quaternion_rotation_matrix(Spine2_quater) @ np.array(RightShoulder_posi)
        RightShoulder_posi_tun_ = [x + y for x, y in zip(np.ravel(Spine2_posi_tun_), list(np.ravel(RightShoulder_posi_tun)))]
        data_position_tun.append(RightShoulder_posi_tun_)

        # RightArm
        # RightArm_posi_tun = _rotation_Y(RightShoulder_euler[1]) @ _rotation_Z(
        #     RightShoulder_euler[2]) @ _rotation_X(
        #     RightShoulder_euler[0]) @ np.array(RightArm_posi)
        RightArm_posi_tun = quaternion_rotation_matrix(hip_quater) @ \
                                 quaternion_rotation_matrix(Spine_quater) @ \
                                 quaternion_rotation_matrix(Spine1_quater) @ \
                                 quaternion_rotation_matrix(Spine2_quater) @ \
                                 quaternion_rotation_matrix(RightShoulder_quater) @np.array(RightArm_posi)
        RightArm_posi_tun_ = [x + y for x, y in zip(np.ravel(RightShoulder_posi_tun_), list(np.ravel(RightArm_posi_tun)))]
        data_position_tun.append(RightArm_posi_tun_)

        # RightForeArm
        # print(RightForeArm_euler)
        # RightForeArm_posi_tun = _rotation_Y(RightArm_euler[1]) @ _rotation_Z(
        #     RightArm_euler[2]) @ _rotation_X(
        #     RightArm_euler[0]) @ np.array(RightForeArm_posi)
        RightForeArm_posi_tun = quaternion_rotation_matrix(hip_quater) @ \
                            quaternion_rotation_matrix(Spine_quater) @ \
                            quaternion_rotation_matrix(Spine1_quater) @ \
                            quaternion_rotation_matrix(Spine2_quater) @ \
                            quaternion_rotation_matrix(RightShoulder_quater) @ \
                            quaternion_rotation_matrix(RightArm_quater) @ np.array(RightForeArm_posi)
        RightForeArm_posi_tun_ = [x + y for x, y in zip(np.ravel(RightArm_posi_tun_), list(np.ravel(RightForeArm_posi_tun)))]
        data_position_tun.append(RightForeArm_posi_tun_)
        # print(data_position_tun)


        # RightHand
        # print(RightHand_euler)
        # RightHand_posi_tun = _rotation_Y(RightForeArm_euler[1]) @ _rotation_Z(
        #     RightForeArm_euler[2]) @ _rotation_X(
        #     RightForeArm_euler[0]) @ np.array(RightHand_posi)
        RightHand_posi_tun = quaternion_rotation_matrix(hip_quater) @ \
                                quaternion_rotation_matrix(Spine_quater) @ \
                                quaternion_rotation_matrix(Spine1_quater) @ \
                                quaternion_rotation_matrix(Spine2_quater) @ \
                                quaternion_rotation_matrix(RightShoulder_quater) @ \
                                quaternion_rotation_matrix(RightArm_quater) @ \
                                quaternion_rotation_matrix(RightForeArm_quater) @np.array(RightHand_posi)
        RightHand_posi_tun_ = [x + y for x, y in zip(np.ravel(RightForeArm_posi_tun_), list(np.ravel(RightHand_posi_tun)))]
        data_position_tun.append(RightHand_posi_tun_)

        # LeftShoulder
        # LeftShoulder_posi_tun = _rotation_Y(Spine2_euler[1]) @ _rotation_Z(
        #     Spine2_euler[2]) @ _rotation_X(
        #     Spine2_euler[0]) @ np.array(LeftShoulder_posi)
        LeftShoulder_posi_tun = quaternion_rotation_matrix(hip_quater) @ \
                                 quaternion_rotation_matrix(Spine_quater) @ \
                                 quaternion_rotation_matrix(Spine1_quater) @ \
                                 quaternion_rotation_matrix(Spine2_quater) @ np.array(LeftShoulder_posi)
        LeftShoulder_posi_tun_ = [x + y for x, y in zip(np.ravel(Spine2_posi_tun_), list(np.ravel(LeftShoulder_posi_tun)))]
        data_position_tun.append(LeftShoulder_posi_tun_)

        # LeftArm
        # LeftArm_posi_tun = _rotation_Y(LeftShoulder_euler[1]) @ _rotation_Z(
        #     LeftShoulder_euler[2]) @ _rotation_X(
        #     LeftShoulder_euler[0]) @ np.array(LeftArm_posi)
        LeftArm_posi_tun = quaternion_rotation_matrix(hip_quater) @ \
                                quaternion_rotation_matrix(Spine_quater) @ \
                                quaternion_rotation_matrix(Spine1_quater) @ \
                                quaternion_rotation_matrix(Spine2_quater) @ \
                                quaternion_rotation_matrix(LeftShoulder_quater) @ np.array(LeftArm_posi)
        LeftArm_posi_tun_ = [x + y for x, y in zip(np.ravel(LeftShoulder_posi_tun_), list(np.ravel(LeftArm_posi_tun)))]
        data_position_tun.append(LeftArm_posi_tun_)

        # LeftForeArm
        # LeftForeArm_posi_tun = _rotation_Y(LeftArm_euler[1]) @ _rotation_Z(
        #     LeftArm_euler[2]) @ _rotation_X(
        #     LeftArm_euler[0]) @ np.array(LeftForeArm_posi)
        LeftForeArm_posi_tun = quaternion_rotation_matrix(hip_quater) @ \
                           quaternion_rotation_matrix(Spine_quater) @ \
                           quaternion_rotation_matrix(Spine1_quater) @ \
                           quaternion_rotation_matrix(Spine2_quater) @ \
                           quaternion_rotation_matrix(LeftShoulder_quater) @ \
                           quaternion_rotation_matrix(LeftArm_quater) @ np.array(LeftForeArm_posi)
        LeftForeArm_posi_tun_ = [x + y for x, y in zip(np.ravel(LeftArm_posi_tun_), list(np.ravel(LeftForeArm_posi_tun)))]
        data_position_tun.append(LeftForeArm_posi_tun_)

        # LeftHand
        # LeftHand_posi_tun = _rotation_Y(LeftForeArm_euler[1]) @ _rotation_Z(
        #     LeftForeArm_euler[2]) @ _rotation_X(
        #     LeftForeArm_euler[0]) @ np.array(LeftHand_posi)
        LeftHand_posi_tun = quaternion_rotation_matrix(hip_quater) @ \
                               quaternion_rotation_matrix(Spine_quater) @ \
                               quaternion_rotation_matrix(Spine1_quater) @ \
                               quaternion_rotation_matrix(Spine2_quater) @ \
                               quaternion_rotation_matrix(LeftShoulder_quater) @ \
                               quaternion_rotation_matrix(LeftArm_quater) @ \
                               quaternion_rotation_matrix(LeftForeArm_quater) @ np.array(LeftHand_posi)
        LeftHand_posi_tun_ = [x + y for x, y in zip(np.ravel(LeftForeArm_posi_tun_), list(np.ravel(LeftHand_posi_tun)))]
        data_position_tun.append(LeftHand_posi_tun_)

        data_position_tun = list(itertools.chain(*data_position_tun))

        # print(data_position_tun)

        return (time_s, data_quaternion, data_euler,  data_position, data_position_tun)


    #####################
    ###### RUNNING ######
    #####################

    ## TODO: Continuously read data from your sensor.
    # Loop until self._running is False.
    # Acquire data from your sensor as desired, and for each timestep
    #  call self.append_data(device_name, stream_name, time_s, data).
    def _run(self):
        try:
            initNum = 0
            while self._running:

                def quaternion_to_AngularVel(Quaternion1, Quaternion2, delTime):

                    Angular_X = 2 / delTime * (Quaternion1[0] * Quaternion2[1] - Quaternion1[1] * Quaternion2[0]
                                               - Quaternion1[2] * Quaternion2[3] + Quaternion1[3] * Quaternion2[2])
                    Angular_Y = 2 / delTime * (Quaternion1[0] * Quaternion2[2] - Quaternion1[1] * Quaternion2[3]
                                               - Quaternion1[2] * Quaternion2[0] + Quaternion1[3] * Quaternion2[1])
                    Angular_Z = 2 / delTime * (Quaternion1[0] * Quaternion2[3] - Quaternion2[1] * Quaternion2[2]
                                               - Quaternion1[2] * Quaternion2[1] + Quaternion1[3] * Quaternion2[0])
                    return Angular_X, Angular_Y, Angular_Z

                # Read and store data for stream 1.
                (time_s, data_quaternion, data_euler, data_local_position, data_position) = self._read_data()
                if time_s is not None:
                    if initNum == 0:
                        time_save = time_s
                        quaternion_save = data_quaternion
                        initNum = 1
                        continue
                        delTime = 0
                    else:
                        if time_save is not None:
                            delTime = time_s - time_save
                        else:
                            delTime = 0.1

                    data_angular_vel = []
                    for i in range(21):
                        Angular_X, Angular_Y, Angular_Z = quaternion_to_AngularVel(quaternion_save[4 * i:4 * i + 4],
                                                                                   data_quaternion[4 * i:4 * i + 4],
                                                                                   delTime)
                        data_angular_vel.append([Angular_X, Angular_Y, Angular_Z])

                if time_s is not None:
                    self.append_data('pns-joint-quaternion', 'angle-values', time_s, data_quaternion)
                    self.append_data('pns-joint-euler', 'angle-values', time_s, data_euler)
                    self.append_data('pns-joint-position', 'cm-values', time_s, data_position)
                    self.append_data('pns-joint-local-position', 'cm-values', time_s, data_local_position)
                    self.append_data('pns-joint-angular-velocity', 'velocity-values', time_s, data_angular_vel)
                time_save = time_s

        except KeyboardInterrupt:  # The program was likely terminated
            pass
        except:
            self._log_error('\n\n***ERROR RUNNING MoticonStreamer:\n%s\n' % traceback.format_exc())
        finally:
            ## TODO: Disconnect from the sensor if desired.
            self._socket.close()

    # Clean up and quit
    def quit(self):
        ## TODO: Add any desired clean-up code.
        self._log_debug('NueronStudioStreamer quitting')
        self._socket.close()
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
        processed_options['pns-joint-position']  = {}
        processed_options['pns-joint-position']['cm-values'] = {
            'class': PnsSkeletonVisualizer,
        }

        for (device_name, device_info) in self._streams_info.items():
            processed_options.setdefault(device_name, {})
            for (stream_name, stream_info) in device_info.items():
                processed_options[device_name].setdefault(stream_name, {'class': None})

        ## TODO: Specify whether some streams should be visualized.
        #        Examples of a line plot and a heatmap are below.
        #        To not visualize data, simply omit the following code and just leave each streamer mapped to the None class as shown above.
        # Use a line plot to visualize the weight.
        # processed_options['pns-joint-position']['cm-values'] = \
        #    {'class': HeatmapVisualizer,
        #     'colorbar_levels': 'auto',  # The range of the colorbar.
        #     # Can be a 2-element list [min, max] to use hard-coded bounds,
        #     # or 'auto' to determine them dynamically based on a buffer of the data.
        #      }
       # processed_options['insole-moticon-right']['pressure-values'] = \
       #     {'class': HeatmapVisualizer,
       #      'colorbar_levels': 'auto',  # The range of the colorbar.
        #     # Can be a 2-element list [min, max] to use hard-coded bounds,
       #      # or 'auto' to determine them dynamically based on a buffer of the data.
        #     }

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
###### TESTING ######
#####################
if __name__ == '__main__':

    # Configuration.
    duration_s = 30

    # Connect to the device(s).
    pns_streamer = PnsStreamer(print_status=True, print_debug=False)
    pns_streamer.connect()

    # Run for the specified duration and periodically print the sample rate.
    print('\nRunning for %gs!' % duration_s)
    pns_streamer.run()
    start_time_s = time.time()
    try:
        while time.time() - start_time_s < duration_s:
            time.sleep(2)
            # Print the sampling rates.
            msg = ' Duration: %6.2fs' % (time.time() - start_time_s)
            for device_name in pns_streamer.get_device_names():
                stream_names = pns_streamer.get_stream_names(device_name=device_name)
                for stream_name in stream_names:
                    num_timesteps = pns_streamer.get_num_timesteps(device_name, stream_name)
                    msg += ' | %s-%s: %6.2f Hz (%4d Timesteps)' % \
                           (device_name, stream_name, ((num_timesteps) / (time.time() - start_time_s)), num_timesteps)
            print(msg)
    except:
        pass

    # Stop the streamer.
    pns_streamer.stop()
    print('\n' * 2)
    print('=' * 75)
    print('Done!')
    print('\n' * 2)
















