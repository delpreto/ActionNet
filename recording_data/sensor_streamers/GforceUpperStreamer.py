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
import pandas as pd

from sensor_streamers.SensorStreamer import SensorStreamer
from visualizers.LinePlotVisualizer import LinePlotVisualizer
from visualizers.HeatmapVisualizer import HeatmapVisualizer

import socket
import numpy as np
import time
import random
from collections import OrderedDict
import traceback

from utils.print_utils import *

################################################
################################################
# A template class for implementing a new sensor.
################################################
################################################
class GforceUpperStreamer(SensorStreamer):

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
        self._log_source_tag = 'Gforce'

        ## TODO: Initialize any state that your sensor needs.
        # Initialize counts
        self._num_segments = None  # Moticon sensor에서 사용하려는 segment 개수 정의함

        # Initialize state
        self._buffer = b''
        self._buffer_read_size = 2048
        self._socket2 = None
        self._gforce_sample_index = None  # The current Moticon timestep being processed (each timestep will send multiple messages)
        self._gforce_message_start_time_s = None  # When a Moticon message was first received
        self._gforce_timestep_receive_time_s = None  # When the first Moticon message for a Moticon timestep was received

        # Specify the Moticon streaming configuration.6
        self._gforce_network_protocol = 'udp'
        self._gforce_network_ip = "192.168.1.23"
        self._gforce_network_port2 = 5556

        ## TODO: Add devices and streams to organize data from your sensor.
        #        Data is organized as devices and then streams.
        #        For example, a Myo device may have streams for EMG and Acceleration.
        #        If desired, this could also be done in the connect() method instead.
        self.add_stream(device_name='armband-gforce-upperarm',
                        stream_name='pressure_values_N_cm2',
                        data_type='float32',
                        sample_size=[8],
                        # the size of data saved for each timestep; here, we expect a 2-element vector per timestep
                        sampling_rate_hz=200,  # the expected sampling rate for the stream
                        extra_data_info={},
                        # can add extra information beyond the data and the timestamp if needed (probably not needed, but see MyoStreamer for an example if desired)
                        # Notes can add metadata about the stream,
                        #  such as an overall description, data units, how to interpret the data, etc.
                        # The SensorStreamer.metadata_data_headings_key is special, and is used to
                        #  describe the headings for each entry in a timestep's data.
                        #  For example - if the data was saved in a spreadsheet with a row per timestep, what should the column headings be.
                        data_notes=OrderedDict([
                            ('Description', 'Pressure data from the upper arm.'
                             ),
                            ('Units', 'N/cm2'),
                            (SensorStreamer.metadata_data_headings_key,
                             ['upper_Channel_1', 'upper_Channel_2', 'upper_Channel_3', 'upper_Channel_4',
                              'upper_Channel_5', 'upper_Channel_6', 'upper_Channel_7', 'upper_Channel_8', ]),
                        ]))
        self.add_stream(device_name='armband-gforce-upperarm',
                        stream_name='Window Unix Time',
                        data_type='float32',
                        sample_size=[1],
                        # the size of data saved for each timestep; here, we expect a 2-element vector per timestep
                        sampling_rate_hz=200,  # the expected sampling rate for the stream
                        extra_data_info={},
                        # can add extra information beyond the data and the timestamp if needed (probably not needed, but see MyoStreamer for an example if desired)
                        # Notes can add metadata about the stream,
                        #  such as an overall description, data units, how to interpret the data, etc.
                        # The SensorStreamer.metadata_data_headings_key is special, and is used to
                        #  describe the headings for each entry in a timestep's data.
                        #  For example - if the data was saved in a spreadsheet with a row per timestep, what should the column headings be.
                        data_notes=OrderedDict([
                            ('Description', 'Pressure data from the upper arm.'
                             ),
                            ('Units', 'second'),
                            (SensorStreamer.metadata_data_headings_key,
                             ['Window Unix Time']),
                        ]))
        # self.add_stream(device_name='armband-gforce-upperarm',
        #                 stream_name='imu-acc-values',
        #                 data_type='float32',
        #                 sample_size=[3],
        #                 # the size of data saved for each timestep; here, we expect a 2-element vector per timestep
        #                 sampling_rate_hz=500,  # the expected sampling rate for the stream
        #                 extra_data_info={},
        #                 # can add extra information beyond the data and the timestamp if needed (probably not needed, but see MyoStreamer for an example if desired)
        #                 # Notes can add metadata about the stream,
        #                 #  such as an overall description, data units, how to interpret the data, etc.
        #                 # The SensorStreamer.metadata_data_headings_key is special, and is used to
        #                 #  describe the headings for each entry in a timestep's data.
        #                 #  For example - if the data was saved in a spreadsheet with a row per timestep, what should the column headings be.
        #                 data_notes=OrderedDict([
        #                     ('Description', 'Pressure data from the upper arm.'
        #                      ),
        #                     ('Units', 'N/cm2'),
        #                     (SensorStreamer.metadata_data_headings_key,
        #                      ['Acc_X', 'Acc_Y', 'Acc_Z']),
        #                 ]))
        # self.add_stream(device_name='armband-gforce-upperarm',
        #                stream_name='imu-gyro-values',
        #                data_type='float32',
        #                sample_size=[3],
        #                # the size of data saved for each timestep; here, we expect a 2-element vector per timestep
        #                sampling_rate_hz=500,  # the expected sampling rate for the stream
        #                extra_data_info={},
        #                # can add extra information beyond the data and the timestamp if needed (probably not needed, but see MyoStreamer for an example if desired)
        #                # Notes can add metadata about the stream,
        #                #  such as an overall description, data units, how to interpret the data, etc.
        #                # The SensorStreamer.metadata_data_headings_key is special, and is used to
        #                #  describe the headings for each entry in a timestep's data.
        #                #  For example - if the data was saved in a spreadsheet with a row per timestep, what should the column headings be.
        #                data_notes=OrderedDict([
        #                    ('Description', 'Pressure data from the upper arm.'
        #                     ),
        #                    ('Units', 'N/cm2'),
        #                    (SensorStreamer.metadata_data_headings_key,
        #                     ['Gyro_X', 'Gyro_Y', 'Gyro_Z']),
        #                ]))
        # self.add_stream(device_name='armband-gforce-upperarm',
        #                stream_name='imu-quaternion-values',
        #                data_type='float32',
        #                sample_size=[3],
        #                # the size of data saved for each timestep; here, we expect a 2-element vector per timestep
        #                sampling_rate_hz=500,  # the expected sampling rate for the stream
        #                extra_data_info={},
        #                # can add extra information beyond the data and the timestamp if needed (probably not needed, but see MyoStreamer for an example if desired)
        #                # Notes can add metadata about the stream,
        #                #  such as an overall description, data units, how to interpret the data, etc.
        #                # The SensorStreamer.metadata_data_headings_key is special, and is used to
        #                #  describe the headings for each entry in a timestep's data.
        #                #  For example - if the data was saved in a spreadsheet with a row per timestep, what should the column headings be.
        #                data_notes=OrderedDict([
        #                    ('Description', 'Pressure data from the upper arm.'
        #                     ),
        #                    ('Units', 'N/cm2'),
        #                    (SensorStreamer.metadata_data_headings_key,
        #                     ['Quat_X', 'Quat_Y', 'Quat_Z', 'Quat_W']),
        #                ]))

    #######################################
    # Connect to the sensor.
    # @param timeout_s How long to wait for the sensor to respond.
    def _connect(self, timeout_s=10):
        # Open a socket to the Moticon network stream
        ## TODO: Add code for connecting to your sensor.
        #        Then return True or False to indicate whether connection was successful.
        self._socket2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket2.settimeout(
            5)  # timeout for all socket operations, such as receiving if the Xsens network stream is inactive
        self._socket2.bind((self._gforce_network_ip, self._gforce_network_port2))
        self._log_status('Successfully connected to the gforce streamer.')
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
            bytesAddressPair2 = self._socket2.recvfrom(self._buffer_read_size)

            message2 = bytesAddressPair2[0].decode("utf-8")
            data2 = message2.split(',')

            if len(data2) == 8:
                time_s_2 = time.time()
                device2_EMG_list = [-1, -1, -1, -1, -1, -1, -1, -1]
            else:
                # print(data2)
                # for i in range(len(data2)):
                #     print(data2[i])
                device2_EMG_list = []
                device2_Acc_list = []
                device2_Quat_list = []
                device2_Gyro_list = []
                time_s_2 = float(data2[0].split("[")[1])
                if data2[1].split("'")[1] == 'EMG':
                    for i in range(8):
                        if i == 7:
                            device2_EMG_list.append(float(data2[i + 2].split(']')[0]))
                        else:
                            device2_EMG_list.append(float(data2[i + 2]))
                elif data2[1].split("'")[1] == 'ACC':
                    device2_Acc_list.append(float(data2[2].split("(")[1]))
                    device2_Acc_list.append(float(data2[3]))
                    device2_Acc_list.append(float(data2[4].split("(")[0]))
                elif data2[1].split("'")[1] == 'GYRO':
                    device2_Gyro_list.append(float(data2[2].split("(")[1]))
                    device2_Gyro_list.append(float(data2[3]))
                    device2_Gyro_list.append(float(data2[4].split("(")[0]))
                elif data2[1].split("'")[1] == 'QUAT':
                    device2_Quat_list.append(float(data2[2].split("(")[1]))
                    device2_Quat_list.append(float(data2[3]))
                    device2_Quat_list.append(float(data2[4]))
                    device2_Quat_list.append(float(data2[5].split(")")[0]))

            return (time_s_2, device2_EMG_list)


        except:
            self._log_error('\n\n***ERROR reading from GforceStreamer:\n%s\n' % traceback.format_exc())
            time.sleep(1)
            # time_s = time.time()
            # device1 = [-1, -1, -1, -1, -1, -1, -1, -1]
            # print('type', type(device1))
            # device2 = [-1, -1, -1, -1, -1, -1, -1, -1]

            return (None, None)

    #####################
    ###### RUNNING ######
    #####################

    ## TODO: Continuously read data from your sensor.
    # Loop until self._running is False.
    # Acquire data from your sensor as desired, and for each timestep
    #  call self.append_data(device_name, stream_name, time_s, data).
    def _run(self):
        try:
            while self._running:
                # Read and store data for stream 1.
                (time_s_2, device2_EMG_list) = self._read_data()
                timess = time.time()

                dfff = pd.DataFrame(device2_EMG_list)
                dd = dfff.isnull().sum().to_numpy()
                # print(dfff)
                # print(device2_EMG_list)
                if (sum(dd) == 0) and (device2_EMG_list is not None) and (len(device2_EMG_list) > 0) :
                    # print(None not in device2_EMG_list)
                    self.append_data('armband-gforce-upperarm', 'pressure_values_N_cm2', time_s_2, device2_EMG_list)
                    self.append_data('armband-gforce-upperarm', 'Window Unix Time', time_s_2, timess)
                    # print('Device 2 EMG List : ', device2_EMG_list)
                    # self.append_data('armband-gforce-lowerarm', 'imu-acc-values', time_s_1, device1_Acc_list)
                    # self.append_data('armband-gforce-lowerarm', 'imu-gyro-values', time_s_1, device1_Gyro_list)
                    # self.append_data('armband-gforce-lowerarm', 'imu-quaternion-values', time_s_1, device1_Quat_list)

        except KeyboardInterrupt:  # The program was likely terminated
            self._socket2.close()
            pass
        except:
            self._socket2.close()
            self._log_error('\n\n***ERROR RUNNING MoticonStreamer:\n%s\n' % traceback.format_exc())
        finally:
            ## TODO: Disconnect from the sensor if desired.
            self._socket2.close()

    # Clean up and quit
    def quit(self):
        ## TODO: Add any desired clean-up code.
        self._log_debug('GforceStreamer quitting')
        self._socket2.close()
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

        ## TODO: Specify whether some streams should be visualized.
        #        Examples of a line plot and a heatmap are below.
        #        To not visualize data, simply omit the following code and just leave each streamer mapped to the None class as shown above.
        # Use a line plot to visualize the weight.
        processed_options['armband-gforce-upperarm']['pressure_values_N_cm2'] = \
            {'class': LinePlotVisualizer,
             'single_graph': False,  # Whether to show each dimension on a subplot or all on the same plot.
             'plot_duration_s': 15,  # The timespan of the x axis (will scroll as more data is acquired).
             'downsample_factor': 1,  # Can optionally downsample data before visualizing to improve performance.
             }

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
    duration_s = 40

    # Connect to the device(s).
    gforce_streamer = GforceUpperStreamer(print_status=True, print_debug=False)
    gforce_streamer.connect()

    # Run for the specified duration and periodically print the sample rate.
    print('\nRunning for %gs!' % duration_s)
    gforce_streamer.run()
    start_time_s = time.time()
    try:
        while time.time() - start_time_s < duration_s:
            time.sleep(2)
            # Print the sampling rates.
            msg = ' Duration: %6.2fs' % (time.time() - start_time_s)
            for device_name in gforce_streamer.get_device_names():
                stream_names = gforce_streamer.get_stream_names(device_name=device_name)
                for stream_name in stream_names:
                    num_timesteps = gforce_streamer.get_num_timesteps(device_name, stream_name)
                    msg += ' | %s-%s: %6.2f Hz (%4d Timesteps)' % \
                           (device_name, stream_name, ((num_timesteps) / (time.time() - start_time_s)), num_timesteps)
            print(msg)
    except:
        pass

    # Stop the streamer.
    gforce_streamer.stop()
    print('\n' * 2)
    print('=' * 75)
    print('Done!')
    print('\n' * 2)
















