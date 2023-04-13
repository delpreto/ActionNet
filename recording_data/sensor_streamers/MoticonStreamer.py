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
from visualizers.HeatmapVisualizer import HeatmapVisualizer

import socket
import pandas as pd
import numpy as np
import time
from collections import OrderedDict
import traceback

from utils.print_utils import *


################################################
################################################
# A template class for implementing a new sensor.
################################################
################################################
class MoticonStreamer(SensorStreamer):

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
        self._log_source_tag = 'moticon'

        ## TODO: Initialize any state that your sensor needs.
        # Initialize counts
        self._num_segments = None # Moticon sensor에서 사용하려는 segment 개수 정의함

        # Initialize state
        self._buffer = b''
        self._buffer_read_size = 1024
        self._socket = None
        self._moticon_sample_index = None  # The current Moticon timestep being processed (each timestep will send multiple messages)
        self._moticon_message_start_time_s = None  # When a Moticon message was first received
        self._moticon_timestep_receive_time_s = None  # When the first Moticon message for a Moticon timestep was received

        # Specify the Moticon streaming configuration.
        self._moticon_network_protocol = 'udp'
        self._moticon_network_ip = '127.0.0.1'
        self._moticon_network_port = 2121

        ## TODO: Add devices and streams to organize data from your sensor.
        #        Data is organized as devices and then streams.
        #        For example, a Myo device may have streams for EMG and Acceleration.
        #        If desired, this could also be done in the connect() method instead.
        self.add_stream(device_name='insole-moticon-left-pressure',
                        stream_name='pressure_values_N_cm2',
                        data_type='float32',
                        sample_size=[16],
                        # the size of data saved for each timestep; here, we expect a 2-element vector per timestep
                        sampling_rate_hz=25,  # the expected sampling rate for the stream
                        extra_data_info={},
                        # can add extra information beyond the data and the timestamp if needed (probably not needed, but see MyoStreamer for an example if desired)
                        # Notes can add metadata about the stream,
                        #  such as an overall description, data units, how to interpret the data, etc.
                        # The SensorStreamer.metadata_data_headings_key is special, and is used to
                        #  describe the headings for each entry in a timestep's data.
                        #  For example - if the data was saved in a spreadsheet with a row per timestep, what should the column headings be.
                        data_notes=OrderedDict([
                            ('Description', 'Pressure data from the left shoe.'
                             ),
                            ('Units', 'N/cm2'),
                            (SensorStreamer.metadata_data_headings_key,
                             ['Left_1', 'Left_2', 'Left_3', 'Left_4', 'Left_5', 'Left_6', 'Left_7', 'Left_8', 'Left_9',
                              'Left_10',
                              'Left_11', 'Left_12', 'Left_13', 'Left_14', 'Left_15', 'Left_16']),
                        ]))
        self.add_stream(device_name='insole-moticon-left-acceleration',
                        stream_name='acceleration',
                        data_type='float32',
                        sample_size=[3],
                        # the size of data saved for each timestep; here, we expect a 2-element vector per timestep
                        sampling_rate_hz=25,  # the expected sampling rate for the stream
                        extra_data_info={},
                        # can add extra information beyond the data and the timestamp if needed (probably not needed, but see MyoStreamer for an example if desired)
                        # Notes can add metadata about the stream,
                        #  such as an overall description, data units, how to interpret the data, etc.
                        # The SensorStreamer.metadata_data_headings_key is special, and is used to
                        #  describe the headings for each entry in a timestep's data.
                        #  For example - if the data was saved in a spreadsheet with a row per timestep, what should the column headings be.
                        data_notes=OrderedDict([
                            ('Description', 'Acceleration data from the left shoe.'
                             ),
                            ('Units', 'g'),
                            (SensorStreamer.metadata_data_headings_key,
                             ['Left_X', 'Left_Y', 'Left_Z']),
                        ]))
        self.add_stream(device_name='insole-moticon-left-angular',
                        stream_name='angular',
                        data_type='float32',
                        sample_size=[3],
                        # the size of data saved for each timestep; here, we expect a 2-element vector per timestep
                        sampling_rate_hz=25,  # the expected sampling rate for the stream
                        extra_data_info={},
                        # can add extra information beyond the data and the timestamp if needed (probably not needed, but see MyoStreamer for an example if desired)
                        # Notes can add metadata about the stream,
                        #  such as an overall description, data units, how to interpret the data, etc.
                        # The SensorStreamer.metadata_data_headings_key is special, and is used to
                        #  describe the headings for each entry in a timestep's data.
                        #  For example - if the data was saved in a spreadsheet with a row per timestep, what should the column headings be.
                        data_notes=OrderedDict([
                            ('Description', 'Angular velocity data from the left shoe.'
                             ),
                            ('Units', 'degree/s'),
                            (SensorStreamer.metadata_data_headings_key,
                             ['Left_X', 'Left_Y', 'Left_Z']),
                        ]))
        self.add_stream(device_name='insole-moticon-left-totalForce',
                        stream_name='totalForce',
                        data_type='float32',
                        sample_size=[1],
                        # the size of data saved for each timestep; here, we expect a 2-element vector per timestep
                        sampling_rate_hz=25,  # the expected sampling rate for the stream
                        extra_data_info={},
                        # can add extra information beyond the data and the timestamp if needed (probably not needed, but see MyoStreamer for an example if desired)
                        # Notes can add metadata about the stream,
                        #  such as an overall description, data units, how to interpret the data, etc.
                        # The SensorStreamer.metadata_data_headings_key is special, and is used to
                        #  describe the headings for each entry in a timestep's data.
                        #  For example - if the data was saved in a spreadsheet with a row per timestep, what should the column headings be.
                        data_notes=OrderedDict([
                            ('Description', 'Total Force data from the left shoe.'
                             ),
                            ('Units', 'N'),
                            (SensorStreamer.metadata_data_headings_key,
                             ['Left']),
                        ]))
        self.add_stream(device_name='insole-moticon-right-pressure',
                        stream_name='pressure_values_N_cm2',
                        data_type='float32',
                        sample_size=[16],
                        # the size of data saved for each timestep; here, we expect a 2-element vector per timestep
                        sampling_rate_hz=25,  # the expected sampling rate for the stream
                        extra_data_info={},
                        # can add extra information beyond the data and the timestamp if needed (probably not needed, but see MyoStreamer for an example if desired)
                        # Notes can add metadata about the stream,
                        #  such as an overall description, data units, how to interpret the data, etc.
                        # The SensorStreamer.metadata_data_headings_key is special, and is used to
                        #  describe the headings for each entry in a timestep's data.
                        #  For example - if the data was saved in a spreadsheet with a row per timestep, what should the column headings be.
                        data_notes=OrderedDict([
                            ('Description', 'Pressure data from the left shoe.'
                             ),
                            ('Units', 'N/cm2'),
                            (SensorStreamer.metadata_data_headings_key,
                             ['right_1', 'right_2', 'right_3', 'right_4', 'right_5', 'right_6', 'right_7', 'right_8', 'right_9',
                              'right_10',
                              'right_11', 'right_12', 'right_13', 'right_14', 'right_15', 'right_16']),
                        ]))
        self.add_stream(device_name='insole-moticon-right-acceleration',
                        stream_name='acceleration',
                        data_type='float32',
                        sample_size=[3],
                        # the size of data saved for each timestep; here, we expect a 2-element vector per timestep
                        sampling_rate_hz=25,  # the expected sampling rate for the stream
                        extra_data_info={},
                        # can add extra information beyond the data and the timestamp if needed (probably not needed, but see MyoStreamer for an example if desired)
                        # Notes can add metadata about the stream,
                        #  such as an overall description, data units, how to interpret the data, etc.
                        # The SensorStreamer.metadata_data_headings_key is special, and is used to
                        #  describe the headings for each entry in a timestep's data.
                        #  For example - if the data was saved in a spreadsheet with a row per timestep, what should the column headings be.
                        data_notes=OrderedDict([
                            ('Description', 'Acceleration data from the right shoe.'
                             ),
                            ('Units', 'g'),
                            (SensorStreamer.metadata_data_headings_key,
                             ['Right_X', 'Right_Y', 'Right_Z']),
                        ]))
        self.add_stream(device_name='insole-moticon-right-angular',
                        stream_name='angular',
                        data_type='float32',
                        sample_size=[3],
                        # the size of data saved for each timestep; here, we expect a 2-element vector per timestep
                        sampling_rate_hz=25,  # the expected sampling rate for the stream
                        extra_data_info={},
                        # can add extra information beyond the data and the timestamp if needed (probably not needed, but see MyoStreamer for an example if desired)
                        # Notes can add metadata about the stream,
                        #  such as an overall description, data units, how to interpret the data, etc.
                        # The SensorStreamer.metadata_data_headings_key is special, and is used to
                        #  describe the headings for each entry in a timestep's data.
                        #  For example - if the data was saved in a spreadsheet with a row per timestep, what should the column headings be.
                        data_notes=OrderedDict([
                            ('Description', 'Pressure data from the left shoe.'
                             ),
                            ('Units', 'degree/s'),
                            (SensorStreamer.metadata_data_headings_key,
                             ['Right_X', 'Right_Y', 'Right_Z']),
                        ]))
        self.add_stream(device_name='insole-moticon-cop',
                        stream_name='cop',
                        data_type='float32',
                        sample_size=[4],
                        # the size of data saved for each timestep; here, we expect a 2-element vector per timestep
                        sampling_rate_hz=25,  # the expected sampling rate for the stream
                        extra_data_info={},
                        # can add extra information beyond the data and the timestamp if needed (probably not needed, but see MyoStreamer for an example if desired)
                        # Notes can add metadata about the stream,
                        #  such as an overall description, data units, how to interpret the data, etc.
                        # The SensorStreamer.metadata_data_headings_key is special, and is used to
                        #  describe the headings for each entry in a timestep's data.
                        #  For example - if the data was saved in a spreadsheet with a row per timestep, what should the column headings be.
                        data_notes=OrderedDict([
                            ('Description', 'Center of Pressure data from the both shoe.'
                             ),
                            ('Units', 'percent of insole length/width'),
                            (SensorStreamer.metadata_data_headings_key,
                             ['Left_X', 'Left_Y', 'right_X', 'right_Y']),
                        ]))
        self.add_stream(device_name='insole-moticon-right-totalForce',
                        stream_name='totalForce',
                        data_type='float32',
                        sample_size=[1],
                        # the size of data saved for each timestep; here, we expect a 2-element vector per timestep
                        sampling_rate_hz=25,  # the expected sampling rate for the stream
                        extra_data_info={},
                        # can add extra information beyond the data and the timestamp if needed (probably not needed, but see MyoStreamer for an example if desired)
                        # Notes can add metadata about the stream,
                        #  such as an overall description, data units, how to interpret the data, etc.
                        # The SensorStreamer.metadata_data_headings_key is special, and is used to
                        #  describe the headings for each entry in a timestep's data.
                        #  For example - if the data was saved in a spreadsheet with a row per timestep, what should the column headings be.
                        data_notes=OrderedDict([
                            ('Description', 'Total Force data from the right shoe.'
                             ),
                            ('Units', 'N'),
                            (SensorStreamer.metadata_data_headings_key,
                             ['right']),
                        ]))

    #######################################
    # Connect to the sensor.
    # @param timeout_s How long to wait for the sensor to respond.
    def _connect(self, timeout_s=10):
        # Open a socket to the Moticon network stream
        ## TODO: Add code for connecting to your sensor.
        #        Then return True or False to indicate whether connection was successful.
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.settimeout(5)  # timeout for all socket operations, such as receiving if the Xsens network stream is inactive
        self._socket.bind((self._moticon_network_ip, self._moticon_network_port))
        self._log_status('Successfully connected to the moticon streamer.')
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
            self._log_error('\n\n***ERROR reading from MoticonStreamer:\n%s\n' % traceback.format_exc())
            time.sleep(1)
            return (None, None, None)
        message = bytesAddressPair[0].decode("utf-8")
        # address = bytesAddressPair[1]
        clientMsg = "{}".format(message)
        # clientIP = "Client IP Address:{}".format(address)

        data = clientMsg.split()
        # print(data)
        # print(len(data))
        
        # Extract the device timestamp.
        time_s = float(data[0])

        # Acceleration data
        data_left_acceleration = [float(x) for x in data[1:4]]
        data_right_acceleration = [float(x) for x in data[26:29]]

        # Angular data
        data_left_angular = [float(x) for x in data[4:7]]
        data_right_angular = [float(x) for x in data[29:32]]

        # COP data

        data_cop = [float(x) for x in data[7:9]]
        data_cop.append(float(data[32]))
        data_cop.append(float(data[33]))
        # print(data_cop)
        # data_right_cop = [float(x) for x in data[32:34]]

        # Total Force data
        data_left_totalForce = float(data[25])
        data_right_totalForce = float(data[50])

        # Parse the pressure data.
        data_left_pressure = [float(x) for x in data[9:25]]
        data_right_pressure = [float(x) for x in data[34:50]]

        return (time_s, data_left_acceleration, data_left_angular, data_cop, data_left_pressure, data_left_totalForce,
                data_right_acceleration, data_right_angular, data_right_pressure, data_right_totalForce)

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
                (time_s, data_left_acceleration, data_left_angular, data_cop, data_left_pressure, data_left_totalForce,
                data_right_acceleration, data_right_angular, data_right_pressure, data_right_totalForce) = self._read_data()

                df = pd.DataFrame(data_left_pressure)
                # print(type(df.isnull().sum()).value())
                d = df.isnull().sum().to_numpy()

                if (sum(d) == 0) and (data_left_pressure is not None) and (len(data_left_pressure) > 0):
                    self.append_data('insole-moticon-left-acceleration', 'acceleration', time_s, data_left_acceleration)
                    self.append_data('insole-moticon-left-angular', 'angular', time_s, data_left_angular)
                    self.append_data('insole-moticon-cop', 'cop', time_s, data_cop)
                    self.append_data('insole-moticon-left-pressure', 'pressure_values_N_cm2', time_s, data_left_pressure)
                    self.append_data('insole-moticon-left-totalForce', 'totalForce', time_s, data_left_totalForce)
                    self.append_data('insole-moticon-right-acceleration', 'acceleration', time_s, data_right_acceleration)
                    self.append_data('insole-moticon-right-angular', 'angular', time_s, data_right_angular)
                    # self.append_data('insole-moticon-right-cop', 'cop', time_s, data_right_cop)
                    self.append_data('insole-moticon-right-pressure', 'pressure_values_N_cm2', time_s, data_right_pressure)
                    self.append_data('insole-moticon-right-totalForce', 'totalForce', time_s, data_right_totalForce)
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
        self._log_debug('MoticonStreamer quitting')
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
        for (device_name, device_info) in self._streams_info.items():
            processed_options.setdefault(device_name, {})
            for (stream_name, stream_info) in device_info.items():
                processed_options[device_name].setdefault(stream_name, {'class': None})

        ## TODO: Specify whether some streams should be visualized.
        #        Examples of a line plot and a heatmap are below.
        #        To not visualize data, simply omit the following code and just leave each streamer mapped to the None class as shown above.
        # Use a line plot to visualize the weight.
        processed_options['insole-moticon-left-pressure']['pressure_values_N_cm2'] = \
            {'class': HeatmapVisualizer,
             'colorbar_levels': 'auto',  # The range of the colorbar.
             # Can be a 2-element list [min, max] to use hard-coded bounds,
             # or 'auto' to determine them dynamically based on a buffer of the data.
             }
        processed_options['insole-moticon-right-pressure']['pressure_values_N_cm2'] = \
            {'class': HeatmapVisualizer,
             'colorbar_levels': 'auto',  # The range of the colorbar.
             # Can be a 2-element list [min, max] to use hard-coded bounds,
             # or 'auto' to determine them dynamically based on a buffer of the data.
             }
        processed_options['insole-moticon-cop']['cop'] = \
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
    duration_s = 30

    # Connect to the device(s).
    moticon_streamer = MoticonStreamer(print_status=True, print_debug=False)
    moticon_streamer.connect()

    # Run for the specified duration and periodically print the sample rate.
    print('\nRunning for %gs!' % duration_s)
    moticon_streamer.run()
    start_time_s = time.time()
    try:
        while time.time() - start_time_s < duration_s:
            time.sleep(2)
            # Print the sampling rates.
            msg = ' Duration: %6.2fs' % (time.time() - start_time_s)
            for device_name in moticon_streamer.get_device_names():
                stream_names = moticon_streamer.get_stream_names(device_name=device_name)
                for stream_name in stream_names:
                    num_timesteps = moticon_streamer.get_num_timesteps(device_name, stream_name)
                    msg += ' | %s-%s: %6.2f Hz (%4d Timesteps)' % \
                           (device_name, stream_name, ((num_timesteps) / (time.time() - start_time_s)), num_timesteps)
            print(msg)
    except:
        pass

    # Stop the streamer.
    moticon_streamer.stop()
    print('\n' * 2)
    print('=' * 75)
    print('Done!')
    print('\n' * 2)
















