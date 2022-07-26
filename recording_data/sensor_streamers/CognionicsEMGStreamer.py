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

import numpy as np
import time
from collections import OrderedDict
import traceback
import pylsl
from pylsl import StreamInlet, resolve_stream

from utils.print_utils import *


################################################
################################################
# A template class for implementing a new sensor.
################################################
################################################
class CognionicsEMGStreamer(SensorStreamer):

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
        self._log_source_tag = 'CogEMG'

        ## TODO: Initialize any state that your sensor needs.
        # Initialize counts
        self._num_segments = None

        # Initialize state
        self._buffer = b''
        self._buffer_read_size = 4096
        self._LSLstream = None
        self._Inlet = None
        self._CogEMG_sample_index = None  # The current Moticon timestep being processed (each timestep will send multiple messages)
        self._CogEMG_message_start_time_s = None  # When a Cognionics message was first received
        self._CogEMG_timestep_receive_time_s = None  # When the first Cognionics message for a Cognionics timestep was received
        # self._device_id = 'D931CD'

        # Specify the Moticon streaming configuration.
        self._CogEMG_network_protocol = 'LSL' #LabStreamingLayer
        # self._CogEMG_network_ip = '127.0.0.1'
        # self._CogEMG_network_port = 50000

        ## TODO: Add devices and streams to organize data from your sensor.
        #        Data is organized as devices and then streams.
        #        For example, a Myo device may have streams for EMG and Acceleration.
        #        If desired, this could also be done in the connect() method instead.
        self.add_stream(device_name='EMGLeft-cognionics',
                        stream_name='emgleft-values',
                        data_type='float32',
                        sample_size=[1],
                        # the size of data saved for each timestep; here, we expect a 2-element vector per timestep
                        sampling_rate_hz=50,  # the expected sampling rate for the stream
                        extra_data_info={},
                        # can add extra information beyond the data and the timestamp if needed (probably not needed, but see MyoStreamer for an example if desired)
                        # Notes can add metadata about the stream,
                        #  such as an overall description, data units, how to interpret the data, etc.
                        # The SensorStreamer.metadata_data_headings_key is special, and is used to
                        #  describe the headings for each entry in a timestep's data.
                        #  For example - if the data was saved in a spreadsheet with a row per timestep, what should the column headings be.
                        data_notes=OrderedDict([
                            ('Description', 'EMG data from EMGLeft-cognionics.'
                             ),
                            ('Units', ''),
                            (SensorStreamer.metadata_data_headings_key,
                             ['emg_left']),
                        ]))

        self.add_stream(device_name='EMGRight-cognionics',
                        stream_name='emgright-values',
                        data_type='float32',
                        sample_size=[1],
                        # the size of data saved for each timestep; here, we expect a 2-element vector per timestep
                        sampling_rate_hz=50,  # the expected sampling rate for the stream
                        extra_data_info={},
                        # can add extra information beyond the data and the timestamp if needed (probably not needed, but see MyoStreamer for an example if desired)
                        # Notes can add metadata about the stream,
                        #  such as an overall description, data units, how to interpret the data, etc.
                        # The SensorStreamer.metadata_data_headings_key is special, and is used to
                        #  describe the headings for each entry in a timestep's data.
                        #  For example - if the data was saved in a spreadsheet with a row per timestep, what should the column headings be.
                        data_notes=OrderedDict([
                            ('Description', 'EMG data from EMGRight-cognionics.'
                             ),
                            ('Units', ''),
                            (SensorStreamer.metadata_data_headings_key,
                             ['emg_right']),
                        ]))


    #######################################
    # Connect to the sensor.
    # @param timeout_s How long to wait for the sensor to respond.
    def _connect(self, timeout_s=10):

        ## TODO: Add code for connecting to your sensor.
        #        Then return True or False to indicate whether connection was successful.
        self._LSLstream = resolve_stream()
        self._Inlet = StreamInlet(self._LSLstream[0])
        # self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self._socket.settimeout(3)

        # print("Connecting to server")
        # self._socket.connect((self._CogEMG_network_ip, self._CogEMG_network_port))
        # print("Connected to server\n")

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
            # print("Starting LSL streaming")

            # first resolve an EEG stream on the lab network
            # print("looking for an EEG stream...")

            time_s = time.time()
            sample, timestamp = self._Inlet.pull_sample()
            data_left = sample[0]
            data_right = sample[1]
            # print('Unix Time: ', time_s, ', EMG Left : ', sample[0], ', EMG Right : ', sample[1])


            return (time_s, data_left, data_right)

        except:
            self._log_error('\n\n***ERROR reading from E4Streamer:\n%s\n' % traceback.format_exc())
            time.sleep(1)
            return (None, None, None)

    #####################
    ###### RUNNING ######
    #####################

    ## TODO: Continuously read data from your sensor.
    # Loop until self._running is False.
    # Acquire data from your sensor as desired, and for each timestep
    #  call self.append_data(device_name, stream_name, time_s, data).
    def _run(self):
        try:
            print("Streaming...")
            while self._running:

                # Read and store data for stream 1.
                (time_s, data_left, data_right) = self._read_data()
                if time_s is not None:
                    self.append_data('EMGLeft-cognionics', 'emgleft-values', time_s, data_left)
                    self.append_data('EMGRight-cognionics', 'emgright-values', time_s, data_right)
        except KeyboardInterrupt:  # The program was likely terminated
            pass
        except:
            self._log_error('\n\n***ERROR RUNNING E4Streamer:\n%s\n' % traceback.format_exc())
        finally:
            ## TODO: Disconnect from the sensor if desired.
            self._log_debug('Keyboard Interrupted ')

    # Clean up and quit
    def quit(self):
        ## TODO: Add any desired clean-up code.
        self._log_debug('E4Streamer quitting')
        # self._socket.close()
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
        processed_options['EMGLeft-cognionics']['emgleft-values'] = \
            {'class': LinePlotVisualizer, #HeatmapVisualizer
             'single_graph': True,   # Whether to show each dimension on a subplot or all on the same plot.
             'plot_duration_s': 15,  # The timespan of the x axis (will scroll as more data is acquired).
             'downsample_factor': 1, # Can optionally downsample data before visualizing to improve performance.
             }

        processed_options['EMGRight-cognionics']['emgright-values'] = \
            {'class': LinePlotVisualizer,  # HeatmapVisualizer
             'single_graph': True,  # Whether to show each dimension on a subplot or all on the same plot.
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
    CognionicsEMG_streamer = CognionicsEMGStreamer(print_status=True, print_debug=False)
    CognionicsEMG_streamer.connect()

    # Run for the specified duration and periodically print the sample rate.
    print('\nRunning for %gs!' % duration_s)
    CognionicsEMG_streamer.run()
    print("Streamer Running Start")
    start_time_s = time.time()
    try:
        while time.time() - start_time_s < duration_s:
            time.sleep(2)
            # Print the sampling rates.
            msg = ' Duration: %6.2fs' % (time.time() - start_time_s)
            for device_name in CognionicsEMG_streamer.get_device_names():
                stream_names = CognionicsEMG_streamer.get_stream_names(device_name=device_name)
                for stream_name in stream_names:
                    num_timesteps = CognionicsEMG_streamer.get_num_timesteps(device_name, stream_name)
                    msg += ' | %s-%s: %6.2f Hz (%4d Timesteps)' % \
                           (device_name, stream_name, ((num_timesteps) / (time.time() - start_time_s)), num_timesteps)
            print(msg)
    except:
        pass

    # Stop the streamer.
    CognionicsEMG_streamer.stop()
    print('\n' * 2)
    print('=' * 75)
    print('Done!')
    print('\n' * 2)
















