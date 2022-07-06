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

import multiprocessing, multiprocessing.managers
import multiprocessing.managers
import itertools
import glob
import h5py
import time
import sys
import os
script_dir = os.path.dirname(os.path.realpath(__file__))

from utils.time_utils import *
from utils.dict_utils import *
from utils.print_utils import *

from sensor_streamers.SensorStreamer import SensorStreamer
from sensor_streamer_handlers.DataLogger import DataLogger
from sensor_streamer_handlers.DataVisualizer import DataVisualizer

################################################
################################################
# A helper class to manage a collection of SensorStreamer objects.
# Will connect/start/stop all streamers, and optionally
#  create DataLoggers to save their data
#  and DataVisualizers to visualize their data.
# Will use a separate process for each streamer.
# Will use a thread in the main process for logging.
# Will use the main thread of the main process for visualizing.
################################################
################################################
class SensorManager:

  ########################
  ###### INITIALIZE ######
  ########################

  # @param sensor_streamer_specs contains dicts that describe what streamers to make.
  #   Each dict should have an entry for 'class' with the class name to instantiate.
  #     It can then have as many keyword arguments for the initializer as desired.
  #   sensor_streamer_specs can be a list of dicts or a list of lists of dicts.
  #   The list structure will determine how many DataLoggers are created if logging is enabled.
  #   A DataLogger will be created for each top-level entry of the list.
  #   For example:
  #     [{'class':'NotesStreamer'}, {'class':'MyoStreamer'}, {'class':'TouchStreamer'}]
  #       will save all Notes, Myo, and Touch data in a single HDF5 file.
  #     [{'class':'NotesStreamer'}, [{'class':'MyoStreamer'}, {'class':'TouchStreamer'}]]
  #       will save Notes in one HDF5 file, and Myo+Touch data in a second HDF5 file.
  # @param data_logger_options is a dict of keyword arguments for DataLogger.
  #   It can be a list of such dicts, which would correspond to each group in sensor_streamer_specs.
  #   It should be None if no logging is desired.
  def __init__(self, sensor_streamer_specs=None,
               log_player_options=None,
               data_logger_options=None,
               data_visualizer_options=None,
               kill_other_python_processes=True,
               print_status=True, print_debug=False,
               log_history_filepath=None):

    # Try to kill all other Python processes.
    if kill_other_python_processes:
      try:
        import psutil
        import os
        for proc in psutil.process_iter():
          pinfo = proc.as_dict(attrs=['pid', 'name'])
          procname = str(pinfo['name'])
          procpid = str(pinfo['pid'])
          if "python" in procname and procpid != str(os.getpid()):
            print("Stopped Python Process ", proc)
            proc.kill()
      except:
        pass

    # Import all classes in the sensor_streamers folder.
    # Assumes the class name matches the filename.
    self._sensor_streamer_classes = {}
    sensor_streamer_files = glob.glob(os.path.join(script_dir, '..', 'sensor_streamers', '*.py'))
    for sensor_streamer_file in sensor_streamer_files:
      try:
        sensor_streamer_class_name = os.path.splitext(os.path.basename(sensor_streamer_file))[0]
        sensor_streamer_module = __import__('sensor_streamers.%s' % sensor_streamer_class_name, fromlist=[sensor_streamer_class_name])
        sensor_streamer_class = getattr(sensor_streamer_module, sensor_streamer_class_name)
        self._sensor_streamer_classes[sensor_streamer_class_name] = sensor_streamer_class
      except:
        pass
    
  # Record various configuration options.
    self._log_player_options = log_player_options
    self._data_visualizer_options = data_visualizer_options or {}
    self._print_status = print_status
    self._print_debug = print_debug
    self._log_source_tag = 'manager'
    self._log_history_filepath = log_history_filepath

    # Create streamer specs from an existing data log if desired.
    if log_player_options is not None:
      sensor_streamer_specs = self.get_sensor_streamer_specs_from_log()

    # Validate the streamer specs, and make a list of them.
    if not isinstance(sensor_streamer_specs, (list, tuple)):
      sensor_streamer_specs = [sensor_streamer_specs]
    sensor_streamer_specs = list(sensor_streamer_specs)
    if len(sensor_streamer_specs) == 0:
      raise AssertionError('At least one SensorStreamer spec must be provided to SensorManager')
    # If a nested list is provided, make sure each element is a list and created a flattened version.
    if True in [isinstance(x, (list, tuple)) for x in sensor_streamer_specs]:
      sensor_streamer_specs_grouped = [x if isinstance(x, (list, tuple)) else [x] for x in sensor_streamer_specs]
      sensor_streamer_specs_flattened = list(itertools.chain(*sensor_streamer_specs_grouped))
    else:
      # Otherwise, make a nested list that indicates all of them are one set.
      sensor_streamer_specs_grouped = [sensor_streamer_specs]
      sensor_streamer_specs_flattened = sensor_streamer_specs
    # Validate that all entries are dicts with a 'class' entry.
    if False in [(isinstance(x, dict) and 'class' in x) for x in sensor_streamer_specs_flattened]:
      raise AssertionError('At least one streamer spec provided to SensorManager is not a dictionary with a \'class\' entry')

    # Register each class that we will want to instantiate with the multiprocessing manager,
    #  so that Proxy classes will be created for them that can share memory.
    self._log_status('SensorManager starting the multiprocessing manager')
    for class_name in [spec['class'] for spec in sensor_streamer_specs_flattened]:
      multiprocessing.managers.BaseManager.register('%s_mp' % class_name, self._sensor_streamer_classes[class_name])
    multiprocessing_manager = multiprocessing.managers.BaseManager()
    multiprocessing_manager.start()

    # Instantiate each desired streamer.
    # Use a multiprocessing manager so that their data can be shared across processes.
    #   Otherwise, their data would only be accessible in the process where the run method is used.
    self._log_status('SensorManager creating sensor streamers')
    self._streamers = []
    self._streamers_grouped = []
    for streamer_specs_group in sensor_streamer_specs_grouped:
      self._streamers_grouped.append([])
      for streamer_spec in streamer_specs_group:
        class_name = streamer_spec['class']
        class_type = eval('multiprocessing_manager.%s_mp' % class_name)
        class_args = streamer_spec.copy()
        class_args['log_player_options'] = self._log_player_options
        class_args['log_history_filepath'] = self._log_history_filepath
        del(class_args['class'])
        # Disable visualization if desired for this class.
        if 'classes_to_visualize' in self._data_visualizer_options:
          if class_name not in self._data_visualizer_options['classes_to_visualize']:
            class_args['visualization_options'] = {'disable_visualization':True}
        # Create the class object.
        class_object = class_type(**class_args)
        # If the streamer says it should be in the main process,
        #  then create an instance of it directly instead of using a proxy class.
        if class_object.get_threading_config('always_run_in_main_process'):
          class_type = self._sensor_streamer_classes[class_name]
          class_object = class_type(**class_args)
        # Store the streamer object.
        self._streamers.append(class_object)
        self._streamers_grouped[-1].append(class_object)

    # Create DataLoggers if desired.
    self._data_loggers = []
    if data_logger_options is not None:
      self._log_status('SensorManager creating data loggers')
      for (group_index, sensor_streamers_group) in enumerate(self._streamers_grouped):
        if isinstance(data_logger_options, dict):
          datalogging_options_forGroup = data_logger_options
        else:
          datalogging_options_forGroup = data_logger_options[group_index]
        datalogging_options_forGroup['log_history_filepath'] = self._log_history_filepath
        self._data_loggers.append(DataLogger(
                                    sensor_streamers_group,
                                    **datalogging_options_forGroup))

    # Set default data visualization options.
    self._data_visualizer_options.setdefault('visualize_streaming_data', False)
    self._data_visualizer_options.setdefault('visualize_all_data_when_stopped', False)
    self._data_visualizer_options.setdefault('wait_while_visualization_windows_open', True)
    # Create DataVisualizers if desired.
    self._data_visualizer = None
    if self._data_visualizer_options is not None \
        and (self._data_visualizer_options['visualize_streaming_data']
              or self._data_visualizer_options['visualize_all_data_when_stopped']):
      self._log_status('SensorManager creating a data visualizer')
      data_visualizer_init_options = {}
      data_visualizer_init_options['update_period_s'] = None
      data_visualizer_init_options['print_debug'] = self._print_debug
      data_visualizer_init_options['print_status'] = self._print_status
      data_visualizer_init_options['log_history_filepath'] = self._log_history_filepath
      data_visualizer_init_options['use_composite_video'] = False
      data_visualizer_init_options['composite_video_layout'] = None
      data_visualizer_init_options['composite_video_filepath'] = None
      for (k, v) in self._data_visualizer_options.items():
        if k in data_visualizer_init_options:
          data_visualizer_init_options[k] = v
      self._data_visualizer = DataVisualizer(
                                  self._streamers,
                                  **data_visualizer_init_options)

    # Initialize lists of processes.
    self._sensor_processes = []


  #############################
  ###### GETTERS/SETTERS ######
  #############################

  # Get the streamer objects.
  # Will return all streamers if class_name is None,
  #  and otherwise will only return the streamers of that type.
  # class_name can be a string or the class type itself.
  def get_streamers(self, class_name=None):
    # Return all streamers
    if class_name is None:
      return self._streamers
    # Find streamers of the desired type
    streamers = []
    if isinstance(class_name, type):
      class_name = class_name.__name__
    for streamer in self._streamers:
      streamer_class_name = type(streamer).__name__
      # Check if the streamer is exactly the desired type.
      if streamer_class_name == class_name:
        streamers.append(streamer)
      # Check if the streamer is a multiprocessing Proxy for the desired class.
      # An example could look like AutoProxy[MyoStreamer_mp] for a MyoStreamer.
      elif '[%s_mp]' % class_name in streamer_class_name:
        streamers.append(streamer)
    return streamers

  # Get the data logger objects.
  def get_data_loggers(self):
    return self._data_loggers

  # Get the data visualizer objects.
  def get_data_visualizer(self):
    return self._data_visualizer


  ##########################################
  ###### REPLAYING EXISTING DATA LOGS ######
  ##########################################

  # Get specs for sensor streamers that should be created
  #  to replay an existing HDF5 data log.
  def get_sensor_streamer_specs_from_log(self):
    sensor_streamer_specs = []
    streamer_class_names_added = [] # will avoid duplicates (only create one class object per active streamer type)

    # Find all HDF5 files in the directory.
    data_log_input_dir = self._log_player_options['log_dir']
    hdf5_filepaths = []
    for file in os.listdir(data_log_input_dir):
      if file.endswith('.hdf5'):
        hdf5_filepaths.append(os.path.join(data_log_input_dir, file))

    # Determine the streams that were active in the log.
    for hdf5_filepath in hdf5_filepaths:
      sensor_streamer_specs.append([])
      hdf5_file = h5py.File(hdf5_filepath, 'r')
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
        if streamer_class_name not in streamer_class_names_added:
          sensor_streamer_specs[-1].append({
            'class': streamer_class_name,
            'log_player_options': self._log_player_options,
            'print_status': self._print_status, 'print_debug': self._print_debug
          })
          streamer_class_names_added.append(streamer_class_name)
      hdf5_file.close()
    if self._print_debug:
      self._log_debug('Extracted the following streamer specs from the HDF5 files:')
      for group in sensor_streamer_specs:
        for spec in group:
          self._log_debug(get_dict_str(spec))
    return sensor_streamer_specs



  #####################
  ###### RUNNING ######
  #####################

  # Helper to connect each of the individual streamers.
  def connect(self):
    for (streamer_index, streamer) in enumerate(self._streamers):
      self._log_status('\nSensorManager connecting streamer %d/%d (class %s)' % (streamer_index+1, len(self._streamers), type(streamer).__name__))
      connected = streamer.connect()
      if not connected:
        raise AssertionError('Error connecting streamer %d/%d (class %s)' % (streamer_index+1, len(self._streamers), type(streamer).__name__))

  # The main run method.
  # Start each streamer in its own process.
  # Start data logging if desired (in a thread of the main process).
  # Start data visualization if desired (in the main thread of the main process).
  # Wait until the specified duration has elapsed or until the provided callback function is True.
  def run(self, duration_s=None, stopping_condition_fn=None):
    # Start external data recording if desired.
    # Do this before the streamers start logging,
    #   in case the external softwares reset indexes and whatnot.
    # Also do it before a streaming start time is chosen,
    #   since it might require user action that takes a while.
    for (data_logger_index, data_logger) in enumerate(self._data_loggers):
      self._log_status('SensorManager starting external recordings via logger %d/%d' % (data_logger_index+1, len(self._data_loggers)))
      data_logger.start_external_recordings()
      
    # Choose a time in the future for streamers to start running.
    # This will allow them to synchronize by starting at the same time if needed.
    # It should account for some delay between telling a process to start and it actually starting.
    start_delay_s = 1.5*len(self._streamers)
    time_to_start_s = time.time() + start_delay_s
    self._log_status('SensorManager chose a time for streamers to start: %0.2f seconds from now (at %s)' % (time_to_start_s - time.time(), get_time_str(time_to_start_s)))

    # Create a callback function that determines whether the run loop should exit.
    def is_finished_running(start_time_s, duration_s, stopping_condition_fn, streamers_are_running_fn):
      finished_duration = (duration_s is not None) and (time.time() - start_time_s > duration_s)
      finished_callback = callable(stopping_condition_fn) and stopping_condition_fn()
      finished_streamers = not streamers_are_running_fn()
      return finished_duration or finished_callback or finished_streamers
    is_finished_running_fn = lambda: is_finished_running(time_to_start_s, duration_s, stopping_condition_fn, self.streamers_are_running)

    # Create a separate process for each streamer,
    #  unless it indicates it should be in the main process.
    self._sensor_processes = []
    for (streamer_index, streamer) in enumerate(self._streamers):
      if not streamer.get_threading_config('always_run_in_main_process'):
        sensor_process = multiprocessing.Process(
                           target=streamer.run,
                           args=(),
                           kwargs={'time_to_start_s': time_to_start_s})
        sensor_process.daemon = True
        self._sensor_processes.append(sensor_process)
      else:
        self._sensor_processes.append(None)
    # Start each streamer with its own process,
    #  and wait to make sure it actually starts.
    for (streamer_index, sensor_process) in enumerate(self._sensor_processes):
      if sensor_process is not None:
        streamer = self._streamers[streamer_index]
        self._log_status('SensorManager starting streamer %d/%d in its own process (%s)' % (streamer_index+1, len(self._streamers), type(streamer).__name__))
        sensor_process.start()
    # Start each streamer that is in the main process.
    for (streamer_index, sensor_process) in enumerate(self._sensor_processes):
      if sensor_process is None:
        self._log_status('SensorManager starting streamer %d/%d in the main process' % (streamer_index+1, len(self._streamers)))
        self._streamers[streamer_index].run(time_to_start_s=time_to_start_s)
    # Wait to make sure the streamers actually start.
    while time_to_start_s - time.time() > 1:
      time.sleep(0.1)
    for (streamer_index, sensor_process) in enumerate(self._sensor_processes):
      streamer = self._streamers[streamer_index]
      self._log_status('SensorManager waiting for streamer %d/%d to start (%s)' % (streamer_index+1, len(self._streamers), type(streamer).__name__))
      start_time_s = time.time()
      while (not streamer.is_running()) and (time.time() - start_time_s < 10):
        time.sleep(0.1)
      if not streamer.is_running():
        raise AssertionError('Streamer %s (1-indexed number %d) did not start running!' % (type(streamer).__name__, streamer_index+1))
        
    # Start any data loggers.
    # Use the main process since otherwise there can be memory sharing and serialization issues?
    for (data_logger_index, data_logger) in enumerate(self._data_loggers):
      self._log_status('SensorManager starting data logger %d/%d' % (data_logger_index+1, len(self._data_loggers)))
      data_logger.run()

    # Start any data visualizations.
    # Use the main process since otherwise there can be memory sharing and serialization issues?  And also GUI issues.
    keyboard_interrupted = False
    if self._data_visualizer_options['visualize_streaming_data']:
      self._log_status('SensorManager running streaming data visualizer and waiting until streaming is complete')
      try:
        self._data_visualizer.visualize_streaming_data(stopping_condition_fn=is_finished_running_fn)
      except KeyboardInterrupt:
        keyboard_interrupted = True

    # Wait for the desired timeout or stopping criteria.
    # Note that if a streaming data visualizer was run above, it already
    #   waited for the same criteria so the below loop should not actually run.
    if self._print_status and (not is_finished_running_fn() and not keyboard_interrupted):
      self._log_status('SensorManager waiting until streaming is complete')
    while not is_finished_running_fn() and not keyboard_interrupted:
      try:
        time.sleep(2)
      except KeyboardInterrupt:
        keyboard_interrupted = True

  # Stop each streamer and data logger.
  def stop(self):
    # Stop each streamer.
    # Note that the stop() method should be blocking and wait for the run thread to finish.
    for (streamer_index, streamer) in enumerate(self._streamers):
      self._log_status('\nSensorManager stopping streamer %d/%d of class %s' % (streamer_index+1, len(self._streamers), type(self._streamers[streamer_index]).__name__))
      streamer.stop()

    # Stop any external data recordings.
    for (data_logger_index, data_logger) in enumerate(self._data_loggers):
      self._log_status('SensorManager stopping external recordings via logger %d/%d' % (data_logger_index+1, len(self._data_loggers)))
      data_logger.stop_external_recordings()
    
    # Stop each data logger.
    # Note that the stop() method should be blocking and wait for writing to finish.
    for (data_logger_index, data_logger) in enumerate(self._data_loggers):
      self._log_status('\nSensorManager stopping data logger %d/%d' % (data_logger_index+1, len(self._data_loggers)))
      data_logger.stop()

    # Visualize all data if desired.
    if self._data_visualizer_options['visualize_all_data_when_stopped']:
      self._data_visualizer.visualize_all_data()

    # Wait for the user to close all visualization windows if desired.
    if self._data_visualizer_options['wait_while_visualization_windows_open']:
      if self._data_visualizer is not None:
        self._data_visualizer.wait_while_windows_open()
    
    # Close all visualization windows.
    if self._data_visualizer is not None:
      self._data_visualizer.close_visualizations()

    # All done!
    self._log_status('\nSensorManager successfully stopped!')

  # A helper to visualize all data (intended for the end of an experiment rather than streaming).
  def visualize_all_data(self):
    if self._data_visualizer is not None:
      self._data_visualizer.visualize_all_data()

  # A helper to wait until the user closes all visualizations.
  def wait_while_visualization_windows_open(self):
    if self._data_visualizer is not None:
      self._data_visualizer.wait_while_windows_open()

  # Check if streamers are still running.
  def streamers_are_running(self, all_streamers=False):
    streamers_running = [streamer.is_running() for streamer in self._streamers]
    if all_streamers:
      return (False not in streamers_running)
    else:
      return (True in streamers_running)

  ##############################
  ###### LOGGING/PRINTING ######
  ##############################

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

#####################
###### TESTING ######
#####################
if __name__ == '__main__':

  print_status = False
  print_debug = False

  # Define the streamers to use.
  sensor_streamer_specs = [
    {'class': 'DummyStreamer',
     'streamer_tag':'dummy1',
     'update_period_s': 0.5,
     'video': True,
     'print_debug': print_debug, 'print_status': print_status
     },
    {'class': 'DummyStreamer',
     'streamer_tag':'dummy2',
     'update_period_s': 0.1,
     'video': False,
     'sample_size': [3,4],
     'visualization_options': {'plot_duration_s': 5}, # override default of DummyStreamer
     'print_debug': print_debug, 'print_status': print_status
     },
    {'class': 'NotesStreamer',
     'print_debug': print_debug, 'print_status': print_status
     },
    # {'class': 'MyoStreamer',
    #  'num_myos': 1,
    #  'print_debug': print_debug, 'print_status': print_status
    #  },
  ]

  # Configure where and how to save sensor data.
  datalogging_options = None
  # (log_time_str, log_time_s) = get_time_str(return_time_s=True)
  # log_tag = 'testing'
  # log_dir_root = '../data_logs'
  # log_subdir = '%s_%s' % (log_time_str, log_tag)
  # log_dir = os.path.join(log_dir_root, log_subdir)
  # datalogging_options = {
  #   'log_dir': log_dir, 'log_tag': log_tag,
  #   # Choose whether to periodically write data to files.
  #   'stream_csv'  : False,
  #   'stream_hdf5' : True,
  #   'stream_video': False,
  #   'stream_audio': False,
  #   'stream_period_s': 5,
  #   # Choose whether to write all data at the end.
  #   'dump_csv'  : False,
  #   'dump_hdf5' : True,
  #   'dump_video': True,
  #   'dump_audio': True,
  #   # Additional configuration.
  #   'videos_format': 'avi', # mp4 occasionally gets openCV errors about a tag not being supported?
  #   'audio_format' : 'wav', # currently only supports WAV
  #   'print_status': print_status, 'print_debug': print_debug
  # }

  # Configure visualization.
  visualization_options = {
    'visualize_streaming_data'       : True,
    'visualize_all_data_when_stopped': True,
    'wait_while_visualization_windows_open': True,
    'update_period_s': 0.5,
    # 'classes_to_visualize': ['DummyStreamer']
  }

  # Create a sensor manager.
  sensor_manager = SensorManager(sensor_streamer_specs=sensor_streamer_specs,
                                 data_logger_options=datalogging_options,
                                 data_visualizer_options=visualization_options,
                                 print_status=print_status, print_debug=print_debug)
  # Run!
  sensor_manager.connect()
  sensor_manager.run(duration_s=3)
  sensor_manager.stop()




