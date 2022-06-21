
# Whether to use OpenCV or pyqtgraph to generate a composite visualization.
#   The OpenCV method is a bit faster, but the pyqtgraph method is interactive.
use_opencv_for_composite = True

if not use_opencv_for_composite:
  import pyqtgraph
  import pyqtgraph.exporters
  import numpy as np
  from pyqtgraph.Qt import QtCore, QtGui

import cv2
import numpy as np
import os
import time
from collections import OrderedDict

from utils.time_utils import *
from utils.dict_utils import *
from utils.print_utils import *
from utils.numpy_utils import *

################################################
################################################
# A class to visualize streaming data.
# SensorStreamer instances are passed to the class, and the data
#  that they stream can be visualized periodically or all at once.
# Each streamer that supports visualization should implement the following:
#   get_default_visualization_options(self, device_name, stream_name)
#   which returns a dict with 'class' and any additional options.
#  The specified class should inherit from visualizers.Visualizer.
#   To not show a visualization, 'class' can map to 'none' or None.
################################################
################################################
class DataVisualizer:

  ########################
  ###### INITIALIZE ######
  ########################

  def __init__(self, sensor_streamers, update_period_s=None,
               use_composite_video=False, composite_video_layout=None, composite_video_filepath=None,
               print_status=True, print_debug=False, log_history_filepath=None):
    # Validate the streamer objects, and make a list of them.
    if not isinstance(sensor_streamers, (list, tuple)):
      sensor_streamers = [sensor_streamers]
    if len(sensor_streamers) == 0:
      raise AssertionError('At least one SensorStreamer must be provided to DataLogger')
    self._streamers = list(sensor_streamers)

    # Record the configuration options.
    self._update_period_s = 2.0 if update_period_s is None else update_period_s
    self._use_composite_video = use_composite_video
    self._composite_video_layout = composite_video_layout
    self._composite_video_filepath = composite_video_filepath
    self._composite_video_tileBorder_width = max(1, round(1/8/100*sum([tile_layout['width'] for tile_layout in composite_video_layout[0]]))) if composite_video_layout is not None else None
    self._composite_video_tileBorder_color = [225, 225, 225] # BGR
    self._composite_frame_width = None # will be computed from the layout
    self._composite_frame_height = None # will be computed from the layout
    self._composite_frame_height_withTimestamp = None # will be computed from the layout and the timestamp height
    self._print_status = print_status
    self._print_debug = print_debug
    self._log_source_tag = 'vis'
    self._log_history_filepath = log_history_filepath
    
    self._print_debug_extra = False # Print debugging information for visualizers that probably isn't needed during normal experiment logging

    self._visualizers = []
    
    if composite_video_layout is not None:
      self._composite_video_timestamp_height = max(20, round(1/2/100*sum([tile_layout['width'] for tile_layout in composite_video_layout[0]])))
      self._composite_video_timestamp_bg_color = [100, 100, 100] # BGR
      self._composite_video_timestamp_color = [255, 255, 0] # BGR
      # Find a font scale that will make the text fit within the desired pad size.
      fontFace = cv2.FONT_HERSHEY_SIMPLEX
      fontScale = 5
      fontThickness = 2 if self._composite_video_timestamp_height > 25 else 1
      textsize = None
      while textsize is None or (textsize[1] > 0.8*self._composite_video_timestamp_height):
        textsize = cv2.getTextSize(get_time_str(0, format='%Y-%m-%d %H:%M:%S.%f'), fontFace, fontScale, fontThickness)[0]
        fontScale -= 0.2
      fontScale += 0.2
      self._composite_video_timestamp_fontScale = fontScale
      self._composite_video_timestamp_textSize = textsize
      self._composite_video_timestamp_fontThickness = fontThickness
      self._composite_video_timestamp_fontFace = fontFace

  # Initialize visualizers.
  # Will use the visualizer specified by each streamer,
  #  which may be a default visualizer defined in this file or a custom one.
  def init_visualizations(self, hide_composite=False):
    # Initialize the composite view.
    self._composite_video_writer = None
    self._composite_parent_layout = None
    self._composite_visualizer_layouts = None
    self._composite_visualizer_layout_sizes = None
    self._hide_composite = hide_composite
    # Validate the composite video configuration.
    if self._composite_video_layout is None and self._use_composite_video:
      raise AssertionError('Automatic composite video layout is not currently supported.')
    if self._use_composite_video:
      num_columns_perRow = np.array([len(row_layout) for row_layout in self._composite_video_layout])
      # assert (not any(np.diff(num_columns_perRow)) != 0), 'All rows of the composite video layout must have the same number of columns'
      heights = np.array([[tile_spec['height'] for tile_spec in row_layout if tile_spec] for row_layout in self._composite_video_layout])
      # assert (not np.any(np.diff(heights, axis=1) != 0)), 'All images in a row of the composite video must have the same height.'
      widths = np.array([[tile_spec['width'] for tile_spec in row_layout] for row_layout in self._composite_video_layout])
      # assert (not np.any(np.diff(widths, axis=0) != 0)), 'All images in a column of the composite video must have the same width.'

      self._composite_frame_width = np.sum(widths, axis=1)[0] # + 2*len(widths)*self._composite_video_tileBorder_width
      self._composite_frame_height = np.sum(heights, axis=0)[0] # + 2*len(widths)*self._composite_video_tileBorder_width
      self._composite_frame_height_withTimestamp = self._composite_frame_height + self._composite_video_timestamp_height
      
      # Get a list of streams included in the composite video.
      streams_in_composite = []
      for (row_index, row_layout) in enumerate(self._composite_video_layout):
        for (column_index, tile_info) in enumerate(row_layout):
          if tile_info['device_name'] is None:
            continue
          if tile_info['stream_name'] is None:
            continue
          streams_in_composite.append((tile_info['device_name'], tile_info['stream_name']))
          
      # Create a parent layout and sub-layouts for the composite visualization.
      if not use_opencv_for_composite:
        pyqtgraph.setConfigOption('background', 'w')
        pyqtgraph.setConfigOption('foreground', 'k')
        self._app = QtGui.QApplication([])
        # Define a top-level widget to hold everything
        self._composite_widget = QtGui.QWidget()
        if hide_composite:
          self._composite_widget.hide()
        # Create a grid layout to manage the widgets size and position
        self._composite_parent_layout = QtGui.QGridLayout()
        self._composite_widget.setLayout(self._composite_parent_layout)
        self._composite_widget.setWindowTitle('Composite Visualization')
        self._composite_widget.setGeometry(10, 50, self._composite_frame_width, self._composite_frame_height)
        self._composite_widget.show()
        # Create widgets for each streamer.
        self._composite_visualizer_layouts = {}
        self._composite_visualizer_layout_sizes = {}
        for (row_index, row_layout) in enumerate(self._composite_video_layout):
          for (column_index, tile_info) in enumerate(row_layout):
            device_name = tile_info['device_name']
            stream_name = tile_info['stream_name']
            rowspan = tile_info['rowspan']
            colspan = tile_info['colspan']
            width = tile_info['width']
            height = tile_info['height']
            if tile_info['device_name'] is None:
              continue
            if tile_info['stream_name'] is None:
              continue
            layout = pyqtgraph.GraphicsLayoutWidget()
            self._composite_parent_layout.addWidget(layout, row_index, column_index,
                                                    rowspan, colspan)
            self._composite_visualizer_layouts.setdefault(device_name, {})
            self._composite_visualizer_layouts[device_name][stream_name] = layout
            self._composite_visualizer_layout_sizes.setdefault(device_name, {})
            self._composite_visualizer_layout_sizes[device_name][stream_name] = (width, height)
            
      # Create a composite video writer if desired.
      if self._composite_video_filepath is not None:
        extension = 'avi'
        fourcc = 'MJPG'
        fps = 1/self._update_period_s
        self._composite_video_filepath = '%s.%s' % (os.path.splitext(self._composite_video_filepath)[0], extension)
        if use_opencv_for_composite:
          composite_video_frame_height = self._composite_frame_height_withTimestamp
        else:
          composite_video_frame_height = self._composite_frame_height
        composite_video_frame_width = self._composite_frame_width
        self._composite_video_writer = cv2.VideoWriter(self._composite_video_filepath,
                                                       cv2.VideoWriter_fourcc(*fourcc),
                                                       fps, (composite_video_frame_width, composite_video_frame_height))

    # Initialize a record of the next indexes that should be fetched for each stream,
    #  and how many timesteps to stay behind of the most recent step (if needed).
    self._next_data_indexes = [OrderedDict() for i in range(len(self._streamers))]
    self._timesteps_before_solidified = [OrderedDict() for i in range(len(self._streamers))]
    for (streamer_index, streamer) in enumerate(self._streamers):
      for (device_name, streams_info) in streamer.get_all_stream_infos().items():
        self._next_data_indexes[streamer_index][device_name] = OrderedDict()
        self._timesteps_before_solidified[streamer_index][device_name] = OrderedDict()
        for (stream_name, stream_info) in streams_info.items():
          self._next_data_indexes[streamer_index][device_name][stream_name] = 0
          self._timesteps_before_solidified[streamer_index][device_name][stream_name] \
                  = stream_info['timesteps_before_solidified']

    # Instantiate and initialize the visualizers.
    self._visualizers = [OrderedDict() for i in range(len(self._streamers))]
    for (streamer_index, streamer) in enumerate(self._streamers):
      for (device_name, streams_info) in streamer.get_all_stream_infos().items():
        self._visualizers[streamer_index][device_name] = OrderedDict()
        for (stream_name, stream_info) in streams_info.items():
          visualizer_options = streamer.get_visualization_options(device_name, stream_name)
          if callable(visualizer_options['class']):
            try:
              composite_visualizer_layout = self._composite_visualizer_layouts[device_name][stream_name]
              composite_visualizer_layout_size = self._composite_visualizer_layout_sizes[device_name][stream_name]
            except:
              composite_visualizer_layout = None
              composite_visualizer_layout_size = None
            visualizer = visualizer_options['class'](
                            visualizer_options,
                            hidden=self._use_composite_video,
                            parent_layout=composite_visualizer_layout,
                            parent_layout_size=composite_visualizer_layout_size,
                            print_debug=self._print_debug, print_status=self._print_status)
            visualizer.init(device_name, stream_name, stream_info)
          else:
            visualizer = None
          self._visualizers[streamer_index][device_name][stream_name] = visualizer

    # Initialize state for visualization control.
    self._last_update_time_s = None

  ##############################
  ###### VISUALIZING DATA ######
  ##############################

  # Periodically update the visualizations until a stopping criteria is met.
  # This function is blocking.
  def visualize_streaming_data(self, duration_s=None, stopping_condition_fn=None):
    # Initialize visualizations.
    self.init_visualizations()

    # Visualize the streaming data.
    self._log_status('DataVisualizer visualizing streaming data')
    start_time_s = time.time()
    while (duration_s is None or time.time() - start_time_s < duration_s) \
          and (not callable(stopping_condition_fn) or not stopping_condition_fn()):
      self.update_visualizations(wait_for_next_update_time=True)
  
  # Visualize data that is already logged, either from streaming or from replaying logs.
  # Periodically update the visualizations until a stopping criteria is met.
  # This function is blocking.
  def visualize_logged_data(self, start_offset_s=None, end_offset_s=None,
                                  start_time_s=None, end_time_s=None,
                                  duration_s=None,
                                  hide_composite=False, realtime=True):
    # Initialize visualizations.
    self.init_visualizations(hide_composite=hide_composite)
    
    # Determine reasonable start and end times.
    (start_time_s, end_time_s) = self.get_loggedData_start_end_times_s(
                                  start_offset_s=start_offset_s, end_offset_s=end_offset_s,
                                  start_time_s=start_time_s, end_time_s=end_time_s, duration_s=duration_s)
      
    # Visualize the existing data.
    self._log_status('DataVisualizer visualizing logged data')
    current_time_s = start_time_s
    current_frame_index = 0
    start_viz_time_s = time.time()
    while current_time_s <= end_time_s:
      self._log_debug('Visualizing for time %0.2f' % current_time_s)
      self.update_visualizations(wait_for_next_update_time=False, verify_next_update_time=False,
                                 ending_time_s=current_time_s, hide_composite=hide_composite)
      current_time_s += self._update_period_s
      current_frame_index += 1
      if realtime:
        next_update_time_s = start_viz_time_s + current_frame_index * self._update_period_s
        time.sleep(max(0, next_update_time_s - time.time()))
    
  # Determine reasonable start and end times for visualizing logged data.
  def get_loggedData_start_end_times_s(self, start_offset_s=None, end_offset_s=None,
                                        start_time_s=None, end_time_s=None,
                                        duration_s=None):
    start_times_s = []
    end_times_s = []
    for (streamer_index, streamer) in enumerate(self._streamers):
      streamer_start_times_s = []
      streamer_end_times_s = []
      for (device_name, streams_info) in streamer.get_all_stream_infos().items():
        for (stream_name, stream_info) in streams_info.items():
          streamer_start_time_s = streamer.get_start_time_s(device_name, stream_name)
          streamer_end_time_s = streamer.get_end_time_s(device_name, stream_name)
          if streamer_start_time_s is not None:
            streamer_start_times_s.append(streamer_start_time_s)
          if streamer_end_time_s is not None:
            streamer_end_times_s.append(streamer_end_time_s)
      # Use the earliest start time and the latest end time
      #  to indicate when any data from the streamer was available.
      if len(streamer_start_times_s) > 0:
        start_times_s.append(min(streamer_start_times_s))
        end_times_s.append(max(streamer_end_times_s))
    # # Choose the latest start time and the earliest end time,
    # #  so that data is always available from every streamer.
    # start_time_s = max(start_times_s)
    # end_time_s = min(end_times_s)
    # Choose the earliest start time and the latest end time,
    #  to cover all data from all streamers.
    if start_time_s is None:
      start_time_s = min(start_times_s)
    if end_time_s is None:
      end_time_s = max(end_times_s)
  
    # Adjust the start and end times if desired.
    if start_offset_s is not None:
      start_time_s += start_offset_s
    if end_offset_s is not None:
      end_time_s -= end_offset_s
    if duration_s is not None:
      end_time_s = start_time_s + duration_s
    
    # Return the results.
    return (start_time_s, end_time_s)
    
  # Fetch recent data from each streamer and visualize it.
  # If a poll period is set by self._visualize_period_s, can opt to check whether
  #  the next update time has been reached before proceeding by setting verify_next_update_time.
  #  If so, can wait for the time to arrive or return immediately by setting wait_for_next_update_time.
  # If an ending time is specified, will only fetch data up to that time.
  #  Otherwise, will fetch up to the end of the current log.
  def update_visualizations(self, wait_for_next_update_time=True, verify_next_update_time=True, ending_time_s=None, hide_composite=False):
    # Check whether it is time to show new data, which is when:
    #  This is the first iteration,
    #  it has been at least self._update_period_s since the last visualization, or
    #  no polling period was specified.
    if verify_next_update_time and self._update_period_s is not None:
      if self._print_debug_extra: self._log_debug('Visualization thread checking if the next update time is reached')
      next_update_time_s = (self._last_update_time_s or 0) + self._update_period_s
      if time.time() < next_update_time_s:
        # Return immediately or wait as appropriate.
        if not wait_for_next_update_time:
          return
        time.sleep(max(0, next_update_time_s - time.time()))
      # Update the last update time now, before the visualization actually starts.
      # This will keep the period more consistent; otherwise, the amount
      #   of time it takes to visualize would be added to the update period.
      #   This would compound over time, leading to longer delays and more data to display each time.
      #   This becomes more severe as the visualization duration increases.
      self._last_update_time_s = time.time()
    if self._print_debug_extra: self._log_debug('Visualization thread starting update')
    # Visualize new data for each stream of each device of each streamer.
    start_update_time_s = time.time()
    for (streamer_index, streamer) in enumerate(self._streamers):
      for (device_name, streams_info) in streamer.get_all_stream_infos().items():
        if self._print_debug_extra: self._log_debug('Visualizing streams for streamer %d device %s' % (streamer_index, device_name))
        for (stream_name, stream_info) in streams_info.items():
          # Check if a visualizer is created for this stream.
          visualizer = self._visualizers[streamer_index][device_name][stream_name]
          if visualizer is None:
            continue
          # Determine the start and end bounds for data to fetch.
          if ending_time_s is None:
            #  End at the most recent data (or back by a few timesteps
            #  if the streamer may still edit the most recent timesteps).
            ending_index = -self._timesteps_before_solidified[streamer_index][device_name][stream_name]
            if ending_index == 0: # no time is needed to solidify, so fetch up to the most recent data
              ending_index = None
          else:
            #  End at the specified time.
            ending_index = None
          # Start with the first timestep that hasn't been shown yet,
          #  or with just the last frame if getting video data.
          starting_index = self._next_data_indexes[streamer_index][device_name][stream_name]
          if stream_info['is_video']:
            if ending_time_s is None:
              if ending_index is None:
                starting_index = streamer.get_num_timesteps(device_name, stream_name) - 1
              else:
                starting_index = ending_index - 1
            else:
              ending_index_forTime = streamer.get_index_for_time_s(device_name, stream_name, ending_time_s, 'before')
              if ending_index_forTime is not None:
                ending_index_forTime += 1 # since will use as a list index and thus exclude the specified index
                starting_index = ending_index_forTime - 1
          # Get the data!
          start_get_data_time_s = time.time()
          new_data = streamer.get_data(device_name, stream_name, return_deepcopy=False,
                                        starting_index=starting_index,
                                        ending_index=ending_index, ending_time_s=ending_time_s)
          # self._log_status('Time to get data: \t%s \t \t%0.3f' % (type(streamer).__name__, time.time() - start_get_data_time_s))
          if new_data is not None:
            # Visualize any new data and save any updated sates.
            if visualizer is not None:
              start_visualizer_update_time_s = time.time()
              visualizer.update(new_data, visualizing_all_data=False)
              # self._log_status('Time to update vis: \t%s \t%s \t%s \t%0.3f' % (type(streamer).__name__, stream_name, type(visualizer).__name__, time.time() - start_visualizer_update_time_s))
            # Update starting indexes for the next write.
            num_new_entries = len(new_data['data'])
            next_starting_index = starting_index + num_new_entries
            self._next_data_indexes[streamer_index][device_name][stream_name] = next_starting_index
            if self._print_debug_extra: self._log_debug('Visualized %d new entries for stream %s.%s' % (num_new_entries, device_name, stream_name))
          else:
            # check if data has been cleared from memory, thus invalidating our start index.
            if streamer.get_num_timesteps(device_name, stream_name) < starting_index:
              self._next_data_indexes[streamer_index][device_name][stream_name] = 0
    if self._print_debug_extra: self._log_debug('Visualization thread finished update of each streamer')
    if self._print_debug_extra: self._log_debug('Time to update visualizers: \t \t \t \t%0.3f' % (time.time() - start_update_time_s))
    # self._log_status('Time to update visualizers: \t \t \t \t%0.3f' % (time.time() - start_update_time_s))
    # If showing a composite video, update it now that the streamers have updated their frames.
    if self._use_composite_video:
      start_composite_update_time_s = time.time()
      if use_opencv_for_composite:
        self._update_composite_video_opencv(hidden=hide_composite, time_s=ending_time_s or self._last_update_time_s)
      else:
        self._update_composite_video_pyqtgraph(hidden=hide_composite, time_s=ending_time_s or self._last_update_time_s)
      if self._print_debug_extra: self._log_debug('Time to update composite visualization: \t \t \t \t%0.3f' % (time.time() - start_composite_update_time_s))
    self._log_debug('Time to update visualizers and composite: %0.4f' % (time.time() - start_update_time_s))

  # Visualize all data currently in the streamers' memory.
  # This is meant to be called at the end of an experiment.
  # This may interfere with windows/figures created by update_visualizations()
  #  so it is recommended to close all existing visualizations first.
  def visualize_all_data(self):
    # Initialize visualizations.
    self.init_visualizations()

    # Visualize all recorded data.
    self._log_status('DataVisualizer visualizing all data')
    for (streamer_index, streamer) in enumerate(self._streamers):
      for (device_name, streams_info) in streamer.get_all_stream_infos().items():
        if self._print_debug_extra: self._log_debug('Visualizing streams for streamer %d device %s' % (streamer_index, device_name))
        for (stream_name, stream_info) in streams_info.items():
          # Fetch data starting with the first timestep,
          #  and ending at the most recent data (or back by a few timesteps
          #  if the streamer may still edit the most recent timesteps).
          starting_index = 0
          ending_index = -self._timesteps_before_solidified[streamer_index][device_name][stream_name]
          if ending_index == 0: # no time is needed to solidify, so fetch up to the most recent data
            ending_index = None
          new_data = streamer.get_data(device_name, stream_name, return_deepcopy=False,
                                        starting_index=starting_index, ending_index=ending_index)
          if new_data is not None:
            # Visualize any new data and save any updates states.
            visualizer = self._visualizers[streamer_index][device_name][stream_name]
            if visualizer is not None:
              visualizer.update(new_data, visualizing_all_data=True)
            if self._print_debug_extra: self._log_debug('Visualized %d new entries for stream %s.%s' % (len(new_data['data']), device_name, stream_name))

  # Create a composite video from all streamers that are creating visualizations.
  def _update_composite_video_opencv(self, hidden=False, time_s=None):
    if self._print_debug_extra: self._log_debug('DataVisualizer updating the composite video using OpenCV')
    # Get the latest images from each streamer.
    imgs = []
    for (row_index, row_layout) in enumerate(self._composite_video_layout):
      imgs.append([])
      for (column_index, tile_info) in enumerate(row_layout):
        if tile_info['device_name'] is None:
          imgs[-1].append(None)
          continue
        # Find the streamer for this tile.
        img = None
        for visualizers_streamer in self._visualizers:
          for (device_name, visualizers_device) in visualizers_streamer.items():
            if device_name != tile_info['device_name']:
              continue
            for (stream_name, visualizer) in visualizers_device.items():
              if stream_name != tile_info['stream_name']:
                continue
              if visualizer is None:
                continue
              # Get the latest image
              try:
                img = visualizer.get_visualization_image(device_name, stream_name)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
              except AttributeError:
                # The streamer likely hasn't actually made a figure yet,
                #  so just use a black image for now.
                img = np.zeros((100, 100, 3), dtype=np.uint8) # will be resized later
        # Append the image, or a blank image if the device/stream was not found.
        if img is None:
          img = np.zeros((100, 100, 3), dtype=np.uint8) # will be resized later
        imgs[-1].append(img)
    # Resize the images.
    for (row_index, row_layout) in enumerate(self._composite_video_layout):
      for (column_index, tile_info) in enumerate(row_layout):
        img = imgs[row_index][column_index]
        if img is None:
          continue
        img_width = img.shape[1]
        img_height = img.shape[0]
        target_width = tile_info['width'] - 2*self._composite_video_tileBorder_width
        target_height = tile_info['height'] - 2*self._composite_video_tileBorder_width
        # Check if the width or height will be the controlling dimension.
        scale_factor_fromWidth = target_width/img_width
        scale_factor_fromHeight = target_height/img_height
        if img_height*scale_factor_fromWidth > target_height:
          scale_factor = scale_factor_fromHeight
        else:
          scale_factor = scale_factor_fromWidth
        # Resize the image.
        img = cv2.resize(img, (0,0), None, scale_factor, scale_factor)
        # Pad the image to fill the tile.
        img_width = img.shape[1]
        img_height = img.shape[0]
        pad_top = int(max(0, (target_height - img_height)/2))
        pad_bottom = (target_height - (img_height+pad_top))
        pad_left = int(max(0, (target_width - img_width)/2))
        pad_right = (target_width - (img_width+pad_left))
        pad_color = [225, 225, 225]
        img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right,
                                 cv2.BORDER_CONSTANT, value=pad_color)
        # Add a border around each tile
        border_color = self._composite_video_tileBorder_color
        border_width = self._composite_video_tileBorder_width
        img = cv2.copyMakeBorder(img, border_width, border_width, border_width, border_width,
                                 cv2.BORDER_CONSTANT, value=border_color)
        imgs[row_index][column_index] = img
    # # Create the composite image.
    # composite_row_imgs = []
    # for (row_index, row_imgs) in enumerate(imgs):
    #   composite_row_imgs.append(cv2.hconcat(row_imgs))
    # composite_img = cv2.vconcat(composite_row_imgs)
    # Create the composite image.
    composite_column_imgs = []
    for column_index in range(len(imgs[0])):
      imgs_for_column = []
      for (row_index, row_imgs) in enumerate(imgs):
        if row_imgs[column_index] is not None:
          imgs_for_column.append(row_imgs[column_index])
      composite_column_imgs.append(cv2.vconcat(imgs_for_column))
    composite_img = cv2.hconcat(composite_column_imgs)
    if time_s is not None:
      # Determine a font size that will fit in the desired pad size.
      timestamp_str = get_time_str(time_s, format='%Y-%m-%d %H:%M:%S.%f')
      # Add the padding and the text.
      composite_img = cv2.copyMakeBorder(composite_img, 0, self._composite_video_timestamp_height, 0, 0,
                                         cv2.BORDER_CONSTANT, value=self._composite_video_timestamp_bg_color)
      composite_img = cv2.putText(composite_img, timestamp_str,
                                  [int(composite_img.shape[1]/2 - self._composite_video_timestamp_textSize[0]/2),
                                   int(composite_img.shape[0] - (self._composite_video_timestamp_height-self._composite_video_timestamp_textSize[1])/2)],
                                  fontFace=self._composite_video_timestamp_fontFace, fontScale=self._composite_video_timestamp_fontScale,
                                  color=self._composite_video_timestamp_color, thickness=self._composite_video_timestamp_fontThickness, lineType=cv2.LINE_AA)
    
    # Display the composite image if desired.
    if not hidden:
      cv2.imshow('Action!', composite_img)
      cv2.waitKey(1)
    # Write the composite image to a video if desired.
    if self._composite_video_writer is not None:
      self._composite_video_writer.write(composite_img)
      
  def _update_composite_video_pyqtgraph(self, hidden=False, time_s=None):
    if self._print_debug_extra: self._log_debug('DataVisualizer updating the composite video using pyqtgraph')
    # Display the composite image if desired.
    if not hidden:
      cv2.waitKey(1) # find a better way?
    # Write the composite image to a video if desired.
    if self._composite_video_writer is not None:
      composite_img = self._composite_widget.grab() # returns a QPixmap
      composite_img = composite_img.toImage() # returns a QImage
      composite_img = self._convertQImageToMat(composite_img)
      composite_img = composite_img[:,:,0:3]
      scale_factor_width = self._composite_frame_width / composite_img.shape[1]
      scale_factor_height = self._composite_frame_height / composite_img.shape[0]
      composite_img = cv2.resize(composite_img, (0,0), None, scale_factor_width, scale_factor_height)
      self._composite_video_writer.write(composite_img)

  # Convert a QImage to a numpy ndarray in BGR format.
  def _convertQImageToMat(self, qimg):
    img = qimg.convertToFormat(QtGui.QImage.Format.Format_RGB32)
    ptr = img.bits()
    ptr.setsize(img.sizeInBytes())
    arr = np.array(ptr).reshape(img.height(), img.width(), 4)  #  Copies the data
    return arr
  
  # Keep line plot figures responsive if any are active.
  # Wait for user to press a key or close the windows if any videos are active.
  # Note that this will only work if called on the main thread.
  def wait_while_windows_open(self):
    # Get a list of waiting functions for the various visualizers.
    waiting_functions = []
    for visualizers_streamer in self._visualizers:
      for (device_name, visualizers_device) in visualizers_streamer.items():
        for (stream_name, visualizer) in visualizers_device.items():
          if visualizer is not None:
            waiting_functions.append(visualizer.wait_for_user_to_close)
    # Wait for all of the functions to be satisfied.
    if len(waiting_functions) > 0:
      self._log_userAction('\n\n*** Close all visualization windows to exit\n\n')
    for wait_fn in waiting_functions:
      wait_fn()
    if self._use_composite_video and not self._hide_composite:
      if use_opencv_for_composite:
        cv2.waitKey(0)
      else:
        self._app.exec()

  # Close line plot figures, video windows, and custom visualizers.
  def close_visualizations(self):
    for visualizers_streamer in self._visualizers:
      for (device_name, visualizers_device) in visualizers_streamer.items():
        for (stream_name, visualizer) in visualizers_device.items():
          if visualizer is not None:
            try:
              visualizer.close()
            except:
              pass
    if self._composite_video_writer is not None:
      self._composite_video_writer.release()
      time.sleep(2)
    try:
      cv2.destroyAllWindows()
    except:
      pass
    try:
      self._app.quit()
    except:
      pass

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

  from sensor_streamers.NotesStreamer import NotesStreamer
  from sensor_streamers.MyoStreamer   import MyoStreamer
  from sensor_streamers.XsensStreamer import XsensStreamer
  from sensor_streamers.TouchStreamer import TouchStreamer
  from sensor_streamers.EyeStreamer   import EyeStreamer
  from sensor_streamers.DummyStreamer   import DummyStreamer
  import time

  # Configure printing to the console.
  print_debug = False
  print_status = True

  # Create the streamers.
  dummy_streamer_video = DummyStreamer(update_period_s=0.2,
                                 streamer_tag='dummy1',
                                 visualization_type='video', # can override sample_size
                                 print_debug=print_debug, print_status=print_status)
  dummy_streamer_lines = DummyStreamer(sample_size=[3,4], update_period_s=0.2,
                                 streamer_tag='dummy2',
                                 visualization_type='line',
                                 print_debug=print_debug, print_status=print_status)
  dummy_streamer_heatmap = DummyStreamer(sample_size=[32,32], update_period_s=0.2,
                                 streamer_tag='dummy3',
                                 visualization_type='heatmap',
                                 print_debug=print_debug, print_status=print_status)
  dummy_streamer_xsens = DummyStreamer(sample_size=[23,3], update_period_s=0.2,
                                 streamer_tag='dummy3',
                                 visualization_type='xsens-skeleton',
                                 print_debug=print_debug, print_status=print_status)
  # notes_streamer = NotesStreamer(print_debug=print_debug, print_status=print_status)
  # myo_streamer = MyoStreamer(num_myos=2, print_debug=print_debug, print_status=print_status)
  # xsens_streamer = XsensStreamer(print_debug=print_debug, print_status=print_status)
  # touch_streamer = TouchStreamer(print_debug=print_debug, print_status=print_status)
  # eye_streamer = EyeStreamer(stream_video_world=True, stream_video_worldGaze=True,
  #                             stream_video_eye=True,
  #                             print_debug=print_debug, print_status=print_status)

  dummy_streamers = [
      dummy_streamer_lines,
      dummy_streamer_heatmap,
      dummy_streamer_video,
      dummy_streamer_xsens,
    ]
  streamers = [
      # notes_streamer,
      # myo_streamer,
      # xsens_streamer,
      # touch_streamer,
      # eye_streamer,
    ]
  streamers.extend(dummy_streamers)

  # Create a visualizer to display the streaming data.
  data_visualizer = DataVisualizer(streamers, update_period_s=0.5,
                                    print_status=print_status, print_debug=print_debug)

  # Start the streamers.
  for dummy_streamer in dummy_streamers:
    dummy_streamer.connect()
  for dummy_streamer in dummy_streamers:
    dummy_streamer.run()

  # Run.
  duration_s = 5
  data_visualizer.visualize_streaming_data(duration_s=duration_s,
                                           stopping_condition_fn=None)

  # Stop the streamers.
  for dummy_streamer in dummy_streamers:
    dummy_streamer.stop()

  # Visualize all data.
  data_visualizer.visualize_all_data()

  # Wait for user to close windows.
  data_visualizer.wait_while_windows_open()









