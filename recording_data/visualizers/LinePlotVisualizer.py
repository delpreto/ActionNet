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

# Whether to use matplotlib or pyqtgraph.
#   pyqtgraph is faster.
use_matplotlib = False

from visualizers.Visualizer import Visualizer
from sensor_streamers.SensorStreamer import SensorStreamer

if use_matplotlib:
  import matplotlib
  import matplotlib.pyplot as plt
else:
  import pyqtgraph
  import pyqtgraph.exporters
  from pyqtgraph.Qt import QtCore, QtGui
  import cv2

import numpy as np

from utils.numpy_scipy_utils import *


################################################
# Plot data as one or more line graphs.
# May put all elements of N-D data on a single graph or in subplots.
################################################
class LinePlotVisualizer(Visualizer):

  def __init__(self, visualizer_options=None, hidden=False,
                     parent_layout=None, parent_layout_size=None,
                     print_debug=False, print_status=False):
    Visualizer.__init__(self, visualizer_options=visualizer_options, hidden=hidden,
                        parent_layout=parent_layout, parent_layout_size=parent_layout_size,
                        print_debug=print_debug, print_status=print_status)
    
    self._plot_length = None
    self._time_s = None
    self._data = None
    self._plotting_start_time_s = None
    self._extract_data_for_axis_fn = None
    if use_matplotlib:
      self._fig = None
      self._axs = None
      self._use_blitting = False # speeds up drawing, but axis labels are fixed
    else:
      self._app = QtGui.QApplication([])
      self._layout = parent_layout
      self._is_sub_layout = parent_layout is not None
      self._layout_size = parent_layout_size
      self._plots = None
    
    if use_matplotlib:
      # If using hidden mode, change matplotlib's backend.
      if self._hidden:
        matplotlib.use("Agg")
      matplotlib.style.use('fast') # doesn't seem to impact speed much for tested plots

  # Initialize a visualization that plots data as line graph(s).
  def init(self, device_name, stream_name, stream_info):
    if self._print_debug: print('LinePlotVisualizer initializing for %s %s' % (device_name, stream_name))

    # Process default plotting options.
    self._visualizer_options.setdefault('single_graph', True)
    self._visualizer_options.setdefault('downsample_factor', 1)
    self._visualizer_options.setdefault('plot_duration_s', 30)
    # Get the plotting options.
    single_graph = self._visualizer_options['single_graph']
    plot_duration_s = self._visualizer_options['plot_duration_s']
    downsample_factor = self._visualizer_options['downsample_factor']
    sample_size = stream_info['sample_size']
    sampling_rate_hz = stream_info['sampling_rate_hz']
    if sampling_rate_hz is not None:
      plot_length = int(plot_duration_s*sampling_rate_hz) + 1
    else:
      # Take a guess at a reasonable sampling rate.
      plot_length = int(plot_duration_s*50) + 1
    plot_length_downsampled = int((plot_length-1)/downsample_factor)+1
    if use_matplotlib:
      figure_size = (5.5, 3.5)
    else:
      if self._layout_size is None:
        # screen_widths = [screen.size().width() for screen in app.screens()]
        # screen_heights = [screen.size().heights() for screen in app.screens()]
        screen_width = self._app.primaryScreen().size().width()
        screen_height = self._app.primaryScreen().size().height()
        figure_width = int(screen_width*0.5)
        figure_height = int(figure_width/1.5)
        figure_size = (figure_width, figure_height)
      else:
        figure_size = self._layout_size
    
    # Create a function pointer that takes an array of new data
    #  and returns a list of 1D arrays to plot on the specified subplot.
    # If single_graph is False, the returned list will always just contain 1 element.
    extract_data_for_axis_fn = None
    if single_graph:
      num_rows = 1
      num_columns = 1
      def f(new_data, row, col):
        data_toParse = new_data.copy()
        if data_toParse.ndim == 1:
          data_toParse = data_toParse[:,None]
        data_toVisualize = []
        for element_index in range(np.prod(sample_size)):
          element_subs = np.unravel_index(element_index, sample_size)
          element_slices = [slice(sub,sub+1) for sub in element_subs]
          element_slices.insert(0, slice(None))
          element_slices = tuple(element_slices)
          data_toVisualize.append(np.atleast_1d(np.squeeze(data_toParse[element_slices])))
        return data_toVisualize
      extract_data_for_axis_fn = f
    else:
      num_elements = np.prod(sample_size)
      if len(sample_size) == 1:
        num_rows = num_elements
        num_columns = 1
        extract_data_for_axis_fn = lambda new_data, row, col: [new_data[:, row]]
      elif len(sample_size) == 2:
        num_rows = sample_size[0]
        num_columns = sample_size[1]
        extract_data_for_axis_fn = lambda new_data, row, col: [new_data[:, row, col]]
      else:
        num_rows = np.ceil(np.sqrt(num_elements))
        num_columns = np.ceil(num_elements/num_rows)
        def f(new_data, row, col):
          element_index = np.ravel_multi_index((row, col), (num_rows, num_columns))
          element_subs = np.unravel_index(element_index, sample_size)
          element_slices = [slice(sub,sub+1) for sub in element_subs]
          element_slices.insert(0, slice(None))
          return [np.squeeze(new_data[element_slices])]
        extract_data_for_axis_fn = f
    # Determine line titles for each line that will be plotted.
    # These will become subplot titles or legend entries as appropriate.
    if isinstance(stream_info['data_notes'], dict) and SensorStreamer.metadata_data_headings_key in stream_info['data_notes']:
      line_titles = stream_info['data_notes'][SensorStreamer.metadata_data_headings_key]
    else:
      # Create a line title for each element in each data sample.
      # Each sample may be a matrix that will be unwrapped into separate lines,
      #  so label them as i-j where i is the original matrix row
      #  and j is the original matrix column (and if more than 2D keep adding more).
      line_titles = []
      subs = np.unravel_index(range(0,np.prod(sample_size)), sample_size)
      subs = np.stack(subs).T
      for title_index in range(subs.shape[0]):
        title = 'Element '
        for sub_index in range(subs.shape[1]):
          title += '%d-' % subs[title_index, sub_index]
        title = title.strip('-')
        line_titles.append(title)
    
    # Create the plot windows
    if use_matplotlib:
      # Set interactive mode.
      # In interactive mode:
      #  - newly created figures will be shown immediately;
      #  - figures will automatically redraw on change;
      #  - pyplot.show will not block by default.
      # In non-interactive mode:
      #  - newly created figures and changes to figures will not be reflected until explicitly asked to be;
      #  - pyplot.show will block by default.
      plt.ioff()
  
      # Create a figure and subplots.
      fig, axs = plt.subplots(nrows=num_rows, ncols=num_columns,
                               squeeze=False, # if False, always return 2D array of axes
                               sharex=True, sharey=True,
                               subplot_kw={'frame_on': True},
                               figsize=figure_size
                               )
    else:
      pyqtgraph.setConfigOption('background', 'w')
      pyqtgraph.setConfigOption('foreground', 'k')
      # Create the main window if one was not provided.
      if not self._is_sub_layout:
        self._layout = pyqtgraph.GraphicsLayoutWidget(show=True)
        self._layout.setGeometry(10, 10, *figure_size)
      self._plots = []
      for row in range(num_rows):
        self._plots.append([])
        for column in range(num_columns):
          self._plots[row].append(self._layout.addPlot(row=row, col=column))
          if single_graph:
            self._plots[row][-1].addLegend(offset=(0, 0))
      self._plots = np.array(self._plots)
      if self._hidden and not self._is_sub_layout:
        self._layout.hide()
      self._exporter = pyqtgraph.exporters.ImageExporter(self._layout.scene())
      
    # Create time/data arrays for each axis,
    #  and use it to initialize the plots.
    # data will then be a 2D array, with data[row][col]
    #  being a list of data arrays to plot in that subplot.
    dummy_data = np.empty((plot_length, *sample_size))
    dummy_data[:] = 0 if (not use_matplotlib or self._use_blitting) else np.nan
    time_s = np.linspace(-plot_duration_s, 0, num=plot_length)
    time_s_downsampled = time_s[0::downsample_factor]
    data = []
    h_lines = []
    for row in range(num_rows):
      data.append([])
      h_lines.append([])
      for column in range(num_columns):
        h_lines[-1].append([])
        axis_datas = extract_data_for_axis_fn(dummy_data, row, column)
        data[-1].append(axis_datas)
        if use_matplotlib:
          axis = axs[row][column]
          plt.sca(axis)
          for axis_data in axis_datas:
            axis_data_downsampled = axis_data[0::downsample_factor]
            (h_line,) = axis.plot(time_s_downsampled, axis_data_downsampled, '-', animated=self._use_blitting)
            h_lines[row][column].append(h_line)
        else:
          plot = self._plots[row][column]
          for (index, axis_data) in enumerate(axis_datas):
            axis_data_downsampled = axis_data[0::downsample_factor]
            color = pyqtgraph.intColor(index, hues=max(9, len(axis_datas)))
            plot_kwargs = {'pen': pyqtgraph.mkPen(color=color, width=2),
                           'symbol': None, # 'o'
                           'symbolPen': pyqtgraph.mkPen(color=color, width=2),
                           'symbolBrush': color}
            if single_graph:
              plot_kwargs['name'] = line_titles[index]
            h_line = plot.plot(time_s_downsampled, axis_data_downsampled, **plot_kwargs)
            h_lines[row][column].append(h_line)
        
    # Set formatting options for each subplot.
    titles_byAxs = []
    for row in range(num_rows):
      titles_byAxs.append([])
      for column in range(num_columns):
        titles_byAxs[row].append(line_titles[np.ravel_multi_index((row, column), (num_rows, num_columns))])
    if use_matplotlib:
      for row in range(num_rows):
        for column in range(num_columns):
          plt.sca(axs[row][column])
          plt.grid(True, color='lightgray')
          if single_graph:
            axs[row][column].legend(line_titles)
          else:
            plt.title(titles_byAxs[row][column])
          if row == (num_rows-1):
            axs[row][column].set_xlabel('Time [s]')
      fig.suptitle('%s: %s' % (device_name, stream_name))
      # Show the figure.
      if not self._hidden:
        fig.show()
      # Wait to make sure the figure is fully drawn before drawing artists or copying it as the background.
      plt.pause(0.1)
      if self._use_blitting:
        # Get a copy of the figure without the animated artists drawn.
        fig_bg = fig.canvas.copy_from_bbox(fig.bbox)
        # Draw the animated artists.
        for row in range(num_rows):
          for column in range(num_columns):
            for h_line in h_lines[row][column]:
              axs[row][column].draw_artist(h_line)
        fig.canvas.blit(fig.clipbox)
      else:
        fig_bg = None
      # Save state for future updates.
      self._fig_bg = fig_bg
    else:
      axis_tick_font = QtGui.QFont('arial', pointSize=10)
      for row in range(num_rows):
        for column in range(num_columns):
          for axis_side in ['left', 'bottom', 'top']:
            plot = self._plots[row][column].getAxis(axis_side)
            plot.setPen('k')
            plot.setGrid(grid=True)
            plot.setTickFont(axis_tick_font)
          self._plots[row][column].showGrid(x=True, y=True, alpha=0.8)
          if row == 0:
            self._plots[row][column].setTitle('%s: %s' % (device_name, stream_name),  size='10pt')
          if row == (num_rows-1):
            labelStyle = {'color': '#000', 'font-size': '10pt'}
            # self._plots[row][column].setLabel('bottom', 'Time [s]', **labelStyle)
          else:
            self._plots[row][column].hideAxis('bottom')
          self._layout.setWindowTitle('%s: %s' % (device_name, stream_name))
      # Update the plot window
      if not self._hidden:
        QtCore.QCoreApplication.processEvents()

    # Save state for future updates.
    self._plot_length = plot_length
    self._plot_length_downsampled = plot_length_downsampled
    self._downsample_factor = downsample_factor
    self._time_s = time_s
    self._data = data
    self._h_lines = h_lines
    self._plotting_start_time_s = None
    self._extract_data_for_axis_fn = extract_data_for_axis_fn
    if use_matplotlib:
      self._fig = fig
      self._axs = axs
    
  # Update the line graph(s) with new data.
  # @param new_data is a dict with 'data', 'time_s', 'time_str', and any other extra channels for the stream.
  #   The data may contain multiple timesteps (each value may be a list).
  # @param visualizing_all_data is whether this is being called
  #   as part of a periodic update loop or in order to show all data at once.
  #   If periodically, new data should be added to the visualization if applicable.
  #   Otherwise the new data should replace the visualization if applicable.
  def update(self, new_data, visualizing_all_data):
    
    # Subtract the starting time, so graph x axes start at 0.
    new_time_s = np.atleast_1d(np.array(new_data['time_s']))
    if self._plotting_start_time_s is None:
      self._plotting_start_time_s = min(new_time_s)
    new_time_s = new_time_s - self._plotting_start_time_s
    # Update the time, which is assumed to be the same across subplots.
    # Insert the new time at the end of the array, shifting back and popping old times.
    self._time_s = add_to_rolling_array(self._time_s, new_time_s)

    # Update the plots!
    
    # Reset the figure to the pre-plot canvas state.
    if use_matplotlib and self._use_blitting:
      self._fig.canvas.restore_region(self._fig_bg)
    # For each subplot, add the relevant portion of the new data.
    # Also trim data to be the appropriate length for the plot.
    new_data_data = np.atleast_1d(np.array(new_data['data']))
    num_rows = self._axs.shape[0] if use_matplotlib else self._plots.shape[0]
    num_columns = self._axs.shape[1] if use_matplotlib else self._plots.shape[1]
    for row in range(num_rows):
      for column in range(num_columns):
        if use_matplotlib:
          ax = self._axs[row][column]
        data_forSubplot = self._data[row][column]
        new_data_forSubplot = self._extract_data_for_axis_fn(new_data_data, row, column)
        # Clear the subplot if visualizing all data at once.
        if use_matplotlib and visualizing_all_data:
          for artist in ax.lines + ax.collections:
            artist.remove()
        # Plot each desired line (each element of the data).
        for (line_index, data_forLine) in enumerate(data_forSubplot):
          # Shift/pop the existing data back, and append the new at the end.
          new_data_forLine = new_data_forSubplot[line_index]
          if not visualizing_all_data:
            data_forLine = add_to_rolling_array(data_forLine, new_data_forLine)
          else:
            self._time_s = new_time_s
            data_forLine = new_data_forLine
          # Save the updated data.
          self._data[row][column][line_index] = data_forLine
          # Downsample if desired.
          time_s_downsampled = self._time_s[0::self._downsample_factor]
          data_forLine_downsampled = data_forLine[0::self._downsample_factor]
          # Update the subplot's time/data
          # (or replace it if visualizing all data, since then sizes may have changed).
          if use_matplotlib:
            h_line = ax.get_lines()[line_index]
            if not visualizing_all_data:
              h_line.set_xdata(time_s_downsampled)
              h_line.set_ydata(data_forLine_downsampled)
            else:
              (h_line,) = ax.plot(time_s_downsampled, data_forLine_downsampled, '-', animated=self._use_blitting)
            # Rescale the axes.
            ax.relim()
            ax.autoscale_view(scalex=True, scaley=True)
            # Re-render the artist.
            if self._use_blitting:
              ax.draw_artist(h_line)
          else:
            h_line = self._h_lines[row][column][line_index]
            h_line.setData(time_s_downsampled, data_forLine_downsampled)
    # Update the figure to see the changes.
    if use_matplotlib:
      if self._use_blitting:
        self._fig.canvas.blit(self._fig.clipbox)
        self._fig.canvas.flush_events()
      else:
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
    else:
      if not self._hidden:
        QtCore.QCoreApplication.processEvents() # update the plot window (yes, this works)

  # Retrieve an image of the most updated visualization.
  # Should return a matrix in RGB format.
  def get_visualization_image(self, device_name, stream_name):
    # start_export_time_s = time.time()
    if use_matplotlib:
      # Convert the figure canvas to an image.
      img = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
      img = img.reshape(self._fig.canvas.get_width_height()[::-1] + (3,))
      # if self._print_debug: print('Time to export line plot for %s.%s: %0.3fs', (device_name, stream_name, time.time() - start_export_time_s))
      return img
    else:
      img = self._exporter.export(toBytes=True)
      img = self._convertQImageToMat(img)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      # if self._print_debug: print('Time to export line plot for %s.%s: %0.3fs', (device_name, stream_name, time.time() - start_export_time_s))
      return img

  # Convert a QImage to a numpy ndarray in BGR format.
  def _convertQImageToMat(self, qimg):
    img = qimg.convertToFormat(QtGui.QImage.Format.Format_RGB32)
    ptr = img.bits()
    ptr.setsize(img.sizeInBytes())
    arr = np.array(ptr).reshape(img.height(), img.width(), 4)  #  Copies the data
    return arr
  
  # Close the figure.
  def close(self):
    if use_matplotlib:
      plt.close(self._fig)
    else:
      if not self._hidden and not self._is_sub_layout:
        self._layout.close()
        self._app.quit()

  # Wait for the user to close the figure.
  def wait_for_user_to_close(self):
    if not self._hidden:
      if use_matplotlib:
        plt.show(block=True)
      else:
        if not self._is_sub_layout:
          self._app.exec()











