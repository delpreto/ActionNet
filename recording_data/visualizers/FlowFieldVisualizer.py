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

from visualizers.Visualizer import Visualizer

import pyqtgraph
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.exporters
import cv2
import numpy as np
import time

################################################
# Visualize 2D matrix data as a heatmap.
################################################
class FlowFieldVisualizer(Visualizer):
  
  def __init__(self, visualizer_options=None, hidden=False,
               parent_layout=None, parent_layout_size=None,
               print_debug=False, print_status=False):
    Visualizer.__init__(self, visualizer_options=visualizer_options, hidden=hidden,
                        parent_layout=parent_layout, parent_layout_size=parent_layout_size,
                        print_debug=print_debug, print_status=print_status)
    
    self._app = QtGui.QApplication([])
    self._layout = parent_layout
    self._is_sub_layout = parent_layout is not None
    self._layout_size = parent_layout_size
    self._data = None
    self._plot = None
    self._auto_magnitude_normalization = True # can be overridden by supplied options in init()
    self._magnitude_normalization_buffer = []
    self._magnitude_normalization_buffer_start_time_s = None
    self._magnitude_normalization_buffer_duration_s = None # will be computed at first update time
    self._magnitude_normalization_buffer_length = None # will be computed at first update time
    self._magnitude_normalization = None
    self._magnitude_normalization_update_period_s = None
    self._next_magnitude_normalization_update_time_s = 0

  # Define a function for getting line segment coordinates and scatter coordinates
  # from matrices of magnitudes/angles.
  def _vector_to_plotPoints(self, magnitudes_scaled, angles_rad):
    
    dx = magnitudes_scaled*np.cos(angles_rad)
    dy = magnitudes_scaled*np.sin(angles_rad)
    
    x1 = self._plotPoints_x0+dx
    y1 = self._plotPoints_y0+dy
    
    x0_1d = np.reshape(self._plotPoints_x0, (1, self._plotData_size_n))
    x1_1d = np.reshape(x1, (1, self._plotData_size_n))
    y0_1d = np.reshape(self._plotPoints_y0, (1, self._plotData_size_n))
    y1_1d = np.reshape(y1, (1, self._plotData_size_n))
    x = np.squeeze(np.concatenate((x0_1d, x1_1d), axis=0).T.reshape((-1, 1)))
    y = np.squeeze(np.concatenate((y0_1d, y1_1d), axis=0).T.reshape((-1, 1)))
    
    nan_mask_scatter = np.squeeze(np.reshape(np.isnan(magnitudes_scaled) | np.isnan(angles_rad), (1, self._plotData_size_n)))
    nan_mask_scatter = np.atleast_2d(nan_mask_scatter)
    
    return {'scatter': (x0_1d, y0_1d),
            'scatter_not_nan': (np.atleast_2d(x0_1d[~nan_mask_scatter]), np.atleast_2d(y0_1d[~nan_mask_scatter])),
            'scatter_nan': (np.atleast_2d(x0_1d[nan_mask_scatter]), np.atleast_2d(y0_1d[nan_mask_scatter])),
            'segments': (x, y),
            }
    
  # Initialize a visualization that plots data as a flow field.
  def init(self, device_name, stream_name, stream_info, subplot_row=0, subplot_col=0):
    if self._print_debug: print('FlowFieldVisualizer initializing for %s %s' % (device_name, stream_name))
    
    # Get the size of each sample, and ensure it is at least 2D.
    self._sample_size = stream_info['sample_size'][1:] # the first entry should be 2, indicating magnitude and angle
    if len(self._sample_size) == 1:
      self._sample_size = [1, self._sample_size[0]]
    
    # Process default plotting options.
    self._visualizer_options.setdefault('magnitude_normalization', 'auto')
    self._visualizer_options.setdefault('magnitude_normalization_min', 0.01)
    self._visualizer_options.setdefault('magnitude_normalization_update_period_s', 10)
    self._visualizer_options.setdefault('magnitude_normalization_buffer_duration_s', 120)
    self._visualizer_options.setdefault('linewidth', 1.5)
    self._visualizer_options.setdefault('data_transform_fn', None)
    # Get the plotting options.
    magnitude_normalization = self._visualizer_options['magnitude_normalization']
    magnitude_normalization_min = self._visualizer_options['magnitude_normalization_min']
    magnitude_normalization_update_period_s = self._visualizer_options['magnitude_normalization_update_period_s']
    magnitude_normalization_buffer_duration_s = self._visualizer_options['magnitude_normalization_buffer_duration_s']
    self._magnitude_normalization_update_period_s = magnitude_normalization_update_period_s
    self._magnitude_normalization_buffer_duration_s = magnitude_normalization_buffer_duration_s
    self._auto_magnitude_normalization = isinstance(magnitude_normalization, str) and magnitude_normalization.lower() == 'auto'
    self._magnitude_normalization_min = magnitude_normalization_min
    if not self._auto_magnitude_normalization:
      assert isinstance(magnitude_normalization, (int, float)) and magnitude_normalization > 0
      self._magnitude_normalization = magnitude_normalization
    else:
      self._magnitude_normalization = 1 # will be adjusted based on auto-scaling later
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
    title = '%s: %s' % (device_name, stream_name)
    
    pyqtgraph.setConfigOption('background', 'w')
    pyqtgraph.setConfigOption('foreground', 'k')
    
    # Create the plot window if one was not provided.
    if not self._is_sub_layout:
      self._layout = pyqtgraph.GraphicsLayoutWidget(show=True)
      self._layout.setGeometry(10, 10, *figure_size)
      self._layout.setWindowTitle(title)
      if self._hidden:
        self._layout.hide()
    
    # Create dummy data to use for initializing the plot.
    data_magnitudes = np.zeros(self._sample_size)
    data_angles_rad = np.zeros(self._sample_size)
    if self._visualizer_options['data_transform_fn'] is not None:
      data_magnitudes = self._visualizer_options['data_transform_fn'](data_magnitudes)
      data_angles_rad = self._visualizer_options['data_transform_fn'](data_angles_rad)
    self._data = np.stack((data_magnitudes, data_angles_rad), axis=0)

    # Compute x and y coordinates for the center of each tile plot.
    self._plotData_size_x = self._data.shape[2]
    self._plotData_size_y = self._data.shape[1]
    self._plotData_size_n = self._plotData_size_x * self._plotData_size_y
    self._plotPoints_x0 = np.tile(np.arange(self._plotData_size_x), (self._plotData_size_y, 1))
    self._plotPoints_y0 = np.flipud(np.tile(np.atleast_2d(np.arange(self._plotData_size_y)).T, (1, self._plotData_size_x)))
    plotPoints = self._vector_to_plotPoints(data_magnitudes, data_angles_rad)
    
    # Initialize the plot with dots at each location.
    h_plot = self._layout.addPlot(subplot_row,subplot_col, 1,1, title=title) # row, col, rowspan, colspan
    scatter_not_nan = pyqtgraph.ScatterPlotItem(pos=np.concatenate(plotPoints['scatter_not_nan'], axis=0).T,
                                                size=0.5, pxMode=False)
    scatter_not_nan.setBrush(color=np.array([1,1,1])*150)
    scatter_not_nan.setPen(color=np.array([1,1,1])*150)
    scatter_nan = pyqtgraph.ScatterPlotItem(pos=np.concatenate(plotPoints['scatter_nan'], axis=0).T,
                                            size=0.1, pxMode=False)
    scatter_nan.setPen(color=np.array([1,1,1])*255)
    path = pyqtgraph.arrayToQPath(*plotPoints['segments'], connect='pairs')
    path_item = pyqtgraph.QtGui.QGraphicsPathItem(path)
    path_item.setPen(pyqtgraph.mkPen('k', width=self._visualizer_options['linewidth'], pxMode=False))
    h_plot.addItem(scatter_not_nan)
    h_plot.addItem(scatter_nan)
    h_plot.addItem(path_item)
    h_plot.setXRange(0, self._plotData_size_x - 1, padding=0.5 / (self._plotData_size_x - 1) if self._plotData_size_x > 1 else 0.5)
    h_plot.setYRange(0, self._plotData_size_y - 1, padding=0.5 / (self._plotData_size_y - 1) if self._plotData_size_y > 1 else 0.5)
    h_plot.setAspectLocked(ratio=1)
    
    # Create an exporter to grab an image of the plot.
    self._exporter = pyqtgraph.exporters.ImageExporter(self._layout.scene())
    
    # Save state for future updates.
    self._plot = h_plot
    self._scatter_not_nan = scatter_not_nan
    self._scatter_nan = scatter_nan
    self._segments = path_item
    self._figure_size = figure_size
    
  
  # Update the visualization with new data.
  # Note that only the most recent data sample will be shown.
  # @param new_data is a dict with 'data' (all other keys will be ignored).
  #   The data may contain multiple timesteps (each value may be a list).
  # @param visualizing_all_data is whether this is being called
  #   as part of a periodic update loop or in order to show all data at once.
  #   If periodically, new data should be added to the visualization if applicable.
  #   Otherwise the new data should replace the visualization if applicable.
  def update(self, new_data, visualizing_all_data):
    # Extract the latest data.
    self._data = np.array(new_data['data'][-1])
    magnitudes = self._data[0][:,:]
    angles_rad = self._data[1][:,:]

    # Apply any user-defined transformation.
    if self._visualizer_options['data_transform_fn'] is not None:
      magnitudes = self._visualizer_options['data_transform_fn'](magnitudes)
      angles_rad = self._visualizer_options['data_transform_fn'](angles_rad)
      self._data = np.stack((magnitudes, angles_rad), axis=0)

    # Update the magnitude scale based on a buffer of recent magnitudes.
    if self._auto_magnitude_normalization:
      # Update the buffer length if it is the first time seeing it filled.
      if len(self._magnitude_normalization_buffer) == 0:
        self._magnitude_normalization_buffer_start_time_s = time.time()
      elif time.time() - self._magnitude_normalization_buffer_start_time_s >= self._magnitude_normalization_buffer_duration_s:
        if self._magnitude_normalization_buffer_length is None:
          self._magnitude_normalization_buffer_length = len(self._magnitude_normalization_buffer)
      # Update the rolling buffer of magnitudes.
      self._magnitude_normalization_buffer.append(magnitudes)
      if self._magnitude_normalization_buffer_length is not None:
        del self._magnitude_normalization_buffer[0]
      # Update the overall magnitude scale.
      if time.time() >= self._next_magnitude_normalization_update_time_s:
        magnitudes_buffer = np.array(self._magnitude_normalization_buffer)
        if np.any(~np.isnan(magnitudes_buffer)):
          self._magnitude_normalization = np.nanquantile(magnitudes_buffer, 0.98)
          self._magnitude_normalization = max(self._magnitude_normalization_min, self._magnitude_normalization)
        self._next_magnitude_normalization_update_time_s = time.time() + self._magnitude_normalization_update_period_s
    
    # Scale the magnitudes using the automatic scale or the user-provided scale.
    magnitudes = magnitudes/self._magnitude_normalization
    
    # Update the visualization.
    plotPoints = self._vector_to_plotPoints(magnitudes, angles_rad)
    path = pyqtgraph.arrayToQPath(*plotPoints['segments'], connect='pairs')
    self._segments.setPath(path)
    
    # Update the figure to see the changes.
    if not self._hidden:
      QtCore.QCoreApplication.processEvents()
  
  
  # Retrieve an image of the most updated visualization.
  # Should return a matrix in RGB format.
  def get_visualization_image(self, device_name, stream_name):
    # start_export_time_s = time.time()
    img = self._exporter.export(toBytes=True)
    img = self._convertQImageToMat(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # if self._print_debug: print('Time to export heatmap for %s.%s: %0.3fs', (device_name, stream_name, time.time() - start_export_time_s))
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
    if not self._hidden and not self._is_sub_layout:
      self._layout.close()
      self._app.quit()
  
  # Wait for the user to close the figure.
  def wait_for_user_to_close(self):
    if not self._hidden and not self._is_sub_layout:
      self._app.exec()











