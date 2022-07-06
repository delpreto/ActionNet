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
class HeatmapVisualizer(Visualizer):
  
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
    self._heatmap = None
    self._colorbar = None
    self._auto_colorbar_levels = True # can be overridden by supplied options in init()
    self._colorbar_levels_buffer = []
    self._colorbar_levels_data_buffer = []
    self._colorbar_levels = None
    self._levels_update_period_s = 15
    self._next_levels_update_time_s = 0

  # Initialize a visualization that plots data as a 2D heatmap.
  def init(self, device_name, stream_name, stream_info):
    if self._print_debug: print('HeatmapVisualizer initializing for %s %s' % (device_name, stream_name))
    
    # Process default plotting options.
    # Default color maps include:
    #   cividis, inferno, magma, plasma, viridis
    #   CET maps: https://colorcet.com/gallery.html#linear
    self._visualizer_options.setdefault('colormap', 'inferno')
    self._visualizer_options.setdefault('colorbar_levels', 'auto')
    # Get the plotting options.
    colormap = self._visualizer_options['colormap']
    colorbar_levels = self._visualizer_options['colorbar_levels']
    self._auto_colorbar_levels = isinstance(colorbar_levels, str) and colorbar_levels.lower() == 'auto'
    if not self._auto_colorbar_levels:
      assert isinstance(colorbar_levels, (list, tuple)) and len(colorbar_levels) == 2
      self._colorbar_levels = colorbar_levels
    sample_size = stream_info['sample_size']
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
    
    # Create dummy data, and use it to initialize the plot.
    self._data = np.zeros(sample_size)
    h_heatmap = pyqtgraph.ImageItem(image=self._data, hoverable=True)
    h_plot = self._layout.addPlot(0,0, 1,1, title=title) # row, col, rowspan, colspan
    h_plot.addItem(h_heatmap, title=title)
    h_plot.hideAxis('bottom')
    h_plot.hideAxis('left')
    # self._h_plot.getAxis('bottom').setLabel('Horizontal Index')
    # self._h_plot.getAxis('right').setLabel('Vertical Index')
    h_plot.setAspectLocked(True)
    # Add a colorbar
    h_colorbar = h_plot.addColorBar(h_heatmap, colorMap=colormap) # , interactive=False)
    # Add a callback to show values.
    h_plot.scene().sigMouseMoved.connect(self._mouse_moved_callback)
    
    # Create an exporter to grab an image of the plot.
    self._exporter = pyqtgraph.exporters.ImageExporter(self._layout.scene())
    
    # Save state for future updates.
    self._plot = h_plot
    self._heatmap = h_heatmap
    self._colorbar = h_colorbar
    self._figure_size = figure_size
  
  # Callback for mouse moving over the heatmap, to add a tool tip with the value.
  def _mouse_moved_callback(self, mouse_position):
    # Check if event is inside heatmap, and convert from screen/pixels to image xy indexes.
    if self._plot.sceneBoundingRect().contains(mouse_position):
      mouse_point = self._plot.getViewBox().mapSceneToView(mouse_position)
      x_i = int(mouse_point.x())
      y_i = int(mouse_point.y())
      if x_i >= 0 and x_i < self._data.shape[0] and y_i >= 0 and y_i < self._data.shape[1]:
        self._layout.window().setToolTip('(y=%d, x=%d): %0.2f' %
                                          (y_i, x_i, self._data[y_i][x_i]))
        return
  
  # Update the heatmap with new data.
  # Note that only the most recent data sample will be shown.
  # @param new_data is a dict with 'data' (all other keys will be ignored).
  #   The data may contain multiple timesteps (each value may be a list).
  # @param visualizing_all_data is whether this is being called
  #   as part of a periodic update loop or in order to show all data at once.
  #   If periodically, new data should be added to the visualization if applicable.
  #   Otherwise the new data should replace the visualization if applicable.
  def update(self, new_data, visualizing_all_data):
    # Update the heatmap with the latest data.
    self._data = np.array(new_data['data'][-1])
    self._heatmap.setImage(self._data.T) # index the image as (x, y) but numpy as (y, x)
    
    # Update the colorbar scale based on a buffer of recent colorbar levels.
    if self._auto_colorbar_levels:
      heatmap_levels = self._heatmap.getLevels()
      # heatmap_levels = abs(heatmap_levels)*0.8 * np.sign(heatmap_levels)
      self._colorbar_levels_buffer.append(heatmap_levels)
      self._colorbar_levels_data_buffer.append(self._data.T)
      if time.time() >= self._next_levels_update_time_s:
        # colorbar_levels_buffer = np.array(self._colorbar_levels_buffer)
        # colorbar_levels_means = np.mean(colorbar_levels_buffer, axis=0)
        # colorbar_levels_stds = np.std(colorbar_levels_buffer, axis=0)
        # self._colorbar_levels = [np.quantile(colorbar_levels_buffer[:,0], 0.2),
        #                          np.quantile(colorbar_levels_buffer[:,1], 0.8)]
        colorbar_levels_data_buffer = np.array(self._colorbar_levels_data_buffer)
        self._colorbar_levels = [np.quantile(colorbar_levels_data_buffer, 0.01),
                                 np.quantile(colorbar_levels_data_buffer, 0.99)]
        self._colorbar_levels_buffer = []
        self._colorbar_levels_data_buffer = []
        self._next_levels_update_time_s = time.time() + self._levels_update_period_s
    
    # Note that the below seems needed to update the heatmap, even if the levels stayed the same.
    self._colorbar.setLevels(self._colorbar_levels)
    
    # Update the figure to see the changes.
    if not self._hidden:
      cv2.waitKey(1) # find a better way?
  
  
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











