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

import cv2
import numpy as np

import pyqtgraph
import pyqtgraph.exporters
from pyqtgraph.Qt import QtCore, QtGui

################################################
# Visualize videos.
# Assumes that the stream's data['data'] field contains raw frames
#   as matrices in BGR format.
################################################
class VideoVisualizer(Visualizer):

  def __init__(self, visualizer_options=None, hidden=False,
                     parent_layout=None, parent_layout_size=None,
                     print_debug=False, print_status=False):
    Visualizer.__init__(self, visualizer_options=visualizer_options, hidden=hidden,
                              parent_layout=parent_layout, parent_layout_size=parent_layout_size,
                              print_debug=print_debug, print_status=print_status)

    self._video_title = None
    self._layout = parent_layout
    self._is_sub_layout = parent_layout is not None
    self._layout_size = parent_layout_size
    self._plot = None
    self._plot_plot = None

  # Initialize a visualization that displays a video.
  # Assumes that the stream's data['data'] field contains raw frames as matrices in BGR format.
  def init(self, device_name, stream_name, stream_info):
    if self._print_debug: print('VideoVisualizer initializing for %s %s' % (device_name, stream_name))

    # Get the plotting options.
    frame_size = stream_info['sample_size']
    fps = stream_info['sampling_rate_hz']
    if fps is None:
      # Take a guess at a reasonable frame rate.
      fps = 30
    video_title = '%s | %s' % (device_name, stream_name)

    # Save state for future updates.
    self._video_title = video_title
    self._latest_frame = 255*np.ones(frame_size, dtype='uint8')
    
    # Create a window or plot to show the video.
    if self._is_sub_layout:
      self._plot = self._layout.addPlot(row=0, col=0)
      self._plot.setAspectLocked(True)
      data = np.zeros(shape=frame_size)
      self._plot_image_item = pyqtgraph.ImageItem(image=data)
      self._plot.addItem(self._plot_image_item)
      self._plot.hideAxis('bottom')
      self._plot.hideAxis('left')
    elif not self._hidden:
      cv2.imshow(self._video_title, self._latest_frame)
      cv2.waitKey(1) # necessary to show/update the window; argument is in ms

  # Update the displayed video.
  # Note that only the most recent frame will be shown when this is called.
  # @param new_data is a dict with 'data' (all other keys will be ignored).
  #   The 'data' entry must contain raw frames as a matrix in BGR format.
  #   The data may contain multiple timesteps (a list of matrices).
  def update(self, new_data, visualizing_all_data):
    # Get the most recent frame.
    self._latest_frame = new_data['data'][-1]
    # Update the layout if one was provided.
    if self._is_sub_layout:
      self._plot_image_item.setImage(
          cv2.rotate(cv2.cvtColor(self._latest_frame, cv2.COLOR_BGR2RGB),
                     cv2.cv2.ROTATE_90_CLOCKWISE)
      )
    # Show the image if appropriate.
    elif not self._hidden:
      cv2.imshow(self._video_title, self._latest_frame)
      cv2.waitKey(1) # necessary to show/update the window; argument is in ms

  # Retrieve an image of the most updated visualization.
  # Should return a matrix in RGB format.
  def get_visualization_image(self, device_name, stream_name):
    img = self._latest_frame.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

  # Close the window.
  def close(self):
    if not self._is_sub_layout:
      try:
        cv2.destroyWindow(self._video_title)
      except:
        pass

  # Wait for the user to close the window.
  def wait_for_user_to_close(self):
    if not self._is_sub_layout:
      if not self._hidden:
        cv2.waitKey(0)











