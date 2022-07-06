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

################################################
# A superclass to visualize SensorStreamer data.
# Will be used by DataVisualizer.
################################################
class Visualizer():

  def __init__(self, visualizer_options=None, hidden=False,
                     parent_layout=None, parent_layout_size=None,
                     print_debug=False, print_status=False):
    self._visualizer_options = visualizer_options
    self._hidden = hidden
    self._parent_layout = parent_layout
    self._parent_layout_size = parent_layout_size
    self._print_debug = print_debug
    self._print_status = print_status

  # Initialize the visualization (create figures, plot dummy data if needed, etc).
  def init(self, device_name, stream_name, stream_info):
    pass

  # Update the visualization with new data.
  # @param new_data is a dict with 'data', 'time_s', 'time_str', and any other extra channels for the stream.
  #   The data may contain multiple timesteps (each value may be a list).
  # @param visualizing_all_data is whether this is being called
  #   as part of a periodic update loop or in order to show all data at once.
  #   If periodically, new data should be added to the visualization if applicable.
  #   Otherwise the new data should replace the visualization if applicable.
  def update(self, new_data, visualizing_all_data):
    pass
  
  # Retrieve an image of the most updated visualization.
  # Should return a matrix in RGB format.
  def get_visualization_image(self, device_name, stream_name):
    return None

  # Close any windows created for the visualization.
  def close(self):
    pass

  # Wait for the user to close any windows created for the visualization.
  def wait_for_user_to_close(self):
    pass
