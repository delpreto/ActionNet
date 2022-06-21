
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
