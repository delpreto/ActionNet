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
#   The pyqtgraph method is configured to show the whole kitchen from a fixed view,
#    while the matplotlib method is configured to always follow the person.
use_matplotlib = False

from visualizers.Visualizer import Visualizer

if use_matplotlib:
  import matplotlib
  import matplotlib.pyplot as plt
else:
  import pyqtgraph
  import pyqtgraph.opengl as gl
  from pyqtgraph.Qt import QtCore, QtGui
  if QtGui.QApplication.instance() is None:
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
  import pyqtgraph.exporters

import cv2
import numpy as np


################################################
# Visualize Xsens data by drawing a wireframe skeleton.
# Assumes that the stream's data['data'] field contains
#  a matrix of segment positions (each row is xyz).
################################################
class XsensSkeletonVisualizer(Visualizer):

  def __init__(self, visualizer_options=None, hidden=False,
                     parent_layout=None, parent_layout_size=None,
                     print_debug=False, print_status=False):
    Visualizer.__init__(self, visualizer_options=visualizer_options or {}, hidden=hidden,
                              parent_layout=parent_layout, parent_layout_size=parent_layout_size,
                              print_debug=print_debug, print_status=print_status)
  
    if use_matplotlib:
      self._fig = None
      self._ax = None
    else:
      self._kitchen_floor_size_cm = (150, 300) # x, y
      self._app = QtGui.QApplication.instance() or QtGui.QApplication([])
      self._layout = parent_layout
      self._is_sub_layout = parent_layout is not None
      self._layout_size = parent_layout_size
      self._plot = None
      self._figure_size = None
    
    self._mode = self._visualizer_options.get('mode', 'zoomed')
    self._suit_id = self._visualizer_options.get('suit_id')
    self._position_units = 'cm'
    self._prop_hand_for_visualization = 'right'
    if 'position_units' in self._visualizer_options:
      self._position_units = self._visualizer_options['position_units']
    if 'prop_hand_for_visualization' in self._visualizer_options:
      self._prop_hand_for_visualization = self._visualizer_options['prop_hand_for_visualization']
    
    # Map segment indexes to labels.
    # See page 137 of MVN_User_Manual and page 18 of MVN_real-time_network_streaming_protocol_specification
    self._segment_labels = [
      'Pelvis',
      'L5',
      'L3',
      'T12',
      'T8',
      'Neck',
      'Head',
      'Right Shoulder',
      'Right Upper Arm',
      'Right Forearm',
      'Right Hand',
      'Left Shoulder',
      'Left Upper Arm',
      'Left Forearm',
      'Left Hand',
      'Right Upper Leg',
      'Right Lower Leg',
      'Right Foot',
      'Right Toe',
      'Left Upper Leg',
      'Left Lower Leg',
      'Left Foot',
      'Left Toe',
      'Prop 1',
    ]
    # Define how to visualize the person by connecting segment positions.
    self._segment_chains_labels_toPlot = {
      'Left Leg':  ['Left Upper Leg', 'Left Lower Leg', 'Left Foot', 'Left Toe'],
      'Right Leg': ['Right Upper Leg', 'Right Lower Leg', 'Right Foot', 'Right Toe'],
      'Spine':     ['Head', 'Neck', 'T8', 'T12', 'L3', 'L5', 'Pelvis'], # top down
      'Hip':       ['Left Upper Leg', 'Pelvis', 'Right Upper Leg'],
      'Shoulders': ['Left Upper Arm', 'Left Shoulder', 'Right Shoulder', 'Right Upper Arm'],
      'Left Arm':  ['Left Upper Arm', 'Left Forearm', 'Left Hand'],
      'Right Arm': ['Right Upper Arm', 'Right Forearm', 'Right Hand'],
      'Prop 1':    ['Right Hand' if self._prop_hand_for_visualization == 'right' else 'Left Hand', 'Prop 1'],
    }
    self._segment_chains_colors_rgba = {
      'Left Leg':  (1,0,0,1),
      'Right Leg': (1,0,0,1),
      'Spine':     (1,0,0,1),
      'Hip':       (1,0,0,1),
      'Shoulders': (1,0,0,1),
      'Left Arm':  (1,0,0,1),
      'Right Arm': (1,0,0,1),
      'Prop 1':    (0,0,1,1),
    }
    self._segment_chains_indexes_toPlot = dict()
    for (chain_name, chain_labels) in self._segment_chains_labels_toPlot.items():
      segment_indexes = []
      for chain_label in chain_labels:
        segment_indexes.append(self._segment_labels.index(chain_label))
      self._segment_chains_indexes_toPlot[chain_name] = segment_indexes
    
    # If using hidden mode, change matplotlib's backend.
    if self._hidden and use_matplotlib:
      matplotlib.use("Agg")

  # Initialize a visualization that displays a skeleton from Xsens data.
  # Assumes that the stream's data['data'] field contains segment positions (each row is xyz).
  def init(self, device_name, stream_name, stream_info):
    if self._print_debug: print('XsensSkeletonVisualizer initializing for %s %s' % (device_name, stream_name))

    self._chain_lines = {}
    self._chain_scatters = {}
    # Set some options.
    if use_matplotlib:
      figure_size = (7, 5)
    else:
      if self._layout_size is None:
        # screen_widths = [screen.size().width() for screen in app.screens()]
        # screen_heights = [screen.size().heights() for screen in app.screens()]
        screen_width = self._app.primaryScreen().size().width()
        screen_height = self._app.primaryScreen().size().height()
        figure_height = int(screen_height*0.8)
        figure_width = int(figure_height/1.2)
        # figure_width = int(screen_width*0.5)
        # figure_height = int(figure_width/1.5)
        figure_size = (figure_width, figure_height)
      else:
        figure_size = self._layout_size
      self._figure_size = figure_size
      

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
      fig, axs = plt.subplots(nrows=1, ncols=1,
                               squeeze=False, # if False, always return 2D array of axes
                               sharex=True, sharey=True,
                               subplot_kw={
                                'projection': '3d',
                                },
                               figsize=figure_size
                               )
      ax = axs[0][0]

      # Set some formatting options.
      # Note that x/y/z will be rearranged to get a good view.
      fig.suptitle('Xsens Skeleton: %s' % stream_name)
      ax.set_xlabel('x')
      ax.set_ylabel('y')
      ax.set_zlabel('z')
      ax.view_init(15, 55) # (elevation, azimuth)
    
      # Save state for future updates.
      self._fig = fig
      self._ax = ax

      # Show the figure.
      if not self._hidden:
        fig.show()
    else:
      pyqtgraph.setConfigOption('background', 'w')
      pyqtgraph.setConfigOption('foreground', 'k')
      # Create the main window if one was not provided.
      if not self._is_sub_layout:
        self._layout = pyqtgraph.GraphicsLayoutWidget(show=True)
        self._layout.setGeometry(10, 10, *figure_size)
      # Create the 3D plot widget.
      self._glWidget = gl.GLViewWidget()
      self._glWidget.setWindowTitle('Xsens Skeleton: %s' % stream_name)
      layout = QtGui.QGridLayout()
      self._layout.setLayout(layout)
      layout.addWidget(self._glWidget, 0,0, 1,1)
      self._app.processEvents()
      self._glWidget.makeCurrent()
      
      # Set the camera position and angle.
      # camera_params = {
      #   'fov': 60,
      #   # 'rotation': (1.0, 0.0, 0.0, 0.0),
      #   'elevation': 23.0,
      #   'center': QtGui.QVector3D(-max(self._kitchen_floor_size_cm)*2/5,
      #                             -max(self._kitchen_floor_size_cm)*1/2,
      #                             0.0),
      #   'azimuth': 53.0,
      #   'distance': max(self._kitchen_floor_size_cm)*6/5
      # }
      # camera_params = {
      #   'fov': 60,
      #    # 'rotation': (1.0, 0.0, 0.0, 0.0),
      #    'elevation': 20.0,
      #    'center': QtGui.QVector3D(-max(self._kitchen_floor_size_cm)*0.55,
      #                              -max(self._kitchen_floor_size_cm)*1,
      #                              0.0),
      #    'azimuth': 70,
      #    'distance': max(self._kitchen_floor_size_cm)*1.6
      # }
      camera_params = {
        'fov': 60,
         # 'rotation': (1.0, 0.0, 0.0, 0.0),
         'elevation': 15.0,
         'center': QtGui.QVector3D(-max(self._kitchen_floor_size_cm)*0.7,
                                   -max(self._kitchen_floor_size_cm)*0.9,
                                   0.0),
         'azimuth': 55,
         'distance': max(self._kitchen_floor_size_cm)*1.6
      }
      self._glWidget.setCameraParams(**camera_params)
      # Create a floor.
      grid_color = (0,0,0,80)
      # Add grid perpendicular to z axis
      gz = gl.GLGridItem(color=grid_color)
      gz.setSize(x=self._kitchen_floor_size_cm[0], y=self._kitchen_floor_size_cm[1])
      gz.setSpacing(x=10, y=10, z=10)
      gz.translate(-int(self._kitchen_floor_size_cm[0]/2), -int(self._kitchen_floor_size_cm[1]/2), 0)
      self._glWidget.addItem(gz)
      # Note: to create grid perpendicular to x axis: gx.rotate(90, 0, 1, 0) before translating
      # Note: to create grid perpendicular to y axis: gy.rotate(90, 1, 0, 0) before translating
      
      # Indicate that no plots have been added yet.
      
      self._exporter = pyqtgraph.exporters.ImageExporter(self._layout.scene())
      
    # Create dummy data, and use it to initialize the plot.
    segment_positions_cm = np.zeros([len(self._segment_labels)-1, 3])
    self.update({'data':[segment_positions_cm.tolist()]},
                 visualizing_all_data=True)

  # Update the skeleton visualization with new segment position data.
  # Only the most recent timestep will be visualized.
  # @param new_data is a dict with 'data' (all other entries will be ignored).
  #   It should contain all segment positions as a matrix (each row is xyz).
  def update(self, new_data, visualizing_all_data):
    # A regular stream supplies one matrix; visualization aliases supply one per suit.
    positions = new_data['data'][-1]
    if not isinstance(positions, dict):
      positions = {self._suit_id or 1: positions}
    positions = {suit_id: np.array(data, dtype=float, copy=True)
                 for suit_id, data in positions.items()
                 if self._suit_id is None or suit_id == self._suit_id}
    if not positions:
      return
    if self._position_units == 'm':
      positions = {suit_id: 100*data for suit_id, data in positions.items()}
    if self._mode == 'zoomed':
      # Place independently centered skeletons beside each other along the camera's horizontal axis.
      side_axis = np.array([np.sin(np.deg2rad(55)), -np.cos(np.deg2rad(55)), 0])
      spacing = max(np.ptp(data @ side_axis) for data in positions.values()) + 50
      for suit_index, data in enumerate(positions.values()):
        data[:, :2] -= data[0, :2].copy()
        projection = data @ side_axis
        offset = spacing * (suit_index - (len(positions)-1)/2)
        data += side_axis * (offset - (projection.min()+projection.max())/2)

    if not use_matplotlib:
      self._glWidget.makeCurrent()
    plotted_positions = []
    active_chains = set()
    for suit_id, segment_positions_cm in positions.items():
      for chain_name, segment_indexes in self._segment_chains_indexes_toPlot.items():
        try:
          segment_xyz_cm = segment_positions_cm[segment_indexes, :]
        except IndexError:
          if 'prop' in chain_name.lower():
            continue
          raise
        key = (suit_id, chain_name)
        active_chains.add(key)
        color = self._segment_chains_colors_rgba[chain_name]
        if suit_id == 2 and 'prop' not in chain_name.lower():
          color = (0, 0.65, 0, 1)
        if use_matplotlib:
          plot_xyz_cm = segment_xyz_cm * np.array([-1, -1, 1])
          if key not in self._chain_lines or visualizing_all_data:
            self._chain_lines[key] = self._ax.plot(*plot_xyz_cm.T, '-o', color=color[:3], markersize=5)[0]
          else:
            self._chain_lines[key].set_data_3d(*plot_xyz_cm.T)
        else:
          # The floor is drawn in the negative x/y quadrant.
          plot_xyz_cm = segment_xyz_cm * np.array([-1, -1, 1])
          if key not in self._chain_lines or visualizing_all_data:
            self._chain_lines[key] = gl.GLLinePlotItem(pos=plot_xyz_cm, color=color, width=3, antialias=True)
            self._glWidget.addItem(self._chain_lines[key])
            self._chain_scatters[key] = gl.GLScatterPlotItem(pos=plot_xyz_cm, color=color, size=5, pxMode=False)
            self._chain_scatters[key].setGLOptions('translucent')
            self._glWidget.addItem(self._chain_scatters[key])
          else:
            self._chain_lines[key].setData(pos=plot_xyz_cm, antialias=True)
            self._chain_scatters[key].setData(pos=plot_xyz_cm)
        plotted_positions.append(plot_xyz_cm)

    if not visualizing_all_data:
      for key in set(self._chain_lines) - active_chains:
        if use_matplotlib:
          self._chain_lines.pop(key).remove()
        else:
          self._glWidget.removeItem(self._chain_lines.pop(key))
          self._glWidget.removeItem(self._chain_scatters.pop(key))
    plot_xyz_cm_all = np.concatenate(plotted_positions)
    lower, upper = plot_xyz_cm_all.min(axis=0), plot_xyz_cm_all.max(axis=0)
    if use_matplotlib:
      padding = np.maximum((upper-lower)*0.05, 5)
      bounds = list(zip(lower-padding, upper+padding))
      self._ax.set_xlim3d(bounds[0])
      self._ax.set_ylim3d(bounds[1])
      self._ax.set_zlim3d(bounds[2])
      self._ax.set_box_aspect([hi-lo for lo, hi in bounds])
      self._fig.canvas.draw()
      self._fig.canvas.flush_events()
    else:
      center = (lower + upper)/2
      distance = max(1, max(upper-lower))*1.5*max(1, self._figure_size[0]/self._figure_size[1])
      self._glWidget.setCameraParams(fov=60, elevation=15.0, azimuth=55,
                                    center=QtGui.QVector3D(*center), distance=distance)
      if not self._is_sub_layout:
        if self._hidden:
          self._layout.hide()
        else:
          self._layout.show()
          self._app.processEvents()

  # Retrieve an image of the most updated visualization.
  # Should return a matrix in RGB format.
  def get_visualization_image(self, device_name, stream_name):
    if use_matplotlib:
      # Convert the figure canvas to an image.
      img = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
      img = img.reshape(self._fig.canvas.get_width_height()[::-1] + (3,))
      return img
    else:
      img = self._glWidget.renderToArray(self._figure_size)
      img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
      return img

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















