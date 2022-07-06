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
from utils.print_utils import *

if use_matplotlib:
  import matplotlib
  import matplotlib.pyplot as plt
else:
  import pyqtgraph
  import pyqtgraph.opengl as gl
  from pyqtgraph.Qt import QtCore, QtGui
  import pyqtgraph.exporters

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


################################################
# Visualize quaternion orientation data by drawing a 3D arrow from the origin.
# Assumes that the stream's data['data'] field contains
#  quaternions of orientations (each row is xyzw).
################################################
class OrientationVisualizer(Visualizer):
  
  def __init__(self, visualizer_options=None, hidden=False,
               parent_layout=None, parent_layout_size=None,
               print_debug=False, print_status=False):
    Visualizer.__init__(self, visualizer_options=visualizer_options, hidden=hidden,
                        parent_layout=parent_layout, parent_layout_size=parent_layout_size,
                        print_debug=print_debug, print_status=print_status)
    
    if use_matplotlib:
      self._fig = None
      self._ax = None
    else:
      self._floor_size_cm = (20, 20) # x, y
      self._floor_grid_spacing_cm = (1, 1, 1) # x, y, z
      self._arrow_length = 5
      self._app = QtGui.QApplication([])
      self._layout = parent_layout
      self._is_sub_layout = parent_layout is not None
      self._layout_size = parent_layout_size
      self._plot = None
      self._figure_size = None
    
    # Define the points to plot when no rotation is applied.
    self._indicator_points = np.array([
      [0, 0, -1],
      [0, 0, 1],
      [0, 0, 0],
      [10, 0, 0],
      [8, -2, 0],
      [10, 0, 0],
      [8, 2, 0],
    ])
    # # Rotate so main axis is along xy instead of along x.
    # theta = 3*np.pi/4
    # self._indicator_points = [Rotation.from_quat([0, 0, np.sin(theta/2), np.cos(theta/2)]).apply(indicator_point) for indicator_point in self._indicator_points]
    # theta = np.pi/8
    # self._indicator_points = [np.matmul([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]], indicator_point) for indicator_point in self._indicator_points]
    # Negate the x axis of the original indicator points so the directionality will match the arm.
    # This will be overridden by calibration though.
    self._indicator_points_toRotate = [np.array(indicator_point) * [-1, 1, 1] for indicator_point in self._indicator_points]
    self._calibrated = False # will calibrate in the first timestep such that applying the first quaternion yields the above indicator orientation
    # If using hidden mode, change matplotlib's backend.
    if self._hidden and use_matplotlib:
      matplotlib.use("Agg")
  
  # Initialize a visualization that displays a 3D shape from the origin.
  # Assumes that the stream's data['data'] field contains quaternions (each row is xyzw).
  def init(self, device_name, stream_name, stream_info):
    if self._print_debug: print('OrientationVisualizer initializing for %s %s' % (device_name, stream_name))
    
    # Set some options.
    if use_matplotlib:
      figure_size = (7, 5)
    else:
      if self._layout_size is None:
        # screen_widths = [screen.size().width() for screen in app.screens()]
        # screen_heights = [screen.size().heights() for screen in app.screens()]
        screen_width = self._app.primaryScreen().size().width()
        screen_height = self._app.primaryScreen().size().height()
        figure_width = int(screen_width*0.8)
        figure_height = int(screen_height/1.2)
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
                                'frame_on': True,
                                'projection': '3d',
                              },
                              figsize=figure_size
                              )
      ax = axs[0][0]
      
      # Set some formatting options.
      # Note that x/y/z will be rearranged to get a good view.
      fig.suptitle('Orientation')
      ax.set_xlabel('z')
      ax.set_ylabel('x')
      ax.set_zlabel('y')
      ax.view_init(-70, 0) # (elevation, azimuth) # (20, -40)
      
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
      self._glWidget.setWindowTitle('Orientation')
      layout = QtGui.QGridLayout()
      self._layout.setLayout(layout)
      layout.addWidget(self._glWidget, 0,0, 1,1)
      
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
      # camera_params = {
      #   'fov': 60,
      #   # 'rotation': (1.0, 0.0, 0.0, 0.0),
      #   'elevation': 12.0,
      #   'center': QtGui.QVector3D(-max(self._floor_size_cm) * 0.15,
      #                             -max(self._floor_size_cm) * 0.21,
      #                              max(self._floor_size_cm) * 0.2),
      #   'azimuth': 50,
      #   'distance': max(self._floor_size_cm) * 1.25
      # }
      # camera_params = {
      #   'fov': 60,
      #   # 'rotation': (1.0, 0.0, 0.0, 0.0),
      #   'elevation': 18.0,
      #   'center': QtGui.QVector3D(-max(self._floor_size_cm) * 0.035,
      #                             -max(self._floor_size_cm) * 0.064,
      #                              max(self._floor_size_cm) * 0.2),
      #   'azimuth': 222,
      #   'distance': max(self._floor_size_cm) * 1.4
      # }
      camera_params = {
        'fov': 60,
        # 'rotation': (1.0, 0.0, 0.0, 0.0),
        'elevation': 22.0,
        'center': QtGui.QVector3D(-max(self._floor_size_cm) * 0.035,
                                   max(self._floor_size_cm) * 0.0025,
                                   max(self._floor_size_cm) * 0.2),
        'azimuth': 180,
        'distance': max(self._floor_size_cm) * 1.2
      }
      self._glWidget.setCameraParams(**camera_params)
      # Create a floor.
      grid_color = (0,0,0,80)
      # Add grid perpendicular to z axis
      gz = gl.GLGridItem(color=grid_color)
      gz.setSize(x=self._floor_size_cm[0], y=self._floor_size_cm[1])
      gz.setSpacing(x=self._floor_grid_spacing_cm[0], y=self._floor_grid_spacing_cm[1], z=self._floor_grid_spacing_cm[2])
      # gz.translate(-int(self._floor_size_cm[0]/2), -int(self._floor_size_cm[1]/2), 0)
      self._glWidget.addItem(gz)
      # Add grid perpendicular to x axis
      gx = gl.GLGridItem(color=grid_color)
      gx.setSize(x=self._floor_size_cm[0], y=self._floor_size_cm[1])
      gx.setSpacing(x=self._floor_grid_spacing_cm[0], y=self._floor_grid_spacing_cm[1], z=self._floor_grid_spacing_cm[2])
      gx.rotate(90, 0, 1, 0)
      gx.translate(int(self._floor_size_cm[0]/2), 0, int(self._floor_size_cm[1]/2))
      self._glWidget.addItem(gx)
      # Add grid perpendicular to y axis
      gy = gl.GLGridItem(color=grid_color)
      gy.setSize(x=self._floor_size_cm[0], y=self._floor_size_cm[1])
      gy.setSpacing(x=self._floor_grid_spacing_cm[0], y=self._floor_grid_spacing_cm[1], z=self._floor_grid_spacing_cm[2])
      gy.rotate(90, 1, 0, 0)
      gy.translate(0, int(self._floor_size_cm[1]/2), int(self._floor_size_cm[0]/2))
      self._glWidget.addItem(gy)
      # Add grid perpendicular to y axis
      gy2 = gl.GLGridItem(color=grid_color)
      gy2.setSize(x=self._floor_size_cm[0], y=self._floor_size_cm[1])
      gy2.setSpacing(x=self._floor_grid_spacing_cm[0], y=self._floor_grid_spacing_cm[1], z=self._floor_grid_spacing_cm[2])
      gy2.rotate(90, 1, 0, 0)
      gy2.translate(0, -int(self._floor_size_cm[1]/2), int(self._floor_size_cm[0]/2))
      self._glWidget.addItem(gy2)
      
      # Note: to create grid perpendicular to x axis: gx.rotate(90, 0, 1, 0) before translating
      # Note: to create grid perpendicular to y axis: gy.rotate(90, 1, 0, 0) before translating
      
      self._exporter = pyqtgraph.exporters.ImageExporter(self._layout.scene())
    
    # Initialize the plot with the initial indicator.
    self.update({'data':[[0, 0, 0, 1]]},
                visualizing_all_data=True,
                real_data=False)
  
  # Update the skeleton visualization with new segment position data.
  # Only the most recent timestep will be visualized.
  # @param new_data is a dict with 'data' (all other entries will be ignored).
  #   It should contain all segment positions as a matrix (each row is xyz).
  def update(self, new_data, visualizing_all_data, real_data=True):
  
    # Extract the latest quaternion and convert it to a rotation matrix.
    quaternion = np.array(new_data['data'][-1])
    # quaternion = [quaternion[1], quaternion[2], quaternion[0], quaternion[3]]
    rotation_matrix = Rotation.from_quat(quaternion)

    # Calibrate if this is the first time seeing real data.
    if real_data and not self._calibrated:
      inverted_matrix = np.linalg.inv(rotation_matrix.as_matrix())
      self._indicator_points_toRotate = [np.matmul(inverted_matrix, indicator_point) for indicator_point in self._indicator_points]
      self._calibrated = True
    
    # Rotate the indicator.
    rotated_indicator_points = [rotation_matrix.apply(indicator_point) for indicator_point in self._indicator_points_toRotate]
    # rotated_indicator_points = self._indicator_points_toRotate
    rotated_indicator_points = np.array(rotated_indicator_points)

    if use_matplotlib:
      plot_x_bounds = np.array([1000, -1000])
      plot_y_bounds = np.array([1000, -1000])
      plot_z_bounds = np.array([1000, -1000])
      ax_lines = self._ax.get_lines()
      # Reorder them to make adjusting the plot view angle easier.
      plot_x = rotated_indicator_points[:, 2]
      plot_y = rotated_indicator_points[:, 0]
      plot_z = rotated_indicator_points[:, 1]
      # Draw each indicator point in a connected chain.
      # Create a fresh plot or update existing data.
      if not visualizing_all_data:
        ax_lines[0].set_data_3d(plot_x, plot_y, plot_z)
      else:
        self._ax.plot(plot_x, plot_y, plot_z, 'r-o', markersize=5)
      # Update axis bounds.
      plot_x_bounds = np.array([min(plot_x_bounds[0], min(plot_x)), max(plot_x_bounds[1], max(plot_x))])
      plot_y_bounds = np.array([min(plot_y_bounds[0], min(plot_y)), max(plot_y_bounds[1], max(plot_y))])
      plot_z_bounds = np.array([min(plot_z_bounds[0], min(plot_z)), max(plot_z_bounds[1], max(plot_z))])
      
      # Expand the limits a bit if they have 0 span.
      plot_x_bounds = plot_x_bounds if abs(np.diff(plot_x_bounds)[0]) > 0 else plot_x_bounds + [-1, 1]
      plot_y_bounds = plot_y_bounds if abs(np.diff(plot_y_bounds)[0]) > 0 else plot_y_bounds + [-1, 1]
      plot_z_bounds = plot_z_bounds if abs(np.diff(plot_z_bounds)[0]) > 0 else plot_z_bounds + [-1, 1]
      # Set the limits to fit all data.
      self._ax.set_xlim3d(plot_x_bounds)
      self._ax.set_ylim3d(plot_y_bounds)
      self._ax.set_zlim3d(plot_z_bounds)
      # Set axis scaling to be 'equal' to preserve proper aspect ratio.
      #  Adapted from https://stackoverflow.com/a/70245893
      self._ax.set_box_aspect([ub - lb for lb, ub in (self._ax.get_xlim(), self._ax.get_ylim(), self._ax.get_zlim())])
      
      # Update the figure to see the changes.
      self._fig.canvas.draw()
      self._fig.canvas.flush_events()
    else:
      # Negate the x and y coordinates since the floor was visualized in the negative quadrant.
      # plot_xyz_cm = rotated_indicator_points * np.array([-1, -1, 1])
      plot_xyz_cm = rotated_indicator_points
      # Draw each indicator point in a connected chain.
      # Create a fresh plot or update existing data.
      if visualizing_all_data:
        self._indicator_shadow_line = gl.GLLinePlotItem(
            pos=plot_xyz_cm*[1,1,0], color=(0.5,0.5,0.5, 0.2),
            width=10, antialias=True)
        self._glWidget.addItem(self._indicator_shadow_line)
        self._indicator_lines = gl.GLLinePlotItem(
            pos=plot_xyz_cm, color=(1,0,0,1),
            width=25, antialias=True)
        self._glWidget.addItem(self._indicator_lines)
        # self._indicator_scatter = gl.GLScatterPlotItem(
        #     pos=plot_xyz_cm, color=(1,0,0,1),
        #     size=1, pxMode=False)
        # self._indicator_scatter.setGLOptions('translucent')
        # self._glWidget.addItem(self._indicator_scatter)
        cv2.waitKey(1) # wait for it to actually draw
        # Show or hide the figure.
        if not self._is_sub_layout:
          if not self._hidden:
            self._layout.show()
          else:
            self._layout.hide()
      else:
        self._indicator_lines.setData(pos=plot_xyz_cm, antialias=True)
        # self._indicator_scatter.setData(pos=plot_xyz_cm)
        self._indicator_shadow_line.setData(pos=plot_xyz_cm*[1,1,0], antialias=True)
      # plot_xyz_cm_mean = np.mean(np.array(plot_xyz_cm), 0)
      # if np.all(np.abs(plot_xyz_cm_mean) > 0):
      #   camera_params = {
      #     'fov': 60,
      #     # 'rotation': (1.0, 0.0, 0.0, 0.0),
      #     'elevation': 15.0,
      #     'center': QtGui.QVector3D(plot_xyz_cm_mean[0]*0.7,
      #                               plot_xyz_cm_mean[1]*0.9,
      #                               plot_xyz_cm_mean[2]),
      #     'azimuth': 55,
      #     'distance': np.max(np.max(plot_xyz_cm, axis=0) - np.min(plot_xyz_cm, axis=0))*1.5
      #   }
      #   self._glWidget.setCameraParams(**camera_params)
      # print_var(self._glWidget.cameraParams())
      
      # Update the plot to see the changes.
      if not self._hidden and not self._is_sub_layout:
        cv2.waitKey(1) # find a better way?
  
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















