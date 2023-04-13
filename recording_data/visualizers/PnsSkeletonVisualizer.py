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
    import pyqtgraph.exporters

import cv2
import numpy as np


################################################
# Visualize Pns data by drawing a wireframe skeleton.
# Assumes that the stream's data['data'] field contains
#  a matrix of segment positions (each row is xyz).
################################################
class PnsSkeletonVisualizer(Visualizer):

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
            self._kitchen_floor_size_cm = (700, 700)  # x, y
            self._app = QtGui.QApplication([])
            self._layout = parent_layout
            self._is_sub_layout = parent_layout is not None
            self._layout_size = parent_layout_size
            self._plot = None
            self._figure_size = None

        # Map segment indexes to labels.
        # See page 137 of MVN_User_Manual and page 18 of MVN_real-time_network_streaming_protocol_specification
        self._segment_labels = [
            'Hips',
            'Right UpLeg',
            'Right Leg',
            'Right Foot',
            'Left UpLeg',
            'Left Leg',
            'Left Foot',
            'Spine',
            'Spine1',
            'Spine2',
            'Neck',
            'Neck1',
            'Head',
            'Right Shoulder',
            'Right Arm',
            'Right ForeArm',
            'Right Hand',
            'Left Shoulder',
            'Left Arm',
            'Left ForeArm',
            'Left Hand',
        ]
        # Define how to visualize the person by connecting segment positions.
        self._segment_chains_labels_toPlot = {
            # 'Left Legs': ['Spine', 'Left UpLeg', 'Left Leg', 'Left Foot'],
            # 'Right Legs': ['Spine', 'Right UpLeg', 'Right Leg', 'Right Foot'],
            'Left Legs': ['Hips', 'Left UpLeg', 'Left Leg', 'Left Foot'],
            'Right Legs': ['Hips', 'Right UpLeg', 'Right Leg', 'Right Foot'],
            # 'Spines': ['Head', 'Neck1', 'Neck1', 'Spine2', 'Spine1', 'Spine', ],
            'Spines': ['Head', 'Neck1', 'Neck1',  'Spine2', 'Spine1', 'Spine', 'Hips', ], #, 'Spine', 'Hips'],  # top down
            # 'Hips': ['Left UpLeg', 'Hips', 'Right UpLeg'],
            'Shoulders': ['Left Shoulder', 'Neck', 'Right Shoulder'],
            'Left Arms': ['Left Shoulder', 'Left Arm', 'Left ForeArm', 'Left Hand'],
            'Right Arms': ['Right Shoulder', 'Right Arm', 'Right ForeArm', 'Right Hand'],
        }
        self._segment_chains_indexes_toPlot = dict()
        for (chain_name, chain_labels) in self._segment_chains_labels_toPlot.items():
            # print(chain_name + str(chain_labels))
            segment_indexes = []
            for chain_label in chain_labels:
                segment_indexes.append(self._segment_labels.index(chain_label))
            self._segment_chains_indexes_toPlot[chain_name] = segment_indexes

        # If using hidden mode, change matplotlib's backend.
        if self._hidden and use_matplotlib:
            matplotlib.use("Agg")

    # Initialize a visualization that displays a skeleton from Pns data.
    # Assumes that the stream's data['data'] field contains segment positions (each row is xyz).
    def init(self, device_name, stream_name, stream_info):
        if self._print_debug: print('PnsSkeletonVisualizer initializing for %s %s' % (device_name, stream_name))

        # Set some options.
        if use_matplotlib:
            figure_size = (7, 5)
        else:
            if self._layout_size is None:
                # screen_widths = [screen.size().width() for screen in app.screens()]
                # screen_heights = [screen.size().heights() for screen in app.screens()]
                screen_width = self._app.primaryScreen().size().width()
                screen_height = self._app.primaryScreen().size().height()
                figure_height = int(screen_height * 0.8)
                figure_width = int(figure_height / 1.2)
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
                                    squeeze=False,  # if False, always return 2D array of axes
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
            fig.suptitle('Pns Skeleton')
            ax.set_xlabel('z')
            ax.set_ylabel('x')
            ax.set_zlabel('y')
            ax.view_init(20, -40)  # (elevation, azimuth) # (20, -40)

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
            self._glWidget.setWindowTitle('Pns Skeleton')
            layout = QtGui.QGridLayout()
            self._layout.setLayout(layout)
            layout.addWidget(self._glWidget, 0, 0, 1, 1)

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
                'fov':  100,
                # 'rotation': (1.0, 0.0, 0.0, 0.0),
                'elevation': 15.0,
                'center': QtGui.QVector3D(-max(self._kitchen_floor_size_cm) * 0.7,
                                          -max(self._kitchen_floor_size_cm) * 0.9,
                                          0.0),
                'azimuth': 30,
                'distance': max(self._kitchen_floor_size_cm) * 1.6
            }
            self._glWidget.setCameraParams(**camera_params)
            # Create a floor.
            grid_color = (0, 0, 0, 80)
            # Add grid perpendicular to z axis
            gz = gl.GLGridItem(color=grid_color)
            gz.setSize(x=self._kitchen_floor_size_cm[0], y=self._kitchen_floor_size_cm[1])
            gz.setSpacing(x=10, y=10, z=10)
            # gz.rotate(90, 1, 0, 0)
            # gz.translate(-int(self._kitchen_floor_size_cm[0] / 2), -int(self._kitchen_floor_size_cm[1] / 2), 0)
            self._glWidget.addItem(gz)
            # Note: to create grid perpendicular to x axis: gx.rotate(90, 0, 1, 0) before translating
            # Note: to create grid perpendicular to y axis: gy.rotate(90, 1, 0, 0) before translating

            # Indicate that no plots have been added yet.
            self._chain_lines = [None] * len(self._segment_chains_indexes_toPlot)
            self._chain_scatters = [None] * len(self._segment_chains_indexes_toPlot)

            self._exporter = pyqtgraph.exporters.ImageExporter(self._layout.scene())

        # Create dummy data, and use it to initialize the plot.
        segment_positions_cm = np.zeros([len(self._segment_labels), 3])
        self.update({'data': [segment_positions_cm.tolist()]},
                    visualizing_all_data=True)

    # Update the skeleton visualization with new segment position data.
    # Only the most recent timestep will be visualized.
    # @param new_data is a dict with 'data' (all other entries will be ignored).
    #   It should contain all segment positions as a matrix (each row is xyz).
    def update(self, new_data, visualizing_all_data):

        # Extract the latest segment positions.
        segment_positions_cm = np.array(new_data['data'][-1]).reshape(21, 3)
        # print(type(segment_positions_cm))
        for i in range(len(segment_positions_cm)):

            # if (i == 16) or (i==17) or (i == 19) or (i==20):
            #     segment_positions_cm[i][0], segment_positions_cm[i][1] = segment_positions_cm[i][1], segment_positions_cm[i][0]
            #     segment_positions_cm[i][2], segment_positions_cm[i][1] = segment_positions_cm[i][1], segment_positions_cm[i][2]
            segment_positions_cm[i][2], segment_positions_cm[i][1] = segment_positions_cm[i][1], segment_positions_cm[i][2]
        # print(segment_positions_cm)
        # print(len(segment_positions_cm))

        if use_matplotlib:
            plot_x_bounds = np.array([1000, -1000])
            plot_y_bounds = np.array([1000, -1000])
            plot_z_bounds = np.array([1000, -1000])
            ax_lines = self._ax.get_lines()
            # Draw each connected chain of segments.
            for (chain_index, chain_name) in enumerate(self._segment_chains_indexes_toPlot.keys()):
                segment_indexes = self._segment_chains_indexes_toPlot[chain_name]
                segment_xyz_cm = segment_positions_cm[segment_indexes, :]
                # Reorder them to make adjusting the plot view angle easier.
                plot_x = segment_xyz_cm[:, 2]
                plot_y = segment_xyz_cm[:, 0]
                plot_z = segment_xyz_cm[:, 1]
                # Create a fresh plot or update existing data.
                if not visualizing_all_data:
                    ax_lines[chain_index].set_data_3d(plot_x, plot_y, plot_z)
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
            self._ax.set_box_aspect(
                [ub - lb for lb, ub in (self._ax.get_xlim(), self._ax.get_ylim(), self._ax.get_zlim())])

            # Update the figure to see the changes.
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()
        else:
            # Draw each connected chain of segments.
            plot_xyz_cm_all = np.zeros(shape=(0, 3))
            for (chain_index, chain_name) in enumerate(self._segment_chains_indexes_toPlot.keys()):

                segment_indexes = self._segment_chains_indexes_toPlot[chain_name]

                segment_xyz_cm = segment_positions_cm[segment_indexes, :]
                # print('segment_XYZ_CM' + chain_name + str(segment_xyz_cm))
                # Negate the x and y coordinates since the floor was visualized in the negative quadrant.
                plot_xyz_cm = segment_xyz_cm * np.array([-1, -1, 1])
                # print(plot_xyz_cm)
                plot_xyz_cm_all = np.concatenate((plot_xyz_cm_all, plot_xyz_cm), axis=0)
                # Create a fresh plot or update existing data.
                if self._chain_lines[chain_index] is None or visualizing_all_data:
                    self._chain_lines[chain_index] = gl.GLLinePlotItem(
                        pos=plot_xyz_cm, color=(1, 0, 0, 1),
                        width=3, antialias=True)
                    self._glWidget.addItem(self._chain_lines[chain_index])
                    self._chain_scatters[chain_index] = gl.GLScatterPlotItem(
                        pos=plot_xyz_cm, color=(1, 0, 0, 1),
                        size=5, pxMode=False)
                    self._chain_scatters[chain_index].setGLOptions('translucent')
                    self._glWidget.addItem(self._chain_scatters[chain_index])
                    cv2.waitKey(1)  # wait for it to actually draw
                    # Show or hide the figure.
                    if not self._is_sub_layout:
                        if not self._hidden:
                            self._layout.show()
                        else:
                            self._layout.hide()
                else:
                    self._chain_lines[chain_index].setData(pos=plot_xyz_cm, antialias=True)
                    self._chain_scatters[chain_index].setData(pos=plot_xyz_cm)
            plot_xyz_cm_mean = np.mean(np.array(plot_xyz_cm_all), 0)
            if np.all(np.abs(plot_xyz_cm_mean) > 0):
                camera_params = {
                    'fov': 85,
                    # 'rotation': (1.0, 0.0, 0.0, 0.0),
                    'elevation': 35.0,
                    'center': QtGui.QVector3D(plot_xyz_cm_mean[0] ,
                                              plot_xyz_cm_mean[1] ,
                                              plot_xyz_cm_mean[2]),
                    'azimuth': 100,
                    'distance': np.max(np.max(plot_xyz_cm_all, axis=0) - np.min(plot_xyz_cm_all, axis=0)) * 1.5
                }
                self._glWidget.setCameraParams(**camera_params)

            # Update the plot to see the changes.
            if not self._hidden and not self._is_sub_layout:
                cv2.waitKey(1)  # find a better way?

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















