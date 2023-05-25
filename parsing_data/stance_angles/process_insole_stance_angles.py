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

import h5py
import numpy as np
from utils.numpy_scipy_utils import convolve2d_strided
from visualizers.FlowFieldVisualizer import FlowFieldVisualizer
from collections import OrderedDict
import os
import pyqtgraph
from pyqtgraph.Qt import QtCore, QtGui
import seaborn as sns
import matplotlib.pyplot as plt
import threading

# NOTE: HDFView is a helpful program for exploring HDF5 contents.
#   The official download page is at https://www.hdfgroup.org/downloads/hdfview.
#   It can also be downloaded without an account from https://www.softpedia.com/get/Others/Miscellaneous/HDFView.shtml.

# Specify the downloaded file to parse.
filepath = os.path.join('P:/MIT/Lab/Wearativity/data/tests',
                        '2023-05-24_shoeAngle_testing',
                        '2023-05-24_17-41-33_shoeAngle_testing_full',
                        '2023-05-24_17-41-53_streamLog_shoeAngle_testing.hdf5')

# Specify the 8x24 submatrix to extract (including the first dimension of time)
insole_submatrix_slice = np.s_[:, 24:32, 6:30] # starts are inclusive, ends are exclusive

# Choose what to plot.
plot_tactile_heatmaps = True
plot_shear_flowfields = True
plot_shear_average_flow = True
plot_magnitude_box_plots = True
plot_magnitudeAngle_scatter = True

label_order_toPlot = [
  '45 toe-down',
  '30 toe-down',
  '15 toe-down',
  '45 heel-down',
  '30 heel-down',
  '15 heel-down',
  '45 inner-down',
  '30 inner-down',
  '15 inner-down',
  '45 outer-down',
  '30 outer-down',
  '15 outer-down',
  ]

####################################################
# Load the raw tactile data
####################################################
print()
print('-'*50)
print('Loading the raw tactile data')

device_name = 'shear-sensor-left'
stream_name = 'tactile_data'

# Open the file.
h5_file = h5py.File(filepath, 'r')

# Extract the raw tactile data and the timestamps.
tactile_time_s = np.squeeze(h5_file[device_name][stream_name]['time_s']) # squeeze (optional) converts from a list of single-element lists to a 1D list
tactile_data_raw = np.squeeze(h5_file[device_name][stream_name]['data'])

# Only use the desired 8x24 slice.
tactile_data_raw = tactile_data_raw[insole_submatrix_slice]

print('  Tactile data summary:')
print('    Matrix shape:', tactile_data_raw.shape)
print('    Duration [s]: %0.2f' % (np.max(tactile_time_s) - np.min(tactile_time_s)))
print('    Mean Fs [Hz]: %0.2f' % ((tactile_data_raw.shape[0]-1)/(np.max(tactile_time_s) - np.min(tactile_time_s))))

####################################################
# Load and process the label data
####################################################
print()
print('-'*50)
print('Extracting activity labels from the HDF5 file')

device_name = 'experiment-activities'
stream_name = 'activities'

# Get the timestamped label data.
# As described in the HDF5 metadata, each row has entries for ['Activity', 'Start/Stop', 'Valid', 'Notes'].
activity_datas = h5_file[device_name][stream_name]['data']
activity_times_s = h5_file[device_name][stream_name]['time_s']
activity_times_s = np.squeeze(np.array(activity_times_s))  # squeeze (optional) converts from a list of single-element lists to a 1D list
# Convert to strings for convenience.
activity_datas = [[x.decode('utf-8') for x in datas] for datas in activity_datas]

# Combine start/stop rows to single activity entries with start/stop times.
#   Each row is either the start or stop of the label.
#   The notes and ratings fields are the same for the start/stop rows of the label, so only need to check one.
exclude_bad_labels = True # some activities may have been marked as 'Bad' or 'Maybe' by the experimenter; submitted notes with the activity typically give more information
activities_labels = []
activities_start_times_s = []
activities_end_times_s = []
activities_ratings = []
activities_notes = []
for (row_index, time_s) in enumerate(activity_times_s):
  label    = activity_datas[row_index][0]
  is_start = activity_datas[row_index][1] == 'Start'
  is_stop  = activity_datas[row_index][1] == 'Stop'
  rating   = activity_datas[row_index][2]
  notes    = activity_datas[row_index][3]
  if exclude_bad_labels and rating in ['Bad', 'Maybe']:
    continue
  # Record the start of a new activity.
  if is_start:
    activities_labels.append(label)
    activities_start_times_s.append(time_s)
    activities_ratings.append(rating)
    activities_notes.append(notes)
  # Record the end of the previous activity.
  if is_stop:
    activities_end_times_s.append(time_s)

print('Activity labels and durations:')
for (label_index, activity_label) in enumerate(activities_labels):
  print('  %02d: [%5.2f s] %s' % (label_index,
                                 activities_end_times_s[label_index] - activities_start_times_s[label_index],
                                 activity_label))

###################################################################
# Map label to a numpy matrix of the tactile data.
###################################################################
print()
print('-'*50)
print('Aggregating data by label')

# Create a dictionary that maps label to
#  a list of numpy matrices recorded for instances of that label.
# Each entry in a list will be Tx32x32 where T is timesteps.
tactile_data_raw_byLabel = OrderedDict()
for (label_index, activity_label) in enumerate(activities_labels):
  # Extract the data for this label segment.
  label_start_time_s = activities_start_times_s[label_index]
  label_end_time_s = activities_end_times_s[label_index]
  data_indexes_forLabel = np.where((tactile_time_s >= label_start_time_s) & (tactile_time_s <= label_end_time_s))[0]
  data_forLabel = tactile_data_raw[data_indexes_forLabel, :]
  # Store the data segment in the dictionary.
  tactile_data_raw_byLabel.setdefault(activity_label, [])
  tactile_data_raw_byLabel[activity_label].append(data_forLabel)

# Create a dictionary that maps label to
#  an average 32x32 matrix recorded across all instances and timesteps for that label.
tactile_data_average_byLabel = OrderedDict()
for (label_index, activity_label) in enumerate(activities_labels):
  # Average the segments recorded with this label.
  data_forLabel = np.concatenate(tactile_data_raw_byLabel[activity_label], axis=0)
  data_averaged = np.mean(data_forLabel, axis=0)
  # Store the data segment in the dictionary.
  tactile_data_average_byLabel[activity_label] = data_averaged

# Subtract a calibration matrix.
calibration_matrix = tactile_data_average_byLabel['flat']
tactile_data_average_calibrated_byLabel = OrderedDict()
for (label_index, activity_label) in enumerate(activities_labels):
  tactile_data_average_calibrated_byLabel[activity_label] = \
    tactile_data_average_byLabel[activity_label] - calibration_matrix

###################################################################
# Compute overall shear quantities for each matrix
###################################################################

shear_magnitude_byLabel = OrderedDict()
shear_angle_rad_byLabel = OrderedDict()
shear_magnitudeAngle_byLabel = OrderedDict()
shear_average_magnitudeAngle_byLabel = OrderedDict()
for (label_index, activity_label) in enumerate(activities_labels):
  data_matrix_averaged_calibrated = tactile_data_average_calibrated_byLabel[activity_label]
  
  # Compute the total force in each shear square.
  toConvolve_tiled_magnitude = np.array([[1,1],[1,1]])
  data_matrix_tiled_magnitude = convolve2d_strided(data_matrix_averaged_calibrated, toConvolve_tiled_magnitude, stride=2)
  
  # Compute the force angle and magnitude in each shear square.
  toConvolve_tiled_x = np.array([[-1,1],[-1,1]])
  toConvolve_tiled_y = np.array([[1,1],[-1,-1]])
  data_matrix_tiled_x = convolve2d_strided(data_matrix_averaged_calibrated, toConvolve_tiled_x, stride=2)
  data_matrix_tiled_y = convolve2d_strided(data_matrix_averaged_calibrated, toConvolve_tiled_y, stride=2)
  data_matrix_tiled_shearAngle_rad = np.arctan2(data_matrix_tiled_y, data_matrix_tiled_x)
  data_matrix_tiled_shearMagnitude = np.linalg.norm(np.stack([data_matrix_tiled_y, data_matrix_tiled_x], axis=0), axis=0)
  
  # Compute average shear over whole sensor.
  average_shear_x = np.mean(data_matrix_tiled_x)
  average_shear_y = np.mean(data_matrix_tiled_y)
  average_shearAngle_rad = np.arctan2(average_shear_y, average_shear_x)
  average_shearMagnitude = np.linalg.norm([average_shear_y, average_shear_x])
  
  # Store the results.
  shear_magnitude_byLabel[activity_label] = data_matrix_tiled_shearMagnitude
  shear_angle_rad_byLabel[activity_label] = data_matrix_tiled_shearAngle_rad
  shear_magnitudeAngle_byLabel[activity_label] = np.stack((data_matrix_tiled_shearMagnitude,
                                                           data_matrix_tiled_shearAngle_rad),
                                                          axis=0)
  shear_average_magnitudeAngle_byLabel[activity_label] = np.stack((np.atleast_2d(average_shearMagnitude),
                                                                   np.atleast_2d(average_shearAngle_rad)),
                                                                  axis=0)

###################################################################
# Create variables that will be used for pyqtgraph plotting
###################################################################

pyqtgraph_app = QtGui.QApplication([])
screen_width = pyqtgraph_app.primaryScreen().size().width()
screen_height = pyqtgraph_app.primaryScreen().size().height()
figure_width = int(screen_width*0.5)
figure_height = int(figure_width/1.5)
figure_size = (figure_width, figure_height)
pyqtgraph.setConfigOption('background', 'w')
pyqtgraph.setConfigOption('foreground', 'k')
pyqtgraph_layouts = []
flowfield_visualizers = []

###################################################################
# Plot a heatmap of tactile data for each label
###################################################################

if plot_tactile_heatmaps:
  labels_to_exclude = ['unworn', 'flat']
  
  layout = pyqtgraph.GraphicsLayoutWidget(show=True)
  pyqtgraph_layouts.append(layout)
  layout.setGeometry(10, 10, *figure_size)
  layout.setWindowTitle('Average Calibrated Tactile Data')
  num_labels_toPlot = len(label_order_toPlot)
  num_cols = 3
  num_rows = np.ceil(num_labels_toPlot/num_cols)
  subplot_row = 0
  subplot_col = 0
  max_level = None
  min_level = None
  h_colorbars = []
  for (label_index, activity_label) in enumerate(label_order_toPlot): # enumerate(activities_labels)
    if activity_label in labels_to_exclude:
      continue
  
    # Transpose since image is indexed as (x, y) but numpy as (y, x).
    # Flip so y=0 is at the top of the heatmap.
    matrix_toPlot = tactile_data_average_calibrated_byLabel[activity_label]
    matrix_toPlot = np.flipud(matrix_toPlot).T
  
    title = '%s' % activity_label
    h_heatmap = pyqtgraph.ImageItem(image=matrix_toPlot, hoverable=True)
    h_plot = layout.addPlot(subplot_row,subplot_col, 1,1, title=title) # row, col, rowspan, colspan
    h_plot.addItem(h_heatmap, title=title)
    h_plot.hideAxis('bottom')
    h_plot.hideAxis('left')
    h_plot.setAspectLocked(True)
    # Add a colorbar, then set the image again to force an update
    h_colorbar = h_plot.addColorBar(h_heatmap, colorMap='inferno') # , interactive=False)
    h_heatmap.setImage(matrix_toPlot)
    h_colorbar.setLevels(h_heatmap.getLevels())
    if max_level is None or max(h_heatmap.getLevels()) > max_level:
      max_level = max(h_heatmap.getLevels())
    if min_level is None or min(h_heatmap.getLevels()) < min_level:
      min_level = min(h_heatmap.getLevels())
    h_colorbars.append(h_colorbar)
    # Add a callback to show values.
    def _mouse_moved_callback(mouse_position):
      # Check if event is inside heatmap, and convert from screen/pixels to image xy indexes.
      if h_plot.sceneBoundingRect().contains(mouse_position):
        mouse_point = h_plot.getViewBox().mapSceneToView(mouse_position)
        x_i = int(mouse_point.x())
        y_i = int(mouse_point.y())
        if x_i >= 0 and x_i < matrix_toPlot.shape[0] and y_i >= 0 and y_i < matrix_toPlot.shape[1]:
          y_i = (matrix_toPlot.shape[0]-1) - y_i # since the matrix was flipped to put y=0 at the top of the heatmap
          layout.window().setToolTip('(y=%d, x=%d): %0.2f' %
                                           (y_i, x_i, matrix_toPlot[y_i][x_i]))
          return
    h_plot.scene().sigMouseMoved.connect(_mouse_moved_callback)
  
    subplot_col = (subplot_col+1) % num_cols
    if subplot_col == 0:
      subplot_row += 1
  
  # # Set all heatmaps to the same levels
  # for h_colorbar in h_colorbars:
  #   h_colorbar.setLevels([min_level, max_level])
  
  QtCore.QCoreApplication.processEvents()

###################################################################
# Plot a flow field for each label
###################################################################

if plot_shear_flowfields:
  labels_to_exclude = ['unworn', 'flat']
  layout = pyqtgraph.GraphicsLayoutWidget(show=True)
  pyqtgraph_layouts.append(layout)
  layout.setGeometry(10, 10, *figure_size)
  layout.setWindowTitle('Average Calibrated Shear Data')
  num_labels_toPlot = len(label_order_toPlot)
  num_cols = 3
  num_rows = np.ceil(num_labels_toPlot/num_cols)
  subplot_row = 0
  subplot_col = 0
  for (label_index, activity_label) in enumerate(label_order_toPlot): # enumerate(activities_labels)
    if activity_label in labels_to_exclude:
      continue

    shear_magnitudeAngle = shear_magnitudeAngle_byLabel[activity_label]

    title = '%s' % activity_label
    
    v = FlowFieldVisualizer(parent_layout=layout, visualizer_options={'linewidth':5})
    v.init(activity_label, '', {'sample_size':shear_magnitudeAngle.shape},
           subplot_row=subplot_row, subplot_col=subplot_col)
    v.update(new_data={'data':[shear_magnitudeAngle]}, visualizing_all_data=False)
    flowfield_visualizers.append(v)
  
    subplot_col = (subplot_col+1) % num_cols
    if subplot_col == 0:
      subplot_row += 1
  
  QtCore.QCoreApplication.processEvents()
  
###################################################################
# Plot the average flow for each label
###################################################################

if plot_shear_average_flow:
  labels_to_exclude = ['unworn', 'flat']
  
  # Will normalize all plots by the same scale factor
  normalization_factor = 1
  for (label_index, activity_label) in enumerate(label_order_toPlot): # enumerate(activities_labels)
    if activity_label in labels_to_exclude:
      continue
    shear_magnitudeAngle = shear_average_magnitudeAngle_byLabel[activity_label]
    shear_magnitude = float(np.squeeze(shear_magnitudeAngle[0]))
    if shear_magnitude > normalization_factor:
      normalization_factor = shear_magnitude
  
  layout = pyqtgraph.GraphicsLayoutWidget(show=True)
  pyqtgraph_layouts.append(layout)
  layout.setGeometry(10, 10, *figure_size)
  layout.setWindowTitle('Average Calibrated Shear Data')
  num_labels_toPlot = len(label_order_toPlot)
  num_cols = 3
  num_rows = np.ceil(num_labels_toPlot/num_cols)
  subplot_row = 0
  subplot_col = 0
  for (label_index, activity_label) in enumerate(label_order_toPlot): # enumerate(activities_labels)
    if activity_label in labels_to_exclude:
      continue

    shear_magnitudeAngle = shear_average_magnitudeAngle_byLabel[activity_label]

    title = '%s' % activity_label
    
    v = FlowFieldVisualizer(parent_layout=layout, visualizer_options={'magnitude_normalization':normalization_factor,
                                                                      'linewidth':5})
    v.init(activity_label, '', {'sample_size':shear_magnitudeAngle.shape},
           subplot_row=subplot_row, subplot_col=subplot_col)
    v.update(new_data={'data':[shear_magnitudeAngle]}, visualizing_all_data=False)
    flowfield_visualizers.append(v)
  
    subplot_col = (subplot_col+1) % num_cols
    if subplot_col == 0:
      subplot_row += 1
  
  QtCore.QCoreApplication.processEvents()
  
###################################################################
# Plot magnitude box plots
###################################################################

if plot_magnitude_box_plots:
  label_order_toBoxPlot = label_order_toPlot
  label_order_toBoxPlot = sorted(label_order_toBoxPlot)

  plt.figure()
  magnitudes = []
  angles_deg = []
  for (label_index, activity_label) in enumerate(label_order_toBoxPlot): # enumerate(activities_labels)
    shear_magnitudeAngle = shear_magnitudeAngle_byLabel[activity_label]
    shear_magnitude = np.squeeze(shear_magnitudeAngle[0])
    shear_angle_rad = np.squeeze(shear_magnitudeAngle[1])
    shear_angle_deg = shear_angle_rad*180/np.pi
    magnitudes.append(np.squeeze(shear_magnitude.reshape((1,-1))))
    angles_deg.append(np.squeeze(shear_angle_deg.reshape((1,-1))))

  sns.set(style="whitegrid")
  
  # The main boxplot
  ax = sns.boxplot(data=magnitudes)
  # Add jitter with the swarmplot function
  sns.swarmplot(data=magnitudes, size=3)
  # Add labels
  ax.set_xticklabels([x.split('-down')[0].replace(' ','\n') for x in label_order_toBoxPlot])
  # plt.xticks(rotation=45)
  plt.ylabel('Shear Magnitudes in Time-Averaged Matrix')
  plt.title('Shear Magnitudes in Time-Averaged Matrix per Label')
  
  figManager = plt.get_current_fig_manager()
  figManager.window.showMaximized()

###################################################################
# Plot magnitude-angle scatter plots
###################################################################

if plot_magnitudeAngle_scatter:
  label_order_toBoxPlot = label_order_toPlot
  label_order_toBoxPlot = sorted(label_order_toBoxPlot)
  
  plt.figure()
  for (label_index, activity_label) in enumerate(label_order_toBoxPlot): # enumerate(activities_labels)
    shear_magnitudeAngle = shear_magnitudeAngle_byLabel[activity_label]
    shear_magnitude = np.squeeze(shear_magnitudeAngle[0])
    shear_angle_rad = np.squeeze(shear_magnitudeAngle[1])
    shear_angle_deg = shear_angle_rad*180/np.pi
    magnitudes = np.squeeze(shear_magnitude.reshape((1,-1)))
    angles_deg = np.squeeze(shear_angle_deg.reshape((1,-1)))

    shear_average_magnitudeAngle = shear_average_magnitudeAngle_byLabel[activity_label]
    shear_average_magnitude = np.squeeze(shear_average_magnitudeAngle[0])
    shear_average_angle_rad = np.squeeze(shear_average_magnitudeAngle[1])
    shear_average_angle_deg = shear_average_angle_rad*180/np.pi
  
    # Plot the scatter for this label
    ax = plt.scatter(magnitudes, angles_deg, label=activity_label)
    plt.draw()
    plt.scatter(shear_average_magnitude, shear_average_angle_deg,
                s=500,
                color=ax.get_facecolors()[0].tolist())
    
  # Format
  plt.legend()
  plt.grid(True, color='lightgray')
  plt.ylabel('Shear Angle [deg]')
  plt.xlabel('Shear Magnitude')
  plt.title('Shear Magnitudes and Angles in Time-Averaged Matrix per Label')
  
  figManager = plt.get_current_fig_manager()
  figManager.window.showMaximized()

###################################################################
# Show all plots
###################################################################

using_pyqtgraph = plot_tactile_heatmaps or plot_shear_flowfields or plot_shear_average_flow
using_pyplot = plot_magnitude_box_plots or plot_magnitudeAngle_scatter

if using_pyqtgraph:
  def run_pyqtgraphs():
    pyqtgraph_app.exec()
  run_pyqtgraph_thread = threading.Thread(target=run_pyqtgraphs)
  run_pyqtgraph_thread.start()
if using_pyplot:
  plt.show()
if using_pyqtgraph:
  run_pyqtgraph_thread.join()




