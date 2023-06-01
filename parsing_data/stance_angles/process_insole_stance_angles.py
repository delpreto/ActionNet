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
                        '2023-05-24_17-41-53_streamLog_shoeAngle_testing_updated.hdf5')

# Specify the 8x24 submatrix to extract (including the first dimension of time)
insole_submatrix_slice = np.s_[:, 24:32, 6:30] # starts are inclusive, ends are exclusive
x = 0
n = np.nan
insole_shear_layout = np.array([
  [x, n, n, n, 1, n, 1, n, 1, n, 1, n, 1, n, 1, n, 1, n, n, x, x, x, x,],
  [1, n, 1, n, 1, n, 1, n, 1, n, 1, n, n, n, n, n, n, n, n, n, n, x, x,],
  [1, n, 1, n, 1, n, 1, n, 1, n, n, n, 1, n, 1, n, 1, n, 1, 1, 1, n, x,],
  [1, n, 1, n, 1, n, 1, n, 1, n, 1, n, n, n, n, n, n, n, 1, 1, 1, n, n,],
  [1, n, 1, n, n, n, n, n, n, n, 1, n, 1, n, 1, n, 1, n, n, n, n, n, n,],
  [x, n, n, n, n, n, n, n, n, n, n, n, 1, n, n, n, n, n, 1, 1, 1, n, n,],
  [x, x, x, x, x, x, x, x, x, x, x, n, n, n, 1, n, 1, n, 1, 1, 1, n, x,],
  [x, x, x, x, x, x, x, x, x, x, x, x, x, x, n, n, n, n, n, n, n, x, x,],
])
insole_tactile_layout = []
for shear_layout_row_index in range(insole_shear_layout.shape[0]):
  shear_row_data = insole_shear_layout[shear_layout_row_index, :]
  tactile_row_data = np.squeeze(np.array([shear_row_data, shear_row_data]).T.reshape((1,-1)))
  insole_tactile_layout.append(tactile_row_data)
  insole_tactile_layout.append(tactile_row_data)
insole_tactile_layout = np.array(insole_tactile_layout)

def matrix_to_layout(data_matrix, layout):
  data_layout = np.nan*np.ones(shape=layout.shape)
  data_column_index = 0
  for layout_column_index in range(layout.shape[1]):
    layout_column = layout[:, layout_column_index]
    column_sensor_indexes = np.where(layout_column == 1)[0]
    column_nan_indexes = np.where(np.isnan(layout_column))[0]
    column_zero_indexes = np.where(layout_column == 0)[0]
    data_layout[column_nan_indexes, layout_column_index] = np.nan
    data_layout[column_zero_indexes, layout_column_index] = np.nan
    if column_sensor_indexes.size > 0:
      data_layout[column_sensor_indexes, layout_column_index] = data_matrix[:, data_column_index]
      data_column_index += 1
  return data_layout
def shear_matrix_to_insole_layout(shear_matrix):
  return matrix_to_layout(shear_matrix, insole_shear_layout)
def tactile_matrix_to_insole_layout(tactile_matrix):
  return matrix_to_layout(tactile_matrix, insole_tactile_layout)

# x = np.array(range(4*12)).reshape((4,12))+1
# print(x)
# print(shear_matrix_to_insole_layout(x))
# x = np.array(range(8*24)).reshape((8,24))+1
# print(x)
# print(tactile_matrix_to_insole_layout(x))

# Choose what to plot.
plot_tactile_heatmaps = True
plot_shear_flowfields = True
plot_shear_flowfields_onlyActive = True
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
  'flat',
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
exp_activities_labels = []
exp_activities_start_times_s = []
exp_activities_end_times_s = []
exp_activities_ratings = []
exp_activities_notes = []
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
    exp_activities_labels.append(label)
    exp_activities_start_times_s.append(time_s)
    exp_activities_ratings.append(rating)
    exp_activities_notes.append(notes)
  # Record the end of the previous activity.
  if is_stop:
    exp_activities_end_times_s.append(time_s)

# Create a unique list of labels, which preserves the original first-seen ordering.
activity_labels = []
for activity_label in exp_activities_labels:
  if activity_label not in activity_labels:
    activity_labels.append(activity_label)

# Print some information about the labels.
print('Activity sequence and durations:')
for (label_index, activity_label) in enumerate(exp_activities_labels):
  print('  %02d: [%5.2f s] %s' % (label_index,
                                  exp_activities_end_times_s[label_index] - exp_activities_start_times_s[label_index],
                                  activity_label))
print()
print('Unique activity labels and durations:')
for (label_index, activity_label) in enumerate(activity_labels):
  durations_s = [exp_activities_end_times_s[i] - exp_activities_start_times_s[i] for i in range(len(exp_activities_labels)) if exp_activities_labels[i] == activity_label]
  print('  %02d: [%5.2f s] %s' % (label_index,
                                  sum(durations_s),
                                  activity_label))

###################################################################
# Map labels to a numpy matrix of the tactile data.
###################################################################
print()
print('-'*50)
print('Aggregating data by label')

# Create a dictionary that maps label to
#  a list of numpy matrices recorded for trials of that label.
# Each entry in a list will be Tx8x24 where T is timesteps.
tactile_data_raw_byLabel_byTrial = OrderedDict()
for (label_index, activity_label) in enumerate(exp_activities_labels):
  # Extract the data for this label segment.
  label_start_time_s = exp_activities_start_times_s[label_index]
  label_end_time_s = exp_activities_end_times_s[label_index]
  data_indexes_forLabel = np.where((tactile_time_s >= label_start_time_s) & (tactile_time_s <= label_end_time_s))[0]
  data_forLabel = tactile_data_raw[data_indexes_forLabel, :]
  # Store the data segment in the dictionary.
  tactile_data_raw_byLabel_byTrial.setdefault(activity_label, [])
  tactile_data_raw_byLabel_byTrial[activity_label].append(data_forLabel)

# Create a dictionary that maps label to
#  an average matrix recorded across all trials and timesteps for that label.
tactile_data_average_byLabel = OrderedDict()
for (label_index, activity_label) in enumerate(activity_labels):
  # Average the segments recorded with this label.
  data_forLabel = np.concatenate(tactile_data_raw_byLabel_byTrial[activity_label], axis=0)
  data_averaged = np.mean(data_forLabel, axis=0)
  # Store the data segment in the dictionary.
  tactile_data_average_byLabel[activity_label] = data_averaged

# Subtract a calibration matrix.
calibration_matrix = tactile_data_average_byLabel['unworn']
tactile_data_calibrated_byLabel_byTrial = OrderedDict()
tactile_data_average_calibrated_byLabel = OrderedDict()
for (label_index, activity_label) in enumerate(activity_labels):
  tactile_data_calibrated_byLabel_byTrial[activity_label] = \
    [tactile_data_raw_byLabel_byTrial[activity_label][i] - calibration_matrix
     for i in range(len(tactile_data_raw_byLabel_byTrial[activity_label]))]
  tactile_data_average_calibrated_byLabel[activity_label] = \
    tactile_data_average_byLabel[activity_label] - calibration_matrix

###################################################################
# Compute shear quantities for each matrix

# tactile_data_raw_byLabel_byTrial			  label: [Tx8x24]
# tactile_data_calibrated_byLabel_byTrial	label: [Tx8x24]
# tactile_data_average_byLabel			      label: 8x24
# tactile_data_average_calibrated_byLabel	label: 8x24
#
# shear_magnitude_byLabel					                    label: [Tx4x12]
# shear_angle_rad_byLabel					                    label: [Tx4x12]
# shear_magnitudeAngle_byLabel			                  label: [Tx2x4x12]
# shear_magnitude_average_byLabel			                label: 4x12
# shear_angle_rad_average_byLabel			                label: 4x12
# shear_magnitudeAngle_average_byLabel		            label: 2x4x12
# shear_magnitude_averageActive_byLabel			          label: 4x12
# shear_angle_rad_averageActive_byLabel			          label: 4x12
# shear_magnitudeAngle_averageActive_byLabel 	        label: 2x4x12
# shear_magnitudeAngle_averageActive_overall_byLabel	label: 2x1x1
###################################################################

toConvolve_tiled_magnitude = np.array([[1,1],[1,1]])
toConvolve_tiled_x = np.array([[-1,1],[-1,1]])
toConvolve_tiled_y = np.array([[1,1],[-1,-1]])

shear_magnitude_byLabel = OrderedDict()
shear_angle_rad_byLabel = OrderedDict()
shear_magnitudeAngle_byLabel = OrderedDict()
shear_magnitude_average_byLabel = OrderedDict()
shear_angle_rad_average_byLabel = OrderedDict()
shear_magnitudeAngle_average_byLabel = OrderedDict()
shear_magnitude_averageActive_byLabel = OrderedDict()
shear_angle_rad_averageActive_byLabel = OrderedDict()
shear_magnitudeAngle_averageActive_byLabel = OrderedDict()
shear_magnitudeAngle_averageActive_overall_byLabel = OrderedDict()
for (label_index, activity_label) in enumerate(activity_labels):
  shear_magnitude_byLabel.setdefault(activity_label, [])
  shear_angle_rad_byLabel.setdefault(activity_label, [])
  shear_magnitudeAngle_byLabel.setdefault(activity_label, [])
  
  data_matrix_calibrated_byTrial = tactile_data_calibrated_byLabel_byTrial[activity_label]

  data_matrix_tiled_x_allForLabel = []
  data_matrix_tiled_y_allForLabel = []
  tactile_data_calibrated_allForLabel = []
  for (trial_index, data_matrix_calibrated_allT) in enumerate(data_matrix_calibrated_byTrial):
    shear_magnitude_byLabel[activity_label].append([]) # will be cast to np.array later
    shear_angle_rad_byLabel[activity_label].append([]) # will be cast to np.array later
    shear_magnitudeAngle_byLabel[activity_label].append([]) # will be cast to np.array later

    for (timestep_index, data_matrix_calibrated) in enumerate(data_matrix_calibrated_allT):
      # Compute the total force in each shear square.
      data_matrix_tiled_magnitude = convolve2d_strided(data_matrix_calibrated, toConvolve_tiled_magnitude, stride=2)
  
      # Compute the force angle and magnitude in each shear square.
      data_matrix_tiled_x = convolve2d_strided(data_matrix_calibrated, toConvolve_tiled_x, stride=2)
      data_matrix_tiled_y = convolve2d_strided(data_matrix_calibrated, toConvolve_tiled_y, stride=2)
      data_matrix_tiled_shearAngle_rad = np.arctan2(data_matrix_tiled_y, data_matrix_tiled_x)
      data_matrix_tiled_shearMagnitude = np.linalg.norm(np.stack([data_matrix_tiled_y, data_matrix_tiled_x], axis=0), axis=0)
  
      # Store the results.
      shear_magnitude_byLabel[activity_label][trial_index].append(data_matrix_tiled_shearMagnitude)
      shear_angle_rad_byLabel[activity_label][trial_index].append(data_matrix_tiled_shearAngle_rad)
      shear_magnitudeAngle_byLabel[activity_label][trial_index].append(
          np.stack((data_matrix_tiled_shearMagnitude,
          data_matrix_tiled_shearAngle_rad),
          axis=0))
      data_matrix_tiled_x_allForLabel.append(data_matrix_tiled_x)
      data_matrix_tiled_y_allForLabel.append(data_matrix_tiled_y)
    tactile_data_calibrated_allForLabel.append(data_matrix_calibrated_allT)
    # Convert lists to numpy arrays.
    shear_magnitude_byLabel[activity_label][trial_index] = np.array(shear_magnitude_byLabel[activity_label][trial_index])
    shear_angle_rad_byLabel[activity_label][trial_index] = np.array(shear_angle_rad_byLabel[activity_label][trial_index])
    shear_magnitudeAngle_byLabel[activity_label][trial_index] = np.array(shear_magnitudeAngle_byLabel[activity_label][trial_index])
  # Compute naive average for this label.
  data_matrix_tiled_x_allForLabel = np.array(data_matrix_tiled_x_allForLabel)
  data_matrix_tiled_y_allForLabel = np.array(data_matrix_tiled_y_allForLabel)
  data_matrix_tiled_average_x = np.mean(data_matrix_tiled_x_allForLabel, axis=0)
  data_matrix_tiled_average_y = np.mean(data_matrix_tiled_y_allForLabel, axis=0)
  shear_magnitude_average_byLabel[activity_label] = np.linalg.norm(np.stack([data_matrix_tiled_average_x, data_matrix_tiled_average_y], axis=0), axis=0)
  shear_angle_rad_average_byLabel[activity_label] = np.arctan2(data_matrix_tiled_average_y, data_matrix_tiled_average_x)
  shear_magnitudeAngle_average_byLabel[activity_label] = np.stack(
                                                          (shear_magnitude_average_byLabel[activity_label],
                                                           shear_angle_rad_average_byLabel[activity_label]),
                                                          axis=0)
  # Determine averages using active, interior cells.
  tactile_data_calibrated_allForLabel = np.concatenate(tactile_data_calibrated_allForLabel, axis=0)
  data_matrix_tiled_x_allForLabel_active = np.zeros_like(data_matrix_tiled_x_allForLabel)
  data_matrix_tiled_y_allForLabel_active = np.zeros_like(data_matrix_tiled_y_allForLabel)
  data_matrix_tiled_x_averageActive = np.zeros(shape=(tactile_data_calibrated_allForLabel.shape[0], 1))
  data_matrix_tiled_y_averageActive = np.zeros(shape=(tactile_data_calibrated_allForLabel.shape[0], 1))
  for t in range(tactile_data_calibrated_allForLabel.shape[0]):
    tactile_data_calibrated = np.squeeze(tactile_data_calibrated_allForLabel[t,:,:])
    # threshold = np.mean([np.max(tactile_data_calibrated), np.min(tactile_data_calibrated)])
    threshold = np.quantile(tactile_data_calibrated, 0.5)
    is_active = np.ones_like(tactile_data_calibrated).astype(bool)
    is_active[tactile_data_calibrated < threshold] = 0
    #is_interior = np.ones_like(tactile_data_calibrated).astype(bool)
    is_interior = np.array([
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
      [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,],
      [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,],
      [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,],
      [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    ]).astype(bool)
    to_select = is_active & is_interior
    tactile_data_calibrated_selected = np.copy(tactile_data_calibrated)
    tactile_data_calibrated_selected[~to_select] = np.nan

    data_matrix_tiled_x = convolve2d_strided(tactile_data_calibrated_selected, toConvolve_tiled_x, stride=2)
    data_matrix_tiled_y = convolve2d_strided(tactile_data_calibrated_selected, toConvolve_tiled_y, stride=2)

    data_matrix_tiled_x_allForLabel_active[t, :,:] = data_matrix_tiled_x
    data_matrix_tiled_y_allForLabel_active[t, :,:] = data_matrix_tiled_y
    data_matrix_tiled_x_averageActive[t] = np.nanmean(data_matrix_tiled_x)
    data_matrix_tiled_y_averageActive[t] = np.nanmean(data_matrix_tiled_y)
    
  data_matrix_tiled_average_x_active = np.nanmean(data_matrix_tiled_x_allForLabel_active, axis=0)
  data_matrix_tiled_average_y_active = np.nanmean(data_matrix_tiled_y_allForLabel_active, axis=0)
  shear_magnitude_averageActive_byLabel[activity_label] = np.linalg.norm(np.stack([data_matrix_tiled_average_x_active, data_matrix_tiled_average_y_active], axis=0), axis=0)
  shear_angle_rad_averageActive_byLabel[activity_label] = np.arctan2(data_matrix_tiled_average_y_active, data_matrix_tiled_average_x_active)
  shear_magnitudeAngle_averageActive_byLabel[activity_label] = np.stack(
      (shear_magnitude_averageActive_byLabel[activity_label],
       shear_angle_rad_averageActive_byLabel[activity_label]),
      axis=0)
  
  data_matrix_tiled_average_x_active = np.nanmean(data_matrix_tiled_x_averageActive)
  data_matrix_tiled_average_y_active = np.nanmean(data_matrix_tiled_y_averageActive)
  shear_magnitude_averageActive_overall = np.linalg.norm(np.stack([data_matrix_tiled_average_x_active, data_matrix_tiled_average_y_active], axis=0), axis=0)
  shear_angle_rad_averageActive_overall = np.arctan2(data_matrix_tiled_average_y_active, data_matrix_tiled_average_x_active)
  shear_magnitudeAngle_averageActive_overall_byLabel[activity_label] = np.stack(
      (np.atleast_2d(shear_magnitude_averageActive_overall),
       np.atleast_2d(shear_angle_rad_averageActive_overall)),
      axis=0)
  
  
  
# # Compute overall average, considering interior activated cells.
# shear_magnitudeAngle_averageActive_byLabel = OrderedDict()
# shear_magnitudeAngle_averageActive_overall_byLabel = OrderedDict()
# for (label_index, activity_label) in enumerate(activity_labels):
#   # Get the [Tx4x12] sequences
#   tactile_data = tactile_data_calibrated_byLabel_byTrial[activity_label]
#   shear_magnitude = shear_magnitude_byLabel[activity_label]
#   shear_angle_rad = shear_angle_rad_byLabel[activity_label]
#
#   is_active = np.zeros_like(shear_magnitude)
#   is_interior = np.zeros_like(shear_magnitude)
#   for t in range(shear_magnitude.shape[0]):
#     # Determine activation threshold.
#     tactile_magnitudes = np.squeeze(tactile_data[t,:,:])
#
#     # Determine interior cells.
#
#   shear_magnitude_selected = np.copy(shear_magnitude)
#   shear_angle_rad_selected = np.copy(shear_angle_rad)
#   shear_magnitude_selected[~is_active] = np.nan
#   shear_magnitude_selected[~is_interior] = np.nan
#   shear_angle_rad_selected[~is_active] = np.nan
#   shear_angle_rad_selected[~is_interior] = np.nan
  
  
  
  

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
  colormap = 'plasma' # ['plasma', 'viridis', 'inferno', 'magma']
  for (label_index, activity_label) in enumerate(label_order_toPlot): # enumerate(activities_labels)
    
    matrix_toPlot = tactile_data_average_calibrated_byLabel[activity_label]
    matrix_toPlot = tactile_matrix_to_insole_layout(matrix_toPlot)
    
    # Flip so y=0 is at the top of the heatmap.
    matrix_toPlot = np.flipud(matrix_toPlot)
    # Transpose since image is indexed as (x, y) but numpy as (y, x).
    matrix_toPlot = matrix_toPlot.T
  
    title = '%s' % activity_label
    h_heatmap = pyqtgraph.ImageItem(image=matrix_toPlot, hoverable=True)
    h_plot = layout.addPlot(subplot_row,subplot_col, 1,1, title=title) # row, col, rowspan, colspan
    h_plot.addItem(h_heatmap, title=title)
    h_plot.hideAxis('bottom')
    h_plot.hideAxis('left')
    h_plot.setAspectLocked(True)
    # Add a colorbar, then set the image again to force an update
    h_colorbar = h_plot.addColorBar(h_heatmap, colorMap=colormap) # , interactive=False)
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

shear_magnitudeAngles_byLabel_toPlot = []
if plot_shear_flowfields:
  shear_magnitudeAngles_byLabel_toPlot.append(shear_magnitudeAngle_average_byLabel)
if plot_shear_flowfields_onlyActive:
  shear_magnitudeAngles_byLabel_toPlot.append(shear_magnitudeAngle_averageActive_byLabel)

for shear_magnitudeAngle_byLabel_toPlot in shear_magnitudeAngles_byLabel_toPlot:
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

    # shear_magnitudeAngle = shear_magnitudeAngle_average_byLabel[activity_label]
    # shear_magnitudeAngle = shear_magnitudeAngle_averageActive_byLabel[activity_label]
    shear_magnitudeAngle = shear_magnitudeAngle_byLabel_toPlot[activity_label]
    
    title = '%s' % activity_label
    
    v = FlowFieldVisualizer(parent_layout=layout, visualizer_options={'linewidth':5,
                                                                      'data_transform_fn':shear_matrix_to_insole_layout})
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
  
  # Will normalize all plots by the same scale factor
  normalization_factor = 1
  for (label_index, activity_label) in enumerate(label_order_toPlot): # enumerate(activities_labels)
    shear_magnitudeAngle = shear_magnitudeAngle_averageActive_overall_byLabel[activity_label]
    shear_magnitude = float(np.squeeze(shear_magnitudeAngle[0]))
    if shear_magnitude > normalization_factor:
      normalization_factor = shear_magnitude
  normalization_factor *= 1
  
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

    shear_magnitudeAngle = shear_magnitudeAngle_averageActive_overall_byLabel[activity_label]

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
  # label_order_toBoxPlot = sorted(label_order_toBoxPlot)

  plt.figure()
  magnitudes = []
  angles_deg = []
  for (label_index, activity_label) in enumerate(label_order_toBoxPlot): # enumerate(activities_labels)
    shear_magnitudeAngle = shear_magnitudeAngle_averageActive_byLabel[activity_label]
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
  # label_order_toBoxPlot = sorted(label_order_toBoxPlot)
  
  plt.figure()
  for (label_index, activity_label) in enumerate(label_order_toBoxPlot): # enumerate(activities_labels)
    shear_magnitudeAngle = shear_magnitudeAngle_averageActive_byLabel[activity_label]
    shear_magnitude = np.squeeze(shear_magnitudeAngle[0])
    shear_angle_rad = np.squeeze(shear_magnitudeAngle[1])
    shear_angle_deg = shear_angle_rad*180/np.pi
    magnitudes = np.squeeze(shear_magnitude.reshape((1,-1)))
    angles_deg = np.squeeze(shear_angle_deg.reshape((1,-1)))

    shear_average_magnitudeAngle = shear_magnitudeAngle_averageActive_overall_byLabel[activity_label]
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
  pyqtgraph_app.exec()
if using_pyplot:
  plt.show()





