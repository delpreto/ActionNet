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
from scipy import interpolate
from scipy.spatial.transform import Rotation
from collections import OrderedDict
import os
import glob
script_dir = os.path.dirname(os.path.realpath(__file__))
import sys
import cv2

import pandas
from bs4 import BeautifulSoup

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d

from utils.numpy_scipy_utils import *
from utils.angle_utils import *
from utils.print_utils import *
from utils.time_utils import *

# Specify the folder of experiments to parse.
data_dir = os.path.realpath(os.path.join(script_dir, '..', '..', '..', 'data'))
experiments_dir = os.path.join(data_dir, 'tests', '2023-08-31_experiment_S00_xsens_joint_angles')
output_dir = os.path.realpath(os.path.join(script_dir, '..', '..', '..', 'results', 'learning_trajectories',
                                           '2023-08-31_test_xsens_joint_angles'))
os.makedirs(output_dir, exist_ok=True)


##############################################

bodySegment_labels = [
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
]
bodyJoint_labels = [
  'L5S1',
  'L4L3',
  'L1T12',
  'T9T8',
  'T1C7',
  'C1 Head',
  'Right T4 Shoulder',
  'Right Shoulder',
  'Right Elbow',
  'Right Wrist',
  'Left T4 Shoulder',
  'Left Shoulder',
  'Left Elbow',
  'Left Wrist',
  'Right Hip',
  'Right Knee',
  'Right Ankle',
  'Right Ball Foot',
  'Left Hip',
  'Left Knee',
  'Left Ankle',
  'Left Ball Foot',
  'Left First CMC',
  'Left First MCP',
  'Left IP',
  'Left Second CMC',
  'Left Second MCP',
  'Left Second PIP',
  'Left Second DIP',
  'Left Third CMC',
  'Left Third MCP',
  'Left Third PIP',
  'Left Third DIP',
  'Left Fourth CMC',
  'Left Fourth MCP',
  'Left Fourth PIP',
  'Left Fourth DIP',
  'Left Fifth CMC',
  'Left Fifth MCP',
  'Left Fifth PIP',
  'Left Fifth DIP',
  'Right First CMC',
  'Right First MCP',
  'Right IP',
  'Right Second CMC',
  'Right Second MCP',
  'Right Second PIP',
  'Right Second DIP',
  'Right Third CMC',
  'Right Third MCP',
  'Right Third PIP',
  'Right Third DIP',
  'Right Fourth CMC',
  'Right Fourth MCP',
  'Right Fourth PIP',
  'Right Fourth DIP',
  'Right Fifth CMC',
  'Right Fifth MCP',
  'Right Fifth PIP',
  'Right Fifth DIP',
]

##############################################

mvnx_filepaths = glob.glob(os.path.join(experiments_dir, '*.mvnx'))
for mvnx_filepath in mvnx_filepaths[0:]:
  print('Reading and parsing %s' % os.path.basename(mvnx_filepath))
  with open(mvnx_filepath, 'r') as fin:
    mvnx_contents = fin.read()
  mvnx_data = BeautifulSoup(mvnx_contents, 'xml')

  # Extract all of the frames, ignoring the first few that are empty rows.
  frames = mvnx_data.find_all('frame')
  frame_indexes = [frame.get('index') for frame in frames]
  frames = [frame for (i, frame) in enumerate(frames) if frame_indexes[i].isnumeric()]
  frame_indexes = [int(frame_index) for frame_index in frame_indexes if frame_index.isnumeric()]
  num_frames = len(frames)

  # Get time information.
  times_since_start_s = [float(frame.get('time'))/1000.0 for frame in frames]
  times_utc_str = [frame.get('tc') for frame in frames]
  times_s = [float(frame.get('ms'))/1000.0 for frame in frames]
  times_str = [get_time_str(time_s, '%Y-%m-%d %H:%M:%S.%f') for time_s in times_s]

  # Check that the timestamps monotonically increase.
  times_s_np = np.array(times_s)
  times_s_diffs = np.diff(times_s_np)
  if np.any(times_s_diffs < 0):
    print_var(times_str, 'times_str')
    msg = '\n'*2
    msg += 'x'*75
    msg += 'XsensStreamer aborting merge due to incorrect timestamps in the MVNX file (they are not monotonically increasing)'
    msg += 'x'*75
    msg = '\n'*2
    print(msg)
    continue

  # A helper to get a matrix of data for a given tag, such as 'position'.
  def get_tagged_data(tag):
    datas = [frame.find_all(tag)[0] for frame in frames]
    data = np.array([[float(x) for x in data.contents[0].split()] for data in datas])
    return data

  # Extract the data!
  segment_positions_body_cm                 = get_tagged_data('position')*100.0 # convert from m to cm
  segment_orientations_body_quaternion      = get_tagged_data('orientation')
  segment_velocities_body_cm_s              = get_tagged_data('velocity')*100.0 # convert from m to cm
  segment_accelerations_body_cm_ss          = get_tagged_data('acceleration')*100.0 # convert from m to cm
  segment_angular_velocities_body_deg_s     = get_tagged_data('angularVelocity')*180.0/np.pi # convert from radians to degrees
  segment_angular_accelerations_body_deg_ss = get_tagged_data('angularAcceleration')*180.0/np.pi # convert from radians to degrees
  foot_contacts                             = get_tagged_data('footContacts')
  sensor_freeAccelerations_cm_ss            = get_tagged_data('sensorFreeAcceleration')*100.0 # convert from m to cm
  sensor_magnetic_fields                    = get_tagged_data('sensorMagneticField')
  sensor_orientations_quaternion            = get_tagged_data('sensorOrientation')
  joint_angles_zxy_body_deg                 = get_tagged_data('jointAngle')
  joint_angles_xzy_body_deg                 = get_tagged_data('jointAngleXZY')
  ergonomic_joint_angles_zxy_deg            = get_tagged_data('jointAngleErgo')
  ergonomic_joint_angles_xzy_deg            = get_tagged_data('jointAngleErgoXZY')
  center_of_mass                            = get_tagged_data('centerOfMass')
  center_of_mass_positions_cm               = center_of_mass[:, 0:3]*100.0 # convert from m to cm
  center_of_mass_velocities_cm_s            = center_of_mass[:, 3:6]*100.0 # convert from m to cm
  center_of_mass_accelerations_cm_ss        = center_of_mass[:, 6:9]*100.0 # convert from m to cm
  try:
    segment_positions_fingersLeft_cm             = get_tagged_data('positionFingersLeft')*100.0
    segment_positions_fingersRight_cm            = get_tagged_data('positionFingersRight')*100.0
    segment_orientations_fingersLeft_quaternion  = get_tagged_data('orientationFingersLeft')
    segment_orientations_fingersRight_quaternion = get_tagged_data('orientationFingersRight')
    joint_angles_zxy_fingersLeft_deg             = get_tagged_data('jointAngleFingersLeft')
    joint_angles_zxy_fingersRight_deg            = get_tagged_data('jointAngleFingersRight')
    joint_angles_xzy_fingersLeft_deg             = get_tagged_data('jointAngleFingersLeftXZY')
    joint_angles_xzy_fingersRight_deg            = get_tagged_data('jointAngleFingersRightXZY')
    segment_positions_all_cm = np.concatenate((segment_positions_body_cm,
                                               segment_positions_fingersLeft_cm,
                                               segment_positions_fingersRight_cm),
                                              axis=1)
    segment_orientations_all_quaternion = np.concatenate((segment_orientations_body_quaternion,
                                                          segment_orientations_fingersLeft_quaternion,
                                                          segment_orientations_fingersRight_quaternion),
                                                         axis=1)
    joint_angles_zxy_all_deg = np.concatenate((joint_angles_zxy_body_deg,
                                               joint_angles_zxy_fingersLeft_deg,
                                               joint_angles_zxy_fingersRight_deg),
                                              axis=1)
    joint_angles_xzy_all_deg = np.concatenate((joint_angles_xzy_body_deg,
                                               joint_angles_xzy_fingersLeft_deg,
                                               joint_angles_xzy_fingersRight_deg),
                                              axis=1)
  except IndexError: # fingers were not included in the data
    segment_positions_all_cm = segment_positions_body_cm
    segment_orientations_all_quaternion = segment_orientations_body_quaternion
    joint_angles_zxy_all_deg = joint_angles_zxy_body_deg
    joint_angles_xzy_all_deg = joint_angles_xzy_body_deg

  # Get the number of segments and sensors
  num_segments_body = segment_orientations_body_quaternion.shape[1]/4
  assert num_segments_body == int(num_segments_body)
  num_segments_body = int(num_segments_body)
  num_segments_all = segment_orientations_all_quaternion.shape[1]/4
  assert num_segments_all == int(num_segments_all)
  num_segments_all = int(num_segments_all)
  num_sensors = sensor_orientations_quaternion.shape[1]/4
  assert num_sensors == int(num_sensors)
  num_sensors = int(num_sensors)

  # Create Euler orientations from quaternion orientations.
  segment_orientations_all_euler_deg = np.empty([num_frames, 3*num_segments_all])
  for segment_index in range(num_segments_all):
    for frame_index in range(num_frames):
      eulers_deg = euler_from_quaternion(*segment_orientations_all_quaternion[frame_index, (segment_index*4):(segment_index*4+4)])
      segment_orientations_all_euler_deg[frame_index, (segment_index*3):(segment_index*3+3)] = eulers_deg
  sensor_orientations_euler_deg = np.empty([num_frames, 3*num_sensors])
  for sensor_index in range(num_sensors):
    for frame_index in range(num_frames):
      eulers_deg = euler_from_quaternion(*sensor_orientations_quaternion[frame_index, (sensor_index*4):(sensor_index*4+4)])
      sensor_orientations_euler_deg[frame_index, (sensor_index*3):(sensor_index*3+3)] = eulers_deg

  ###################################################

  # Plot
  joint_labels_toPlot = ['Right Elbow', 'Left Elbow', 'Right Shoulder', 'Left Shoulder']
  num_joints_toPlot = len(joint_labels_toPlot)
  figure_size = (5,7)
  fig, axs = plt.subplots(nrows=num_joints_toPlot, ncols=2,
                               squeeze=False, # if False, always return 2D array of axes
                               sharex=True, sharey=True,
                               subplot_kw={'frame_on': True},
                               figsize=figure_size
                               )
  joint_angles_zxy_body_deg = joint_angles_zxy_body_deg.reshape(joint_angles_zxy_body_deg.shape[0], -1, 3)
  joint_angles_xzy_body_deg = joint_angles_xzy_body_deg.reshape(joint_angles_xzy_body_deg.shape[0], -1, 3)
  for (plt_index, joint_label) in enumerate(joint_labels_toPlot):
    data = joint_angles_zxy_body_deg[:, bodyJoint_labels.index(joint_label), :]
    data = data[:, [1, 2, 0]]
    data = np.unwrap(data, axis=0, discont=None, period=2*np.pi)
    axs[plt_index][0].plot(times_s_np, data)
    axs[plt_index][0].set_title('ZXY: %s' % joint_label)
    axs[plt_index][0].grid(True, color='lightgray')
    data = joint_angles_xzy_body_deg[:, bodyJoint_labels.index(joint_label), :]
    data = data[:, [0, 2, 1]]
    data = np.unwrap(data, axis=0, discont=None, period=2*np.pi)
    axs[plt_index][1].plot(times_s_np, data)
    axs[plt_index][1].set_title('XZY: %s' % joint_label)
    axs[plt_index][1].grid(True, color='lightgray')
    if plt_index == num_joints_toPlot-1:
      axs[plt_index][0].legend(['x', 'y', 'z'])
      axs[plt_index][1].legend(['x', 'y', 'z'])
  title = '%s' % os.path.splitext(os.path.basename(mvnx_filepath))[0]
  fig.suptitle(title.replace('_', ' '))
  fig.canvas.manager.set_window_title(title.replace('_', ' '))
  plt.savefig(os.path.join(output_dir, '%s.jpg' % title), dpi=300)

plt.show()







