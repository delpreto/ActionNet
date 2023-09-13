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
data_dir = os.path.realpath(os.path.join(script_dir, '..', '..', '..', '..', 'data'))
# experiments_dir = os.path.join(data_dir, 'tests', '2023-08-31_experiment_S00_xsens_joint_angles')
# experiments_dir = os.path.join(data_dir, 'experiments', '2023-09-10_xsens_controlled_rotations_S00')
experiments_dir = os.path.join(data_dir, 'experiments', '2023-09-10_xsens_controlled_rotations_S11')
# output_dir = os.path.realpath(os.path.join(script_dir, '..', '..', '..', '..', 'results', 'learning_trajectories',
#                                            '2023-08-31_test_xsens_joint_angles'))
output_dir = os.path.join(experiments_dir, 'plots')
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
  mvnx_xml = BeautifulSoup(mvnx_contents, 'xml')

  # Extract all frames
  frames = mvnx_xml.find_all('frame')

  # Extract the calibration frames.
  calibration_types = ['identity', 'Tpose', 'Tpose_ISB'] # should match MVNX types if make lowercase and replace underscores with hyphones
  calibration_data = dict([(key, None) for key in calibration_types])
  for (frame_index, frame) in enumerate(frames[0:5]): # should be the first 3 frames, but increase a bit just to be safe
    for calibration_type in calibration_types:
      if frame.get('type') == calibration_type.lower().replace('_', '-'):
        calibration_data[calibration_type] = {
          'segment_orientations_body_quaternion': [round(float(q), 6) for q in frame.find('orientation').contents[0].split()],
          'segment_positions_body_m': [round(float(q), 6) for q in frame.find('position').contents[0].split()],
        }

  # Extract data from all non-calibration frames.

  frames = [frame for (i, frame) in enumerate(frames) if frame.get('type') == 'normal']
  frame_indexes = [int(frame.get('index')) for frame in frames]
  num_frames = len(frames)

  # Get time information.
  times_since_start_s = [round(float(frame.get('time')))/1000.0 for frame in frames]
  times_utc_str = [frame.get('tc') for frame in frames]
  times_s = [float(frame.get('ms'))/1000.0 for frame in frames]
  times_str = [get_time_str(time_s, '%Y-%m-%d %H:%M:%S.%f') for time_s in times_s]

  # Check that the timestamps monotonically increase.
  if np.any(np.diff(times_s) < 0):
    print_var(times_str, 'times_str')
    msg = '\n'*2
    msg += 'x'*75
    msg += 'XsensStreamer aborting merge due to incorrect timestamps in the MVNX file (they are not monotonically increasing)'
    msg += 'x'*75
    msg = '\n'*2
    print(msg)
    continue

  # A helper to get a matrix of data from all frames for a given tag, such as 'position'.
  def get_tagged_data(tag):
    datas = [frame.find(tag) for frame in frames]
    data = np.array([[round(float(x), 6) for x in data.contents[0].split()] for data in datas])
    return data

  # Extract the data!
  segment_orientations_body_quaternion_wijk = get_tagged_data('orientation')
  segment_positions_body_m                  = get_tagged_data('position')
  segment_velocities_body_m_s               = get_tagged_data('velocity')
  segment_accelerations_body_m_ss           = get_tagged_data('acceleration')
  segment_angular_velocities_body_rad_s     = get_tagged_data('angularVelocity')
  segment_angular_accelerations_body_rad_ss = get_tagged_data('angularAcceleration')
  foot_contacts                             = get_tagged_data('footContacts')
  sensor_freeAccelerations_m_ss             = get_tagged_data('sensorFreeAcceleration')
  sensor_magnetic_fields_au                 = get_tagged_data('sensorMagneticField')
  sensor_orientations_quaternion_wijk       = get_tagged_data('sensorOrientation')
  joint_angles_body_eulerZXY_xyz_rad        = get_tagged_data('jointAngle')*np.pi/180.0 # convert from degrees to radians
  joint_angles_body_eulerXZY_xyz_rad        = get_tagged_data('jointAngleXZY')*np.pi/180.0 # convert from degrees to radians
  joint_angles_ergonomic_eulerZXY_xyz_rad   = get_tagged_data('jointAngleErgo')*np.pi/180.0 # convert from degrees to radians
  joint_angles_ergonomic_eulerXZY_xyz_rad   = get_tagged_data('jointAngleErgoXZY')*np.pi/180.0 # convert from degrees to radians
  center_of_mass                            = get_tagged_data('centerOfMass')
  center_of_mass_positions_m                = center_of_mass[:, 0:3]
  center_of_mass_velocities_m_s             = center_of_mass[:, 3:6]
  center_of_mass_accelerations_m_ss         = center_of_mass[:, 6:9]
  

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
  joint_angles_body_eulerZXY_xyz_rad = joint_angles_body_eulerZXY_xyz_rad.reshape(joint_angles_body_eulerZXY_xyz_rad.shape[0], -1, 3)
  joint_angles_body_eulerXZY_xyz_rad = joint_angles_body_eulerXZY_xyz_rad.reshape(joint_angles_body_eulerXZY_xyz_rad.shape[0], -1, 3)
  for (plt_index, joint_label) in enumerate(joint_labels_toPlot):
    data = joint_angles_body_eulerZXY_xyz_rad[:, bodyJoint_labels.index(joint_label), :]
    data = data[:, [0, 1, 2]]
    data = np.unwrap(data, axis=0, discont=None, period=2*np.pi)
    axs[plt_index][0].plot(times_s, data)
    axs[plt_index][0].set_title('ZXY: %s' % joint_label)
    axs[plt_index][0].grid(True, color='lightgray')
    data = joint_angles_body_eulerXZY_xyz_rad[:, bodyJoint_labels.index(joint_label), :]
    data = data[:, [0, 1, 2]]
    data = np.unwrap(data, axis=0, discont=None, period=2*np.pi)
    axs[plt_index][1].plot(times_s, data)
    axs[plt_index][1].set_title('XZY: %s' % joint_label)
    axs[plt_index][1].grid(True, color='lightgray')
    if plt_index == num_joints_toPlot-1:
      axs[plt_index][0].legend(['x (abduction)', 'y (internal)', 'z (flexion)'])
      axs[plt_index][1].legend(['x (abduction)', 'y (internal)', 'z (flexion)'])
  title = '%s' % os.path.splitext(os.path.basename(mvnx_filepath))[0]
  fig.suptitle(title.replace('_', ' '))
  fig.canvas.manager.set_window_title(title.replace('_', ' '))
  plt.savefig(os.path.join(output_dir, '%s.jpg' % title), dpi=300)

plt.show()







