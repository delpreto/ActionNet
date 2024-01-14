
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
import os
script_dir = os.path.dirname(os.path.realpath(__file__))

# Specify the folder of experiments to parse.
data_dir = os.path.realpath(os.path.join(script_dir, '..', '..', '..', 'results', 'learning_trajectories'))
# Specify the input file of labeled feature matrices.
training_data_filepath = os.path.join(data_dir, 'scooping_training_data.hdf5')

###################################################################
###################################################################
###################################################################

# Open the file of labeled feature matrices.
training_data_file = h5py.File(training_data_filepath, 'r')

# Get the feature matrices and their labels.
# Feature_matrices will be NxTx31, where
#   N is the number of examples
#   T is the number of timesteps in each trial
#   30 is the concatenation of:
#     [9] positions: (xyz hand), (xyz elbow), (xyz shoulder)
#    [12] orientation quaternions: (wijk hand), (wijk forearm), (wijk upper arm)
#     [9] joint angles: (xyz wrist), (xyz elbow), (xyz shoulder)
#     [1] time since trial start for each sample
feature_matrices = np.squeeze(np.array(training_data_file['feature_matrices']))
labels = training_data_file['labels']
labels = [label.decode('utf-8') for label in labels]

num_examples = len(labels)
num_timesteps_per_example = feature_matrices[0].shape[0]
num_features = feature_matrices[0].shape[1]
assert(feature_matrices.shape[0] == num_examples)

print()
print('Loaded training data')
print('  Shape of feature_matrices:', feature_matrices.shape)
print('  Number of examples   :', num_examples)
print('  Timesteps per example:', num_timesteps_per_example)
print('  Feature matrix shape :', feature_matrices[0].shape)
print('  Label breakdown:')
for label in set(labels):
  print('    %02d: %s' % (len([x for x in labels if x == label]), label))
print()

# Example of parsing a feature row:
time_index = 0
for row_index in range(2):
  row_time_features = feature_matrices[row_index, time_index, :]
  hand_position_xyz_cm = row_time_features[0:3]
  elbow_position_xyz_cm = row_time_features[3:6]
  shoulder_position_xyz_cm = row_time_features[6:9]
  hand_quaternion_wxyz = row_time_features[9:13]
  lowerarm_quaternion_wxyz = row_time_features[13:17]
  upperarm_quaternion_wxyz = row_time_features[17:21]
  wrist_angles_xzy_rad = row_time_features[21:24]
  elbow_angles_xzy_rad = row_time_features[24:27]
  shoulder_angles_xzy_rad = row_time_features[27:30]

  print('Sample parsed data using example %d timestep %d:' % (row_index, time_index))
  print('  Label                      : %s' % (labels[row_index]))
  print('  Hand position [xyz]        : (%g, %g, %g) cm'  % tuple(hand_position_xyz_cm))
  print('  Elbow position [xyz]       : (%g, %g, %g) cm'  % tuple(elbow_position_xyz_cm))
  print('  Shoulder position [xyz]    : (%g, %g, %g) cm'  % tuple(shoulder_position_xyz_cm))
  print('  Hand quaternion [wxyz]     : (%g, %g, %g %g)'  % tuple(hand_quaternion_wxyz))
  print('  Lower arm quaternion [wxyz]: (%g, %g, %g %g)'  % tuple(lowerarm_quaternion_wxyz))
  print('  Upper arm quaternion [wxyz]: (%g, %g, %g %g)'  % tuple(upperarm_quaternion_wxyz))
  print('  Wrist angles [xzy]         : (%g, %g, %g) rad (Ulnar Deviation/Radial Deviation, Pronation/Supination, Flexion/Extension)' % tuple(wrist_angles_xzy_rad))
  print('  Elbow angles [xzy]         : (%g, %g, %g) rad (Ulnar Deviation/Radial Deviation, Pronation/Supination, Flexion/Extension)' % tuple(elbow_angles_xzy_rad))
  print('  Shoulder angles [xzy]      : (%g, %g, %g) rad (Abduction/Adduction, Internal/External Rotation, Flexion/Extension)' % tuple(shoulder_angles_xzy_rad))


