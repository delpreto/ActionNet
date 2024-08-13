############
#
# Copyright (c) 2024 MIT CSAIL and Joseph DelPreto
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
# Created 2021-2024 for the MIT ActionSense project by Joseph DelPreto [https://josephdelpreto.com].
#
############

try:
  import pyperclip
except ModuleNotFoundError:
  pass

try:
  import numpy as np
except ModuleNotFoundError:
  pass

# A helper to insert new entries into the end of an array,
#  shifting back and deleting earlier entries.
# Do it in-place instead of using append() and then trimming,
#  to avoid making inefficient copies.
# Assumes arr and new_entries are 1D arrays.
def add_to_rolling_array(arr, new_entries):
  # Determine start/end indexes of the new entries such that it
  #  can fit in the array to update.
  new_entries_end_index = new_entries.shape[0] - 1
  new_entries_start_index = new_entries_end_index - arr.shape[0] + 1
  new_entries_start_index = max(0, new_entries_start_index)
  # Shift original entries back and add the new entries.
  num_new_elements = new_entries_end_index - new_entries_start_index + 1
  arr[0:-num_new_elements] = arr[num_new_elements:]
  arr[-num_new_elements:] = new_entries[new_entries_start_index:(new_entries_end_index+1)]
  return arr

# Get a matrix as a string that can be printed/copied.
def get_matrix_str(matrix, column_delim='\t', row_delim='\n'):
  return row_delim.join([column_delim.join([str(x) for x in matrix[row_index,:]])
                         for row_index in range(matrix.shape[0])])

# Copy a matrix as a string so it can be pasted into Excel or a similar program.
def copy_matrix(matrix, column_delim='\t', row_delim='\n'):
  pyperclip.copy(get_matrix_str(matrix, column_delim=column_delim, row_delim=row_delim))

# SciPy utils if SciPy is available
try:
  from scipy import signal
  scipy_is_available = True
except:
  scipy_is_available = False

# Perform a 2D convolution with a stride parameter.
# Code from https://stackoverflow.com/a/49064179
def convolve2d_strided(input_1, input_2, stride, mode='valid'):
  assert scipy_is_available
  return signal.convolve2d(input_1, input_2[::-1, ::-1], mode=mode)[::stride, ::stride]

# Calculate a 3D rotation matrix to align two vectors.
# Code from https://stackoverflow.com/a/59204638
def rotation_matrix_from_vectors(source_vector, target_vector):
  """ Find the rotation matrix that aligns source_vector to target_vector
  :param source_vector: A 3d "source" vector
  :param target_vector: A 3d "destination" vector
  :return mat: A transform matrix (3x3) which when applied to source_vector, aligns it with target_vector.
  """
  a, b = (source_vector / np.linalg.norm(source_vector)).reshape(3), (target_vector / np.linalg.norm(target_vector)).reshape(3)
  v = np.cross(a, b)
  c = np.dot(a, b)
  s = np.linalg.norm(v)
  kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
  rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
  return rotation_matrix








