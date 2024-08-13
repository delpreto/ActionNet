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

from learning_trajectories.helpers.configuration import *
import numpy as np
from scipy.spatial.transform import Rotation

# ================================================================
# Rotate a box in 3D.
# If center_preRotation_cm is provided, will translate box to that location before applying the given quaternion.
def rotate_3d_box(quaternion_localToGlobal_wijk, center_preRotation_cm, box_dimensions_cm):
  # Define vertices of a unit box in the global frame
  corners = np.array([
    [-1, -1, -1],
    [-1, -1, 1],
    [-1, 1, -1],
    [-1, 1, 1],
    [1, -1, -1],
    [1, -1, 1],
    [1, 1, -1],
    [1, 1, 1]
  ]) * 0.5
  # Define faces of the box in the global frame, using corner indexes
  faces = np.array([
    [0, 1, 3, 2], # bottom face
    [0, 2, 6, 4],
    [0, 1, 5, 4],
    [4, 5, 7, 6], # top face
    [1, 3, 7, 5],
    [2, 3, 7, 6], # hand-side face
  ])
  # Scale the box
  corners = corners * box_dimensions_cm
  
  # Translate the box
  corners = corners + center_preRotation_cm
  
  # Invert quaternion.
  quaternion_globalToLocal_ijkw = [
    -quaternion_localToGlobal_wijk[1],
    -quaternion_localToGlobal_wijk[2],
    -quaternion_localToGlobal_wijk[3],
    quaternion_localToGlobal_wijk[0],
    ]
  # Rotate the box using the quaternion,
  rot = Rotation.from_quat(quaternion_globalToLocal_ijkw).as_matrix()
  corners = np.dot(corners, rot)
  
  return (corners, faces)




















