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

import numpy as np
from scipy.spatial.transform import Rotation

resampled_fs_hz = 50

stationary_position_buffer_duration_s = 2
stationary_position_min_ratio = 0.10
stationary_position_max_ratio = 0.90
stationary_position_hardcoded_time_fraction = 0.3 # if not doing pouring, will use the hand position/pose at this ratio of time into the trial

hand_box_dimensions_cm = np.array([4.8, 3, 1.3]) # np.array([2, 9, 18]) # open hand
pitcher_box_dimensions_cm = np.array([23, 23, 10.8]) # [height, top length, width]
hand_to_pitcher_rotation = Rotation.from_rotvec(np.array([np.radians(90),
                                                          np.radians(0),
                                                          np.radians(-5)]))
hand_to_pitcherTop_cm = 8
hand_to_pitcher_offset_cm = np.array([hand_to_pitcherTop_cm - pitcher_box_dimensions_cm[0]/2,
                                      -(0+pitcher_box_dimensions_cm[1]/2),
                                      2])
# hand_to_pitcher_offset_cm = np.array([-3, -13, 0]) # used -15 for Baxter videos so the orange hides behind pitcher less

referenceObject_bodySegment_name = 'LeftHand'
referenceObject_diameter_cm = 7.3 # glass top 7.3 bottom 6.3
referenceObject_height_cm = 15.8
hand_to_referenceObject_bottom_cm = 6
hand_to_referenceObject_top_cm = referenceObject_height_cm - hand_to_referenceObject_bottom_cm
referenceObject_offset_cm = np.array([hand_to_referenceObject_top_cm, 8, -4]) # [up along thumb, out along forearm axis, out from back of hand]

hand_box_color = 0.8*np.array([1, 0.6, 0])
pitcher_box_color = 0.8*np.array([1, 1, 1])

animation_view_angle_backLeft = (16, -28)
animation_view_angle_backRight = (16, 44)
animation_view_angle_forBaxter = (30, -179.9)
# animation_view_angle = animation_view_angle_backLeft
# animation_view_angle = animation_view_angle_backRight
animation_view_angle = animation_view_angle_forBaxter


bodySegment_chains_labels_toPlot = {
  # 'Left Leg':  ['Left Upper Leg', 'Left Lower Leg', 'Left Foot', 'Left Toe'],
  # 'Right Leg': ['RightUpperLeg', 'Right Lower Leg', 'Right Foot', 'Right Toe'],
  'Spine':     ['Head', 'Neck', 'T8', 'T12', 'L3', 'L5', 'Pelvis'], # top down
  # 'Hip':       ['Left Upper Leg', 'Pelvis', 'RightUpperLeg'],
  'Shoulders': ['LeftUpperArm', 'LeftShoulder', 'RightShoulder', 'RightUpperArm'],
  'Left Arm':  ['LeftUpperArm', 'LeftForeArm', 'LeftHand'],
  'Right Arm': ['RightUpperArm', 'RightForeArm', 'RightHand'],
}

# Used to artificially shift distributions for demonstration purposes.
example_types_to_offset = []#['model']
if len(example_types_to_offset) > 0:
  print()
  print('*'*50)
  print('*'*50)
  print('NOTE THAT DISTRIBUTIONS ARE BEING ARTIFICALLY')
  print('SHIFTED FOR DEMONSTRATION PURPOSES FOR')
  print('THE FOLLOWING EXAMPLE TYPES')
  print(example_types_to_offset)
  print('*'*50)
  print('*'*50)
  input('Press Enter to confirm and continue')
  print()
  print()



















