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

# Adjust activity start/end times.
# Will add the given offset to the original time.
start_offsets_s = {
  'pouring': {
    'S00': -0.5,
    'S11': 0,
    'S10': 0,
  },
  'scooping': {
    'S00': 0,
    'S11': 0,
    'S10': 0,
  }
}
end_offsets_s = {
  'pouring': {
    'S00': 0,
    'S11': 0,
    'S10': 0,
  },
  'scooping': {
    'S00': 0,
    'S11': 0,
    'S10': 0,
  }
}

stationary_position_use_variance = {
  'pouring': True,
  'scooping': False,
}
stationary_position_buffer_duration_s = 2
stationary_position_min_ratio = 0.10
stationary_position_max_ratio = 0.90
stationary_position_hardcoded_time_fraction = {
  'pouring': None,
  'scooping': 0.3
}

motionObject_name = {
  'pouring': 'Pitcher',
  'scooping': 'Spoon',
}
referenceObject_name = {
  'pouring': 'Glass',
  'scooping': 'Plate',
}
motionObjectKeypoint_name = {
  'pouring': 'Spout',
  'scooping': 'Ladle',
}
hand_box_dimensions_cm = np.array([4.8, 3, 1.3]) # np.array([2, 9, 18]) # open hand
motionObject_shape_dimensions_cm = {
  'pouring': np.array([23, 23, 10.8]), # [height, top length, width]
  'scooping': np.array([7, 32.5, 2]), # [width, top length, height]
}
hand_to_motionObject_angles_rad = {
  'pouring': np.array([np.radians(90), # tilt left/right (positive/negative)
                       np.radians(0),  # tilt down/up (positive/negative)
                       np.radians(0)   # tilt inward/outward (positive/negative)
                       ]),
  'scooping': np.array([np.radians(15), # around knuckle axis (positive down)
                        np.radians(0), # around forearm axis (positive CCW)
                        np.radians(30) # around vertical axis (positive CCW looking down)
                       ]),
}
hand_to_motionObject_rotation = dict([(k, Rotation.from_rotvec(v)) for (k, v) in hand_to_motionObject_angles_rad.items()])
corner_indexes_forTilt = { # keypointside and handside on the left, to connect for tilting estimate
  'pouring': [4, 6],
  'scooping': [5, 7],
}
corner_indexes_forKeypoint = { # front top two points, to average for the keypoint position
  'pouring': [4, 5],
  'scooping': [1, 5],
}
hand_to_motionObject_top_cm = {
  'pouring': 8,
}
hand_to_motionObject_offset_cm = {
  'pouring': np.array([
              hand_to_motionObject_top_cm['pouring'] - motionObject_shape_dimensions_cm['pouring'][0]/2,
              -(0+motionObject_shape_dimensions_cm['pouring'][1]/2),
              2
             ]),
  'scooping': np.array([
              0,
              -(0+motionObject_shape_dimensions_cm['pouring'][1]/2),
              -2-4, # estimate sensor to hand top then hand top to spoon
             ]),
}
# hand_to_pitcher_offset_cm = np.array([-3, -13, 0]) # used -15 for Baxter videos so the orange hides behind pitcher less

table_height_cm = 88.7 + 0.5 # Table height plus foam board thickness
target_starting_height_cm = {
  'pouring': {'RightHand': np.mean([19.4132, 11.3471])}, # average of medians for S00-S11, which seems to match measurements on the pitcher (note S10 was average 10.60645)
  'scooping': None,
}

infer_motionObjectKeypoint_position_m_fn = {
  'pouring': 'infer_motionObjectKeypoint_position_m',
  'scooping': 'infer_motionObjectKeypoint_position_m',
}
motionObject_bodySegment_name = {
  'pouring': 'RightHand',
  'scooping': 'RightHand',
}
referenceObject_bodySegment_name = {
  'pouring': 'LeftHand',
  'scooping': 'LeftHand',
}
referenceObject_use_motionObjectKeypoint_position_xy = {
  'pouring': True,
  'scooping': False,
}
referenceObject_diameter_cm = {
  'pouring': 7.3, # glass top 7.3 bottom 6.3
  'scooping': 20 ,
}
referenceObject_height_cm = {
  'pouring': 15.8,
  'scooping': 1, # roughly to interior surface of plate
}
hand_to_referenceObject_bottom_cm = {
  'pouring': 6,
}
hand_to_referenceObject_top_cm = {
  'pouring': referenceObject_height_cm['pouring'] - hand_to_referenceObject_bottom_cm['pouring'],
}
referenceObject_offset_cm = {
  'pouring': np.array([hand_to_referenceObject_top_cm['pouring'], 8, -4]), # [up along thumb, out along forearm axis, out from back of hand]
  'scooping': np.array([15, 10
                         
                         
                         , -2]), # [left/right, in/out, up/down not used?]
}

hand_box_color = 0.8*np.array([1, 0.6, 0])
motionObject_box_color = 0.8*np.array([1, 1, 1])

animation_view_angle_backLeft = (16, -28)
animation_view_angle_backRight = (16, 44)
animation_view_angle_forBaxter = (30, -179.9)
animation_view_angle_forAllTrajectories = (10, -160)
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



















