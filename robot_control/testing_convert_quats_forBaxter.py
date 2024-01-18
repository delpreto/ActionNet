
import numpy as np
from scipy.spatial.transform import Rotation

# This script was run, and the printed result was manually copied to
#  testing_converted_baxter_quaternions_positions open on the Linux computer
#  (since that computer doesn't have the Rotation module)

# Process that worked:
#   Move Baxter to a unit quaternion
#     Move Baxter to a unit quaternion rotated about each axis, to see the coordinate frame
#   Show the skeleton with a unit quaternion
#   Imagine a series of rotations from Baxter's unit quaternion to the skeleton's unit quaternion
#   Baxter hand box (imagined as protruding from gripper) matches skeleton hand box
#   Rotate the skeleton quaternion 45 about an axis, and visualize it
#   Baxter poured the wrong way
#   Add a 180 about z first to Baxter's rotation sequence
#   Baxter still poured the wrong way
#   Manually negate i and j components of the skeleton quaternion
#     Not quite sure of the theory behind this since it seems different than doing a 180 about z
#   Baxter poured the correct way
#   Rotate the skeleton rotation about another axis
#   Rotate the skeleton about a combination of axes
#   This worked with the test case, but not with the human example data :(
#     It seemed the test case was rotated 180 about some axes so it looks the same but was different
#   Add a rotation to make the human data example match the previous example
#     The result can probably be simplified

# quat_wijk = [1,0,0,0]

# quat_wijk = [0.5, -0.5, 0.5, 0.5] # pitcher start (idealized)
# quat_wijk = [0.6532814824381883, -0.27059805007309834, 0.2705980500730985, 0.6532814824381883] # pour 45 degrees
# quat_wijk = [0.29883623873011994, -0.6408563820557884, 0.2988362387301199, 0.6408563820557885] # rotate inwards 45 degrees
# quat_wijk = [0.6532814824381884, -0.27059805007309845, 0.6532814824381883, 0.2705980500730985] # rotate outwards 45 degrees
# quat_wijk = [0.3535533905932738, -0.35355339059327373, 0.1464466094067262, 0.8535533905932738] # pour 45 and inwards 45
# quat_wijk = [0.6532814824381884, -0.6532814824381882, 0.2705980500730985, 0.2705980500730986] # away 45
quat_wijk = [-0.34846608187950395, 0.6646458463561292, 0.267898035291973, 0.6041919656763008] # human start
# quat_wijk = [-0.775413100771162, 0.22540388321922272, 0.27472229555907957, 0.521972444913735] # human pour

# For hand holding glass:
# quat_wijk = [-0.5, -0.5, -0.5, -0.5] # glass start (idealized)
# rotates_by_deg = [
#     [0, 0, 90],
#     [0, -30, 0],
#     ]
# Then use the result *before* i and j components are negated

rotates_by_deg = [
    [0, 0, -180],
    ]
print()
print(quat_wijk)
print(rotates_by_deg)
# Apply the rotations.
rotation_quat = Rotation.from_quat([quat_wijk[1], quat_wijk[2], quat_wijk[3], quat_wijk[0]])
for i in range(len(rotates_by_deg)-1, -1, -1):
  rotate_by_deg = rotates_by_deg[i]
  rotation_toApply = Rotation.from_rotvec(np.radians(rotate_by_deg))
  rotation_quat = rotation_quat * rotation_toApply
ijkw = rotation_quat.as_quat()
quat_wijk = [ijkw[3], ijkw[0], ijkw[1], ijkw[2]]
print(quat_wijk)
print()

rotates_by_deg = [
    # [90, 0, 0],
    # [0, 90, 0],
    # [0, 0, 90],
    # [0, 90, 0],
    # [0, 0, -90],
    
    # ***
    [0, 0, 180],
    [0, -90, 0],
    [0, 0, 90],
    [0, 180, 0],
  
    # [0, 90, 0],
    # [0, 0, 90],
    # [0, 0, 180],
    ]
print()
print(quat_wijk)
print(rotates_by_deg)
# print('Rotating %s by %s' % (quat_wijk, rotate_by_deg))

# Negate the i and j components.
quat_wijk = [quat_wijk[0], -quat_wijk[1], -quat_wijk[2], quat_wijk[3]]

# Apply the rotations.
rotation_quat = Rotation.from_quat([quat_wijk[1], quat_wijk[2], quat_wijk[3], quat_wijk[0]])
for i in range(len(rotates_by_deg)-1, -1, -1):
  rotate_by_deg = rotates_by_deg[i]
  rotation_toApply = Rotation.from_rotvec(np.radians(rotate_by_deg))
  rotation_quat = rotation_quat * rotation_toApply
  
# rotation_toApply = Rotation.from_rotvec(np.radians(rotate_by_deg))
# rotation_quat = Rotation.from_quat([quat_wijk[1], quat_wijk[2], quat_wijk[3], quat_wijk[0]])
# rotation_quat = rotation_quat * rotation_toApply
ijkw = rotation_quat.as_quat()
quat_wijk = [ijkw[3], ijkw[0], ijkw[1], ijkw[2]]

# # Negate the i and j components.
# quat_wijk = [quat_wijk[0], -quat_wijk[1], -quat_wijk[2], quat_wijk[3]]

print(list(quat_wijk))
print()

