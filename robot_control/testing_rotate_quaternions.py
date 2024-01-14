
import numpy as np
from scipy.spatial.transform import Rotation

# wijk = [1, 0.1, 0, 0.2]
quat_wijk = [-0.35, -0.66, -0.27, -0.60]

rotates_by_deg = [
  [48-180, 10, 0],
  ]

print()
print(quat_wijk)

# Apply the rotations.
rotation_quat = Rotation.from_quat([quat_wijk[1], quat_wijk[2], quat_wijk[3], quat_wijk[0]])
for i in range(len(rotates_by_deg)-1, -1, -1):
  rotate_by_deg = rotates_by_deg[i]
  rotation_toApply = Rotation.from_rotvec(np.radians(rotate_by_deg))
  rotation_quat = rotation_quat * rotation_toApply

quat_ijkw = rotation_quat.as_quat()
quat_wijk = [quat_ijkw[3], quat_ijkw[0], quat_ijkw[1], quat_ijkw[2]]

print(quat_wijk)
print([round(a,2) for a in quat_wijk])
print()


