
import numpy as np
from scipy.spatial.transform import Rotation

# wijk = [1, 0.1, 0, 0.2]
wijk = [-0.35, -0.66, -0.27, -0.60]

rotate_by_deg = [48-180, 10, 0]

print()
print(wijk)
rotation_toApply = Rotation.from_rotvec(np.radians(rotate_by_deg))
rotation_quat = Rotation.from_quat([wijk[1], wijk[2], wijk[3], wijk[0]])
rotation = rotation_toApply*rotation_quat
ijkw = rotation.as_quat()
wijk = [ijkw[3], ijkw[0], ijkw[1], ijkw[2]]
print(wijk)
print([round(a,2) for a in wijk])
print()