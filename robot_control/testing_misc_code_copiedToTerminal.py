

from __future__ import print_function
from BaxterController import BaxterController
c = BaxterController(limb_name='right', print_debug=True)
c.move_to_resting()

xyz = [0.656982770038, -0.852598021641, 0.0388609422173]
wijk_1 = [0.261868353356,  0.367048116303, 0.885911751787, -0.108908281936]
wijk_2 = [-0.34846608,  0.66464585,  0.26789804,  0.60419197]

print()
print()
angles_1 = c.get_joint_angles_rad_for_gripper_pose(xyz, [0.26,  0.37, 0.89, -0.11], None)

print()
print()
angles_2 = c.get_joint_angles_rad_for_gripper_pose(xyz, [-0.35,  -0.66, -0.27,  -0.60], None)

c.move_to_gripper_pose(xyz, [0.26,  0.37, 0.89, -0.11], wait_for_completion=True)

xyz = [1, -0.26, -0.03]
def f(w,i,j,k):
  c.move_to_gripper_pose(xyz, [w,i,j,k], wait_for_completion=True)

import itertools
import time
a = [0, 1.5, 0.8, -0.25, 0.01, -1.3, 0.02]
all_a = itertools.permutations(a)
good_a = []
for a_i in all_a:
  res = c.get_joint_angles_rad_for_gripper_pose([ 0.99281162, -0.26284867, -0.025269], [-0.34846608, -0.66464585, -0.26789804, -0.60419197], seed_joint_angles_rad=a_i)
  if res is not None:
    good_a.append(a_i)
  time.sleep(0.1)
 

