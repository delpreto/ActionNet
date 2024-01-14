
from __future__ import print_function

from BaxterController import BaxterController

import time

################################################

quaternions_localToGlobal_wijk = [
  # Original from feature matrix.
  [-0.34846608187950395, 0.6646458463561292, 0.267898035291973, 0.6041919656763008],
  [-0.7753348516423536, 0.22563123751422992, 0.27495695772012557, 0.521866841026997],
  # Rotation vector [0, 0, 180]
  [0.6041919656763008, -0.26789803529197304, 0.6646458463561292, -0.348466081879504],
  [0.5218668629778457, -0.27495696928541025, 0.22563124700476803, -0.7753348842546154],
  ]

xyz = [
  [1.2, -0.25, 0.25],
  [1.2, -0.25, 0.25],
  ]

################################################

limb_name = 'right'
controller = BaxterController(limb_name=limb_name, print_debug=True)

for i in range(len(quaternions_localToGlobal_wijk)):
  print('-'*50)
  pos = xyz[i]
  quat_wijk = quaternions_localToGlobal_wijk[i]
  quat_wijk = [quat_wijk[0], -quat_wijk[1], -quat_wijk[2], -quat_wijk[3]]
  print('Moving to xyz %s quaternion %s' % (str(pos), str(quat_wijk)))
  controller.move_to_gripper_pose(gripper_position_m=pos,
                                  gripper_orientation_quaternion_wijk=quat_wijk,
                                  seed_joint_angles_rad=controller.get_resting_joint_angles_rad(should_print=False),
                                  wait_for_completion=True)
  time.sleep(3)
  print()

controller.quit()
print()

