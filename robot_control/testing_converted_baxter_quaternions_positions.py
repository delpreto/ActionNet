
from __future__ import print_function

from BaxterController import BaxterController

import time

################################################

quaternions_wijk = [
  # Original from feature matrix.
  # [-0.34846608187950395, 0.6646458463561292, 0.267898035291973, 0.6041919656763008],
  # [-0.7753348516423536, 0.22563123751422992, 0.27495695772012557, 0.521866841026997],
  # Original converted to global-to-local.
  # [-0.34846608187950395, -0.6646458463561292, -0.267898035291973, -0.6041919656763008],
  # [-0.7753348516423536, -0.22563123751422992, -0.27495695772012557, -0.521866841026997],

  # Rotation vector [0, 0, 180]
  # [-0.6041919656763008, -0.26789803529197304, 0.6646458463561292, 0.348466081879504],
  # [-0.5218668629778457, -0.27495696928541025, 0.22563124700476803, 0.7753348842546154],

  # Rotation vector [0, 180, 0]
  # [-0.267, 0.604, 0.348, -0.664],
  # [-0.274, 0.521, 0.775, -0.225],

  # Rotation vector [180, 0, 0]
  # [-0.664, 0.348, -0.604, 0.267],
  # [-0.225, 0.775, -0.521, 0.274],


  # Identity.
  # [1, 0, 0, 0],
  # Identity rotate by [45, 0, 0]
  # [0.923, 0.382, 0, 0],
  # Identity rotate by [0, 45, 0]
  # [0.923, 0, 0.382, 0]
  # Identity rotate by [0, 0, 45]
  # [0.923, 0, 0, 0.382]

  # ID > 180 about z
  # [0, 0, 0, 1],

  # ID > 180 about z > -90 about y > 90 about z
  # [-0.5, -0.5, -0.5, 0.5],

  # ID > -90 about x > -90 about z
  # [0.5, -0.5, 0.5, -0.5],

  # ID > 90 about y > 90 about z
  # [0.5, -0.5, 0.5, 0.5],

  # ID > 90 y > 90 z > 180 z
  # [-0.5, -0.5, -0.5, 0.5],

  # Pitcher start > 90 about y > 90 about z
  # [-0.5, -0.5, 0.5, 0.5],

  # Pitcher start > 180 about z > -90 about y > 90 about z
  # [-0.5, 0.5, -0.5, 0.5],

  # Pour 45 deg > 180 about z > -90 about y > 90 about z
  # [-0.653, 0.270, -0.653, 0.270],

  # Pitcher start > 180 about z > -90 about y > 90 about z > 180 about x
  # [0.5, -0.5, -0.5, 0.5],

  # Pitcher start > 180 about z > -90 about y > 90 about z > 180 about y
  # [0.5, 0.5, 0.5, 0.5],

  # Pour 45 > 180 about z > -90 about y > 90 about z > 180 about y
  # [0.270, 0.653, 0.27, 0.653],

  # *** Pitcher start > negate i j > 180 about z > -90 about y > 90 about z > 180 about y
  # [-0.5, 0.5, -0.5, 0.5],

  # *** Pour 45 > negate i j > 180 about z > -90 about y > 90 about z > 180 about y
  # [-0.270, 0.653, -0.270, 0.653],

  # *** Hand in 45 > negate i j > 180 about z > -90 about y > 90 about z > 180 about y
  # [-0.640, 0.640, -0.298, 0.298],

  # *** Hand out 45 > negate i j > 180 about z > -90 about y > 90 about z > 180 about y
  # [-0.270, 0.270, -0.653, 0.653],

  # *** Pour 45 and Hand in 45 > negate i j > 180 about z > -90 about y > 90 about z > 180 about y
  # [-0.500, 0.707, 0.000, 0.500],

  # *** Away 45 > negate i j > 180 about z > -90 about y > 90 about z > 180 about y
  # [-0.270, 0.653, -0.653, 0.270],

  # Human start > negate i j > 180 about z > -90 about y > 90 about z > 180 about y > negate i j
  # [-0.2779, -0.3384, 0.6747, 0.5941],
  # [-0.9426, 0.0100, 0.0705, 0.3262],
  # [-0.2779, 0.3384, -0.6747, 0.5941],
  # Human pour > negate i j > 180 about z > -90 about y > 90 about z > 180 about y > negate i j
  # [-0.6733, -0.3767, 0.6240, 0.1233],
  # [-0.8987, 0.3986, 0.1020, -0.1513],
  # [-0.6733, 0.3767, -0.6240, 0.1233],

  # *** Human start > -180 z >> negate i j > 180 about z > -90 about y > 90 about z > 180 about y > negate i j
  [-0.338, 0.277, -0.594, 0.6674],

  # *** Human pour > -180 z >> negate i j > 180 about z > -90 about y > 90 about z > 180 about y > negate i j
  [-0.376, 0.673, -0.123, 0.624],
  
  # Testing for the left hand.
  # [-0.6123724356957947, -0.6123724356957945, -0.35355339059327384, -0.3535533905932738],
  # [-0.683, -0.183, 0.183, -0.683],
  # [-0.683, -0.683, 0.183, 0.183],
  # [-0.5, -0.5, -0.5, -0.5],
  # [0.612, -0.612, 0.353, -0.353],
  # [-0.612, -0.612, -0.353, -0.353],
  # [-0.183, -0.683, 0.183, -0.683], *** gripper horizontal, forearm rotated out 30 degrees
  # [0, -0.866, 0, -0.5],
  ]

xyz = [
  # [1, -0.25, 0],
  # [0.9, -0.4, 0],
  # [1, -0.25, 0],

  # Identity.
  # [0.4, -0.4, 1],
  # [0.4, -0.4, 1],

  # ID > -90 about x > -90 about z
  # [0.9, -0.4, 0],

  # ID > 90 about y > 90 about z
  # [0.4, 0, 0],

  # Pitcher start > 90 about y > 90 about z
  # Pitcher start > 180 about z > -90 about y > 90 about z
  # Pour 45 deg > 180 about z > -90 about y > 90 about z
  # Pitcher start > 180 about z > -90 about y > 90 about z > 180 about y
  # [0.9, -0.4, 0],
  # [0.9, -0.4, 0],

  # Human start, converted
  [0.8928, -0.4128, -0.0052],
  # Human pour, converted
  [1.0246, -0.1332, 0.1419],
  ]

################################################

limb_name = 'right'
controller = BaxterController(limb_name=limb_name, print_debug=True)
controller.move_to_resting()
time.sleep(1)

for i in range(len(quaternions_wijk)):
  print('-'*50)
  pos = xyz[i]
  quat_wijk = quaternions_wijk[i]
  # quat_wijk = [quat_wijk[0], -quat_wijk[1], -quat_wijk[2], -quat_wijk[3]]
  print('Moving to xyz %s quaternion %s' % (str(pos), str(quat_wijk)))
  controller.move_to_gripper_pose(gripper_position_m=pos,
                                  gripper_orientation_quaternion_wijk=quat_wijk,
                                  seed_joint_angles_rad=controller.get_resting_joint_angles_rad(should_print=False),
                                  wait_for_completion=True)
  print('Waiting')
  time.sleep(3)
  # raw_input()


controller.quit()
print()

