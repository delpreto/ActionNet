

from __future__ import print_function

import sys

from BaxterController import BaxterController

################################################

limb_names = ['left', 'right']
if len(sys.argv) > 1:
  limb_names = sys.argv[1:]

for limb_name in limb_names:
  print()
  print('='*50)
  print('Moving the %s limb' % limb_name)
  print('='*50)
  controller = BaxterController(limb_name=limb_name, print_debug=True)
  controller.move_to_resting()
  controller.quit()

print()
print('='*50)
print('Done!')
print('='*50)
print()
print()