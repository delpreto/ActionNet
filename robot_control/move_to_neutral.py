

from __future__ import print_function

from BaxterController import BaxterController

################################################

for limb_name in ['left', 'right']:
  print()
  print('='*50)
  print('Moving the %s limb' % limb_name)
  print('='*50)
  controller = BaxterController(limb_name=limb_name, print_debug=True)
  controller.move_to_neutral()
  controller.quit()

print()
print('='*50)
print('Done!')
print('='*50)
print()
print()