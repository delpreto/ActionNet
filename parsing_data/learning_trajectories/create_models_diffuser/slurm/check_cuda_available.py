
import sys
print()
print('Python version:', sys.version)
print()

import torch

print()
print('CUDA is available?', torch.cuda.is_available())
print('Device count:', torch.cuda.device_count())
for device_index in range(torch.cuda.device_count()):
  print('Device %d: ' % device_index, torch.cuda.get_device_name(device_index))

print()
