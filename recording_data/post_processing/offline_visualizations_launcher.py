############
#
# Copyright (c) 2022 MIT CSAIL and Joseph DelPreto
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
# Created 2021-2022 for the MIT ActionNet project by Joseph DelPreto [https://josephdelpreto.com].
#
############

import subprocess
import os

# data_dir = os.path.realpath(os.path.join(script_dir, '..', '..', '..', 'data'))
# experiments_dir = os.path.join(data_dir, 'experiments')

script_dir = os.path.dirname(os.path.realpath(__file__))
actionsense_root_dir = script_dir
while os.path.split(actionsense_root_dir)[-1] != 'ActionSense':
  actionsense_root_dir = os.path.realpath(os.path.join(actionsense_root_dir, '..'))
data_dir = os.path.realpath(os.path.join(actionsense_root_dir, 'data'))
experiments_dir = os.path.join(data_dir, 'experiments')
tests_dir = os.path.join(data_dir, 'tests')

log_dirs = [
  # os.path.join(experiments_dir, 'my_set_of_experiments_1', 'my_experiment_folder'),
  # os.path.join(experiments_dir, 'my_set_of_experiments_2', 'my_experiment_folder'),
  # os.path.join(tests_dir, '2026-07-24_tennis_S16', '2026-07-24_11-19-03_tennis_S16'),
  # os.path.join(tests_dir, '2026-07-24_tennis_S16', '2026-07-24_11-52-26_tennis_S16'),
  os.path.join(experiments_dir, '2026-08-18_tennis_S16', 'for_merging', '2026-08-18_12-17-07_tennis_S16'),
  os.path.join(experiments_dir, '2026-08-18_tennis_S16', 'for_merging', '2026-08-18_12-48-40_tennis_S16_short'),
  os.path.join(experiments_dir, '2026-08-18_tennis_S16', 'for_merging', '2026-08-18_12-51-12_tennis_S16'),
  ]

for log_dir in log_dirs:
  print('\n'*5)
  print('|'*75)
  print('Calling script with log directory', log_dir)
  print('\n')
  stream = os.system('py offline_visualizations.py "%s"' % log_dir)
  # stream = os.popen('py offline_visualizations.py %s' % log_dir)
  # output = stream.read()
  # print(output)


