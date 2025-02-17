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

log_dirs = [
  # os.path.join(experiments_dir, '2023-08-18_experiment_S10', '2023-08-18_19-49-19_actionNet-wearables_S10'),
  # os.path.join(experiments_dir, '2023-08-18_experiment_S10', '2023-08-18_20-50-18_actionNet-wearables_S10'),
  # os.path.join(experiments_dir, '2023-10-13_experiment_S07', '2023-10-13_15-35-30_actionNet-wearables_S07'),
  # os.path.join(experiments_dir, '2023-10-13_experiment_S12', '2023-10-13_16-54-46_actionNet-wearables_S12'),
  # os.path.join(experiments_dir, '2023-10-13_experiment_S12', '2023-10-13_17-17-32_actionNet-wearables_S12'),
  # os.path.join(experiments_dir, '2023-10-13_experiment_S13', '2023-10-13_18-01-27_actionNet-wearables_S13'),
  # os.path.join(experiments_dir, '2023-10-13_experiment_S13', '2023-10-13_18-15-33_actionNet-wearables_S13'),
  # os.path.join(experiments_dir, '2023-10-13_experiment_S13', '2023-10-13_18-41-49_actionNet-wearables_S13'),
  # os.path.join(experiments_dir, '2024-12-20_experiment_S14', '2024-12-20_15-48-10_actionSense_S14_stir'),
  # os.path.join(experiments_dir, '2024-12-20_experiment_S14', '2024-12-20_16-27-23_actionSense_S14_scoop'),
  # os.path.join(experiments_dir, '2024-12-20_experiment_S15', '2024-12-20_17-46-00_actionSense_S15_scoop'),
  # os.path.join(experiments_dir, '2024-12-20_experiment_S15', '2024-12-20_18-11-12_actionSense_S15_stir'),
  os.path.join(experiments_dir, '2025-02-14_experiment_S11', '2025-02-14_14-52-51_actionSense_S11'),
  os.path.join(experiments_dir, '2025-02-14_experiment_S11', '2025-02-14_15-11-46_actionSense_S11'),
  os.path.join(experiments_dir, '2025-02-14_experiment_S11', '2025-02-14_15-34-50_actionSense_S11'),
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


