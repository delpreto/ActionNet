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
script_dir = os.path.dirname(os.path.realpath(__file__))


data_dir = os.path.realpath(os.path.join(script_dir, '..', '..', 'data'))
experiments_dir = os.path.join(data_dir, 'experiments')

log_dirs = [
  os.path.join(experiments_dir, 'my_set_of_experiments_1', 'my_experiment_folder'),
  os.path.join(experiments_dir, 'my_set_of_experiments_2', 'my_experiment_folder'),
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


