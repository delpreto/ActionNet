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

import h5py
import numpy as np
from utils.time_utils import *

filepath = 'P:/MIT/Lab/Wearativity/data/experiments/2022-06-07_experiment_S00/2022-06-07_18-10-55_actionNet-wearables_S00/2022-06-07_18-11-37_streamLog_actionNet-wearables_S00_addCuttingBoardLabels.hdf5'
hout = h5py.File(filepath, 'a')

activities = hout['experiment-activities']['activities']

new_data = np.concatenate((activities['data'][:,:], [[b'Clear cutting board', b'Start', b'Good', b'']],
                                                    [[b'Clear cutting board', b'Stop', b'Good', b'']],
                                                    [[b'Clear cutting board', b'Start', b'Good', b'']],
                                                    [[b'Clear cutting board', b'Stop', b'Good', b'']],
                                                    [[b'Clear cutting board', b'Start', b'Good', b'']],
                                                    [[b'Clear cutting board', b'Stop', b'Good', b'']],
                           ))

date_local_str = '2022-06-07'
times_str = [
  '%s %s' % (date_local_str, '18:27:57.200000'),
  '%s %s' % (date_local_str, '18:28:10.600000'),
  '%s %s' % (date_local_str, '18:30:31.400000'),
  '%s %s' % (date_local_str, '18:30:51.700000'),
  '%s %s' % (date_local_str, '18:41:02.100000'),
  '%s %s' % (date_local_str, '18:41:24.300000'),
  ]

new_times_s = activities['time_s']
new_times_str = activities['time_str']

for time_str in times_str:
  time_s = get_time_s_from_local_str(time_str.split()[1], input_time_format='%H:%M:%S.%f', date_local_str=date_local_str, input_date_format='%Y-%m-%d')
  new_times_s = np.concatenate((new_times_s,
                                [[time_s]]))
  new_times_str = np.concatenate((new_times_str,
                                  [[time_str.encode('utf-8')]]))

del hout['experiment-activities']['activities']['data']
del hout['experiment-activities']['activities']['time_s']
del hout['experiment-activities']['activities']['time_str']
hout['experiment-activities']['activities']['data'] = new_data
hout['experiment-activities']['activities']['time_s'] = new_times_s
hout['experiment-activities']['activities'].create_dataset('time_str', data=new_times_str, dtype='S26')

hout.close()

