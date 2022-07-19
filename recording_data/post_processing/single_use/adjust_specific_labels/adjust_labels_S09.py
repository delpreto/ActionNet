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
from utils.print_utils import *

filepath = 'P:/MIT/Lab/Wearativity/data/experiments/2022-07-14_experiment_S09/2022-07-14_09-58-40_actionNet-wearables_S09/2022-07-14_09-59-00_streamLog_actionNet-wearables_S09.hdf5'
hout = h5py.File(filepath, 'a')

activities = hout['experiment-activities']['activities']

new_data = np.concatenate((activities['data'][:,:],
                           [[b'Unload dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils', b'Start', b'Good', b'']],
                           [[b'Unload dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils', b'Stop', b'Good', b'']]
                           ))

# start unloading 2022-07-14 10:58:08.300000 and stop 2022-07-14 11:01:01.619618
date_local_str = '2022-07-14'
start_str = '%s %s' % (date_local_str, '10:58:08.300000')
stop_str  = '%s %s' % (date_local_str, '11:01:01.619618')
start_s = get_time_s_from_local_str(start_str.split()[1], input_time_format='%H:%M:%S.%f', date_local_str=date_local_str, input_date_format='%Y-%m-%d')
stop_s = get_time_s_from_local_str(stop_str.split()[1], input_time_format='%H:%M:%S.%f', date_local_str=date_local_str, input_date_format='%Y-%m-%d')
new_times_s = np.concatenate((activities['time_s'][:,:],
                              [[start_s]],
                              [[stop_s]],
                              ))
new_times_str = np.concatenate((activities['time_str'][:,:],
                                [[start_str.encode('utf-8')]],
                                [[stop_str.encode('utf-8')]],
                                ))

del hout['experiment-activities']['activities']['data']
del hout['experiment-activities']['activities']['time_s']
del hout['experiment-activities']['activities']['time_str']
hout['experiment-activities']['activities']['data'] = new_data
hout['experiment-activities']['activities']['time_s'] = new_times_s
hout['experiment-activities']['activities'].create_dataset('time_str', data=new_times_str, dtype='S26')


hout.close()

