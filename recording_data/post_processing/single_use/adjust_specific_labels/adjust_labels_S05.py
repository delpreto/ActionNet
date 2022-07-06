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

filepath = 'P:/MIT/Lab/Wearativity/data/experiments/2022-06-14_experiment_S05/2022-06-14_20-45-43_actionNet-wearables_S05/2022-06-14_20-46-12_streamLog_actionNet-wearables_S05.hdf5'
hout = h5py.File(filepath, 'a')

activities = hout['experiment-activities']['activities']

new_data = np.concatenate((activities['data'][:,:], [[b'Unload dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils', b'Start', b'Good', b'']], [[b'Unload dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils', b'Stop', b'Good', b'']]))

# stop loading 21:37:04	start unloading 21:37:23	stop unloading 21:40:08
x = activities['time_s'][-1][0]  
unload_start_s = x + 19
unload_stop_s = unload_start_s + 37+120+8
new_times_s = np.concatenate((activities['time_s'][:,:], [[unload_start_s]], [[unload_stop_s]]))

new_times_str = [[get_time_str(time_s[0], format='%Y-%m-%d %H:%M:%S.%f').encode('utf-8')] for time_s in new_times_s]

del hout['experiment-activities']['activities']['data']
del hout['experiment-activities']['activities']['time_s']
del hout['experiment-activities']['activities']['time_str']
hout['experiment-activities']['activities']['data'] = new_data
hout['experiment-activities']['activities']['time_s'] = new_times_s
hout['experiment-activities']['activities'].create_dataset('time_str', data=new_times_str, dtype='S26')

hout.close()

