import h5py
import numpy as np
from utils.time_utils import *

filepath = 'P:/MIT/Lab/Wearativity/data/experiments/2022-06-07_experiment_S00/2022-06-07_18-10-55_actionNet-wearables_S00/2022-06-07_18-11-37_streamLog_actionNet-wearables_S00.hdf5'
hout = h5py.File(filepath, 'a')

activities = hout['experiment-activities']['activities']

new_data = np.concatenate((activities['data'][:,:], [[b'Unload dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils', b'Start', b'Good', b'']], [[b'Unload dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils', b'Stop', b'Good', b'']]))

# start unload 19:03:00 (stop load + 12 seconds)
# stop unloading 19:05:46.7 (start unload + 166.7 seconds)
x = activities['time_s'][-1][0]
unload_start_s = x + 12
unload_stop_s = unload_start_s + 166.7
new_times_s = np.concatenate((activities['time_s'][:,:], [[unload_start_s]], [[unload_stop_s]]))

new_times_str = [[get_time_str(time_s[0], format='%Y-%m-%d %H:%M:%S.%f').encode('utf-8')] for time_s in new_times_s]

del hout['experiment-activities']['activities']['data']
del hout['experiment-activities']['activities']['time_s']
del hout['experiment-activities']['activities']['time_str']
hout['experiment-activities']['activities']['data'] = new_data
hout['experiment-activities']['activities']['time_s'] = new_times_s
hout['experiment-activities']['activities'].create_dataset('time_str', data=new_times_str, dtype='S26')

hout.close()

