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

filepath = 'P:/MIT/Lab/Wearativity/data/to_process_xsens/2022-07-13_experiment_S07/2022-07-13_11-01-18_actionNet-wearables_S07/2022-07-13_11-02-03_streamLog_actionNet-wearables_S07.hdf5'
hout = h5py.File(filepath, 'a')

activities = hout['experiment-activities']['activities']

new_data = np.concatenate((activities['data'][0:38,:],
                           [[b'Slice a potato', b'Start', b'Good', b'']],
                           [[b'Slice a potato', b'Stop', b'Good', b'']],
                           activities['data'][38:,:]))

# start slicing 2022-07-13 11:36:22.604390 end slicing 2022-07-13 11:36:45.719894
date_local_str = '2022-07-13'
start_slicing_str = '%s %s' % (date_local_str, '11:36:22.604390')
stop_slicing_str  = '%s %s' % (date_local_str, '11:36:45.719894')
start_slicing_s = get_time_s_from_local_str(start_slicing_str.split()[1], input_time_format='%H:%M:%S.%f', date_local_str=date_local_str, input_date_format='%Y-%m-%d')
stop_slicing_s = get_time_s_from_local_str(stop_slicing_str.split()[1], input_time_format='%H:%M:%S.%f', date_local_str=date_local_str, input_date_format='%Y-%m-%d')
new_times_s = np.concatenate((activities['time_s'][0:38,:],
                              [[start_slicing_s]], [[stop_slicing_s]],
                              activities['time_s'][38:,:]))
new_times_str = np.concatenate((activities['time_str'][0:38,:],
                                [[start_slicing_str.encode('utf-8')]], [[stop_slicing_str.encode('utf-8')]],
                                activities['time_str'][38:,:]))

del hout['experiment-activities']['activities']['data']
del hout['experiment-activities']['activities']['time_s']
del hout['experiment-activities']['activities']['time_str']
hout['experiment-activities']['activities']['data'] = new_data
hout['experiment-activities']['activities']['time_s'] = new_times_s
hout['experiment-activities']['activities'].create_dataset('time_str', data=new_times_str, dtype='S26')



calibrations_thirdParty = hout['experiment-calibration']['third_party']

new_data = np.concatenate((calibrations_thirdParty['data'][:,:],
                           [[b'Start', b'Good', b'', b'PupilLabs: Single Target', b'', b'', b'']],
                           [[b'Stop',  b'Good', b'', b'PupilLabs: Single Target', b'', b'', b'']],
                           ))

# start calibrating 2022-07-13 12:11:29.444824 end calibrating 2022-07-13 12:12:02.504825
date_local_str = '2022-07-13'
start_calibrating_str = '%s %s' % (date_local_str, '12:11:29.444824')
stop_calibrating_str  = '%s %s' % (date_local_str, '12:12:02.504825')
start_calibrating_s = get_time_s_from_local_str(start_calibrating_str.split()[1], input_time_format='%H:%M:%S.%f', date_local_str=date_local_str, input_date_format='%Y-%m-%d')
stop_calibrating_s = get_time_s_from_local_str(stop_calibrating_str.split()[1], input_time_format='%H:%M:%S.%f', date_local_str=date_local_str, input_date_format='%Y-%m-%d')
new_times_s = np.concatenate((calibrations_thirdParty['time_s'][:,:],
                              [[start_calibrating_s]], [[stop_calibrating_s]],
                              ))
new_times_str = np.concatenate((calibrations_thirdParty['time_str'][:,:],
                                [[start_calibrating_str.encode('utf-8')]], [[stop_calibrating_str.encode('utf-8')]],
                                ))

del hout['experiment-calibration']['third_party']['data']
del hout['experiment-calibration']['third_party']['time_s']
del hout['experiment-calibration']['third_party']['time_str']
hout['experiment-calibration']['third_party']['data'] = new_data
hout['experiment-calibration']['third_party']['time_s'] = new_times_s
hout['experiment-calibration']['third_party'].create_dataset('time_str', data=new_times_str, dtype='S26')

hout.close()

