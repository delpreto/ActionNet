
from sensor_streamer_handlers.SensorManager import SensorManager
from sensor_streamer_handlers.DataVisualizer import DataVisualizer

import time
import traceback
from utils.time_utils import *
import sys
import os
script_dir = os.path.dirname(os.path.realpath(__file__))

import h5py
import glob
import shutil

import numpy as np
from utils.numpy_scipy_utils import convolve2d_strided

device_name = 'shear-sensor-left'

data_dir = os.path.realpath(os.path.join(script_dir, '..', '..', '..', 'data'))
experiments_dir = os.path.join(data_dir, 'tests')
log_dir = os.path.join(experiments_dir, '2023-05-24_shoeAngle_testing', '2023-05-24_17-41-33_shoeAngle_testing_full')

hdf5_filepath_in = glob.glob(os.path.join(log_dir, '*.hdf5'))[0]
hdf5_filepath_out = hdf5_filepath_in.replace('.hdf5', '_updated.hdf5')

hdf5_file_in = h5py.File(hdf5_filepath_in, 'r')
hdf5_file_out = h5py.File(hdf5_filepath_out, 'w')
tactile_data = hdf5_file_in[device_name]['tactile_data']['data']

calibration_matrix = np.mean(tactile_data[100:600], axis=0)
tactile_data_calibrated = tactile_data - calibration_matrix

toConvolve_tiled_x = np.array([[-1,1],[-1,1]])
toConvolve_tiled_y = np.array([[1,1],[-1,-1]])
toConvolve_tiled_magnitude = np.array([[1,1],[1,1]])

force_vector = []
force_magnitude = []
tactile_tiled = []
for i in range(tactile_data_calibrated.shape[0]):
  if i % 10000 == 0:
    print('Processing timestep %6d/%6d (%0.1f%%)' % (i, tactile_data_calibrated.shape[0], 100*i/tactile_data_calibrated.shape[0]))
  data_matrix_calibrated = np.squeeze(tactile_data_calibrated[i,:,:])
  data_matrix_tiled_magnitude = convolve2d_strided(data_matrix_calibrated, toConvolve_tiled_magnitude, stride=2)
  data_matrix_tiled_x = convolve2d_strided(data_matrix_calibrated, toConvolve_tiled_x, stride=2)
  data_matrix_tiled_y = convolve2d_strided(data_matrix_calibrated, toConvolve_tiled_y, stride=2)
  data_matrix_tiled_shearAngle_rad = np.arctan2(data_matrix_tiled_y, data_matrix_tiled_x)
  data_matrix_tiled_shearMagnitude = np.linalg.norm(np.stack([data_matrix_tiled_y, data_matrix_tiled_x], axis=0), axis=0)
  tactile_tiled.append(data_matrix_tiled_magnitude)
  force_vector.append(np.stack((data_matrix_tiled_shearMagnitude,
                                data_matrix_tiled_shearAngle_rad),
                               axis=0))
  force_magnitude.append(data_matrix_tiled_shearMagnitude)

# Copy all devices except the one being edited
for key in hdf5_file_in:
  if key != device_name:
    hdf5_file_in.copy(hdf5_file_in[key], hdf5_file_out,
                     name=None, shallow=False,
                     expand_soft=True, expand_external=True, expand_refs=True,
                     without_attrs=False)
# Copy the metadata of the device being edited
hdf5_file_in.copy(hdf5_file_in[device_name], hdf5_file_out,
                  name=None, shallow=True,
                  expand_soft=True, expand_external=True, expand_refs=True,
                  without_attrs=False)
device_group = hdf5_file_out[device_name]

# Copy data for all streams except the one being edited
for stream_name in hdf5_file_in[device_name]:
  if stream_name not in device_group:
    device_group.create_group(stream_name)
  stream_group = device_group[stream_name]
  for data_type in hdf5_file_in[device_name][stream_name]:
    if data_type != 'data':
      hdf5_file_in.copy(hdf5_file_in[device_name][stream_name][data_type],
                        stream_group,
                        name=None, shallow=False,
                        expand_soft=True, expand_external=True, expand_refs=True,
                        without_attrs=False)


# try: del hdf5_file_out[device_name]['tactile_data']['data']
# except: pass
# try: del hdf5_file_out[device_name]['tactile_data_calibrated']['data']
# except: pass
# try: del hdf5_file_out[device_name]['tactile_tiled']['data']
# except: pass
# try: del hdf5_file_out[device_name]['force_vector']['data']
# except: pass
# try: del hdf5_file_out[device_name]['force_magnitude']['data']
# except: pass

# Add the new data
hdf5_file_out[device_name]['tactile_data'].create_dataset('data', data=tactile_data)
hdf5_file_out[device_name]['tactile_data_calibrated'].create_dataset('data', data=tactile_data_calibrated)
hdf5_file_out[device_name]['tactile_tiled'].create_dataset('data', data=tactile_tiled)
hdf5_file_out[device_name]['force_vector'].create_dataset('data', data=force_vector)

magnitude_group = hdf5_file_out[device_name].create_group('force_magnitude')
magnitude_group.create_dataset('data', data=force_magnitude)
hdf5_file_out.copy(hdf5_file_out[device_name]['force_vector']['time_s'],
                    magnitude_group,
                    name=None, shallow=False,
                    expand_soft=True, expand_external=True, expand_refs=True,
                    without_attrs=False)
hdf5_file_out.copy(hdf5_file_out[device_name]['force_vector']['time_str'],
                    magnitude_group,
                    name=None, shallow=False,
                    expand_soft=True, expand_external=True, expand_refs=True,
                    without_attrs=False)

hdf5_file_in.close()
hdf5_file_out.close()