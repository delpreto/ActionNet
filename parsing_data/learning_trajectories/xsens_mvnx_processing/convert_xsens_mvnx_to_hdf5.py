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
from scipy import interpolate
from scipy.spatial.transform import Rotation
from collections import OrderedDict
import os
import glob
script_dir = os.path.dirname(os.path.realpath(__file__))
import copy
import sys
import cv2

import pandas
from bs4 import BeautifulSoup

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d

from utils.numpy_scipy_utils import *
from utils.angle_utils import *
from utils.dict_utils import *
from utils.print_utils import *
from utils.time_utils import *


##############################################

class XsensConverter():
  def __init__(self):
    # self._metadata_data_headings_key = 'Data headings'
    # self._data_notes_mvnx = {}
    # self._define_data_notes()
    pass
  
  def convert_mvnx_files(self, mvnx_filepaths):
    if not isinstance(mvnx_filepaths, (list, tuple)):
      mvnx_filepaths = [mvnx_filepaths]
    for mvnx_filepath in mvnx_filepaths[0:]:
      # Load the MVNX data.
      print('Reading and processing %s' % os.path.basename(mvnx_filepath))
      with open(mvnx_filepath, 'r') as fin:
        mvnx_contents = fin.read()
      mvnx_xml = BeautifulSoup(mvnx_contents, 'xml')
  
      # Get metadata.
      metadata = self._get_mvnx_metadata(mvnx_xml)
      metadata_segments = dict([(key, value) for (key, value) in metadata.items() if 'joint' not in key and 'sensor' not in key and 'contact' not in key])
      metadata_joints = dict([(key, value) for (key, value) in metadata.items() if 'segment' not in key and 'sensor' not in key and 'contact' not in key])
      metadata_com = dict([(key, value) for (key, value) in metadata.items() if 'segment' not in key and 'joint' not in key and 'sensor' not in key and 'contact' not in key])
      metadata_sensors = dict([(key, value) for (key, value) in metadata.items() if 'segment' not in key and 'joint' not in key and 'contact' not in key])
      metadata_footContacts = dict([(key, value) for (key, value) in metadata.items() if 'segment' not in key and 'joint' not in key and 'sensor' not in key])
      
      have_fingers_left = 'segment_names_fingers_left' in metadata
      have_fingers_right = 'segment_names_fingers_right' in metadata
      # print_var(metadata, 'metadata')
  
      # Open an output HDF5 file.
      hdf5_filepath = '%s.hdf5' % os.path.splitext(mvnx_filepath)[0]
      hdf5_file_toUpdate = h5py.File(hdf5_filepath, 'w')
  
      # Extract all frames
      frames = mvnx_xml.find_all('frame')
  
      # Extract the calibration frames.
      calibration_types = ['identity', 'Tpose', 'Tpose_ISB'] # should match MVNX types if make lowercase and replace underscores with hyphones
      calibration_data = dict([(key, None) for key in calibration_types])
      for (frame_index, frame) in enumerate(frames[0:5]): # should be the first 3 frames, but increase a bit just to be safe
        for calibration_type in calibration_types:
          if frame.get('type') == calibration_type.lower().replace('_', '-'):
            calibration_data[calibration_type] = {
              'segment_orientations_body_quaternion': np.atleast_2d([round(float(q), 6) for q in frame.find('orientation').contents[0].split()]),
              'segment_positions_body_m': np.atleast_2d([round(float(q), 6) for q in frame.find('position').contents[0].split()]),
            }
  
      # Extract data from all non-calibration frames.
  
      frames = [frame for (i, frame) in enumerate(frames) if frame.get('type') == 'normal']
      frame_indexes = [int(frame.get('index')) for frame in frames]
      num_frames = len(frames)
  
      # Get time information.
      times_since_start_s = [round(float(frame.get('time')))/1000.0 for frame in frames]
      times_utc_str = [frame.get('tc') for frame in frames]
      times_s = [float(frame.get('ms'))/1000.0 for frame in frames]
      times_str = [get_time_str(time_s, '%Y-%m-%d %H:%M:%S.%f') for time_s in times_s]
  
      # Check that the timestamps monotonically increase.
      if np.any(np.diff(times_s) < 0):
        print_var(times_str, 'times_str')
        msg = '\n'*2
        msg += 'x'*75
        msg += 'XsensStreamer aborting merge due to incorrect timestamps in the MVNX file (they are not monotonically increasing)'
        msg += 'x'*75
        msg = '\n'*2
        print(msg)
        continue
  
      # A helper to get a matrix of data from all frames for a given tag, such as 'position'.
      def get_tagged_data(tag):
        datas = [frame.find(tag) for frame in frames]
        data = np.array([[round(float(x), 6) for x in data.contents[0].split()] for data in datas])
        return data
  
      # Extract the data!
      segment_orientations_body_quaternion_wijk = get_tagged_data('orientation')
      segment_positions_body_m                  = get_tagged_data('position')
      segment_velocities_body_m_s               = get_tagged_data('velocity')
      segment_accelerations_body_m_ss           = get_tagged_data('acceleration')
      segment_angular_velocities_body_rad_s     = get_tagged_data('angularVelocity')
      segment_angular_accelerations_body_rad_ss = get_tagged_data('angularAcceleration')
      foot_contacts                             = get_tagged_data('footContacts')
      sensor_freeAccelerations_m_ss             = get_tagged_data('sensorFreeAcceleration')
      sensor_magnetic_fields_au                 = get_tagged_data('sensorMagneticField')
      sensor_orientations_quaternion_wijk       = get_tagged_data('sensorOrientation')
      joint_angles_body_eulerZXY_xyz_rad        = get_tagged_data('jointAngle')*np.pi/180.0 # convert from degrees to radians
      joint_angles_body_eulerXZY_xyz_rad        = get_tagged_data('jointAngleXZY')*np.pi/180.0 # convert from degrees to radians
      joint_angles_ergonomic_eulerZXY_xyz_rad   = get_tagged_data('jointAngleErgo')*np.pi/180.0 # convert from degrees to radians
      joint_angles_ergonomic_eulerXZY_xyz_rad   = get_tagged_data('jointAngleErgoXZY')*np.pi/180.0 # convert from degrees to radians
      center_of_mass                            = get_tagged_data('centerOfMass')
      center_of_mass_positions_m                = center_of_mass[:, 0:3]
      center_of_mass_velocities_m_s             = center_of_mass[:, 3:6]
      center_of_mass_accelerations_m_ss         = center_of_mass[:, 6:9]
      # Extract data for the left hand if it is available.
      if have_fingers_left:
        segment_orientations_fingersLeft_quaternion_wijk = get_tagged_data('orientationFingersLeft')
        segment_positions_fingersLeft_m                  = get_tagged_data('positionFingersLeft')
        joint_angles_fingersLeft_eulerZXY_xyz_rad        = get_tagged_data('jointAngleFingersLeft')*np.pi/180.0 # convert from degrees to radians
        joint_angles_fingersLeft_eulerXZY_xyz_rad        = get_tagged_data('jointAngleFingersLeftXZY')*np.pi/180.0 # convert from degrees to radians
      else:
        segment_orientations_fingersLeft_quaternion_wijk = None
        segment_positions_fingersLeft_m = None
        joint_angles_fingersLeft_eulerZXY_xyz_rad = None
        joint_angles_fingersLeft_eulerXZY_xyz_rad = None
      # Extract data for the right hand if it is available.
      if have_fingers_right:
        segment_orientations_fingersRight_quaternion_wijk = get_tagged_data('orientationFingersRight')
        segment_positions_fingersRight_m                  = get_tagged_data('positionFingersRight')
        joint_angles_fingersRight_eulerZXY_xyz_rad        = get_tagged_data('jointAngleFingersRight')*np.pi/180.0 # convert from degrees to radians
        joint_angles_fingersRight_eulerXZY_xyz_rad        = get_tagged_data('jointAngleFingersRightXZY')*np.pi/180.0 # convert from degrees to radians
      else:
        segment_orientations_fingersRight_quaternion_wijk = None
        segment_positions_fingersRight_m = None
        joint_angles_fingersRight_eulerZXY_xyz_rad = None
        joint_angles_fingersRight_eulerXZY_xyz_rad = None
      
      # Check a few dimensions.
      assert metadata['num_segments_body'] == segment_orientations_body_quaternion_wijk.shape[1]/4
      if have_fingers_left:
        assert metadata['num_segments_fingers_left'] == segment_orientations_fingersLeft_quaternion_wijk.shape[1]/4
      if have_fingers_right:
        assert metadata['num_segments_fingers_right'] == segment_orientations_fingersRight_quaternion_wijk.shape[1]/4
      
      # Create Euler orientations from quaternion orientations.
      segment_orientations_body_eulerZXY_xyz_rad = np.empty([num_frames, 3*metadata['num_segments_body']])
      for segment_index in range(metadata['num_segments_body']):
        for frame_index in range(num_frames):
          quat_wijk = segment_orientations_body_quaternion_wijk[frame_index, (segment_index*4):(segment_index*4+4)]
          eulers_rad = euler_from_quaternion(w=quat_wijk[0], x=quat_wijk[1], y=quat_wijk[2], z=quat_wijk[3], euler_sequence='ZXY', degrees=False)
          segment_orientations_body_eulerZXY_xyz_rad[frame_index, (segment_index*3):(segment_index*3+3)] = eulers_rad
      if have_fingers_left:
        segment_orientations_fingersLeft_eulerZXY_xyz_rad = np.empty([num_frames, 3*metadata['num_segments_fingers_left']])
        for segment_index in range(metadata['num_segments_fingers_left']):
          for frame_index in range(num_frames):
            quat_wijk = segment_orientations_fingersLeft_quaternion_wijk[frame_index, (segment_index*4):(segment_index*4+4)]
            eulers_rad = euler_from_quaternion(w=quat_wijk[0], x=quat_wijk[1], y=quat_wijk[2], z=quat_wijk[3], euler_sequence='ZXY', degrees=False)
            segment_orientations_fingersLeft_eulerZXY_xyz_rad[frame_index, (segment_index*3):(segment_index*3+3)] = eulers_rad
      if have_fingers_right:
        segment_orientations_fingersRight_eulerZXY_xyz_rad = np.empty([num_frames, 3*metadata['num_segments_fingers_right']])
        for segment_index in range(metadata['num_segments_fingers_right']):
          for frame_index in range(num_frames):
            quat_wijk = segment_orientations_fingersRight_quaternion_wijk[frame_index, (segment_index*4):(segment_index*4+4)]
            eulers_rad = euler_from_quaternion(w=quat_wijk[0], x=quat_wijk[1], y=quat_wijk[2], z=quat_wijk[3], euler_sequence='ZXY', degrees=False)
            segment_orientations_fingersRight_eulerZXY_xyz_rad[frame_index, (segment_index*3):(segment_index*3+3)] = eulers_rad
      sensor_orientations_eulerZXY_xyz_rad = np.empty([num_frames, 3*metadata['num_sensors']])
      for sensor_index in range(metadata['num_sensors']):
        for frame_index in range(num_frames):
          quat_wijk = sensor_orientations_quaternion_wijk[frame_index, (sensor_index*4):(sensor_index*4+4)]
          eulers_rad = euler_from_quaternion(w=quat_wijk[0], x=quat_wijk[1], y=quat_wijk[2], z=quat_wijk[3], euler_sequence='ZXY', degrees=False)
          sensor_orientations_eulerZXY_xyz_rad[frame_index, (sensor_index*3):(sensor_index*3+3)] = eulers_rad
      
      # Helper to import the MVNX data into the HDF5 file.
      def add_hdf5_data(device_name, stream_name,
                        data, target_data_shape_per_frame,
                        stream_group_metadata=None):
        # Reshape so that data for each frame has the desired shape.
        data = data.reshape((data.shape[0], *target_data_shape_per_frame))
        assert data.shape[0] == num_frames
        # Create the stream group.
        if device_name not in hdf5_file_toUpdate:
          hdf5_file_toUpdate.create_group(device_name)
        if stream_name in hdf5_file_toUpdate[device_name]:
          del hdf5_file_toUpdate[device_name][stream_name]
        hdf5_file_toUpdate[device_name].create_group(stream_name)
        stream_group = hdf5_file_toUpdate[device_name][stream_name]
        # Create the datasets.
        stream_group.create_dataset('data', data=data)
        stream_group.create_dataset('xsens_sample_number', [num_frames, 1],
                                    data=frame_indexes)
        stream_group.create_dataset('xsens_time_since_start_s', [num_frames, 1],
                                    data=times_since_start_s)
        stream_group.create_dataset('time_s', [num_frames, 1], dtype='float64',
                                    data=times_s)
        stream_group.create_dataset('time_str', [num_frames, 1], dtype='S26',
                                    data=times_str)
        # Create a basic device-level metadata if there was none to copy from the original file.
        device_group_metadata = {}
        hdf5_file_toUpdate[device_name].attrs.update(device_group_metadata)
        # Override with provided stream-level metadata if desired.
        if stream_group_metadata is not None:
          hdf5_file_toUpdate[device_name][stream_name].attrs.update(
            convert_dict_values_to_str(stream_group_metadata, preserve_nested_dicts=False))
  
      # Import the data!
      
      # Segment orientation data (quaternion)
      add_hdf5_data(device_name='xsens-segments',
                    stream_name='body_orientation_quaternion_wijk',
                    data=segment_orientations_body_quaternion_wijk,
                    target_data_shape_per_frame=[-1, 4], # 4-element quaternion per segment per frame
                    stream_group_metadata=metadata_segments)
      if have_fingers_left:
        add_hdf5_data(device_name='xsens-segments',
                      stream_name='fingersLeft_orientation_quaternion_wijk',
                      data=segment_orientations_fingersLeft_quaternion_wijk,
                      target_data_shape_per_frame=[-1, 4], # 4-element quaternion per segment per frame
                      stream_group_metadata=metadata_segments)
      if have_fingers_right:
        add_hdf5_data(device_name='xsens-segments',
                      stream_name='fingersRight_orientation_quaternion_wijk',
                      data=segment_orientations_fingersRight_quaternion_wijk,
                      target_data_shape_per_frame=[-1, 4], # 4-element quaternion per segment per frame
                      stream_group_metadata=metadata_segments)
      # Segment orientation data (Euler)
      add_hdf5_data(device_name='xsens-segments',
                    stream_name='body_orientation_eulerZXY_xyz_rad',
                    data=segment_orientations_body_eulerZXY_xyz_rad,
                    target_data_shape_per_frame=[-1, 3], # 3-element Euler vector per segment per frame
                    stream_group_metadata=metadata_segments)
      if have_fingers_left:
        add_hdf5_data(device_name='xsens-segments',
                      stream_name='fingersLeft_orientation_eulerZXY_xyz_rad',
                      data=segment_orientations_fingersLeft_eulerZXY_xyz_rad,
                      target_data_shape_per_frame=[-1, 3], # 3-element Euler vector per segment per frame
                      stream_group_metadata=metadata_segments)
      if have_fingers_right:
        add_hdf5_data(device_name='xsens-segments',
                      stream_name='fingersRight_orientation_eulerZXY_xyz_rad',
                      data=segment_orientations_fingersRight_eulerZXY_xyz_rad,
                      target_data_shape_per_frame=[-1, 3], # 3-element Euler vector per segment per frame
                      stream_group_metadata=metadata_segments)
  
      # Segment position data.
      add_hdf5_data(device_name='xsens-segments',
                    stream_name='body_position_xyz_m',
                    data=segment_positions_body_m,
                    target_data_shape_per_frame=[-1, 3], # 3-element x/y/z vector per segment per frame
                    stream_group_metadata=metadata_segments)
      if have_fingers_left:
        add_hdf5_data(device_name='xsens-segments',
                      stream_name='fingersLeft_position_xyz_m',
                      data=segment_positions_fingersLeft_m,
                      target_data_shape_per_frame=[-1, 3], # 3-element x/y/z vector per segment per frame
                      stream_group_metadata=metadata_segments)
      if have_fingers_right:
        add_hdf5_data(device_name='xsens-segments',
                      stream_name='fingersRight_position_xyz_m',
                      data=segment_positions_fingersRight_m,
                      target_data_shape_per_frame=[-1, 3], # 3-element x/y/z vector per segment per frame
                      stream_group_metadata=metadata_segments)
      # Segment velocity data.
      add_hdf5_data(device_name='xsens-segments',
                    stream_name='body_velocity_xyz_m_s',
                    data=segment_velocities_body_m_s,
                    target_data_shape_per_frame=[-1, 3], # 3-element x/y/z vector per segment per frame
                    stream_group_metadata=metadata_segments)
      # Segment acceleration data.
      add_hdf5_data(device_name='xsens-segments',
                    stream_name='body_acceleration_xyz_m_ss',
                    data=segment_accelerations_body_m_ss,
                    target_data_shape_per_frame=[-1, 3], # 3-element x/y/z vector per segment per frame
                    stream_group_metadata=metadata_segments)
      # Segment angular velocity.
      add_hdf5_data(device_name='xsens-segments',
                    stream_name='body_angular_velocity_xyz_rad_s',
                    data=segment_angular_velocities_body_rad_s,
                    target_data_shape_per_frame=[-1, 3], # 3-element x/y/z vector per segment per frame
                    stream_group_metadata=metadata_segments)
      # Segment angular acceleration.
      add_hdf5_data(device_name='xsens-segments',
                    stream_name='body_angular_acceleration_xyz_rad_ss',
                    data=segment_angular_accelerations_body_rad_ss,
                    target_data_shape_per_frame=[-1, 3], # 3-element x/y/z vector per segment per frame
                    stream_group_metadata=metadata_segments)
  
      # Joint angles ZXY.
      add_hdf5_data(device_name='xsens-joints',
                    stream_name='body_joint_angles_eulerZXY_xyz_rad',
                    data=joint_angles_body_eulerZXY_xyz_rad,
                    target_data_shape_per_frame=[-1, 3], # 3-element vector per segment per frame
                    stream_group_metadata=metadata_joints)
      if have_fingers_left:
        add_hdf5_data(device_name='xsens-joints',
                      stream_name='fingersLeft_joint_angles_eulerZXY_xyz_rad',
                      data=joint_angles_fingersLeft_eulerZXY_xyz_rad,
                      target_data_shape_per_frame=[-1, 3], # 3-element vector per segment per frame
                      stream_group_metadata=metadata_joints)
      if have_fingers_right:
        add_hdf5_data(device_name='xsens-joints',
                      stream_name='fingersRight_joint_angles_eulerZXY_xyz_rad',
                      data=joint_angles_fingersRight_eulerZXY_xyz_rad,
                      target_data_shape_per_frame=[-1, 3], # 3-element vector per segment per frame
                      stream_group_metadata=metadata_joints)
      # Joint angles XZY.
      add_hdf5_data(device_name='xsens-joints',
                    stream_name='body_joint_angles_eulerXZY_xyz_rad',
                    data=joint_angles_body_eulerXZY_xyz_rad,
                    target_data_shape_per_frame=[-1, 3], # 3-element vector per segment per frame
                    stream_group_metadata=metadata_joints)
      if have_fingers_left:
        add_hdf5_data(device_name='xsens-joints',
                    stream_name='fingersLeft_joint_angles_eulerXZY_xyz_rad',
                    data=joint_angles_fingersLeft_eulerXZY_xyz_rad,
                    target_data_shape_per_frame=[-1, 3], # 3-element vector per segment per frame
                    stream_group_metadata=metadata_joints)
      if have_fingers_right:
        add_hdf5_data(device_name='xsens-joints',
                    stream_name='fingersRight_joint_angles_eulerXZY_xyz_rad',
                    data=joint_angles_fingersRight_eulerXZY_xyz_rad,
                    target_data_shape_per_frame=[-1, 3], # 3-element vector per segment per frame
                    stream_group_metadata=metadata_joints)
      # Ergonomic joint angles ZXY.
      add_hdf5_data(device_name='xsens-ergonomic-joints',
                    stream_name='ergonomic_joint_angles_eulerZXY_xyz_rad',
                    data=joint_angles_ergonomic_eulerZXY_xyz_rad,
                    target_data_shape_per_frame=[-1, 3], # 3-element vector per segment per frame
                    stream_group_metadata=metadata_joints)
      # Ergonomic joint angles XZY.
      add_hdf5_data(device_name='xsens-ergonomic-joints',
                    stream_name='ergonomic_joint_angles_eulerXZY_xyz_rad',
                    data=joint_angles_ergonomic_eulerXZY_xyz_rad,
                    target_data_shape_per_frame=[-1, 3], # 3-element vector per segment per frame
                    stream_group_metadata=metadata_joints)
  
      # Center of mass position.
      add_hdf5_data(device_name='xsens-CoM',
                    stream_name='position_xyz_m',
                    data=center_of_mass_positions_m,
                    target_data_shape_per_frame=[3], # 3-element x/y/z vector per frame
                    stream_group_metadata=metadata_com)
      # Center of mass velocity.
      add_hdf5_data(device_name='xsens-CoM',
                    stream_name='velocity_xyz_m_s',
                    data=center_of_mass_velocities_m_s,
                    target_data_shape_per_frame=[3], # 3-element x/y/z vector per frame
                    stream_group_metadata=metadata_com)
      # Center of mass acceleration.
      add_hdf5_data(device_name='xsens-CoM',
                    stream_name='acceleration_xyz_m_ss',
                    data=center_of_mass_accelerations_m_ss,
                    target_data_shape_per_frame=[3], # 3-element x/y/z vector per frame
                    stream_group_metadata=metadata_com)
  
      # Sensor data - acceleration
      add_hdf5_data(device_name='xsens-sensors',
                    stream_name='free_acceleration_xyz_m_ss',
                    data=sensor_freeAccelerations_m_ss,
                    target_data_shape_per_frame=[-1, 3], # 3-element x/y/z vector per segment per frame
                    stream_group_metadata=metadata_sensors)
      # Sensor data - magnetic field
      add_hdf5_data(device_name='xsens-sensors',
                    stream_name='magnetic_field_xyz_au',
                    data=sensor_magnetic_fields_au,
                    target_data_shape_per_frame=[-1, 3], # 3-element x/y/z vector per segment per frame
                    stream_group_metadata=metadata_sensors)
      # Sensor data - orientation
      add_hdf5_data(device_name='xsens-sensors',
                    stream_name='sensor_orientation_quaternion_wijk',
                    data=sensor_orientations_quaternion_wijk,
                    target_data_shape_per_frame=[-1, 4], # 4-element quaternion vector per segment per frame
                    stream_group_metadata=metadata_sensors)
      # Sensor data - orientation
      add_hdf5_data(device_name='xsens-sensors',
                    stream_name='sensor_orientation_eulerZXY_xyz_rad',
                    data=sensor_orientations_eulerZXY_xyz_rad,
                    target_data_shape_per_frame=[-1, 3], # 3-element x/y/z vector per segment per frame
                    stream_group_metadata=metadata_sensors)
  
      # Foot contacts
      add_hdf5_data(device_name='xsens-foot-contacts',
                    stream_name='is_contacting_ground',
                    data=foot_contacts,
                    target_data_shape_per_frame=[4], # 4-element contact vector per segment per frame
                    stream_group_metadata=metadata_footContacts)
      
      # Calibration poses.
      device_group = hdf5_file_toUpdate.create_group('xsens-segments-tpose')
      device_group.create_dataset('body_orientation_identity_quaternion_wijk',
                                  data=calibration_data['identity']['segment_orientations_body_quaternion'].reshape((-1, 4)))
      device_group.create_dataset('body_orientation_Tpose_quaternion_wijk',
                                  data=calibration_data['Tpose']['segment_orientations_body_quaternion'].reshape((-1, 4)))
      device_group.create_dataset('body_orientation_TposeISB_quaternion_wijk',
                                  data=calibration_data['Tpose_ISB']['segment_orientations_body_quaternion'].reshape((-1, 4)))
      device_group.create_dataset('body_position_identity_xyz_m',
                                  data=calibration_data['identity']['segment_positions_body_m'].reshape((-1, 3)))
      device_group.create_dataset('body_position_Tpose_xyz_m',
                                  data=calibration_data['Tpose']['segment_positions_body_m'].reshape((-1, 3)))
      device_group.create_dataset('body_position_TposeISB_xyz_m',
                                  data=calibration_data['Tpose_ISB']['segment_positions_body_m'].reshape((-1, 3)))
      
      hdf5_file_toUpdate.close()
  

  ##############################################################
  ##############################################################
  
  def _get_mvnx_metadata(self, mvnx_xml):
    # Extract metadata about MVN version.
    mvnx_version = mvnx_xml.find('mvnx').get('version')
    mvn_version = mvnx_xml.find('mvn').get('build')
    if mvnx_version != '4':
      raise AssertionError('The MVNX import script currently assumes MVNX version 4, but version %s is being used instead. The script should be checked for compatibility.' % mvnx_version)
    # Extract the sensor names.
    sensor_labels = [sensor_xml.get('label') for sensor_xml in mvnx_xml.find_all('sensor')]
    
    # Extract segment metadata for the main skeleton.
    body_segment_mesh_points_xyz_cm = OrderedDict()
    for segment_xml in mvnx_xml.find('segments').find_all('segment'):
      segment_mesh_points_xyz_cm_forSegment = OrderedDict()
      for segment_point_xml in segment_xml.find_all('point'):
        point_position_xyz_cm = [round(100*float(d), 6) for d in segment_point_xml.find('pos_b').contents[0].split()]
        segment_mesh_points_xyz_cm_forSegment[segment_point_xml.get('label')] = point_position_xyz_cm
      body_segment_mesh_points_xyz_cm[segment_xml.get('label')] = segment_mesh_points_xyz_cm_forSegment
    body_segment_labels = list(body_segment_mesh_points_xyz_cm.keys())
    # Extract finger segment metadata for the left hand if it is available.
    fingerLeft_segment_mesh_points_xyz_cm = OrderedDict()
    if mvnx_xml.find('fingerTrackingSegmentsLeft') is not None:
      for segment_xml in mvnx_xml.find('fingerTrackingSegmentsLeft').find_all('segment'):
        segment_mesh_points_xyz_cm_forSegment = OrderedDict()
        for segment_point_xml in segment_xml.find_all('point'):
          point_position_xyz_cm = [round(100*float(d), 6) for d in segment_point_xml.find('pos_b').contents[0].split()]
          segment_mesh_points_xyz_cm_forSegment[segment_point_xml.get('label')] = point_position_xyz_cm
        fingerLeft_segment_mesh_points_xyz_cm[segment_xml.get('label')] = segment_mesh_points_xyz_cm_forSegment
    fingerLeft_segment_labels = list(fingerLeft_segment_mesh_points_xyz_cm.keys())
    # Extract finger segment metadata for the right hand if it is available.
    fingerRight_segment_mesh_points_xyz_cm = OrderedDict()
    if mvnx_xml.find('fingerTrackingSegmentsRight') is not None:
      for segment_xml in mvnx_xml.find('fingerTrackingSegmentsRight').find_all('segment'):
        segment_mesh_points_xyz_cm_forSegment = OrderedDict()
        for segment_point_xml in segment_xml.find_all('point'):
          point_position_xyz_cm = [round(100*float(d), 6) for d in segment_point_xml.find('pos_b').contents[0].split()]
          segment_mesh_points_xyz_cm_forSegment[segment_point_xml.get('label')] = point_position_xyz_cm
        fingerRight_segment_mesh_points_xyz_cm[segment_xml.get('label')] = segment_mesh_points_xyz_cm_forSegment
    fingerRight_segment_labels = list(fingerRight_segment_mesh_points_xyz_cm.keys())
    
    # Extract joint metadata for the main skeleton.
    body_joint_segment_connections = OrderedDict()
    for joint_xml in mvnx_xml.find('joints').find_all('joint'):
      body_joint_segment_connections[joint_xml.get('label')[1:]] = [ # trim the initial 'j' from the label
        joint_xml.find('connector1').contents[0],
        joint_xml.find('connector2').contents[0],
        ]
    body_joint_labels = list(body_joint_segment_connections.keys())
    # Extract finger joint metadata for the left hand if it is available.
    fingerLeft_joint_segment_connections = OrderedDict()
    if mvnx_xml.find('fingerTrackingJointsLeft') is not None:
      for joint_xml in mvnx_xml.find('fingerTrackingJointsLeft').find_all('joint'):
        fingerLeft_joint_segment_connections[joint_xml.get('label')[1:]] = [ # trim the initial 'j' from the label
          joint_xml.find('connector1').contents[0],
          joint_xml.find('connector2').contents[0],
          ]
    fingerLeft_joint_labels = list(fingerLeft_joint_segment_connections.keys())
    # Extract finger joint metadata for the right hand if it is available.
    fingerRight_joint_segment_connections = OrderedDict()
    if mvnx_xml.find('fingerTrackingJointsRight') is not None:
      for joint_xml in mvnx_xml.find('fingerTrackingJointsRight').find_all('joint'):
        fingerRight_joint_segment_connections[joint_xml.get('label')[1:]] = [ # trim the initial 'j' from the label
          joint_xml.find('connector1').contents[0],
          joint_xml.find('connector2').contents[0],
          ]
    fingerRight_joint_labels = list(fingerRight_joint_segment_connections.keys())
    
    # Extract metadata about the ergonomic joint angles.
    ergonomic_joint_segment_connections = OrderedDict()
    for joint_xml in mvnx_xml.find_all('ergonomicJointAngle'):
      ergonomic_joint_segment_connections[joint_xml.get('label')] = [
        joint_xml.get('parentSegment'),
        joint_xml.get('childSegment'),
        ]
    ergonomic_joint_labels = list(ergonomic_joint_segment_connections.keys())
    
    # Extract the foot contact names.
    foot_contact_labels = [tag.get('label') for tag in mvnx_xml.find_all('contactDefinition')]
    
    # Define joint rotation names for each entry.
    body_joint_rotation_type_ordering_raw = OrderedDict([
      ('L5S1',   ('Lateral Bending', 'Axial Bending',  'Flexion/Extension')),
      ('L4L3',   ('Lateral Bending', 'Axial Rotation', 'Flexion/Extension')),
      ('L1T12',  ('Lateral Bending', 'Axial Rotation', 'Flexion/Extension')),
      ('T9T8',   ('Lateral Bending', 'Axial Rotation', 'Flexion/Extension')),
      ('T1C7',   ('Lateral Bending', 'Axial Rotation', 'Flexion/Extension')),
      ('C1Head', ('Lateral Bending', 'Axial Rotation', 'Flexion/Extension')),
      ('RightT4Shoulder', ('Abduction/Adduction',              'Internal/External Rotation', 'Flexion/Extension')),
      ('RightShoulder',   ('Abduction/Adduction',              'Internal/External Rotation', 'Flexion/Extension')),
      ('RightElbow',      ('Ulnar Deviation/Radial Deviation', 'Pronation/Supination',       'Flexion/Extension')),
      ('RightWrist',      ('Ulnar Deviation/Radial Deviation', 'Pronation/Supination',       'Flexion/Extension')),
      ('LeftT4Shoulder',  ('Abduction/Adduction',              'Internal/External Rotation', 'Flexion/Extension')),
      ('LeftShoulder',    ('Abduction/Adduction',              'Internal/External Rotation', 'Flexion/Extension')),
      ('LeftElbow',       ('Ulnar Deviation/Radial Deviation', 'Pronation/Supination',       'Flexion/Extension')),
      ('LeftWrist',       ('Ulnar Deviation/Radial Deviation', 'Pronation/Supination',       'Flexion/Extension')),
      ('RightHip',        ('Abduction/Adduction',              'Internal/External Rotation', 'Flexion/Extension')),
      ('RightKnee',       ('Abduction/Adduction',              'Internal/External Rotation', 'Flexion/Extension')),
      ('RightAnkle',      ('Abduction/Adduction',              'Internal/External Rotation', 'Dorsiflexion/Plantarflexion')),
      ('RightBallFoot',   ('Abduction/Adduction',              'Internal/External Rotation', 'Flexion/Extension')),
      ('LeftHip',         ('Abduction/Adduction',              'Internal/External Rotation', 'Flexion/Extension')),
      ('LeftKnee',        ('Abduction/Adduction',              'Internal/External Rotation', 'Flexion/Extension')),
      ('LeftAnkle',       ('Abduction/Adduction',              'Internal/External Rotation', 'Dorsiflexion/Plantarflexion')),
      ('LeftBallFoot',    ('Abduction/Adduction',              'Internal/External Rotation', 'Flexion/Extension')),
    ])
    body_joint_rotation_type_ordering = [(key, body_joint_rotation_type_ordering_raw[key]) for key in body_joint_labels]
    fingerLeft_joint_rotation_type_ordering = OrderedDict([
      (key, ('Abduction/Adduction', 'Internal/External Rotation', 'Flexion/Extension'))
       for key in fingerLeft_joint_labels])
    fingerRight_joint_rotation_type_ordering = OrderedDict([
      (key, ('Abduction/Adduction', 'Internal/External Rotation', 'Flexion/Extension'))
       for key in fingerRight_joint_labels])
    
    # Compile the metadata.
    metadata = OrderedDict([
      ('num_segments_body', len(body_segment_labels)),
      ('num_segments_fingers_left', len(fingerLeft_segment_labels)),
      ('num_segments_fingers_right', len(fingerRight_segment_labels)),
      ('num_joints_body', len(body_joint_labels)),
      ('num_joints_fingers_left', len(fingerLeft_joint_labels)),
      ('num_joints_fingers_right', len(fingerRight_joint_labels)),
      ('num_joints_ergonomic', len(ergonomic_joint_labels)),
      ('num_sensors', len(sensor_labels)),
      ('segment_names_body', body_segment_labels),
      ('segment_names_fingers_left', fingerLeft_segment_labels),
      ('segment_names_fingers_right', fingerRight_segment_labels),
      ('joint_names_body', body_joint_labels),
      ('joint_names_fingers_left', fingerLeft_joint_labels),
      ('joint_names_fingers_right', fingerRight_joint_labels),
      ('joint_rotation_order_body', body_joint_rotation_type_ordering),
      ('joint_rotation_order_fingers_left', fingerLeft_joint_rotation_type_ordering),
      ('joint_rotation_order_fingers_right', fingerRight_joint_rotation_type_ordering),
      ('ergonomic_joint_names', ergonomic_joint_labels),
      ('sensor_names', sensor_labels),
      ('foot_contact_names', foot_contact_labels),
      ('joint_segment_connections_body', body_joint_segment_connections),
      ('joint_segment_connections_fingers_left', fingerLeft_joint_segment_connections),
      ('joint_segment_connections_fingers_right', fingerRight_joint_segment_connections),
      ('ergonomic_joint_segment_connections', ergonomic_joint_segment_connections),
      ('segment_mesh_points_body_xyz_cm', body_segment_mesh_points_xyz_cm),
      ('segment_mesh_points_fingers_left_xyz_cm', fingerLeft_segment_mesh_points_xyz_cm),
      ('segment_mesh_points_fingers_right_xyz_cm', fingerRight_segment_mesh_points_xyz_cm),
      ('mvn_version', mvn_version),
      ('mvnx_version', mvnx_version),
    ])
    if len(fingerLeft_segment_labels) == 0:
      for key in list(metadata.keys()):
        if 'fingers_left' in key:
          del metadata[key]
    if len(fingerRight_segment_labels) == 0:
      for key in list(metadata.keys()):
        if 'fingers_right' in key:
          del metadata[key]
    return metadata
    
    # # Record the parent/child segment and point for each streamed joint.
    # # The long lists were copied from a test data stream.
    # joint_parents_segmentIDsPointIDs = [1.002, 2.002, 3.002, 4.002, 5.002, 6.002, 5.003, 8.002, 9.002, 10.002, 5.004, 12.002, 13.002, 14.002, 1.003, 16.002, 17.002, 18.002, 1.004, 20.002, 21.002, 22.002, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0]
    # joint_parents_segmentIDs = [int(x) for x in joint_parents_segmentIDsPointIDs]
    # joint_parents_pointIDs = [round((x - int(x))*1000) for x in joint_parents_segmentIDsPointIDs]
    # joint_children_segmentIDsPointIDs = [2.001, 3.001, 4.001, 5.001, 6.001, 7.001, 8.001, 9.001, 10.001, 11.001, 12.001, 13.001, 14.001, 15.001, 16.001, 17.001, 18.001, 19.001, 20.001, 21.001, 22.001, 23.001, 7.0, 13.0, 9.0, 5.0, 1.0, 5.0]
    # joint_children_segmentIDs = [int(x) for x in joint_children_segmentIDsPointIDs]
    # joint_children_pointIDs = [round((x - int(x))*1000) for x in joint_children_segmentIDsPointIDs]
    # # Convert to dictionaries mapping joint names to segment names and point IDs.
    # #  to avoid dealing with orderings and indexes.
    # # Note that the segment IDs are 1-indexed.
    # joint_parents_segmentIDs = OrderedDict(
    #     [(joint_names_body[i], joint_parent_segmentID)
    #       for (i, joint_parent_segmentID) in enumerate(joint_parents_segmentIDs[0:22])]
    #   + [(joint_names_ergonomic[i], joint_parent_segmentID)
    #       for (i, joint_parent_segmentID) in enumerate(joint_parents_segmentIDs[22:])]
    # )
    # joint_parents_segmentNames = OrderedDict(
    #     [(joint_name, segment_names_body[segmentID-1])
    #       for (joint_name, segmentID) in joint_parents_segmentIDs.items()]
    # )
    # joint_parents_pointIDs = OrderedDict(
    #     [(joint_names_body[i], joint_parent_pointID)
    #       for (i, joint_parent_pointID) in enumerate(joint_parents_pointIDs[0:22])]
    #   + [(joint_names_ergonomic[i], joint_parent_pointID)
    #       for (i, joint_parent_pointID) in enumerate(joint_parents_pointIDs[22:])]
    # )
    # joint_children_segmentIDs = OrderedDict(
    #     [(joint_names_body[i], joint_child_segmentID)
    #       for (i, joint_child_segmentID) in enumerate(joint_children_segmentIDs[0:22])]
    #   + [(joint_names_ergonomic[i], joint_child_segmentID)
    #       for (i, joint_child_segmentID) in enumerate(joint_children_segmentIDs[22:])]
    # )
    # joint_children_segmentNames = OrderedDict(
    #     [(joint_name, segment_names_body[segmentID-1])
    #       for (joint_name, segmentID) in joint_children_segmentIDs.items()]
    # )
    # joint_children_pointIDs = OrderedDict(
    #     [(joint_names_body[i], joint_child_pointID)
    #       for (i, joint_child_pointID) in enumerate(joint_children_pointIDs[0:22])]
    #   + [(joint_names_ergonomic[i], joint_child_pointID)
    #       for (i, joint_child_pointID) in enumerate(joint_children_pointIDs[22:])]
    # )
    
  # def _define_data_notes(self):
  #   metadata = self._get_mvnx_metadata()
  #
  #   ######
  #   # Notes for streaming data.
  #   ######
  #
  #   self._data_notes_stream = {}
  #   self._data_notes_stream.setdefault('xsens-segments', {})
  #   self._data_notes_stream.setdefault('xsens-joints', {})
  #   self._data_notes_stream.setdefault('xsens-CoM', {})
  #   self._data_notes_stream.setdefault('xsens-time', {})
  #
  #   # Segments
  #   self._data_notes_stream['xsens-segments']['position_cm'] = OrderedDict([
  #     ('Units', 'cm'),
  #     ('Coordinate frame', 'A Y-up right-handed frame if Euler data is streamed, otherwise a Z-up right-handed frame'),
  #     ('Matrix ordering', 'To align with data headings, unwrap a frame\'s matrix as data[frame_index][0][0], data[frame_index][0][1], data[frame_index][0][2], data[frame_index][1][0], ...' \
  #      + '   | And if no fingers were included in the data, only use the first 69 data headings (the first 23 segments)'),
  #     (self._metadata_data_headings_key, self._headings['xsens-segments']['position_cm']),
  #     ('Segment Names', self._headings['xsens-segments']['segmentNames']),
  #   ])
  #   self._data_notes_stream['xsens-segments']['orientation_euler_deg'] = OrderedDict([
  #     ('Units', 'degrees'),
  #     ('Coordinate frame', 'A Y-Up, right-handed coordinate system'),
  #     ('Matrix ordering', 'To align with data headings, unwrap a frame\'s matrix as data[frame_index][0][0], data[frame_index][0][1], data[frame_index][0][2], data[frame_index][1][0], ...' \
  #      + '   | And if no fingers were included in the data, only use the first 69 data headings (the first 23 segments)'),
  #     (self._metadata_data_headings_key, self._headings['xsens-segments']['orientation_euler_deg']),
  #     ('Segment Names', self._headings['xsens-segments']['segmentNames']),
  #     # ('Developer note', 'Streamed data did not seem to match Excel data exported from Xsens; on recent tests it was close, while on older tests it seemed very different.'),
  #   ])
  #   self._data_notes_stream['xsens-segments']['orientation_quaternion'] = OrderedDict([
  #     ('Coordinate frame', 'A Z-Up, right-handed coordinate system'),
  #     ('Normalization', 'Normalized but not necessarily positive-definite'),
  #     ('Matrix ordering', 'To align with data headings, unwrap a frame\'s matrix as data[frame_index][0][0], data[frame_index][0][1], data[frame_index][0][2], data[frame_index][0][3], data[frame_index][1][0], ...' \
  #      + '   | And if no fingers were included in the data, only use the first 92 data headings (the first 23 segments)'),
  #     (self._metadata_data_headings_key, self._headings['xsens-segments']['orientation_quaternion']),
  #     ('Segment Names', self._headings['xsens-segments']['segmentNames']),
  #   ])
  #   # Joints
  #   self._data_notes_stream['xsens-joints']['rotation_deg'] = OrderedDict([
  #     ('Units', 'degrees'),
  #     ('Coordinate frame', 'A Z-Up, right-handed coordinate system'),
  #     ('Matrix ordering', 'To align with data headings, unwrap a frame\'s matrix as data[frame_index][0][0], data[frame_index][0][1], data[frame_index][0][2], data[frame_index][1][0], ...'),
  #     ('Joint parents - segment IDs',    self._headings['xsens-joints']['joint_rotation_streamed_parents_segmentIDs']['streamed']),
  #     ('Joint parents - segment Names',  self._headings['xsens-joints']['joint_rotation_streamed_parents_segmentNames']['streamed']),
  #     ('Joint parents - point IDs',      self._headings['xsens-joints']['joint_rotation_streamed_parents_pointIDs']['streamed']),
  #     ('Joint children - segment IDs',   self._headings['xsens-joints']['joint_rotation_streamed_children_segmentIDs']['streamed']),
  #     ('Joint children - segment Names', self._headings['xsens-joints']['joint_rotation_streamed_children_segmentNames']['streamed']),
  #     ('Joint children - point IDs',     self._headings['xsens-joints']['joint_rotation_streamed_children_pointIDs']['streamed']),
  #     ('Segment ID to Name mapping', self._headings['xsens-joints']['segmentIDsToNames']),
  #     ('Joint Names', self._headings['xsens-joints']['jointNames']),
  #     (self._metadata_data_headings_key, self._headings['xsens-joints']['joint_rotation_names_streamed'])
  #   ])
  #   self._data_notes_stream['xsens-joints']['parent'] = OrderedDict([
  #     ('Format', 'segmentID.pointID'),
  #     ('Segment ID to Name mapping', self._headings['xsens-joints']['segmentIDsToNames']),
  #     (self._metadata_data_headings_key, self._headings['xsens-joints']['joint_names_streamed']),
  #   ])
  #   self._data_notes_stream['xsens-joints']['child'] = OrderedDict([
  #     ('Format', 'segmentID.pointID'),
  #     ('Segment ID to Name mapping', self._headings['xsens-joints']['segmentIDsToNames']),
  #     (self._metadata_data_headings_key, self._headings['xsens-joints']['joint_names_streamed'])
  #   ])
  #   # Center of mass
  #   self._data_notes_stream['xsens-CoM']['position_cm'] = OrderedDict([
  #     ('Units', 'cm'),
  #     ('Coordinate frame', 'A Z-up, right-handed coordinate system'),
  #     (self._metadata_data_headings_key, self._headings['xsens-CoM']['position_cm']),
  #     ('Joint Names', self._headings['xsens-joints']['jointNames']),
  #   ])
  #   self._data_notes_stream['xsens-CoM']['velocity_cm_s'] = OrderedDict([
  #     ('Units', 'cm/s'),
  #     ('Coordinate frame', 'A Z-up, right-handed coordinate system'),
  #     (self._metadata_data_headings_key, self._headings['xsens-CoM']['velocity_cm_s']),
  #     ('Joint Names', self._headings['xsens-joints']['jointNames']),
  #   ])
  #   self._data_notes_stream['xsens-CoM']['acceleration_cm_ss'] = OrderedDict([
  #     ('Units', 'cm/s/s'),
  #     ('Coordinate frame', 'A Z-up, right-handed coordinate system'),
  #     (self._metadata_data_headings_key, self._headings['xsens-CoM']['acceleration_cm_ss']),
  #     ('Joint Names', self._headings['xsens-joints']['jointNames']),
  #   ])
  #   # Time
  #   self._data_notes_stream['xsens-time']['device_timestamp_s'] = OrderedDict([
  #     ('Description', 'The timestamp recorded by the Xsens device, which is more precise than the system time when the data was received (the time_s field)'),
  #   ])
  #
  #   ######
  #   # Notes for data imported from Excel exports.
  #   ######
  #
  #   self._data_notes_excel = {}
  #   self._data_notes_excel.setdefault('xsens-segments', {})
  #   self._data_notes_excel.setdefault('xsens-joints', {})
  #   self._data_notes_excel.setdefault('xsens-ergonomic-joints', {})
  #   self._data_notes_excel.setdefault('xsens-CoM', {})
  #   self._data_notes_excel.setdefault('xsens-sensors', {})
  #   self._data_notes_excel.setdefault('xsens-time', {})
  #
  #   # Segments
  #   self._data_notes_excel['xsens-segments']['position_cm'] = self._data_notes_stream['xsens-segments']['position_cm'].copy()
  #   self._data_notes_excel['xsens-segments']['position_cm']['Coordinate frame'] = 'A Z-up right-handed frame'
  #
  #   self._data_notes_excel['xsens-segments']['velocity_cm_s'] = self._data_notes_excel['xsens-segments']['position_cm'].copy()
  #   self._data_notes_excel['xsens-segments']['velocity_cm_s']['Units'] = 'cm/s'
  #   self._data_notes_excel['xsens-segments']['velocity_cm_s']['Matrix ordering'] = 'To align with data headings, unwrap a frame\'s matrix as data[frame_index][0][0], data[frame_index][0][1], data[frame_index][0][2], data[frame_index][1][0], ...' \
  #    + '   | And only use the first 69 data headings (the first 23 segments)'
  #
  #   self._data_notes_excel['xsens-segments']['acceleration_cm_ss'] = self._data_notes_excel['xsens-segments']['velocity_cm_s'].copy()
  #   self._data_notes_excel['xsens-segments']['acceleration_cm_ss']['Units'] = 'cm/s/s'
  #
  #   self._data_notes_excel['xsens-segments']['angular_velocity_deg_s'] = self._data_notes_excel['xsens-segments']['velocity_cm_s'].copy()
  #   self._data_notes_excel['xsens-segments']['angular_velocity_deg_s']['Units'] = 'degrees/s'
  #
  #   self._data_notes_excel['xsens-segments']['angular_acceleration_deg_ss'] = self._data_notes_excel['xsens-segments']['velocity_cm_s'].copy()
  #   self._data_notes_excel['xsens-segments']['angular_acceleration_deg_ss']['Units'] = 'degrees/s/s'
  #
  #   self._data_notes_excel['xsens-segments']['orientation_euler_deg'] = self._data_notes_stream['xsens-segments']['orientation_euler_deg'].copy()
  #   self._data_notes_excel['xsens-segments']['orientation_quaternion'] = self._data_notes_stream['xsens-segments']['orientation_quaternion'].copy()
  #
  #   # Joints
  #   self._data_notes_excel['xsens-joints']['rotation_zxy_deg'] = self._data_notes_stream['xsens-joints']['rotation_deg'].copy()
  #   self._data_notes_excel['xsens-joints']['rotation_zxy_deg'][self._metadata_data_headings_key] = self._headings['xsens-joints']['joint_rotation_names_bodyFingers']
  #   self._data_notes_excel['xsens-joints']['rotation_zxy_deg']['Matrix ordering'] = 'To align with data headings, unwrap a frame\'s matrix as data[frame_index][0][0], data[frame_index][0][1], data[frame_index][0][2], data[frame_index][1][0], ...' \
  #           + '   | And if no fingers were included in the data, only use the first 66 data headings (the first 22 joints)'
  #   self._data_notes_excel['xsens-joints']['rotation_zxy_deg']['Joint parents - segment IDs']    = self._headings['xsens-joints']['joint_rotation_streamed_parents_segmentIDs']['body']
  #   self._data_notes_excel['xsens-joints']['rotation_zxy_deg']['Joint parents - segment Names']  = self._headings['xsens-joints']['joint_rotation_streamed_parents_segmentNames']['body']
  #   self._data_notes_excel['xsens-joints']['rotation_zxy_deg']['Joint parents - point IDs']      = self._headings['xsens-joints']['joint_rotation_streamed_parents_pointIDs']['body']
  #   self._data_notes_excel['xsens-joints']['rotation_zxy_deg']['Joint children - segment IDs']   = self._headings['xsens-joints']['joint_rotation_streamed_children_segmentIDs']['body']
  #   self._data_notes_excel['xsens-joints']['rotation_zxy_deg']['Joint children - segment Names'] = self._headings['xsens-joints']['joint_rotation_streamed_children_segmentNames']['body']
  #   self._data_notes_excel['xsens-joints']['rotation_zxy_deg']['Joint children - point IDs']     = self._headings['xsens-joints']['joint_rotation_streamed_children_pointIDs']['body']
  #
  #   self._data_notes_excel['xsens-joints']['rotation_xzy_deg'] = self._data_notes_excel['xsens-joints']['rotation_zxy_deg'].copy()
  #
  #   self._data_notes_excel['xsens-ergonomic-joints']['rotation_zxy_deg'] = self._data_notes_excel['xsens-joints']['rotation_zxy_deg'].copy()
  #   self._data_notes_excel['xsens-ergonomic-joints']['rotation_zxy_deg']['Matrix ordering'] = 'To align with data headings, unwrap a frame\'s matrix as data[frame_index][0][0], data[frame_index][0][1], data[frame_index][0][2], data[frame_index][1][0], ...'
  #   self._data_notes_excel['xsens-ergonomic-joints']['rotation_zxy_deg'][self._metadata_data_headings_key] = self._headings['xsens-joints']['joint_rotation_names_ergonomic']
  #   self._data_notes_excel['xsens-ergonomic-joints']['rotation_zxy_deg']['Joint parents - segment IDs']    = self._headings['xsens-joints']['joint_rotation_streamed_parents_segmentIDs']['ergonomic']
  #   self._data_notes_excel['xsens-ergonomic-joints']['rotation_zxy_deg']['Joint parents - segment Names']  = self._headings['xsens-joints']['joint_rotation_streamed_parents_segmentNames']['ergonomic']
  #   self._data_notes_excel['xsens-ergonomic-joints']['rotation_zxy_deg']['Joint parents - point IDs']      = self._headings['xsens-joints']['joint_rotation_streamed_parents_pointIDs']['ergonomic']
  #   self._data_notes_excel['xsens-ergonomic-joints']['rotation_zxy_deg']['Joint children - segment IDs']   = self._headings['xsens-joints']['joint_rotation_streamed_children_segmentIDs']['ergonomic']
  #   self._data_notes_excel['xsens-ergonomic-joints']['rotation_zxy_deg']['Joint children - segment Names'] = self._headings['xsens-joints']['joint_rotation_streamed_children_segmentNames']['ergonomic']
  #   self._data_notes_excel['xsens-ergonomic-joints']['rotation_zxy_deg']['Joint children - point IDs']     = self._headings['xsens-joints']['joint_rotation_streamed_children_pointIDs']['ergonomic']
  #
  #   self._data_notes_excel['xsens-ergonomic-joints']['rotation_xzy_deg'] = self._data_notes_excel['xsens-ergonomic-joints']['rotation_zxy_deg'].copy()
  #
  #   # Center of mass
  #   self._data_notes_excel['xsens-CoM']['position_cm'] = self._data_notes_stream['xsens-CoM']['position_cm'].copy()
  #   self._data_notes_excel['xsens-CoM']['velocity_cm_s'] = self._data_notes_stream['xsens-CoM']['velocity_cm_s'].copy()
  #   self._data_notes_excel['xsens-CoM']['acceleration_cm_ss'] = self._data_notes_stream['xsens-CoM']['acceleration_cm_ss'].copy()
  #
  #   # Sensors
  #   self._data_notes_excel['xsens-sensors']['free_acceleration_cm_ss'] = self._data_notes_stream['xsens-segments']['position_cm'].copy()
  #   self._data_notes_excel['xsens-sensors']['free_acceleration_cm_ss']['Matrix ordering'] = 'To align with data headings, unwrap a frame\'s matrix as data[frame_index][0][0], data[frame_index][0][1], data[frame_index][0][2], data[frame_index][1][0], ...' \
  #       + '   | And only use data headings for which the data is not all 0 or all NaN'
  #   self._data_notes_excel['xsens-sensors']['free_acceleration_cm_ss']['Units'] = 'cm/s/s'
  #   del self._data_notes_excel['xsens-sensors']['free_acceleration_cm_ss']['Coordinate frame']
  #   self._data_notes_excel['xsens-sensors']['magnetic_field'] = self._data_notes_excel['xsens-sensors']['free_acceleration_cm_ss'].copy()
  #   self._data_notes_excel['xsens-sensors']['magnetic_field']['Units'] = 'a.u. according to the manual, but more likely gauss based on the magnitudes'
  #
  #   self._data_notes_excel['xsens-sensors']['orientation_quaternion'] = self._data_notes_stream['xsens-segments']['orientation_quaternion'].copy()
  #   self._data_notes_excel['xsens-sensors']['orientation_quaternion']['Matrix ordering'] = 'To align with data headings, unwrap a frame\'s matrix as data[frame_index][0][0], data[frame_index][0][1], data[frame_index][0][2], data[frame_index][0][3], data[frame_index][1][0], ...' \
  #       + '   | And only use data headings for which the data is not all 0 or all NaN'
  #   del self._data_notes_excel['xsens-sensors']['orientation_quaternion']['Coordinate frame']
  #
  #   self._data_notes_excel['xsens-sensors']['orientation_euler_deg'] = self._data_notes_stream['xsens-segments']['orientation_euler_deg'].copy()
  #   self._data_notes_excel['xsens-sensors']['orientation_euler_deg']['Matrix ordering'] = 'To align with data headings, unwrap a frame\'s matrix as data[frame_index][0][0], data[frame_index][0][1], data[frame_index][0][2], data[frame_index][1][0], ...' \
  #       + '   | And only use data headings for which the data is not all 0 or all NaN'
  #   del self._data_notes_excel['xsens-sensors']['orientation_euler_deg']['Coordinate frame']
  #
  #   # Time
  #   self._data_notes_excel['xsens-time']['stream_receive_time_s'] = OrderedDict([
  #     ('Description', 'The estimated system time at which each frame was received by Python during live streaming'),
  #   ])
  #
  #   ######
  #   # Notes for data imported from MVNX exports.
  #   ######
  #   self._data_notes_mvnx = copy.deepcopy(self._data_notes_excel)
  #
  #   # Update the data headings for the sensors.
  #   #  The Excel file contains all segment names and has hidden columns of 0 for ones that don't have sensors,
  #   #  while the MVNX only lists actual sensor locations.
  #   for sensors_key in self._data_notes_mvnx['xsens-sensors'].keys():
  #     if 'quaternion' in sensors_key:
  #       self._data_notes_mvnx['xsens-sensors'][sensors_key][self._metadata_data_headings_key] = self._headings['xsens-sensors']['sensors-quaternion']
  #       self._data_notes_mvnx['xsens-sensors'][sensors_key]['Matrix ordering'] = 'To align with data headings, unwrap a frame\'s matrix as data[frame_index][0][0], data[frame_index][0][1], data[frame_index][0][2], data[frame_index][0][3], data[frame_index][1][0], ...'
  #     else:
  #       self._data_notes_mvnx['xsens-sensors'][sensors_key][self._metadata_data_headings_key] = self._headings['xsens-sensors']['sensors-xyz']
  #       self._data_notes_mvnx['xsens-sensors'][sensors_key]['Matrix ordering'] = 'To align with data headings, unwrap a frame\'s matrix as data[frame_index][0][0], data[frame_index][0][1], data[frame_index][0][2], data[frame_index][1][0], ...'
  #
  #   # Foot contacts
  #   self._data_notes_mvnx.setdefault('xsens-foot-contacts', {})
  #   self._data_notes_mvnx['xsens-foot-contacts']['foot-contacts'] = OrderedDict([
  #     ('Description', 'Which points of the foot are estimated to be in contact with the ground'),
  #     (self._metadata_data_headings_key, self._headings['xsens-foot-contacts']['foot-contacts']),
  #   ])

##############################################

if __name__ == '__main__':

  # Specify the folder of experiments to parse.
  data_dir = os.path.realpath(os.path.join(script_dir, '..', '..', '..', '..', 'data'))
  # experiments_dir = os.path.join(data_dir, 'tests', '2023-08-31_experiment_S00_xsens_joint_angles')
  # experiments_dir = os.path.join(data_dir, 'experiments', '2023-09-10_xsens_controlled_rotations_S00')
  experiments_dir = os.path.join(data_dir, 'experiments', '2023-09-10_xsens_controlled_rotations_S11')
  output_dir = experiments_dir
  os.makedirs(output_dir, exist_ok=True)
  
  mvnx_filepaths = glob.glob(os.path.join(experiments_dir, '*.mvnx'))

  xsens_converter = XsensConverter()
  xsens_converter.convert_mvnx_files(mvnx_filepaths)
  # xsens_converter.convert_mvnx_files(mvnx_filepaths[0])
  # xsens_converter.convert_mvnx_files('C:/Users/jdelp/Desktop/ActionSense/data/experiments/2023-08-18_experiment_S10/2023-08-18_20-50-18_actionNet-wearables_S10/externally_recorded_data/xsens/New Session-001_HD.mvnx')
