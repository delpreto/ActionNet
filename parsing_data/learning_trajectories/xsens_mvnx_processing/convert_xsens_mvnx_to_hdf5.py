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

# Specify the folder of experiments to parse.
data_dir = os.path.realpath(os.path.join(script_dir, '..', '..', '..', 'data'))
experiments_dir = os.path.join(data_dir, 'tests', '2023-08-31_experiment_S00_xsens_joint_angles')
output_dir = experiments_dir
os.makedirs(output_dir, exist_ok=True)

mvnx_filepaths = glob.glob(os.path.join(experiments_dir, '*.mvnx'))

##############################################

class XsensConverter():
  def __init__(self):
    self._metadata_data_headings_key = 'Data headings'
    self._data_notes_mvnx = {}
    self._define_data_notes()
  
  def convert_mvnx_files(self, mvnx_filepaths):
    for mvnx_filepath in mvnx_filepaths[0:]:
      # Load the MVNX data.
      print('Reading and processing %s' % os.path.basename(mvnx_filepath))
      with open(mvnx_filepath, 'r') as fin:
        mvnx_contents = fin.read()
      mvnx_data = BeautifulSoup(mvnx_contents, 'xml')

      # Open an output HDF5 file.
      hdf5_filepath = '%s.hdf5' % os.path.splitext(mvnx_filepath)[0]
      hdf5_file_toUpdate = h5py.File(hdf5_filepath, 'w')
    
      # Extract all of the frames, ignoring the first few that are empty rows.
      frames = mvnx_data.find_all('frame')
      frame_indexes = [frame.get('index') for frame in frames]
      frames = [frame for (i, frame) in enumerate(frames) if frame_indexes[i].isnumeric()]
      frame_indexes = [int(frame_index) for frame_index in frame_indexes if frame_index.isnumeric()]
      num_frames = len(frames)
    
      # Get time information.
      times_since_start_s = [float(frame.get('time'))/1000.0 for frame in frames]
      times_utc_str = [frame.get('tc') for frame in frames]
      times_s = [float(frame.get('ms'))/1000.0 for frame in frames]
      times_str = [get_time_str(time_s, '%Y-%m-%d %H:%M:%S.%f') for time_s in times_s]
    
      # Check that the timestamps monotonically increase.
      times_s_np = np.array(times_s)
      times_s_diffs = np.diff(times_s_np)
      if np.any(times_s_diffs < 0):
        print_var(times_str, 'times_str')
        msg = '\n'*2
        msg += 'x'*75
        msg += 'XsensStreamer aborting merge due to incorrect timestamps in the MVNX file (they are not monotonically increasing)'
        msg += 'x'*75
        msg = '\n'*2
        print(msg)
        continue
    
      # A helper to get a matrix of data for a given tag, such as 'position'.
      def get_tagged_data(tag):
        datas = [frame.find_all(tag)[0] for frame in frames]
        data = np.array([[float(x) for x in data.contents[0].split()] for data in datas])
        return data
    
      # Extract the data!
      segment_positions_body_cm                 = get_tagged_data('position')*100.0 # convert from m to cm
      segment_orientations_body_quaternion      = get_tagged_data('orientation')
      segment_velocities_body_cm_s              = get_tagged_data('velocity')*100.0 # convert from m to cm
      segment_accelerations_body_cm_ss          = get_tagged_data('acceleration')*100.0 # convert from m to cm
      segment_angular_velocities_body_deg_s     = get_tagged_data('angularVelocity')*180.0/np.pi # convert from radians to degrees
      segment_angular_accelerations_body_deg_ss = get_tagged_data('angularAcceleration')*180.0/np.pi # convert from radians to degrees
      foot_contacts                             = get_tagged_data('footContacts')
      sensor_freeAccelerations_cm_ss            = get_tagged_data('sensorFreeAcceleration')*100.0 # convert from m to cm
      sensor_magnetic_fields                    = get_tagged_data('sensorMagneticField')
      sensor_orientations_quaternion            = get_tagged_data('sensorOrientation')
      joint_angles_zxy_body_deg                 = get_tagged_data('jointAngle')
      joint_angles_xzy_body_deg                 = get_tagged_data('jointAngleXZY')
      ergonomic_joint_angles_zxy_deg            = get_tagged_data('jointAngleErgo')
      ergonomic_joint_angles_xzy_deg            = get_tagged_data('jointAngleErgoXZY')
      center_of_mass                            = get_tagged_data('centerOfMass')
      center_of_mass_positions_cm               = center_of_mass[:, 0:3]*100.0 # convert from m to cm
      center_of_mass_velocities_cm_s            = center_of_mass[:, 3:6]*100.0 # convert from m to cm
      center_of_mass_accelerations_cm_ss        = center_of_mass[:, 6:9]*100.0 # convert from m to cm
      try:
        segment_positions_fingersLeft_cm             = get_tagged_data('positionFingersLeft')*100.0
        segment_positions_fingersRight_cm            = get_tagged_data('positionFingersRight')*100.0
        segment_orientations_fingersLeft_quaternion  = get_tagged_data('orientationFingersLeft')
        segment_orientations_fingersRight_quaternion = get_tagged_data('orientationFingersRight')
        joint_angles_zxy_fingersLeft_deg             = get_tagged_data('jointAngleFingersLeft')
        joint_angles_zxy_fingersRight_deg            = get_tagged_data('jointAngleFingersRight')
        joint_angles_xzy_fingersLeft_deg             = get_tagged_data('jointAngleFingersLeftXZY')
        joint_angles_xzy_fingersRight_deg            = get_tagged_data('jointAngleFingersRightXZY')
        segment_positions_all_cm = np.concatenate((segment_positions_body_cm,
                                                   segment_positions_fingersLeft_cm,
                                                   segment_positions_fingersRight_cm),
                                                  axis=1)
        segment_orientations_all_quaternion = np.concatenate((segment_orientations_body_quaternion,
                                                              segment_orientations_fingersLeft_quaternion,
                                                              segment_orientations_fingersRight_quaternion),
                                                             axis=1)
        joint_angles_zxy_all_deg = np.concatenate((joint_angles_zxy_body_deg,
                                                   joint_angles_zxy_fingersLeft_deg,
                                                   joint_angles_zxy_fingersRight_deg),
                                                  axis=1)
        joint_angles_xzy_all_deg = np.concatenate((joint_angles_xzy_body_deg,
                                                   joint_angles_xzy_fingersLeft_deg,
                                                   joint_angles_xzy_fingersRight_deg),
                                                  axis=1)
      except IndexError: # fingers were not included in the data
        segment_positions_all_cm = segment_positions_body_cm
        segment_orientations_all_quaternion = segment_orientations_body_quaternion
        joint_angles_zxy_all_deg = joint_angles_zxy_body_deg
        joint_angles_xzy_all_deg = joint_angles_xzy_body_deg
    
      # Get the number of segments and sensors
      num_segments_body = segment_orientations_body_quaternion.shape[1]/4
      assert num_segments_body == int(num_segments_body)
      num_segments_body = int(num_segments_body)
      num_segments_all = segment_orientations_all_quaternion.shape[1]/4
      assert num_segments_all == int(num_segments_all)
      num_segments_all = int(num_segments_all)
      num_sensors = sensor_orientations_quaternion.shape[1]/4
      assert num_sensors == int(num_sensors)
      num_sensors = int(num_sensors)
    
      # Create Euler orientations from quaternion orientations.
      segment_orientations_all_euler_deg = np.empty([num_frames, 3*num_segments_all])
      for segment_index in range(num_segments_all):
        for frame_index in range(num_frames):
          eulers_deg = euler_from_quaternion(*segment_orientations_all_quaternion[frame_index, (segment_index*4):(segment_index*4+4)])
          segment_orientations_all_euler_deg[frame_index, (segment_index*3):(segment_index*3+3)] = eulers_deg
      sensor_orientations_euler_deg = np.empty([num_frames, 3*num_sensors])
      for sensor_index in range(num_sensors):
        for frame_index in range(num_frames):
          eulers_deg = euler_from_quaternion(*sensor_orientations_quaternion[frame_index, (sensor_index*4):(sensor_index*4+4)])
          sensor_orientations_euler_deg[frame_index, (sensor_index*3):(sensor_index*3+3)] = eulers_deg
    
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
                    stream_name='orientation_quaternion',
                    data=segment_orientations_all_quaternion,
                    target_data_shape_per_frame=[-1, 4], # 4-element quaternion per segment per frame
                    stream_group_metadata=self._data_notes_mvnx['xsens-segments']['orientation_quaternion'])
      # Segment orientation data (Euler)
      add_hdf5_data(device_name='xsens-segments',
                    stream_name='orientation_euler_deg',
                    data=segment_orientations_all_euler_deg,
                    target_data_shape_per_frame=[-1, 3], # 3-element Euler vector per segment per frame
                    stream_group_metadata=self._data_notes_mvnx['xsens-segments']['orientation_euler_deg'])
    
      # Segment position data.
      add_hdf5_data(device_name='xsens-segments',
                    stream_name='position_cm',
                    data=segment_positions_all_cm,
                    target_data_shape_per_frame=[-1, 3], # 3-element x/y/z vector per segment per frame
                    stream_group_metadata=self._data_notes_mvnx['xsens-segments']['position_cm'])
      # Segment velocity data.
      add_hdf5_data(device_name='xsens-segments',
                    stream_name='velocity_cm_s',
                    data=segment_velocities_body_cm_s,
                    target_data_shape_per_frame=[-1, 3], # 3-element x/y/z vector per segment per frame
                    stream_group_metadata=self._data_notes_mvnx['xsens-segments']['velocity_cm_s'])
      # Segment acceleration data.
      add_hdf5_data(device_name='xsens-segments',
                    stream_name='acceleration_cm_ss',
                    data=segment_accelerations_body_cm_ss,
                    target_data_shape_per_frame=[-1, 3], # 3-element x/y/z vector per segment per frame
                    stream_group_metadata=self._data_notes_mvnx['xsens-segments']['acceleration_cm_ss'])
      # Segment angular velocity.
      add_hdf5_data(device_name='xsens-segments',
                    stream_name='angular_velocity_deg_s',
                    data=segment_angular_velocities_body_deg_s,
                    target_data_shape_per_frame=[-1, 3], # 3-element x/y/z vector per segment per frame
                    stream_group_metadata=self._data_notes_mvnx['xsens-segments']['angular_velocity_deg_s'])
      # Segment angular acceleration.
      add_hdf5_data(device_name='xsens-segments',
                    stream_name='angular_acceleration_deg_ss',
                    data=segment_angular_accelerations_body_deg_ss,
                    target_data_shape_per_frame=[-1, 3], # 3-element x/y/z vector per segment per frame
                    stream_group_metadata=self._data_notes_mvnx['xsens-segments']['angular_acceleration_deg_ss'])
    
      # Joint angles ZXY.
      add_hdf5_data(device_name='xsens-joints',
                    stream_name='rotation_zxy_deg',
                    data=joint_angles_zxy_all_deg,
                    target_data_shape_per_frame=[-1, 3], # 3-element vector per segment per frame
                    stream_group_metadata=self._data_notes_mvnx['xsens-joints']['rotation_zxy_deg'])
      # Joint angles XZY.
      add_hdf5_data(device_name='xsens-joints',
                    stream_name='rotation_xzy_deg',
                    data=joint_angles_xzy_all_deg,
                    target_data_shape_per_frame=[-1, 3], # 3-element vector per segment per frame
                    stream_group_metadata=self._data_notes_mvnx['xsens-joints']['rotation_xzy_deg'])
      # Ergonomic joint angles ZXY.
      add_hdf5_data(device_name='xsens-ergonomic-joints',
                    stream_name='rotation_zxy_deg',
                    data=ergonomic_joint_angles_zxy_deg,
                    target_data_shape_per_frame=[-1, 3], # 3-element vector per segment per frame
                    stream_group_metadata=self._data_notes_mvnx['xsens-ergonomic-joints']['rotation_zxy_deg'])
      # Ergonomic joint angles XZY.
      add_hdf5_data(device_name='xsens-ergonomic-joints',
                    stream_name='rotation_xzy_deg',
                    data=ergonomic_joint_angles_xzy_deg,
                    target_data_shape_per_frame=[-1, 3], # 3-element vector per segment per frame
                    stream_group_metadata=self._data_notes_mvnx['xsens-ergonomic-joints']['rotation_xzy_deg'])
      
      # Center of mass position.
      add_hdf5_data(device_name='xsens-CoM',
                    stream_name='position_cm',
                    data=center_of_mass_positions_cm,
                    target_data_shape_per_frame=[3], # 3-element x/y/z vector per frame
                    stream_group_metadata=self._data_notes_mvnx['xsens-CoM']['position_cm'])
      # Center of mass velocity.
      add_hdf5_data(device_name='xsens-CoM',
                    stream_name='velocity_cm_s',
                    data=center_of_mass_velocities_cm_s,
                    target_data_shape_per_frame=[3], # 3-element x/y/z vector per frame
                    stream_group_metadata=self._data_notes_mvnx['xsens-CoM']['velocity_cm_s'])
      # Center of mass acceleration.
      add_hdf5_data(device_name='xsens-CoM',
                    stream_name='acceleration_cm_ss',
                    data=center_of_mass_accelerations_cm_ss,
                    target_data_shape_per_frame=[3], # 3-element x/y/z vector per frame
                    stream_group_metadata=self._data_notes_mvnx['xsens-CoM']['acceleration_cm_ss'])
    
      # Sensor data - acceleration
      add_hdf5_data(device_name='xsens-sensors',
                    stream_name='free_acceleration_cm_ss',
                    data=sensor_freeAccelerations_cm_ss,
                    target_data_shape_per_frame=[-1, 3], # 3-element x/y/z vector per segment per frame
                    stream_group_metadata=self._data_notes_mvnx['xsens-sensors']['free_acceleration_cm_ss'])
      # Sensor data - magnetic field
      add_hdf5_data(device_name='xsens-sensors',
                    stream_name='magnetic_field',
                    data=sensor_magnetic_fields,
                    target_data_shape_per_frame=[-1, 3], # 3-element x/y/z vector per segment per frame
                    stream_group_metadata=self._data_notes_mvnx['xsens-sensors']['magnetic_field'])
      # Sensor data - orientation
      add_hdf5_data(device_name='xsens-sensors',
                    stream_name='orientation_quaternion',
                    data=sensor_orientations_quaternion,
                    target_data_shape_per_frame=[-1, 4], # 4-element quaternion vector per segment per frame
                    stream_group_metadata=self._data_notes_mvnx['xsens-sensors']['orientation_quaternion'])
      # Sensor data - orientation
      add_hdf5_data(device_name='xsens-sensors',
                    stream_name='orientation_euler_deg',
                    data=sensor_orientations_euler_deg,
                    target_data_shape_per_frame=[-1, 3], # 3-element x/y/z vector per segment per frame
                    stream_group_metadata=self._data_notes_mvnx['xsens-sensors']['orientation_euler_deg'])
    
      # Foot contacts
      add_hdf5_data(device_name='xsens-foot-contacts',
                    stream_name='foot_contact_points',
                    data=foot_contacts,
                    target_data_shape_per_frame=[4], # 4-element contact vector per segment per frame
                    stream_group_metadata=self._data_notes_mvnx['xsens-foot-contacts']['foot-contacts'])

      hdf5_file_toUpdate.close()
  

  ##############################################################
  ##############################################################
  
  def _define_data_headings(self):
      segment_names_body = [
        # Main-body segments
        'Pelvis', 'L5', 'L3', 'T12', 'T8', 'Neck', 'Head',
        'Right Shoulder',  'Right Upper Arm', 'Right Forearm', 'Right Hand',
        'Left Shoulder',   'Left Upper Arm',  'Left Forearm',  'Left Hand',
        'Right Upper Leg', 'Right Lower Leg', 'Right Foot',    'Right Toe',
        'Left Upper Leg',  'Left Lower Leg',  'Left Foot',     'Left Toe',
        ]
        # Note: props 1-4 would be between body and fingers here if there are any
      segment_names_fingers = [
        # Fingers of left hand
        'Left Carpus',            'Left First Metacarpal',         'Left First Proximal Phalange', 'Left First Distal Phalange',
        'Left Second Metacarpal', 'Left Second Proximal Phalange', 'Left Second Middle Phalange',  'Left Second Distal Phalange',
        'Left Third Metacarpal',  'Left Third Proximal Phalange',  'Left Third Middle Phalange',   'Left Third Distal Phalange',
        'Left Fourth Metacarpal', 'Left Fourth Proximal Phalange', 'Left Fourth Middle Phalange',  'Left Fourth Distal Phalange',
        'Left Fifth Metacarpal',  'Left Fifth Proximal Phalange',  'Left Fifth Middle Phalange',   'Left Fifth Distal Phalange',
        # Fingers of right hand
        'Right Carpus',            'Right First Metacarpal',         'Right First Proximal Phalange', 'Right First Distal Phalange',
        'Right Second Metacarpal', 'Right Second Proximal Phalange', 'Right Second Middle Phalange',  'Right Second Distal Phalange',
        'Right Third Metacarpal',  'Right Third Proximal Phalange',  'Right Third Middle Phalange',   'Right Third Distal Phalange',
        'Right Fourth Metacarpal', 'Right Fourth Proximal Phalange', 'Right Fourth Middle Phalange',  'Right Fourth Distal Phalange',
        'Right Fifth Metacarpal',  'Right Fifth Proximal Phalange',  'Right Fifth Middle Phalange',   'Right Fifth Distal Phalange',
      ]
      sensor_names = segment_names_body[0:1] + segment_names_body[4:5] + segment_names_body[6:18] + segment_names_body[19:22]
      joint_rotation_names_body = [
        'L5S1 Lateral Bending',    'L5S1 Axial Bending',     'L5S1 Flexion/Extension',
        'L4L3 Lateral Bending',    'L4L3 Axial Rotation',    'L4L3 Flexion/Extension',
        'L1T12 Lateral Bending',   'L1T12 Axial Rotation',   'L1T12 Flexion/Extension',
        'T9T8 Lateral Bending',    'T9T8 Axial Rotation',    'T9T8 Flexion/Extension',
        'T1C7 Lateral Bending',    'T1C7 Axial Rotation',    'T1C7 Flexion/Extension',
        'C1 Head Lateral Bending', 'C1 Head Axial Rotation', 'C1 Head Flexion/Extension',
        'Right T4 Shoulder Abduction/Adduction', 'Right T4 Shoulder Internal/External Rotation', 'Right T4 Shoulder Flexion/Extension',
        'Right Shoulder Abduction/Adduction',    'Right Shoulder Internal/External Rotation',    'Right Shoulder Flexion/Extension',
        'Right Elbow Ulnar Deviation/Radial Deviation', 'Right Elbow Pronation/Supination', 'Right Elbow Flexion/Extension',
        'Right Wrist Ulnar Deviation/Radial Deviation', 'Right Wrist Pronation/Supination', 'Right Wrist Flexion/Extension',
        'Left T4 Shoulder Abduction/Adduction', 'Left T4 Shoulder Internal/External Rotation', 'Left T4 Shoulder Flexion/Extension',
        'Left Shoulder Abduction/Adduction',    'Left Shoulder Internal/External Rotation',    'Left Shoulder Flexion/Extension',
        'Left Elbow Ulnar Deviation/Radial Deviation', 'Left Elbow Pronation/Supination', 'Left Elbow Flexion/Extension',
        'Left Wrist Ulnar Deviation/Radial Deviation', 'Left Wrist Pronation/Supination', 'Left Wrist Flexion/Extension',
        'Right Hip Abduction/Adduction',       'Right Hip Internal/External Rotation',       'Right Hip Flexion/Extension',
        'Right Knee Abduction/Adduction',      'Right Knee Internal/External Rotation',      'Right Knee Flexion/Extension',
        'Right Ankle Abduction/Adduction',     'Right Ankle Internal/External Rotation',     'Right Ankle Dorsiflexion/Plantarflexion',
        'Right Ball Foot Abduction/Adduction', 'Right Ball Foot Internal/External Rotation', 'Right Ball Foot Flexion/Extension',
        'Left Hip Abduction/Adduction',        'Left Hip Internal/External Rotation',        'Left Hip Flexion/Extension',
        'Left Knee Abduction/Adduction',       'Left Knee Internal/External Rotation',       'Left Knee Flexion/Extension',
        'Left Ankle Abduction/Adduction',      'Left Ankle Internal/External Rotation',      'Left Ankle Dorsiflexion/Plantarflexion',
        'Left Ball Foot Abduction/Adduction',  'Left Ball Foot Internal/External Rotation',  'Left Ball Foot Flexion/Extension',
        ]
      joint_rotation_names_fingers = [
        'Left First CMC Abduction/Adduction',  'Left First CMC Internal/External Rotation',  'Left First CMC Flexion/Extension',
        'Left First MCP Abduction/Adduction',  'Left First MCP Internal/External Rotation',  'Left First MCP Flexion/Extension',
        'Left IP Abduction/Adduction', 'Left IP Internal/External Rotation', 'Left IP Flexion/Extension',
        'Left Second CMC Abduction/Adduction', 'Left Second CMC Internal/External Rotation', 'Left Second CMC Flexion/Extension',
        'Left Second MCP Abduction/Adduction', 'Left Second MCP Internal/External Rotation', 'Left Second MCP Flexion/Extension',
        'Left Second PIP Abduction/Adduction', 'Left Second PIP Internal/External Rotation', 'Left Second PIP Flexion/Extension',
        'Left Second DIP Abduction/Adduction', 'Left Second DIP Internal/External Rotation', 'Left Second DIP Flexion/Extension',
        'Left Third CMC Abduction/Adduction',  'Left Third CMC Internal/External Rotation',  'Left Third CMC Flexion/Extension',
        'Left Third MCP Abduction/Adduction',  'Left Third MCP Internal/External Rotation',  'Left Third MCP Flexion/Extension',
        'Left Third PIP Abduction/Adduction',  'Left Third PIP Internal/External Rotation',  'Left Third PIP Flexion/Extension',
        'Left Third DIP Abduction/Adduction',  'Left Third DIP Internal/External Rotation',  'Left Third DIP Flexion/Extension',
        'Left Fourth CMC Abduction/Adduction', 'Left Fourth CMC Internal/External Rotation', 'Left Fourth CMC Flexion/Extension',
        'Left Fourth MCP Abduction/Adduction', 'Left Fourth MCP Internal/External Rotation', 'Left Fourth MCP Flexion/Extension',
        'Left Fourth PIP Abduction/Adduction', 'Left Fourth PIP Internal/External Rotation', 'Left Fourth PIP Flexion/Extension',
        'Left Fourth DIP Abduction/Adduction', 'Left Fourth DIP Internal/External Rotation', 'Left Fourth DIP Flexion/Extension',
        'Left Fifth CMC Abduction/Adduction',  'Left Fifth CMC Internal/External Rotation',  'Left Fifth CMC Flexion/Extension',
        'Left Fifth MCP Abduction/Adduction',  'Left Fifth MCP Internal/External Rotation',  'Left Fifth MCP Flexion/Extension',
        'Left Fifth PIP Abduction/Adduction',  'Left Fifth PIP Internal/External Rotation',  'Left Fifth PIP Flexion/Extension',
        'Left Fifth DIP Abduction/Adduction',  'Left Fifth DIP Internal/External Rotation',  'Left Fifth DIP Flexion/Extension',
        'Right First CMC Abduction/Adduction', 'Right First CMC Internal/External Rotation', 'Right First CMC Flexion/Extension',
        'Right First MCP Abduction/Adduction', 'Right First MCP Internal/External Rotation', 'Right First MCP Flexion/Extension',
        'Right IP Abduction/Adduction',         'Right IP Internal/External Rotation',         'Right IP Flexion/Extension',
        'Right Second CMC Abduction/Adduction', 'Right Second CMC Internal/External Rotation', 'Right Second CMC Flexion/Extension',
        'Right Second MCP Abduction/Adduction', 'Right Second MCP Internal/External Rotation', 'Right Second MCP Flexion/Extension',
        'Right Second PIP Abduction/Adduction', 'Right Second PIP Internal/External Rotation', 'Right Second PIP Flexion/Extension',
        'Right Second DIP Abduction/Adduction', 'Right Second DIP Internal/External Rotation', 'Right Second DIP Flexion/Extension',
        'Right Third CMC Abduction/Adduction',  'Right Third CMC Internal/External Rotation',  'Right Third CMC Flexion/Extension',
        'Right Third MCP Abduction/Adduction',  'Right Third MCP Internal/External Rotation',  'Right Third MCP Flexion/Extension',
        'Right Third PIP Abduction/Adduction',  'Right Third PIP Internal/External Rotation',  'Right Third PIP Flexion/Extension',
        'Right Third DIP Abduction/Adduction',  'Right Third DIP Internal/External Rotation',  'Right Third DIP Flexion/Extension',
        'Right Fourth CMC Abduction/Adduction', 'Right Fourth CMC Internal/External Rotation', 'Right Fourth CMC Flexion/Extension',
        'Right Fourth MCP Abduction/Adduction', 'Right Fourth MCP Internal/External Rotation', 'Right Fourth MCP Flexion/Extension',
        'Right Fourth PIP Abduction/Adduction', 'Right Fourth PIP Internal/External Rotation', 'Right Fourth PIP Flexion/Extension',
        'Right Fourth DIP Abduction/Adduction', 'Right Fourth DIP Internal/External Rotation', 'Right Fourth DIP Flexion/Extension',
        'Right Fifth CMC Abduction/Adduction',  'Right Fifth CMC Internal/External Rotation',  'Right Fifth CMC Flexion/Extension',
        'Right Fifth MCP Abduction/Adduction',  'Right Fifth MCP Internal/External Rotation',  'Right Fifth MCP Flexion/Extension',
        'Right Fifth PIP Abduction/Adduction',  'Right Fifth PIP Internal/External Rotation',  'Right Fifth PIP Flexion/Extension',
        'Right Fifth DIP Abduction/Adduction',  'Right Fifth DIP Internal/External Rotation',  'Right Fifth DIP Flexion/Extension',
      ]
      joint_rotation_names_ergonomic = [
        'T8_Head Lateral Bending',          'T8_Head Axial Bending',          'T8_Head Flexion/Extension',
        'T8_LeftUpperArm Lateral Bending',  'T8_LeftUpperArm Axial Bending',  'T8_LeftUpperArm Flexion/Extension',
        'T8_RightUpperArm Lateral Bending', 'T8_RightUpperArm Axial Bending', 'T8_RightUpperArm Flexion/Extension',
        'Pelvis_T8 Lateral Bending',        'Pelvis_T8 Axial Bending',        'Pelvis_T8 Flexion/Extension',
        'Vertical_Pelvis Lateral Bending',  'Vertical_Pelvis Axial Bending',  'Vertical_Pelvis Flexion/Extension',
        'Vertical_T8 Lateral Bending',      'Vertical_T8 Axial Bending',      'Vertical_T8 Flexion/Extension',
      ]
      joint_names_body = [
        'L5S1', 'L4L3', 'L1T12', 'T9T8', 'T1C7', 'C1 Head',
        'Right T4 Shoulder', 'Right Shoulder', 'Right Elbow', 'Right Wrist',
        'Left T4 Shoulder',  'Left Shoulder',  'Left Elbow',  'Left Wrist',
        'Right Hip', 'Right Knee', 'Right Ankle', 'Right Ball Foot',
        'Left Hip',  'Left Knee',  'Left Ankle',  'Left Ball Foot',
        ]
      joint_names_fingers = [
        'Left First CMC',   'Left First MCP',   'Left IP',
        'Left Second CMC',  'Left Second MCP',  'Left Second PIP',  'Left Second DIP',
        'Left Third CMC',   'Left Third MCP',   'Left Third PIP',   'Left Third DIP',
        'Left Fourth CMC',  'Left Fourth MCP',  'Left Fourth PIP',  'Left Fourth DIP',
        'Left Fifth CMC',   'Left Fifth MCP',   'Left Fifth PIP',   'Left Fifth DIP',
        'Right First CMC',  'Right First MCP',  'Right IP',
        'Right Second CMC', 'Right Second MCP', 'Right Second PIP', 'Right Second DIP',
        'Right Third CMC',  'Right Third MCP',  'Right Third PIP',  'Right Third DIP',
        'Right Fourth CMC', 'Right Fourth MCP', 'Right Fourth PIP', 'Right Fourth DIP',
        'Right Fifth CMC',  'Right Fifth MCP',  'Right Fifth PIP',  'Right Fifth DIP',
      ]
      joint_names_ergonomic = [
        'T8_Head',
        'T8_LeftUpperArm',
        'T8_RightUpperArm',
        'Pelvis_T8',
        'Vertical_Pelvis',
        'Vertical_T8',
      ]
      # Record the parent/child segment and point for each streamed joint.
      # The long lists were copied from a test data stream.
      joint_parents_segmentIDsPointIDs = [1.002, 2.002, 3.002, 4.002, 5.002, 6.002, 5.003, 8.002, 9.002, 10.002, 5.004, 12.002, 13.002, 14.002, 1.003, 16.002, 17.002, 18.002, 1.004, 20.002, 21.002, 22.002, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0]
      joint_parents_segmentIDs = [int(x) for x in joint_parents_segmentIDsPointIDs]
      joint_parents_pointIDs = [round((x - int(x))*1000) for x in joint_parents_segmentIDsPointIDs]
      joint_children_segmentIDsPointIDs = [2.001, 3.001, 4.001, 5.001, 6.001, 7.001, 8.001, 9.001, 10.001, 11.001, 12.001, 13.001, 14.001, 15.001, 16.001, 17.001, 18.001, 19.001, 20.001, 21.001, 22.001, 23.001, 7.0, 13.0, 9.0, 5.0, 1.0, 5.0]
      joint_children_segmentIDs = [int(x) for x in joint_children_segmentIDsPointIDs]
      joint_children_pointIDs = [round((x - int(x))*1000) for x in joint_children_segmentIDsPointIDs]
      # Convert to dictionaries mapping joint names to segment names and point IDs.
      #  to avoid dealing with orderings and indexes.
      # Note that the segment IDs are 1-indexed.
      joint_parents_segmentIDs = OrderedDict(
          [(joint_names_body[i], joint_parent_segmentID)
            for (i, joint_parent_segmentID) in enumerate(joint_parents_segmentIDs[0:22])]
        + [(joint_names_ergonomic[i], joint_parent_segmentID)
            for (i, joint_parent_segmentID) in enumerate(joint_parents_segmentIDs[22:])]
      )
      joint_parents_segmentNames = OrderedDict(
          [(joint_name, segment_names_body[segmentID-1])
            for (joint_name, segmentID) in joint_parents_segmentIDs.items()]
      )
      joint_parents_pointIDs = OrderedDict(
          [(joint_names_body[i], joint_parent_pointID)
            for (i, joint_parent_pointID) in enumerate(joint_parents_pointIDs[0:22])]
        + [(joint_names_ergonomic[i], joint_parent_pointID)
            for (i, joint_parent_pointID) in enumerate(joint_parents_pointIDs[22:])]
      )
      joint_children_segmentIDs = OrderedDict(
          [(joint_names_body[i], joint_child_segmentID)
            for (i, joint_child_segmentID) in enumerate(joint_children_segmentIDs[0:22])]
        + [(joint_names_ergonomic[i], joint_child_segmentID)
            for (i, joint_child_segmentID) in enumerate(joint_children_segmentIDs[22:])]
      )
      joint_children_segmentNames = OrderedDict(
          [(joint_name, segment_names_body[segmentID-1])
            for (joint_name, segmentID) in joint_children_segmentIDs.items()]
      )
      joint_children_pointIDs = OrderedDict(
          [(joint_names_body[i], joint_child_pointID)
            for (i, joint_child_pointID) in enumerate(joint_children_pointIDs[0:22])]
        + [(joint_names_ergonomic[i], joint_child_pointID)
            for (i, joint_child_pointID) in enumerate(joint_children_pointIDs[22:])]
      )
      # Foot contact points
      foot_contact_names = ['LeftFoot_Heel', 'LeftFoot_Toe', 'RightFoot_Heel', 'RightFoot_Toe']
  
      self._headings = {}
      # Center of Mass
      self._headings.setdefault('xsens-CoM', {})
      self._headings['xsens-CoM']['position_cm'] = ['x', 'y', 'z']
      self._headings['xsens-CoM']['velocity_cm_s'] = ['x', 'y', 'z']
      self._headings['xsens-CoM']['acceleration_cm_ss'] = ['x', 'y', 'z']
      # Segment orientation - quaternion
      self._headings.setdefault('xsens-segments', {})
      quaternion_elements = ['q0_re', 'q1_i', 'q2_j', 'q3_k']
      self._headings['xsens-segments']['orientation_quaternion'] = \
        ['%s (%s)' % (name, element) for name in (segment_names_body + segment_names_fingers)
                                     for element in quaternion_elements]
      # Segment orientation - Euler
      self._headings.setdefault('xsens-segments', {})
      euler_elements = ['x', 'y', 'z']
      self._headings['xsens-segments']['orientation_euler_deg'] = \
        ['%s (%s)' % (name, element) for name in (segment_names_body + segment_names_fingers)
                                     for element in euler_elements]
      # Segment positions
      self._headings.setdefault('xsens-segments', {})
      position_elements = ['x', 'y', 'z']
      self._headings['xsens-segments']['position_cm'] = \
        ['%s (%s)' % (name, element) for name in (segment_names_body + segment_names_fingers)
                                     for element in position_elements]
      # Segment names
      self._headings['xsens-segments']['segmentNames'] = segment_names_body + segment_names_fingers
      # Sensors
      self._headings.setdefault('xsens-sensors', {})
      sensor_elements = ['x', 'y', 'z']
      self._headings['xsens-sensors']['sensors-xyz'] = \
        ['%s (%s)' % (name, element) for name in (sensor_names)
                                     for element in sensor_elements]
      sensor_elements = ['q0_re', 'q1_i', 'q2_j', 'q3_k']
      self._headings['xsens-sensors']['sensors-quaternion'] = \
        ['%s (%s)' % (name, element) for name in (sensor_names)
                                     for element in sensor_elements]
      # Joint rotation names
      self._headings.setdefault('xsens-joints', {})
      self._headings['xsens-joints']['joint_rotation_names_body'] = joint_rotation_names_body
      self._headings['xsens-joints']['joint_rotation_names_fingers'] = joint_rotation_names_fingers
      self._headings['xsens-joints']['joint_rotation_names_ergonomic'] = joint_rotation_names_ergonomic
      self._headings['xsens-joints']['joint_rotation_names_streamed'] = joint_rotation_names_body + joint_rotation_names_ergonomic
      self._headings['xsens-joints']['joint_rotation_names_bodyFingers'] = joint_rotation_names_body + joint_rotation_names_fingers
      # Joint names (used within the rotation names above)
      self._headings['xsens-joints']['joint_names_body'] = joint_names_body
      self._headings['xsens-joints']['joint_names_fingers'] = joint_names_fingers
      self._headings['xsens-joints']['joint_names_ergonomic'] = joint_names_ergonomic
      self._headings['xsens-joints']['joint_names_streamed'] = joint_names_body + joint_names_ergonomic
      # Joint parent/child segment/point IDs/names
      self._headings.setdefault('xsens-joints', {})
      self._headings['xsens-joints']['joint_rotation_streamed_parents_segmentIDs'] = {
        'streamed' : joint_parents_segmentIDs,
        'body'     : OrderedDict(list(joint_parents_segmentIDs.items())[0:22]),
        'ergonomic': OrderedDict(list(joint_parents_segmentIDs.items())[22:])}
      self._headings['xsens-joints']['joint_rotation_streamed_parents_segmentNames'] = {
        'streamed' : joint_parents_segmentNames,
        'body'     : OrderedDict(list(joint_parents_segmentNames.items())[0:22]),
        'ergonomic': OrderedDict(list(joint_parents_segmentNames.items())[22:])}
      self._headings['xsens-joints']['joint_rotation_streamed_parents_pointIDs'] = {
        'streamed' : joint_parents_pointIDs,
        'body'     : OrderedDict(list(joint_parents_pointIDs.items())[0:22]),
        'ergonomic': OrderedDict(list(joint_parents_pointIDs.items())[22:])}
      self._headings['xsens-joints']['joint_rotation_streamed_children_segmentIDs'] = {
        'streamed' : joint_children_segmentIDs,
        'body'     : OrderedDict(list(joint_children_segmentIDs.items())[0:22]),
        'ergonomic': OrderedDict(list(joint_children_segmentIDs.items())[22:])}
      self._headings['xsens-joints']['joint_rotation_streamed_children_segmentNames'] = {
        'streamed' : joint_children_segmentNames,
        'body'     : OrderedDict(list(joint_children_segmentNames.items())[0:22]),
        'ergonomic': OrderedDict(list(joint_children_segmentNames.items())[22:])}
      self._headings['xsens-joints']['joint_rotation_streamed_children_pointIDs'] = {
        'streamed' : joint_children_pointIDs,
        'body'     : OrderedDict(list(joint_children_pointIDs.items())[0:22]),
        'ergonomic': OrderedDict(list(joint_children_pointIDs.items())[22:])}
      self._headings['xsens-joints']['segmentIDsToNames'] = OrderedDict([(i+1, name) for (i, name) in enumerate(segment_names_body + segment_names_fingers)])
      self._headings['xsens-joints']['jointNames'] = joint_names_body + joint_names_fingers
      # Foot contacts
      self._headings.setdefault('xsens-foot-contacts', {})
      self._headings['xsens-foot-contacts']['foot-contacts'] = foot_contact_names
      
  def _define_data_notes(self):
    self._define_data_headings()
    
    ######
    # Notes for streaming data.
    ######
    
    self._data_notes_stream = {}
    self._data_notes_stream.setdefault('xsens-segments', {})
    self._data_notes_stream.setdefault('xsens-joints', {})
    self._data_notes_stream.setdefault('xsens-CoM', {})
    self._data_notes_stream.setdefault('xsens-time', {})
    
    # Segments
    self._data_notes_stream['xsens-segments']['position_cm'] = OrderedDict([
      ('Units', 'cm'),
      ('Coordinate frame', 'A Y-up right-handed frame if Euler data is streamed, otherwise a Z-up right-handed frame'),
      ('Matrix ordering', 'To align with data headings, unwrap a frame\'s matrix as data[frame_index][0][0], data[frame_index][0][1], data[frame_index][0][2], data[frame_index][1][0], ...' \
       + '   | And if no fingers were included in the data, only use the first 69 data headings (the first 23 segments)'),
      (self._metadata_data_headings_key, self._headings['xsens-segments']['position_cm']),
      ('Segment Names', self._headings['xsens-segments']['segmentNames']),
    ])
    self._data_notes_stream['xsens-segments']['orientation_euler_deg'] = OrderedDict([
      ('Units', 'degrees'),
      ('Coordinate frame', 'A Y-Up, right-handed coordinate system'),
      ('Matrix ordering', 'To align with data headings, unwrap a frame\'s matrix as data[frame_index][0][0], data[frame_index][0][1], data[frame_index][0][2], data[frame_index][1][0], ...' \
       + '   | And if no fingers were included in the data, only use the first 69 data headings (the first 23 segments)'),
      (self._metadata_data_headings_key, self._headings['xsens-segments']['orientation_euler_deg']),
      ('Segment Names', self._headings['xsens-segments']['segmentNames']),
      # ('Developer note', 'Streamed data did not seem to match Excel data exported from Xsens; on recent tests it was close, while on older tests it seemed very different.'),
    ])
    self._data_notes_stream['xsens-segments']['orientation_quaternion'] = OrderedDict([
      ('Coordinate frame', 'A Z-Up, right-handed coordinate system'),
      ('Normalization', 'Normalized but not necessarily positive-definite'),
      ('Matrix ordering', 'To align with data headings, unwrap a frame\'s matrix as data[frame_index][0][0], data[frame_index][0][1], data[frame_index][0][2], data[frame_index][0][3], data[frame_index][1][0], ...' \
       + '   | And if no fingers were included in the data, only use the first 92 data headings (the first 23 segments)'),
      (self._metadata_data_headings_key, self._headings['xsens-segments']['orientation_quaternion']),
      ('Segment Names', self._headings['xsens-segments']['segmentNames']),
    ])
    # Joints
    self._data_notes_stream['xsens-joints']['rotation_deg'] = OrderedDict([
      ('Units', 'degrees'),
      ('Coordinate frame', 'A Z-Up, right-handed coordinate system'),
      ('Matrix ordering', 'To align with data headings, unwrap a frame\'s matrix as data[frame_index][0][0], data[frame_index][0][1], data[frame_index][0][2], data[frame_index][1][0], ...'),
      ('Joint parents - segment IDs',    self._headings['xsens-joints']['joint_rotation_streamed_parents_segmentIDs']['streamed']),
      ('Joint parents - segment Names',  self._headings['xsens-joints']['joint_rotation_streamed_parents_segmentNames']['streamed']),
      ('Joint parents - point IDs',      self._headings['xsens-joints']['joint_rotation_streamed_parents_pointIDs']['streamed']),
      ('Joint children - segment IDs',   self._headings['xsens-joints']['joint_rotation_streamed_children_segmentIDs']['streamed']),
      ('Joint children - segment Names', self._headings['xsens-joints']['joint_rotation_streamed_children_segmentNames']['streamed']),
      ('Joint children - point IDs',     self._headings['xsens-joints']['joint_rotation_streamed_children_pointIDs']['streamed']),
      ('Segment ID to Name mapping', self._headings['xsens-joints']['segmentIDsToNames']),
      ('Joint Names', self._headings['xsens-joints']['jointNames']),
      (self._metadata_data_headings_key, self._headings['xsens-joints']['joint_rotation_names_streamed'])
    ])
    self._data_notes_stream['xsens-joints']['parent'] = OrderedDict([
      ('Format', 'segmentID.pointID'),
      ('Segment ID to Name mapping', self._headings['xsens-joints']['segmentIDsToNames']),
      (self._metadata_data_headings_key, self._headings['xsens-joints']['joint_names_streamed']),
    ])
    self._data_notes_stream['xsens-joints']['child'] = OrderedDict([
      ('Format', 'segmentID.pointID'),
      ('Segment ID to Name mapping', self._headings['xsens-joints']['segmentIDsToNames']),
      (self._metadata_data_headings_key, self._headings['xsens-joints']['joint_names_streamed'])
    ])
    # Center of mass
    self._data_notes_stream['xsens-CoM']['position_cm'] = OrderedDict([
      ('Units', 'cm'),
      ('Coordinate frame', 'A Z-up, right-handed coordinate system'),
      (self._metadata_data_headings_key, self._headings['xsens-CoM']['position_cm']),
      ('Joint Names', self._headings['xsens-joints']['jointNames']),
    ])
    self._data_notes_stream['xsens-CoM']['velocity_cm_s'] = OrderedDict([
      ('Units', 'cm/s'),
      ('Coordinate frame', 'A Z-up, right-handed coordinate system'),
      (self._metadata_data_headings_key, self._headings['xsens-CoM']['velocity_cm_s']),
      ('Joint Names', self._headings['xsens-joints']['jointNames']),
    ])
    self._data_notes_stream['xsens-CoM']['acceleration_cm_ss'] = OrderedDict([
      ('Units', 'cm/s/s'),
      ('Coordinate frame', 'A Z-up, right-handed coordinate system'),
      (self._metadata_data_headings_key, self._headings['xsens-CoM']['acceleration_cm_ss']),
      ('Joint Names', self._headings['xsens-joints']['jointNames']),
    ])
    # Time
    self._data_notes_stream['xsens-time']['device_timestamp_s'] = OrderedDict([
      ('Description', 'The timestamp recorded by the Xsens device, which is more precise than the system time when the data was received (the time_s field)'),
    ])
  
    ######
    # Notes for data imported from Excel exports.
    ######
  
    self._data_notes_excel = {}
    self._data_notes_excel.setdefault('xsens-segments', {})
    self._data_notes_excel.setdefault('xsens-joints', {})
    self._data_notes_excel.setdefault('xsens-ergonomic-joints', {})
    self._data_notes_excel.setdefault('xsens-CoM', {})
    self._data_notes_excel.setdefault('xsens-sensors', {})
    self._data_notes_excel.setdefault('xsens-time', {})
  
    # Segments
    self._data_notes_excel['xsens-segments']['position_cm'] = self._data_notes_stream['xsens-segments']['position_cm'].copy()
    self._data_notes_excel['xsens-segments']['position_cm']['Coordinate frame'] = 'A Z-up right-handed frame'
    
    self._data_notes_excel['xsens-segments']['velocity_cm_s'] = self._data_notes_excel['xsens-segments']['position_cm'].copy()
    self._data_notes_excel['xsens-segments']['velocity_cm_s']['Units'] = 'cm/s'
    self._data_notes_excel['xsens-segments']['velocity_cm_s']['Matrix ordering'] = 'To align with data headings, unwrap a frame\'s matrix as data[frame_index][0][0], data[frame_index][0][1], data[frame_index][0][2], data[frame_index][1][0], ...' \
     + '   | And only use the first 69 data headings (the first 23 segments)'
    
    self._data_notes_excel['xsens-segments']['acceleration_cm_ss'] = self._data_notes_excel['xsens-segments']['velocity_cm_s'].copy()
    self._data_notes_excel['xsens-segments']['acceleration_cm_ss']['Units'] = 'cm/s/s'
    
    self._data_notes_excel['xsens-segments']['angular_velocity_deg_s'] = self._data_notes_excel['xsens-segments']['velocity_cm_s'].copy()
    self._data_notes_excel['xsens-segments']['angular_velocity_deg_s']['Units'] = 'degrees/s'
    
    self._data_notes_excel['xsens-segments']['angular_acceleration_deg_ss'] = self._data_notes_excel['xsens-segments']['velocity_cm_s'].copy()
    self._data_notes_excel['xsens-segments']['angular_acceleration_deg_ss']['Units'] = 'degrees/s/s'
  
    self._data_notes_excel['xsens-segments']['orientation_euler_deg'] = self._data_notes_stream['xsens-segments']['orientation_euler_deg'].copy()
    self._data_notes_excel['xsens-segments']['orientation_quaternion'] = self._data_notes_stream['xsens-segments']['orientation_quaternion'].copy()
    
    # Joints
    self._data_notes_excel['xsens-joints']['rotation_zxy_deg'] = self._data_notes_stream['xsens-joints']['rotation_deg'].copy()
    self._data_notes_excel['xsens-joints']['rotation_zxy_deg'][self._metadata_data_headings_key] = self._headings['xsens-joints']['joint_rotation_names_bodyFingers']
    self._data_notes_excel['xsens-joints']['rotation_zxy_deg']['Matrix ordering'] = 'To align with data headings, unwrap a frame\'s matrix as data[frame_index][0][0], data[frame_index][0][1], data[frame_index][0][2], data[frame_index][1][0], ...' \
            + '   | And if no fingers were included in the data, only use the first 66 data headings (the first 22 joints)'
    self._data_notes_excel['xsens-joints']['rotation_zxy_deg']['Joint parents - segment IDs']    = self._headings['xsens-joints']['joint_rotation_streamed_parents_segmentIDs']['body']
    self._data_notes_excel['xsens-joints']['rotation_zxy_deg']['Joint parents - segment Names']  = self._headings['xsens-joints']['joint_rotation_streamed_parents_segmentNames']['body']
    self._data_notes_excel['xsens-joints']['rotation_zxy_deg']['Joint parents - point IDs']      = self._headings['xsens-joints']['joint_rotation_streamed_parents_pointIDs']['body']
    self._data_notes_excel['xsens-joints']['rotation_zxy_deg']['Joint children - segment IDs']   = self._headings['xsens-joints']['joint_rotation_streamed_children_segmentIDs']['body']
    self._data_notes_excel['xsens-joints']['rotation_zxy_deg']['Joint children - segment Names'] = self._headings['xsens-joints']['joint_rotation_streamed_children_segmentNames']['body']
    self._data_notes_excel['xsens-joints']['rotation_zxy_deg']['Joint children - point IDs']     = self._headings['xsens-joints']['joint_rotation_streamed_children_pointIDs']['body']
  
    self._data_notes_excel['xsens-joints']['rotation_xzy_deg'] = self._data_notes_excel['xsens-joints']['rotation_zxy_deg'].copy()
    
    self._data_notes_excel['xsens-ergonomic-joints']['rotation_zxy_deg'] = self._data_notes_excel['xsens-joints']['rotation_zxy_deg'].copy()
    self._data_notes_excel['xsens-ergonomic-joints']['rotation_zxy_deg']['Matrix ordering'] = 'To align with data headings, unwrap a frame\'s matrix as data[frame_index][0][0], data[frame_index][0][1], data[frame_index][0][2], data[frame_index][1][0], ...'
    self._data_notes_excel['xsens-ergonomic-joints']['rotation_zxy_deg'][self._metadata_data_headings_key] = self._headings['xsens-joints']['joint_rotation_names_ergonomic']
    self._data_notes_excel['xsens-ergonomic-joints']['rotation_zxy_deg']['Joint parents - segment IDs']    = self._headings['xsens-joints']['joint_rotation_streamed_parents_segmentIDs']['ergonomic']
    self._data_notes_excel['xsens-ergonomic-joints']['rotation_zxy_deg']['Joint parents - segment Names']  = self._headings['xsens-joints']['joint_rotation_streamed_parents_segmentNames']['ergonomic']
    self._data_notes_excel['xsens-ergonomic-joints']['rotation_zxy_deg']['Joint parents - point IDs']      = self._headings['xsens-joints']['joint_rotation_streamed_parents_pointIDs']['ergonomic']
    self._data_notes_excel['xsens-ergonomic-joints']['rotation_zxy_deg']['Joint children - segment IDs']   = self._headings['xsens-joints']['joint_rotation_streamed_children_segmentIDs']['ergonomic']
    self._data_notes_excel['xsens-ergonomic-joints']['rotation_zxy_deg']['Joint children - segment Names'] = self._headings['xsens-joints']['joint_rotation_streamed_children_segmentNames']['ergonomic']
    self._data_notes_excel['xsens-ergonomic-joints']['rotation_zxy_deg']['Joint children - point IDs']     = self._headings['xsens-joints']['joint_rotation_streamed_children_pointIDs']['ergonomic']
    
    self._data_notes_excel['xsens-ergonomic-joints']['rotation_xzy_deg'] = self._data_notes_excel['xsens-ergonomic-joints']['rotation_zxy_deg'].copy()
    
    # Center of mass
    self._data_notes_excel['xsens-CoM']['position_cm'] = self._data_notes_stream['xsens-CoM']['position_cm'].copy()
    self._data_notes_excel['xsens-CoM']['velocity_cm_s'] = self._data_notes_stream['xsens-CoM']['velocity_cm_s'].copy()
    self._data_notes_excel['xsens-CoM']['acceleration_cm_ss'] = self._data_notes_stream['xsens-CoM']['acceleration_cm_ss'].copy()
    
    # Sensors
    self._data_notes_excel['xsens-sensors']['free_acceleration_cm_ss'] = self._data_notes_stream['xsens-segments']['position_cm'].copy()
    self._data_notes_excel['xsens-sensors']['free_acceleration_cm_ss']['Matrix ordering'] = 'To align with data headings, unwrap a frame\'s matrix as data[frame_index][0][0], data[frame_index][0][1], data[frame_index][0][2], data[frame_index][1][0], ...' \
        + '   | And only use data headings for which the data is not all 0 or all NaN'
    self._data_notes_excel['xsens-sensors']['free_acceleration_cm_ss']['Units'] = 'cm/s/s'
    del self._data_notes_excel['xsens-sensors']['free_acceleration_cm_ss']['Coordinate frame']
    self._data_notes_excel['xsens-sensors']['magnetic_field'] = self._data_notes_excel['xsens-sensors']['free_acceleration_cm_ss'].copy()
    self._data_notes_excel['xsens-sensors']['magnetic_field']['Units'] = 'a.u. according to the manual, but more likely gauss based on the magnitudes'
    
    self._data_notes_excel['xsens-sensors']['orientation_quaternion'] = self._data_notes_stream['xsens-segments']['orientation_quaternion'].copy()
    self._data_notes_excel['xsens-sensors']['orientation_quaternion']['Matrix ordering'] = 'To align with data headings, unwrap a frame\'s matrix as data[frame_index][0][0], data[frame_index][0][1], data[frame_index][0][2], data[frame_index][0][3], data[frame_index][1][0], ...' \
        + '   | And only use data headings for which the data is not all 0 or all NaN'
    del self._data_notes_excel['xsens-sensors']['orientation_quaternion']['Coordinate frame']
    
    self._data_notes_excel['xsens-sensors']['orientation_euler_deg'] = self._data_notes_stream['xsens-segments']['orientation_euler_deg'].copy()
    self._data_notes_excel['xsens-sensors']['orientation_euler_deg']['Matrix ordering'] = 'To align with data headings, unwrap a frame\'s matrix as data[frame_index][0][0], data[frame_index][0][1], data[frame_index][0][2], data[frame_index][1][0], ...' \
        + '   | And only use data headings for which the data is not all 0 or all NaN'
    del self._data_notes_excel['xsens-sensors']['orientation_euler_deg']['Coordinate frame']
    
    # Time
    self._data_notes_excel['xsens-time']['stream_receive_time_s'] = OrderedDict([
      ('Description', 'The estimated system time at which each frame was received by Python during live streaming'),
    ])
  
    ######
    # Notes for data imported from MVNX exports.
    ######
    self._data_notes_mvnx = copy.deepcopy(self._data_notes_excel)
    
    # Update the data headings for the sensors.
    #  The Excel file contains all segment names and has hidden columns of 0 for ones that don't have sensors,
    #  while the MVNX only lists actual sensor locations.
    for sensors_key in self._data_notes_mvnx['xsens-sensors'].keys():
      if 'quaternion' in sensors_key:
        self._data_notes_mvnx['xsens-sensors'][sensors_key][self._metadata_data_headings_key] = self._headings['xsens-sensors']['sensors-quaternion']
        self._data_notes_mvnx['xsens-sensors'][sensors_key]['Matrix ordering'] = 'To align with data headings, unwrap a frame\'s matrix as data[frame_index][0][0], data[frame_index][0][1], data[frame_index][0][2], data[frame_index][0][3], data[frame_index][1][0], ...'
      else:
        self._data_notes_mvnx['xsens-sensors'][sensors_key][self._metadata_data_headings_key] = self._headings['xsens-sensors']['sensors-xyz']
        self._data_notes_mvnx['xsens-sensors'][sensors_key]['Matrix ordering'] = 'To align with data headings, unwrap a frame\'s matrix as data[frame_index][0][0], data[frame_index][0][1], data[frame_index][0][2], data[frame_index][1][0], ...'
  
    # Foot contacts
    self._data_notes_mvnx.setdefault('xsens-foot-contacts', {})
    self._data_notes_mvnx['xsens-foot-contacts']['foot-contacts'] = OrderedDict([
      ('Description', 'Which points of the foot are estimated to be in contact with the ground'),
      (self._metadata_data_headings_key, self._headings['xsens-foot-contacts']['foot-contacts']),
    ])

##############################################

if __name__ == '__main__':
  xsens_converter = XsensConverter()
  xsens_converter.convert_mvnx_files(mvnx_filepaths)
