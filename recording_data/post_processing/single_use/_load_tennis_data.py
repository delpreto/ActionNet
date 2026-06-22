
import h5py
import numpy as np
import os

hdf5_filepath = r'P:\MIT\Lab\Wearativity\data\tests\2026-06-14_tennis_S02\2026-06-14_tennis_S02_xsens_myo_data_01.hdf5'
# hdf5_filepath = r'P:\MIT\Lab\Wearativity\data\tests\2026-06-14_tennis_S02\2026-06-14_tennis_S02_xsens_myo_data_02.hdf5'
# hdf5_filepath = '2026-06-14_tennis_S02_xsens_myo_data_01.hdf5'
# hdf5_filepath = '2026-06-14_tennis_S02_xsens_myo_data_02.hdf5'

#####################################################
# Define metadata about how to interpret the matrices.
#####################################################

# Define the sequence of Xsens body (and racket) segments.
xsens_segment_labels = [
      'Pelvis', 'L5', 'L3', 'T12', 'T8', 'Neck', 'Head',
      'Right Shoulder',  'Right Upper Arm', 'Right Forearm', 'Right Hand',
      'Left Shoulder',   'Left Upper Arm',  'Left Forearm',  'Left Hand',
      'Right Upper Leg', 'Right Lower Leg', 'Right Foot',    'Right Toe',
      'Left Upper Leg',  'Left Lower Leg',  'Left Foot',     'Left Toe',
      'Racket',
      ]

# To help describe where each segment is located,
# define chains that would create a skeleton if visualized.
xsens_segment_chains_labels_toPlot = {
  'Left Leg':  ['Left Upper Leg', 'Left Lower Leg', 'Left Foot', 'Left Toe'],
  'Right Leg': ['Right Upper Leg', 'Right Lower Leg', 'Right Foot', 'Right Toe'],
  'Spine':     ['Head', 'Neck', 'T8', 'T12', 'L3', 'L5', 'Pelvis'], # top down
  'Hip':       ['Left Upper Leg', 'Pelvis', 'Right Upper Leg'],
  'Shoulders': ['Left Upper Arm', 'Left Shoulder', 'Right Shoulder', 'Right Upper Arm'],
  'Left Arm':  ['Left Upper Arm', 'Left Forearm', 'Left Hand'],
  'Right Arm': ['Right Upper Arm', 'Right Forearm', 'Right Hand'],
  'Prop 1':    ['Right Hand', 'Prop 1'],
}

# Specify where the IMU sensors are placed on the body.
xsens_imu_sensor_labels = [
   'Pelvis', 'T8', 'Head', 
   'Right Shoulder',  'Right Upper Arm', 'Right Forearm', 'Right Hand',
   'Left Shoulder',   'Left Upper Arm',  'Left Forearm',  'Left Hand',
   'Right Upper Leg', 'Right Lower Leg', 'Right Foot',
   'Left Upper Leg',  'Left Lower Leg',  'Left Foot', 
   'Racket'
]

# Specify the list of joint angle names.
xsens_joint_labels = [
      ['L5S1 Lateral Bending',    'L5S1 Axial Bending',     'L5S1 Flexion/Extension',],
      ['L4L3 Lateral Bending',    'L4L3 Axial Rotation',    'L4L3 Flexion/Extension',],
      ['L1T12 Lateral Bending',   'L1T12 Axial Rotation',   'L1T12 Flexion/Extension',],
      ['T9T8 Lateral Bending',    'T9T8 Axial Rotation',    'T9T8 Flexion/Extension',],
      ['T1C7 Lateral Bending',    'T1C7 Axial Rotation',    'T1C7 Flexion/Extension',],
      ['C1 Head Lateral Bending', 'C1 Head Axial Rotation', 'C1 Head Flexion/Extension',],
      ['Right T4 Shoulder Abduction/Adduction', 'Right T4 Shoulder Internal/External Rotation', 'Right T4 Shoulder Flexion/Extension',],
      ['Right Shoulder Abduction/Adduction',    'Right Shoulder Internal/External Rotation',    'Right Shoulder Flexion/Extension',],
      ['Right Elbow Ulnar Deviation/Radial Deviation', 'Right Elbow Pronation/Supination', 'Right Elbow Flexion/Extension',],
      ['Right Wrist Ulnar Deviation/Radial Deviation', 'Right Wrist Pronation/Supination', 'Right Wrist Flexion/Extension',],
      ['Left T4 Shoulder Abduction/Adduction', 'Left T4 Shoulder Internal/External Rotation', 'Left T4 Shoulder Flexion/Extension',],
      ['Left Shoulder Abduction/Adduction',    'Left Shoulder Internal/External Rotation',    'Left Shoulder Flexion/Extension',],
      ['Left Elbow Ulnar Deviation/Radial Deviation', 'Left Elbow Pronation/Supination', 'Left Elbow Flexion/Extension',],
      ['Left Wrist Ulnar Deviation/Radial Deviation', 'Left Wrist Pronation/Supination', 'Left Wrist Flexion/Extension',],
      ['Right Hip Abduction/Adduction',       'Right Hip Internal/External Rotation',       'Right Hip Flexion/Extension',],
      ['Right Knee Abduction/Adduction',      'Right Knee Internal/External Rotation',      'Right Knee Flexion/Extension',],
      ['Right Ankle Abduction/Adduction',     'Right Ankle Internal/External Rotation',     'Right Ankle Dorsiflexion/Plantarflexion',],
      ['Right Ball Foot Abduction/Adduction', 'Right Ball Foot Internal/External Rotation', 'Right Ball Foot Flexion/Extension',],
      ['Left Hip Abduction/Adduction',        'Left Hip Internal/External Rotation',        'Left Hip Flexion/Extension',],
      ['Left Knee Abduction/Adduction',       'Left Knee Internal/External Rotation',       'Left Knee Flexion/Extension',],
      ['Left Ankle Abduction/Adduction',      'Left Ankle Internal/External Rotation',      'Left Ankle Dorsiflexion/Plantarflexion',],
      ['Left Ball Foot Abduction/Adduction',  'Left Ball Foot Internal/External Rotation',  'Left Ball Foot Flexion/Extension',],
      ['Prop X', 'Prop Y', 'Prop Z'], # Not sure exactly what this joint represents at the moment - will need to investigate
    ]

# Specify the ergonomic joints which are virtual "joints" defined by Xsens.
xsens_ergonomic_joint_labels = [
      'T8_Head',
      'T8_LeftUpperArm',
      'T8_RightUpperArm',
      'Pelvis_T8',
      'Vertical_Pelvis',
      'Vertical_T8',
    ]

#####################################################
# Helpers
#####################################################
def print_hdf5_structure(h5_file):
    """
    Recursively prints the group/dataset hierarchy, shape, and size of an HDF5 file.
    """
    def inspect_node(name, node):
      # Calculate depth for tree indentation
      indent = "  " * (name.count("/")) + "|-- "
      if isinstance(node, h5py.Dataset):
        size_mb = node.id.get_storage_size() / (1024 * 1024)
        print(f"{indent}{name.split('/')[-1]} (Dataset) | Shape: {node.shape} | Size: {size_mb:.2f} MB")
      elif isinstance(node, h5py.Group):
        print(f"{indent}{name.split('/')[-1]} (Group)")
    # Recursively visit every group and dataset
    h5_file.visititems(inspect_node)


#####################################################
# Open the HDF5 file and print its structure.
#####################################################

h5_file = h5py.File(hdf5_filepath, 'r')

# The h5 file can be treated like a dictionary, with keys representing groups (folders) and datasets.
# Datasets are kept on disk until needed, rather than requiring large amounts of RAM for large datasets.
# To load a dataset into RAM, an easy way is to just cast it to a numpy array.
h5_toplevel_groups = h5_file.keys()
sample_datasset = np.array(h5_file['myo-right']['emg']['data'])

print()
print('='*75)
print('Contents of the HDF5 file: %s' % os.path.basename(hdf5_filepath))
print_hdf5_structure(h5_file)
print('='*75)
print()


#####################################################
# Load Xsens data.
#####################################################

print()
print('='*75)
print('Loading Xsens data')

# Load the array of timestamps.
# Each stream has its own timestamp array in the HDF5 file, but they should all be the same.
# time_s contains epoch timestamps, which is the number of seconds since January 1, 1970, 00:00:00 UTC.
#   A handy tool for converting individual dates is at https://www.epochconverter.com/
# time_str contains human-readable date strings.
print('  Loading timestamp data')
xsens_time_s = np.squeeze(h5_file['xsens-CoM']['position_xyz_m']['time_s'])
xsens_time_str = [time_str.decode('utf-8') for time_str in np.squeeze(h5_file['xsens-CoM']['position_xyz_m']['time_str'])]
xsens_duration_s = xsens_time_s[-1] - xsens_time_s[0]
print('    Start epoch time: %0.3f' % xsens_time_s[0])
print('    End epoch time  : %0.3f' % xsens_time_s[-1])
print('    Start timestamp : %s' % xsens_time_str[0])
print('    End timestamp   : %s' % xsens_time_str[-1])
print('    Duration        : %0.3f s = %0.3f min' % (xsens_duration_s, xsens_duration_s/60))
print('    Sampling rate   : %0.2f Hz' % ((len(xsens_time_s)-1)/xsens_duration_s))

# Load the original frame numbers recorded by Xsens.
# These may be helpful if using the BVH format.
xsens_frame_numers = np.squeeze(h5_file['xsens-CoM']['position_xyz_m']['xsens_sample_number'])


# Load center of mass data.
print('  Loading center of mass data')
xsens_com = {
   'position_xyz_m': np.array(h5_file['xsens-CoM']['position_xyz_m']['data']),
   'velocity_xyz_m_s': np.array(h5_file['xsens-CoM']['velocity_xyz_m_s']['data']),
   'acceleration_xyz_m_ss': np.array(h5_file['xsens-CoM']['acceleration_xyz_m_ss']['data']),
}
assert np.array_equal(xsens_time_s, np.squeeze(h5_file['xsens-CoM']['position_xyz_m']['time_s']))
assert np.array_equal(xsens_frame_numers, np.squeeze(h5_file['xsens-CoM']['position_xyz_m']['xsens_sample_number']))


# Load joint angle data.
# Each frame has a 23 x 3 matrix, which correspond to the labels in "joint_rotation_names_body" defined above.
# They are available as Euler angles, computed using XZY or ZXY conventions as described in https://base.xsens.com/s/article/Euler-sequences-in-joint-angles-Gimbal-lock?language=en_US
#   Xsens recommends using XZY for joints that are primarily abduction/adduction,
#   and using ZXY for joints that are primarily flexion/extension.
print('  Loading joint angle data')
xsens_joints = {
   'eulerXZY_xyz_rad': np.array(h5_file['xsens-joints']['body_joint_angles_eulerXZY_xyz_rad']['data']),
   'eulerZXY_xyz_rad': np.array(h5_file['xsens-joints']['body_joint_angles_eulerZXY_xyz_rad']['data']),
}
assert np.array_equal(xsens_time_s, np.squeeze(h5_file['xsens-joints']['body_joint_angles_eulerXZY_xyz_rad']['time_s']))
assert np.array_equal(xsens_frame_numers, np.squeeze(h5_file['xsens-joints']['body_joint_angles_eulerXZY_xyz_rad']['xsens_sample_number']))

# Load ergonomic joint angles, which are virtual joints defined by Xsens to represent body ergonomics.
# The connections are specified above in "joint_names_ergonomic".
# Joint angles are available as Euler angles, computed using XZY or ZXY conventions as described in https://base.xsens.com/s/article/Euler-sequences-in-joint-angles-Gimbal-lock?language=en_US
#   Xsens recommends using XZY for joints that are primarily abduction/adduction,
#   and using ZXY for joints that are primarily flexion/extension.
xsens_ergonomic_joints = {
   'ergonomic_joint_angles_eulerXZY_xyz_rad': np.array(h5_file['xsens-ergonomic-joints']['ergonomic_joint_angles_eulerXZY_xyz_rad']['data']),
   'ergonomic_joint_angles_eulerZXY_xyz_rad': np.array(h5_file['xsens-ergonomic-joints']['ergonomic_joint_angles_eulerZXY_xyz_rad']['data']),
}
assert np.array_equal(xsens_time_s, np.squeeze(h5_file['xsens-ergonomic-joints']['ergonomic_joint_angles_eulerXZY_xyz_rad']['time_s']))
assert np.array_equal(xsens_frame_numers, np.squeeze(h5_file['xsens-ergonomic-joints']['ergonomic_joint_angles_eulerXZY_xyz_rad']['xsens_sample_number']))


# Load body segment position, orientation, and motion data.
# Each frame has a 24 x 3 matrix, which correspond to the labels in "xsens_segment_labels" defined above.
# Orientations are available as Euler angles and as quaternions.
print('  Loading body segment position, orientation, and motion data')
xsens_body_segments = {
   'position_xyz_m': np.array(h5_file['xsens-segments']['body_position_xyz_m']['data']),
   'velocity_xyz_m_s': np.array(h5_file['xsens-segments']['body_velocity_xyz_m_s']['data']),
   'acceleration_xyz_m_ss': np.array(h5_file['xsens-segments']['body_acceleration_xyz_m_ss']['data']),
   'angular_velocity_xyz_rad_s': np.array(h5_file['xsens-segments']['body_angular_velocity_xyz_rad_s']['data']),
   'angular_acceleration_xyz_rad_ss': np.array(h5_file['xsens-segments']['body_angular_acceleration_xyz_rad_ss']['data']),
   'orientation_eulerZXY_xyz_rad': np.array(h5_file['xsens-segments']['body_orientation_eulerZXY_xyz_rad']['data']),
   'orientation_quaternion_wijk': np.array(h5_file['xsens-segments']['body_orientation_quaternion_wijk']['data']),
}
assert np.array_equal(xsens_time_s, np.squeeze(h5_file['xsens-segments']['body_acceleration_xyz_m_ss']['time_s']))
assert np.array_equal(xsens_frame_numers, np.squeeze(h5_file['xsens-segments']['body_acceleration_xyz_m_ss']['xsens_sample_number']))


# Load T-Pose data (standing with legs neutral, looking straight ahead, arms outstetched horizontally at the sides to form a T).
# This is used by some programs for calibration and whatnot.
xsens_tpose = {
   'body_orientation_TposeISB_quaternion_wijk': np.array(h5_file['xsens-segments-tpose']['body_orientation_TposeISB_quaternion_wijk']),
   'body_orientation_Tpose_quaternion_wijk': np.array(h5_file['xsens-segments-tpose']['body_orientation_Tpose_quaternion_wijk']),
   'body_orientation_identity_quaternion_wijk': np.array(h5_file['xsens-segments-tpose']['body_orientation_identity_quaternion_wijk']),
   'body_position_TposeISB_xyz_m': np.array(h5_file['xsens-segments-tpose']['body_position_TposeISB_xyz_m']),
   'body_position_Tpose_xyz_m': np.array(h5_file['xsens-segments-tpose']['body_position_Tpose_xyz_m']),
   'body_position_identity_xyz_m': np.array(h5_file['xsens-segments-tpose']['body_position_identity_xyz_m']),
}


# Load foot contact data.
print('  Loading heel/toe contact data')
is_contacting_ground = np.array(h5_file['xsens-foot-contacts']['is_contacting_ground']['data'])
xsens_foot_contacts = {
   'left_heel':  is_contacting_ground[:, 0],
   'left_toe':   is_contacting_ground[:, 1],
   'right_heel': is_contacting_ground[:, 2],
   'right_toe':  is_contacting_ground[:, 3],
}
assert np.array_equal(xsens_time_s, np.squeeze(h5_file['xsens-foot-contacts']['is_contacting_ground']['time_s']))
assert np.array_equal(xsens_frame_numers, np.squeeze(h5_file['xsens-foot-contacts']['is_contacting_ground']['xsens_sample_number']))


# Load IMU data from the sensors themselves.
# Each frame has a 18 x 3 matrix, where the 18 sensors are specified in "xsens_imu_sensor_labels" above.
print('  Loading sensor IMU data')
xsens_imu_sensors = {
   'free_acceleration_xyz_m_ss': np.array(h5_file['xsens-sensors']['free_acceleration_xyz_m_ss']['data']),
   'magnetic_field_xyz_au': np.array(h5_file['xsens-sensors']['magnetic_field_xyz_au']['data']),
   'orientation_eulerZXY_xyz_rad': np.array(h5_file['xsens-sensors']['sensor_orientation_eulerZXY_xyz_rad']['data']),
   'orientation_quaternion_wijk': np.array(h5_file['xsens-sensors']['sensor_orientation_quaternion_wijk']['data']),
}
assert np.array_equal(xsens_time_s, np.squeeze(h5_file['xsens-sensors']['free_acceleration_xyz_m_ss']['time_s']))
assert np.array_equal(xsens_frame_numers, np.squeeze(h5_file['xsens-sensors']['free_acceleration_xyz_m_ss']['xsens_sample_number']))


#####################################################
# Load Myo data.
#####################################################

print()
print('='*75)
print('Loading Myo data')

# Load the array of timestamps for EMG data.
# time_s contains epoch timestamps, which is the number of seconds since January 1, 1970, 00:00:00 UTC.
#   A handy tool for converting individual dates is at https://www.epochconverter.com/
# time_str contains human-readable date strings.
print('  Loading EMG timestamp data')
myo_emg_time_s = np.squeeze(h5_file['myo-right']['emg']['time_s'])
myo_emg_time_str = [time_str.decode('utf-8') for time_str in np.squeeze(h5_file['myo-right']['emg']['time_str'])]
myo_emg_duration_s = myo_emg_time_s[-1] - myo_emg_time_s[0]
print('    Start epoch time: %0.3f' % myo_emg_time_s[0])
print('    End epoch time  : %0.3f' % myo_emg_time_s[-1])
print('    Start timestamp : %s' % myo_emg_time_str[0])
print('    End timestamp   : %s' % myo_emg_time_str[-1])
print('    Duration        : %0.3f s = %0.3f min' % (myo_emg_duration_s, myo_emg_duration_s/60))
print('    Sampling rate   : %0.2f Hz' % ((len(myo_emg_time_s)-1)/myo_emg_duration_s))

# Load the array of timestamps for IMU data.
# time_s contains epoch timestamps, which is the number of seconds since January 1, 1970, 00:00:00 UTC.
#   A handy tool for converting individual dates is at https://www.epochconverter.com/
# time_str contains human-readable date strings.
print('  Loading IMU timestamp data')
myo_imu_time_s = np.squeeze(h5_file['myo-right']['acceleration_g']['time_s'])
myo_imu_time_str = [time_str.decode('utf-8') for time_str in np.squeeze(h5_file['myo-right']['acceleration_g']['time_str'])]
myo_imu_duration_s = myo_imu_time_s[-1] - myo_imu_time_s[0]
print('    Start epoch time: %0.3f' % myo_imu_time_s[0])
print('    End epoch time  : %0.3f' % myo_imu_time_s[-1])
print('    Start timestamp : %s' % myo_imu_time_str[0])
print('    End timestamp   : %s' % myo_imu_time_str[-1])
print('    Duration        : %0.3f s = %0.3f min' % (myo_imu_duration_s, myo_imu_duration_s/60))
print('    Sampling rate   : %0.2f Hz' % ((len(myo_imu_time_s)-1)/myo_imu_duration_s))


# Load the EMG data.
# Each frame as 8 channels of EMG data.
print('  Loading EMG data')
myo_emg = np.array(h5_file['myo-right']['emg']['data'])


# Load the IMU motion and orientation data.
print('  Loading IMU motion and orientation data')
myo_imu = {
   'acceleration_g': np.array(h5_file['myo-right']['acceleration_g']['data']),
   'angular_velocity_deg_s': np.array(h5_file['myo-right']['orientation_quaternion']['data']),
   'orientation_quaternion': np.array(h5_file['myo-right']['angular_velocity_deg_s']['data']),
}
assert np.array_equal(myo_imu_time_s, np.squeeze(h5_file['myo-right']['orientation_quaternion']['time_s']))
assert np.array_equal(myo_imu_time_s, np.squeeze(h5_file['myo-right']['angular_velocity_deg_s']['time_s']))


#####################################################
# Clean up.
#####################################################

h5_file.close()
print()
print('='*75)
print('Done!')
print()





