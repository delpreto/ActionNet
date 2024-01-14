
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
import os

################################################

# Specify the file with trajectory data.
data_dir = 'C:/Users/jdelp/Desktop/ActionSense/results/learning_trajectories'
input_data_filepath = os.path.join(data_dir, 'pouring_training_data_S00.hdf5')
output_data_filepath = os.path.join(data_dir, '%s_forBaxter.npy' % os.path.splitext(input_data_filepath)[0])

data_file_times_s_column = [30]
data_file_gripper_position_columns_xyz_m = [0,1,2]
data_file_gripper_quaternion_columns_wijk = [9,10,11,12]

def convert_quaternion(quat_xsens_wijk):
  # Do an initial rotation, to make the xsens quat match the example quat used during testing.
  quat_wijk = quat_xsens_wijk
  rotates_by_deg = [
    [0, 0, -180],
    ]
  # Apply the rotations.
  rotation_quat = Rotation.from_quat([quat_wijk[1], quat_wijk[2], quat_wijk[3], quat_wijk[0]])
  for i in range(len(rotates_by_deg)-1, -1, -1):
    rotate_by_deg = rotates_by_deg[i]
    rotation_toApply = Rotation.from_rotvec(np.radians(rotate_by_deg))
    rotation_quat = rotation_quat * rotation_toApply
  ijkw = rotation_quat.as_quat()
  quat_wijk = [ijkw[3], ijkw[0], ijkw[1], ijkw[2]]
  # print(quat_wijk)
  # print()
  
  # Negate the i and j components.
  quat_wijk = [quat_wijk[0], -quat_wijk[1], -quat_wijk[2], quat_wijk[3]]
  
  # Apply the rotations determined during testing.
  rotates_by_deg = [
      [0, 0, 180],
      [0, -90, 0],
      [0, 0, 90],
      [0, 180, 0],
      ]
  # Apply the rotations.
  rotation_quat = Rotation.from_quat([quat_wijk[1], quat_wijk[2], quat_wijk[3], quat_wijk[0]])
  for i in range(len(rotates_by_deg)-1, -1, -1):
    rotate_by_deg = rotates_by_deg[i]
    rotation_toApply = Rotation.from_rotvec(np.radians(rotate_by_deg))
    rotation_quat = rotation_quat * rotation_toApply
  ijkw = rotation_quat.as_quat()
  quat_wijk = [ijkw[3], ijkw[0], ijkw[1], ijkw[2]]
  
  # Return the result.
  return quat_wijk

def convert_position(xyz_xsens_m):
  # A sample home for a human demonstration was [-0.3, 0.25, 0.18]
  #  with x point backwards, y pointing right, and z pointing up.
  # Baxter has a home position of about [0.9, -0.4, 0]
  #  with x pointing forwards, y pointing left, and z pointing down.
  # Human and Baxter are rotated 180 about z (x and y are negated)
  return [
    -xyz_xsens_m[0] + 0.6,
    -xyz_xsens_m[1] - 0.15,
     xyz_xsens_m[2] - 0.18,
  ]

print()

################################################

# Extract the data.
print('Loading the data from %s' % input_data_filepath)
h5_file = h5py.File(input_data_filepath, 'r')
feature_matrices = np.squeeze(h5_file['feature_matrices'])
labels = np.array([x.decode('utf-8') for x in h5_file['labels']])
h5_file.close()

# Only use the human examples.
feature_matrices = feature_matrices[np.where(labels == 'human')[0], :, :]

# Adjust for the Baxter frame.
print('Rotating quaternions and translating positions to the Baxter frame')
for example_index in range(feature_matrices.shape[0]):
  feature_matrix = np.squeeze(feature_matrices[example_index, :, :])
  # Rotate the quaternions.
  gripper_quaternions_wijk = list(feature_matrix[:, data_file_gripper_quaternion_columns_wijk])
  gripper_quaternions_wijk = [convert_quaternion(quat_wijk) for quat_wijk in gripper_quaternions_wijk]
  gripper_quaternions_wijk = np.array(gripper_quaternions_wijk)
  feature_matrix[:, data_file_gripper_quaternion_columns_wijk] = gripper_quaternions_wijk
  feature_matrices[example_index, :, :] = feature_matrix
  # Translate the positions.
  gripper_positions_xyz_m = feature_matrix[:, data_file_gripper_position_columns_xyz_m]
  gripper_positions_xyz_m = [convert_position(xyz_m) for xyz_m in gripper_positions_xyz_m]
  feature_matrix[:, data_file_gripper_position_columns_xyz_m] = gripper_positions_xyz_m
  feature_matrices[example_index, :, :] = feature_matrix
  
# Save a matrix with only the time, position, and quaternion.
print('Saving the data to %s' % output_data_filepath)
output_data = feature_matrices[:, :, data_file_times_s_column + data_file_gripper_position_columns_xyz_m + data_file_gripper_quaternion_columns_wijk]
np.save(output_data_filepath, output_data, fix_imports=True)

print('Done!')
print()
