
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
import os

################################################

# Specify the files with trajectory feature matrices and reference object positions.
# data_dir = 'C:/Users/jdelp/Desktop/ActionSense/results/learning_trajectories'
data_dir = 'C:/Users/jdelp/Desktop/ActionSense/code/results/learning_trajectories'
input_featureMatrices_filepath = os.path.join(data_dir, 'pouring_training_data_ted_S00.hdf5')
input_referenceObjects_filepath = os.path.join(data_dir, 'pouring_training_referenceObject_positions_ted_S00.hdf5')
output_featureMatrices_filepath = os.path.join(data_dir, '%s_forBaxter.npy' % os.path.splitext(input_featureMatrices_filepath)[0])
output_referenceObjects_filepath = os.path.join(data_dir, '%s_forBaxter.npy' % os.path.splitext(input_referenceObjects_filepath)[0])

data_file_times_s_column = [30]
data_file_gripper_position_columns_xyz_m = [0,1,2]
data_file_gripper_quaternion_columns_wijk = [9,10,11,12]

referenceObject_height_cm = 15.8

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
  # A sample pitcher home for a human demonstration was [-0.3, 0.25, 0.18]
  #  and a sample glass home was [-0.38	-0.20	  0.18]
  #  with x point backwards, y pointing right, and z pointing up.
  # Baxter has a pitcher home position of about [0.9, -0.4, 0]
  #  with x pointing forwards, y pointing left, and z pointing down.
  # Human and Baxter are rotated 180 about z (x and y are negated)
  return [
    -xyz_xsens_m[0] + 0.6,
    -xyz_xsens_m[1] - 0.15,
     xyz_xsens_m[2] - 0.18,
  ]

def convert_position_trajectoryHand(xyz_xsens_m):
  # Lift it a bit to be safe.
  xyz_baxter_m = convert_position(xyz_xsens_m)
  return [
    xyz_baxter_m[0],
    xyz_baxter_m[1],
    xyz_baxter_m[2] + (2)/100,
  ]

def convert_position_referenceHand(xyz_xsens_m):
  # The reference object position is at the top of the glass,
  #  but Baxter's hand should be in the center of the glass
  #  so decrease the z axis by half of the glass height.
  # Also move the glass back a bit to account for Baxter's
  #  parallel attachment thickness.
  #  Note that may also need to account for the gripper itself,
  #   but then would need to do that on the pitcher hand too.
  xyz_baxter_m = convert_position(xyz_xsens_m)
  return [
    xyz_baxter_m[0] - (1)/100,
    xyz_baxter_m[1] + (1)/100,
    xyz_baxter_m[2] - (referenceObject_height_cm/2)/100,
  ]

print()

################################################

# Extract the data.
print('Loading feature matrices from %s' % input_featureMatrices_filepath)
h5_file = h5py.File(input_featureMatrices_filepath, 'r')
feature_matrices = np.squeeze(h5_file['feature_matrices'])
if 'labels' in h5_file:
  labels = np.array([x.decode('utf-8') for x in h5_file['labels']])
else:
  labels = None
h5_file.close()

print('Loading reference object positions from %s' % input_referenceObjects_filepath)
h5_file = h5py.File(input_referenceObjects_filepath, 'r')
referenceObject_positions_m = np.squeeze(h5_file['position_m'])
h5_file.close()


# Only use the human examples.
if labels is not None:
  example_indices_toUse = np.where(labels == 'human')[0]
  feature_matrices = feature_matrices[example_indices_toUse, :, :]
  referenceObject_positions_m = referenceObject_positions_m[example_indices_toUse, :]

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
  gripper_positions_xyz_m = [convert_position_trajectoryHand(xyz_m) for xyz_m in gripper_positions_xyz_m]
  feature_matrix[:, data_file_gripper_position_columns_xyz_m] = gripper_positions_xyz_m
  feature_matrices[example_index, :, :] = feature_matrix
  
  # Translate the reference object position to a Baxter hand position.
  referenceObject_position_m = referenceObject_positions_m[example_index, :]
  referenceObject_position_m = convert_position_referenceHand(referenceObject_position_m)
  referenceObject_positions_m[example_index, :] = referenceObject_position_m
  
# Save a matrix with only the time, position, and quaternion.
print('Saving trajectory-hand data to %s' % output_featureMatrices_filepath)
output_data = feature_matrices[:, :, data_file_times_s_column + data_file_gripper_position_columns_xyz_m + data_file_gripper_quaternion_columns_wijk]
np.save(output_featureMatrices_filepath, output_data, fix_imports=True)
print('Saving reference-hand data to %s' % output_referenceObjects_filepath)
output_data = referenceObject_positions_m
np.save(output_referenceObjects_filepath, output_data, fix_imports=True)

print('Done!')
print()


