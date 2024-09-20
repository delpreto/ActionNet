
import h5py
import numpy as np
import os

from helpers.parse_process_feature_matrices import *
from helpers.plot_animations import *

##################################################################
# Configuration.
##################################################################

# Specify the directory with HDF5 files.
# data_dir = 'C:/Users/jdelp/Desktop/ActionSense/results/learning_trajectories'
# data_dir = 'C:/Users/jdelp/Desktop/ActionSense/code/results/learning_trajectories'
data_dir = 'C:/Users/jdelp/Desktop/ActionSense/code/parsing_data/learning_trajectories/from_zahra/'

# Specify the output feature matrices to evaluate.
# For example, may have entries for each subject
#  and may have an entries for model outputs.
feature_matrices_filepaths = {
  'S00': os.path.join(data_dir, 'pouring_training_data_S00.hdf5'),
  # 'ted_S00': os.path.join(data_dir, 'pouring_training_data_ted_S00.hdf5'),
  'S10': os.path.join(data_dir, 'pouring_training_data_S10.hdf5'),
  'S11': os.path.join(data_dir, 'pouring_training_data_S11.hdf5'),
  'model': os.path.join(data_dir, 'model_output_data.hdf5'),
}
referenceObjects_filepaths = {
  'S00': os.path.join(data_dir, 'pouring_training_referenceObject_positions_S00.hdf5'),
  # 'ted_S00': os.path.join(data_dir, 'pouring_training_referenceObject_positions_ted_S00.hdf5'),
  'S10': os.path.join(data_dir, 'pouring_training_referenceObject_positions_S10.hdf5'),
  'S11': os.path.join(data_dir, 'pouring_training_referenceObject_positions_S11.hdf5'),
  'model': os.path.join(data_dir, 'model_referenceObject_positions.hdf5'),
}

# Specify which outputs to process.
# Animations.
plot_example_timestep = True

print()

##################################################################
# Load the data.
##################################################################

# Determine the types of examples provided, and the ones used for comparisons to the primary.
example_types = list(feature_matrices_filepaths.keys())

# Load the feature matrices and reference object positions.
feature_matrices_byType = {}
referenceObject_positions_m_byType = {}
for (example_type, feature_matrices_filepath) in feature_matrices_filepaths.items():
  if feature_matrices_filepath is not None:
    # Load the feature matrices.
    feature_matrices_file = h5py.File(feature_matrices_filepath, 'r')
    feature_matrices_byType[example_type] = np.array(feature_matrices_file['feature_matrices'])
    # If labels are provided, only use the human examples.
    if 'labels' in feature_matrices_file:
      labels = [x.decode('utf-8') for x in np.array(feature_matrices_file['labels'])]
    else:
      labels = None
    feature_matrices_file.close()
    
    # Load the reference object data.
    referenceObjects_file = h5py.File(referenceObjects_filepaths[example_type], 'r')
    referenceObject_positions_m_byType[example_type] = np.array(referenceObjects_file['position_m'])
    referenceObjects_file.close()
    
    # If labels are provided, only use the human examples.
    if labels is not None:
      example_indices_toUse = [i for (i, label) in enumerate(labels) if label == 'human']
      feature_matrices_byType[example_type] = feature_matrices_byType[example_type][example_indices_toUse,:,:]
      referenceObject_positions_m_byType[example_type] = referenceObject_positions_m_byType[example_type][example_indices_toUse,:]


##################################################################
# Examples of inferring various quantities from the data.
##################################################################

print()
print('='*70)
print('Demonstrating getting various trajectory quantities or metrics')

# Will just demonstrate on the first example type.
feature_matrices = feature_matrices_byType[example_types[0]]
referenceObject_positions_m = referenceObject_positions_m_byType[example_types[0]]
previous_plot_handles_trajectoryTimestep = None

# Loop through each trial.
for trial_index in range(len(feature_matrices)):
  feature_matrix = feature_matrices[trial_index]
  referenceObject_position_m = referenceObject_positions_m[trial_index]
  times_s = get_trajectory_time_s(feature_matrix)

  # Get body segment position/speed/acceleration/jerk.
  # Will be a dictionary with keys 'hand', 'elbow', and 'shoulder'.
  #   For position, dict value is a Tx3 matrix where T is number of timesteps and 3 is xyz.
  #   For others, dict value is a T-length vector T is number of timesteps.
  body_segment_positions_m = get_body_position_m(feature_matrix)
  body_segment_speeds_m_s = get_body_speed_m_s(feature_matrix)
  body_segment_acceleration_m_s_s = get_body_acceleration_m_s_s(feature_matrix)
  body_segment_jerk_m_s_s_s = get_body_jerk_m_s_s_s(feature_matrix)

  # Get spout position/speed/acceleration/jerk.
  # Position will be a Tx3 matrix where T is number of timesteps and 3 is xyz.
  # The others will be T-length vectors where T is number of timesteps.
  spout_position_m = infer_spout_position_m(feature_matrix)
  spout_speed_m_s = infer_spout_speed_m_s(feature_matrix)
  spout_speed_acceleration_m_s_s = infer_spout_acceleration_m_s_s(feature_matrix)
  spout_speed_jerk_m_s_s_s = infer_spout_jerk_m_s_s_s(feature_matrix)

  # Get the spout tilt angle.
  # Will infer the tilt angle of the top of the pitcher.
  spout_tilt_angle_rad = infer_spout_tilting(feature_matrix)
  # Get the spout yaw vector, which is the direction of pouring projected onto the XY plane.
  spout_pouring_direction = infer_spout_yawvector(feature_matrix)

  # Infer when the pitcher is being held stationary over the glass,
  #  and the body pose during that moment.
  # Will return a dict with keys 'position_m', 'quaternion_wijk', 'time_index', 'start_time_index', and 'end_time_index'
  #  position_m and quaternion_wijk will have keys for 'hand', 'elbow', and 'shoulder'
  #  start_time_index and end_time_index indicate when the inferred pouring segment starts/stops.
  #  time_index is the midpoint of the inferred pouring segment
  pouring_pose_info = infer_pour_pose(feature_matrix)

  # Plot a single timestep of the trajectory.
  #  The pitcher will be a gray box.
  #  The human arm will be two connected black lines, with gray circles at the joints.
  #  A black circle will be at the origin.
  #  The light blue circle represents the glass, with a dark circle at its center.
  #  The magenta circle represents the projection of the spout onto the table.
  #  The red line from the magenta circle represents the pouring angle of the pitcher.
  if plot_example_timestep and trial_index == 0:
    previous_plot_handles_trajectoryTimestep = plot_timestep(
      feature_matrix,
      time_index=round(len(times_s)/2), time_s=times_s,
      referenceObject_position_m=referenceObject_position_m,
      previous_handles=previous_plot_handles_trajectoryTimestep,
      wait_for_user_after_plotting=True)
    




















