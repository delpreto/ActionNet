
import h5py
import os

import matplotlib
default_matplotlib_backend = matplotlib.rcParams['backend']
import matplotlib.pyplot as plt

from helpers import *

##################################################################
# Configuration.
##################################################################

subject_id_toProcess = 'S00'
data_dir = 'C:/Users/jdelp/Desktop/ActionSense/results/learning_trajectories'
# Specify the output feature matrices to evaluate.
# Also specify feature matrices to be used for comparison purposes.
#  For example, human demonstrations that correspond to model outputs.
feature_matrices_filepaths = {
  # 'model': os.path.join(data_dir, 'pouring_training_data_%s.hdf5' % subject_id_toProcess),
  'human': os.path.join(data_dir, 'pouring_training_data_%s.hdf5' % subject_id_toProcess),
}
referenceObjects_filepaths = {
  # 'model': os.path.join(data_dir, 'pouring_training_referenceObject_positions_%s.hdf5' % subject_id_toProcess),
  'human': os.path.join(data_dir, 'pouring_training_referenceObject_positions_%s.hdf5' % subject_id_toProcess),
}
# Specify the main example type to evaluate. Others specified above will be used for comparisons.
primary_example_type = 'model'

# Specify which trials to parse, or None to parse all.
trial_indexes_to_parse = None # [0,1,2,3,4,5]

# Specify which outputs to compute.
animate_trajectories = False
save_trajectory_animations = False
plot_spout_tilt = False
plot_spout_pouring_projection = False
plot_spout_height = False

keep_plots_open = True

# Specify where outputs should be saved.
# Can be None to not save any outputs.
output_dir = 'C:/Users/jdelp/Desktop/ActionSense/results/learning_trajectories/output_plots'

##################################################################
# Load the data.
##################################################################
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
# Get trajectory information.
# These can be used to create distributions for statistical comparisons.
##################################################################

# Determine the types of examples provided, and the ones used for comparisons to the primary.
example_types = list(feature_matrices_filepaths.keys())
comparison_example_types = [example_type for example_type in example_types if example_type != primary_example_type]

# Initialize some state for plotting.
previous_plot_handles_trajectoryTimestep_byType = dict.fromkeys(example_types, None)
previous_plot_handles_bodyDynamics_byType = dict.fromkeys(example_types, None)
previous_plot_handles_spoutProjection_byType = dict.fromkeys(example_types, None)
previous_plot_handles_spoutTilt_byType = dict.fromkeys(example_types, None)
previous_plot_handles_spoutDynamics_byType = dict.fromkeys(example_types, None)

##################################################################
# Examples of inferring various quantities from the data.
# Will just demonstrate on the primary example type.
##################################################################

# # Loop through each trial.
# for (trial_index, label) in enumerate(labels):
#   if trial_indexes_to_parse is not None:
#     if trial_index not in trial_indexes_to_parse:
#       continue
#     trial_indexes_remaining = [x for x in range(trial_index+1, len(labels)) if x in trial_indexes_to_parse]
#   else:
#     trial_indexes_remaining = [x for x in range(trial_index+1, len(labels))]
#   is_last_trial_to_parse = len(trial_indexes_remaining) == 0
#   print('Processing trial index %d' % trial_index)
#
#   # Get the data for this trial for each type of example provided.
#   feature_matrices_byType_forTrial = {}
#   times_s_byType_forTrial = {}
#   durations_s_byType_forTrial = {}
#   referenceObject_positions_m_byType_forTrial = {}
#   for example_type in example_types:
#     feature_matrices_byType_forTrial[example_type] = np.squeeze(feature_matrices_byType[example_type][trial_index])
#     times_s_byType_forTrial[example_type] = feature_matrices_byType_forTrial[example_type][:, -1]
#     durations_s_byType_forTrial[example_type] = times_s_byType_forTrial[example_type][-1]
#
#     referenceObject_positions_m_byType_forTrial[example_type] = np.squeeze(referenceObject_positions_m_byType[example_type][trial_index])
#
#   feature_matrix = feature_matrices_byType_forTrial[primary_example_type]
#   referenceObject_position_m = referenceObject_positions_m_byType_forTrial[primary_example_type]
#   times_s = times_s_byType_forTrial[primary_example_type]
#
#   # Get body segment position/speed/acceleration/jerk.
#   # Will be a dictionary with keys 'hand', 'elbow', and 'shoulder'.
#   #   For position, dict value is a Tx3 matrix where T is number of timesteps and 3 is xyz.
#   #   For others, dict value is a T-length vector T is number of timesteps.
#   body_segment_positions_m = get_body_position_m(feature_matrix)
#   body_segment_speeds_m_s = get_body_speed_m_s(feature_matrix)
#   body_segment_acceleration_m_s_s = get_body_acceleration_m_s_s(feature_matrix)
#   body_segment_jerk_m_s_s_s = get_body_jerk_m_s_s_s(feature_matrix)
#
#   # Get spout position/speed/acceleration/jerk.
#   # Position will be a Tx3 matrix where T is number of timesteps and 3 is xyz.
#   # The others will be T-length vectors where T is number of timesteps.
#   spout_position_m = infer_spout_position_m(feature_matrix)
#   spout_speed_m_s = infer_spout_speed_m_s(feature_matrix)
#   spout_speed_acceleration_m_s_s = infer_spout_acceleration_m_s_s(feature_matrix)
#   spout_speed_jerk_m_s_s_s = infer_spout_jerk_m_s_s_s(feature_matrix)
#
#   # Get the spout tilt angle.
#   # Will infer the tilt angle of the top of the pitcher.
#   spout_tilt_angle_rad = infer_spout_tilting(feature_matrix)
#   # Get the spout yaw vector, which is the direction of pouring projected onto the XY plane.
#   spout_pouring_direction = infer_spout_yawvector(feature_matrix)
#
#   # Infer when the pitcher is being held stationary over the glass,
#   #  and the body pose during that moment.
#   # Will return a dict with keys 'position_m', 'quaternion_wijk', 'time_index', 'start_time_index', and 'end_time_index'
#   #  position_m and quaternion_wijk will have keys for 'hand', 'elbow', and 'shoulder'
#   #  start_time_index and end_time_index indicate when the inferred pouring segment starts/stops.
#   #  time_index is the midpoint of the inferred pouring segment
#   pouring_pose_info = infer_pour_pose(feature_matrix)
#
#   # Plot a single timestep of the trajectory.
#   #  The pitcher will be a gray box.
#   #  The human arm will be two connected black lines, with gray circles at the joints.
#   #  A black circle will be at the origin.
#   #  The light blue circle represents the glass, with a dark circle at its center.
#   #  The magenta circle represents the projection of the spout onto the table.
#   #  The red line from the magenta circle represents the pouring angle of the pitcher.
#   print(' Visualizing a single timestep')
#   previous_plot_handles_trajectoryTimestep = plot_timestep(
#     feature_matrix,
#     time_index=round(len(times_s)/2), times_s=times_s,
#     referenceObject_position_m=referenceObject_position_m,
#     previous_handles=previous_plot_handles_trajectoryTimestep,
#     pause_after_plotting=False)
  
##################################################################
# Results: Visualizations
##################################################################

# Animate the trajectory and optionally save it as a video.
#  The pitcher will be a gray box.
#  The human arm will be two connected black lines, with gray circles at the joints.
#  A black circle will be at the origin.
#  The light blue circle represents the glass, with a dark circle at its center.
#  The magenta circle represents the projection of the spout onto the table.
#  The red line from the magenta circle represents the pouring angle of the pitcher.
# Will create a subplot for each example type in the dictionaries provided.
if save_trajectory_animations or animate_trajectories:
  # Loop through each trial.
  # Currently assumes that each example type has the same number of trials.
  for trial_index in range(len(feature_matrices_byType[example_types[0]])):
    # Get the data for this trial for each type of example provided.
    feature_matrices_byType_forTrial = {}
    times_s_byType_forTrial = {}
    durations_s_byType_forTrial = {}
    referenceObject_positions_m_byType_forTrial = {}
    for example_type in example_types:
      feature_matrices_byType_forTrial[example_type] = np.squeeze(feature_matrices_byType[example_type][trial_index])
      times_s_byType_forTrial[example_type] = feature_matrices_byType_forTrial[example_type][:, -1]
      durations_s_byType_forTrial[example_type] = times_s_byType_forTrial[example_type][-1]
      referenceObject_positions_m_byType_forTrial[example_type] = np.squeeze(referenceObject_positions_m_byType[example_type][trial_index])
    
    # Animate the trajectory; press enter to advance to the next timestep.
    if animate_trajectories:
      plot_trajectory(
        feature_matrices_byType_forTrial['human'], durations_s_byType_forTrial['human'],
        referenceObject_positions_m_byType_forTrial['human'],
        pause_after_timesteps=True)
    
    # Save the animation as a video.
    if save_trajectory_animations:
      print('Saving trajectory animations as a video for trial index %02d' % trial_index)
      save_trajectory_animation(
        feature_matrices_byType_forTrial, durations_s_byType_forTrial, referenceObject_positions_m_byType_forTrial,
        subject_id=subject_id_toProcess, trial_index=trial_index,
        output_filepath=os.path.join(output_dir, 'trajectory_animations_%s' % subject_id_toProcess, 'trajectory_animation_%s_trial%02d.mp4' % (subject_id_toProcess, trial_index)) if output_dir is not None else None
      )
  
# Plot the spout tilt over time.
if plot_spout_tilt:
  print('Plotting the spout tilt, relative to the table plane')
  # Plot each trial as a trace on the plot, with a separate plot for each example type.
  for example_type in example_types:
    plot_pour_tilting(
      feature_matrices_byType[example_type],
      plot_mean=True, plot_std_shading=False, plot_all_trials=True,
      subtitle=example_type, label=None,
      output_filepath=os.path.join(output_dir, 'spout_tilt_angle_%s_%s.jpg' % (example_type, subject_id_toProcess)) if output_dir is not None else None,
      shade_pouring_region=False,
      fig=None, hide_figure_window=False)
  # Plot mean and standard deviation shading for each example type, on the same plot.
  previous_plot_handles = None
  for example_type in example_types:
    is_last_type = example_type == example_types[-1]
    previous_plot_handles = plot_pour_tilting(
      feature_matrices_byType[example_type],
      plot_mean=True, plot_std_shading=True, plot_all_trials=False,
      subtitle=None, label=example_type,
      output_filepath=(os.path.join(output_dir, 'spout_tilt_angle_allTypes_%s.jpg' % (subject_id_toProcess)) if is_last_type else None) if output_dir is not None else None,
      shade_pouring_region=False,
      fig=previous_plot_handles, hide_figure_window=False)

if plot_spout_pouring_projection:
  # Plot the spout projection onto the table relative to the glass.
  #   The blue shaded circle represents the glass.
  #   The y-axis is the pouring direction, as inferred from the yaw of the pitcher in each trial.
  #   A point will be plotted at the spout's projection onto the table, relative to the glass.
  #   So the water would pour upward on the plot from the plotted spout position.
  print('Plotting the spout pouring position relative to the glass, along the pitcher\'s pouring direction')
  # Plot each trial as a separate point, with a separate plot for each example type.
  for example_type in example_types:
    plot_pour_relativePosition(
      feature_matrices_byType[example_type], referenceObject_positions_m_byType[example_type],
      plot_mean=False, plot_std_shading=False, plot_all_trials=True,
      subtitle=example_type, label=None,
      output_filepath=os.path.join(output_dir, 'spout_projection_relativeToGlass_%s_%s.jpg' % (example_type, subject_id_toProcess)) if output_dir is not None else None,
      fig=None, hide_figure_window=False)
  # Plot mean and standard deviation shading for each example type, on the same plot.
  previous_plot_handles = None
  for (example_index, example_type) in enumerate(example_types):
    is_last_type = example_type == example_types[-1]
    previous_plot_handles = plot_pour_relativePosition(
      feature_matrices_byType[example_type], referenceObject_positions_m_byType[example_type],
      plot_mean=True, plot_std_shading=True, plot_all_trials=False,
      subtitle=None, label=example_type,
      color=plt.rcParams['axes.prop_cycle'].by_key()['color'][example_index],
      output_filepath=(os.path.join(output_dir, 'spout_projection_relativeToGlass_allTypes_%s.jpg' % (subject_id_toProcess)) if is_last_type else None) if output_dir is not None else None,
      fig=previous_plot_handles, hide_figure_window=False)

if plot_spout_height:
  # Plot the spout height over time, relative to the glass height.
  print('Plotting the spout height relative to the glass')
  # Plot each trial as a trace on the plot, with a separate plot for each example type.
  for example_type in example_types:
    plot_pour_relativeHeight(
      feature_matrices_byType[example_type], referenceObject_positions_m_byType[example_type],
      plot_mean=True, plot_std_shading=False, plot_all_trials=True,
      subtitle=example_type, label=None,
      output_filepath=os.path.join(output_dir, 'spout_height_%s_%s.jpg' % (example_type, subject_id_toProcess)) if output_dir is not None else None,
      shade_pouring_region=False,
      fig=None, hide_figure_window=False)
  # Plot mean and standard deviation shading for each example type, on the same plot.
  previous_plot_handles = None
  for example_type in example_types:
    is_last_type = example_type == example_types[-1]
    previous_plot_handles = plot_pour_relativeHeight(
      feature_matrices_byType[example_type], referenceObject_positions_m_byType[example_type],
      plot_mean=True, plot_std_shading=True, plot_all_trials=False,
      subtitle=None, label=example_type,
      output_filepath=(os.path.join(output_dir, 'spout_height_allTypes_%s.jpg' % (subject_id_toProcess)) if is_last_type else None) if output_dir is not None else None,
      shade_pouring_region=False,
      fig=previous_plot_handles, hide_figure_window=False)
  
#   # Plot the body speed and jerk over time.
#   # If you want to add this trial to a plot with all trials, set fig to the previous output of this function.
#   #   If you want to plot a single trial, set fig=None.
#   print(' Plotting body segment dynamics')
#   previous_plot_handles_bodyDynamics_byType = plot_body_dynamics(
#     feature_matrix,
#     output_filepath=os.path.join(output_dir, 'body_dynamics_%s.jpg' % subject_id_toProcess) if is_last_trial_to_parse else None,
#     shade_pouring_region=False, fig=previous_plot_handles_bodyDynamics_byType, hide_figure_window=False)
#
#   # Plot the spout projection onto the table relative to the glass.
#   #   The blue shaded circle represents the glass.
#   #   The y-axis is the pouring direction, as inferred from the yaw of the pitcher in each trial.
#   #   A point will be plotted at the spout's projection onto the table, relative to the glass.
#   #   So the water would pour upward on the plot from the plotted spout position.
#   # If you want to add this trial to a plot with all trials, set fig to the previous output of this function.
#   #   If you want to plot a single trial, set fig=None.
#   print(' Plotting the spout pouring position relative to the glass, along the pitcher\'s pouring direction')
#   previous_plot_handles_spoutProjection_byType = plot_pour_relativePosition(
#     feature_matrix, referenceObject_position_m,
#     output_filepath=os.path.join(output_dir, 'spout_projection_relativeToGlass_%s.jpg' % subject_id_toProcess) if is_last_trial_to_parse else None,
#     fig=previous_plot_handles_spoutProjection_byType, hide_figure_window=False)
#
#   # Plot the spout speed and jerk over time.
#   # If you want to add this trial to a plot with all trials, set fig to the previous output of this function.
#   #   If you want to plot a single trial, set fig=None.
#   print(' Plotting the spout dynamics')
#   previous_plot_handles_spoutDynamics_byType = plot_spout_dynamics(
#     feature_matrix,
#     output_filepath=os.path.join(output_dir, 'spout_dynamics_%s.jpg' % subject_id_toProcess) if is_last_trial_to_parse else None,
#     shade_pouring_region=False,
#     fig=previous_plot_handles_spoutDynamics_byType, hide_figure_window=False)
#
# # Show the plot windows.
# matplotlib.use(default_matplotlib_backend)
# plt.show(block=True)
#
#
#
#
#
if keep_plots_open:
  plt.show(block=True)