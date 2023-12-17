
import h5py
import os

import matplotlib
default_matplotlib_backend = matplotlib.rcParams['backend']
import matplotlib.pyplot as plt

from helpers import *

#############################
# Configuration.
#############################

subject_id_toProcess = 'S00'
data_dir = 'C:/Users/jdelp/Desktop/ActionSense/results/learning_trajectories'
feature_matrices_filepath = os.path.join(data_dir, 'pouring_training_data_%s.hdf5' % subject_id_toProcess)
referenceObjects_filepath = os.path.join(data_dir, 'pouring_training_referenceObject_positions_%s.hdf5' % subject_id_toProcess)

output_dir = 'C:/Users/jdelp/Desktop/ActionSense/results/learning_trajectories/output_plots'

#############################
# Load the data.
#############################
# Load the feature matrices and trial labels (human or robot).
feature_matrices_file = h5py.File(feature_matrices_filepath, 'r')
feature_matrices = np.array(feature_matrices_file['feature_matrices'])
labels = [x.decode('utf-8') for x in np.array(feature_matrices_file['labels'])]
feature_matrices_file.close()
# Load the reference object data.
referenceObjects_file = h5py.File(referenceObjects_filepath, 'r')
referenceObject_positions_m = np.array(referenceObjects_file['position_m'])
referenceObjects_file.close()

#############################
# Get trajectory information.
# These can be used to create distributions for statistical comparisons.
#############################

# Specify which trials to parse, or None to parse all.
trial_indexes_to_parse = None # [0,1,2,3,4,5] # note that some indexes may be skipped since they are not human trials

# Initialize some state for plotting.
previous_plot_handles_trajectoryTimestep = None
previous_plot_handles_bodyDynamics = None
previous_plot_handles_spoutProjection = None
previous_plot_handles_spoutTilt = None
previous_plot_handles_spoutDynamics = None

# Loop through each trial.
for (trial_index, label) in enumerate(labels):
  if trial_indexes_to_parse is not None:
    if trial_index not in trial_indexes_to_parse:
      continue
    trial_indexes_remaining = [x for x in range(trial_index+1, len(labels)) if x in trial_indexes_to_parse and labels[x] == 'human']
  else:
    trial_indexes_remaining = [x for x in range(trial_index+1, len(labels)) if labels[x] == 'human']
  is_last_trial_to_parse = len(trial_indexes_remaining) == 0
  print('Processing trial %d' % trial_index)
  
  if label == 'human':
    # Get the feature matrix.
    feature_matrix = np.squeeze(feature_matrices[trial_index])
    # Get the time of each sample.
    times_s = feature_matrix[:, -1]
    duration_s = times_s[-1]
    
    # Get the glass position.
    referenceObject_position_m = np.squeeze(referenceObject_positions_m[trial_index])
    
    print(' Getting body segment dynamics')
    # Get body segment position/speed/acceleration/jerk.
    # Will be a dictionary with keys 'hand', 'elbow', and 'shoulder'.
    #   For position, dict value is a Tx3 matrix where T is number of timesteps and 3 is xyz.
    #   For others, dict value is a T-length vector T is number of timesteps.
    body_segment_positions_m = get_body_position_m(feature_matrix)
    body_segment_speeds_m_s = get_body_speed_m_s(feature_matrix)
    body_segment_acceleration_m_s_s = get_body_acceleration_m_s_s(feature_matrix)
    body_segment_jerk_m_s_s_s = get_body_jerk_m_s_s_s(feature_matrix)
    
    print(' Getting spout dynamics')
    # Get spout position/speed/acceleration/jerk.
    # Position will be a Tx3 matrix where T is number of timesteps and 3 is xyz.
    # The others will be T-length vectors where T is number of timesteps.
    spout_position_m = infer_spout_position_m(feature_matrix)
    spout_speed_m_s = infer_spout_speed_m_s(feature_matrix)
    spout_speed_acceleration_m_s_s = infer_spout_acceleration_m_s_s(feature_matrix)
    spout_speed_jerk_m_s_s_s = infer_spout_jerk_m_s_s_s(feature_matrix)
    
    print(' Getting spout orientations')
    # Get the spout tilt angle.
    # Will infer the tilt angle of the top of the pitcher.
    spout_tilt_angle_rad = infer_spout_tilting(feature_matrix)
    # Get the spout yaw vector, which is the direction of pouring projected onto the XY plane.
    spout_pouring_direction = infer_spout_yawvector(feature_matrix)
    
    print(' Inferring the pouring window and associated body pose')
    # Infer when the pitcher is being held stationary over the glass,
    #  and the body pose during that moment.
    # Will return a dict with keys 'position_m', 'quaternion_wijk', 'time_index', 'start_time_index', and 'end_time_index'
    #  position_m and quaternion_wijk will have keys for 'hand', 'elbow', and 'shoulder'
    #  start_time_index and end_time_index indicate when the inferred pouring segment starts/stops.
    #  time_index is the midpoint of the inferred pouring segment
    pouring_pose_info = infer_pour_pose(feature_matrix)

    #############################
    # Visualizations
    #############################

    # Plot a single timestep of the trajectory.
    #  The pitcher will be a gray box.
    #  The human arm will be two connected black lines, with gray circles at the joints.
    #  A black circle will be at the origin.
    #  The light blue circle represents the glass, with a dark circle at its center.
    #  The magenta circle represents the projection of the spout onto the table.
    #  The red line from the magenta circle represents the pouring angle of the pitcher.
    print(' Visualizing a single timestep')
    previous_plot_handles_trajectoryTimestep = plot_timestep(
      feature_matrix,
      time_index=round(len(times_s)/2), times_s=times_s,
      referenceObject_position_m=referenceObject_position_m,
      previous_handles=previous_plot_handles_trajectoryTimestep,
      pause_after_plotting=False)

    # Animate the trajectory and save it as a video.
    #  The pitcher will be a gray box.
    #  The human arm will be two connected black lines, with gray circles at the joints.
    #  A black circle will be at the origin.
    #  The light blue circle represents the glass, with a dark circle at its center.
    #  The magenta circle represents the projection of the spout onto the table.
    #  The red line from the magenta circle represents the pouring angle of the pitcher.
    print(' Saving a trajectory animation as a video')
    save_trajectory_animation(
      feature_matrix, duration_s, referenceObject_position_m,
      subject_id=subject_id_toProcess, trial_index=trial_index,
      output_filepath=os.path.join(output_dir, 'trajectory_animations_%s' % subject_id_toProcess, 'trajectory_animation_%s_trial%02d.mp4' % (subject_id_toProcess, trial_index)))
                              
    # Plot the body speed and jerk over time.
    # If you want to add this trial to a plot with all trials, set fig to the previous output of this function.
    #   If you want to plot a single trial, set fig=None.
    print(' Plotting body segment dynamics')
    previous_plot_handles_bodyDynamics = plot_body_dynamics(
      feature_matrix,
      output_filepath=os.path.join(output_dir, 'body_dynamics_%s.jpg' % subject_id_toProcess) if is_last_trial_to_parse else None,
      shade_pouring_region=False, fig=previous_plot_handles_bodyDynamics, hide_figure_window=False)
    
    # Plot the spout projection onto the table relative to the glass.
    #   The blue shaded circle represents the glass.
    #   The y-axis is the pouring direction, as inferred from the yaw of the pitcher in each trial.
    #   A point will be plotted at the spout's projection onto the table, relative to the glass.
    #   So the water would pour upward on the plot from the plotted spout position.
    # If you want to add this trial to a plot with all trials, set fig to the previous output of this function.
    #   If you want to plot a single trial, set fig=None.
    print(' Plotting the spout pouring position relative to the glass, along the pitcher\'s pouring direction')
    previous_plot_handles_spoutProjection = plot_pour_relativePosition(
      feature_matrix, referenceObject_position_m,
      output_filepath=os.path.join(output_dir, 'spout_projection_relativeToGlass_%s.jpg' % subject_id_toProcess) if is_last_trial_to_parse else None,
      fig=previous_plot_handles_spoutProjection, hide_figure_window=False)
    
    # Plot the spout tilt over time.
    # If you want to add this trial to a plot with all trials, set fig to the previous output of this function.
    #   If you want to plot a single trial, set fig=None.
    print(' Plotting the spout tilt, relative to the table plane')
    previous_plot_handles_spoutTilt = plot_pour_tilting(
      feature_matrix,
      output_filepath=os.path.join(output_dir, 'spout_tilt_angle_%s.jpg' % subject_id_toProcess) if is_last_trial_to_parse else None,
      shade_pouring_region=False,
      fig=previous_plot_handles_spoutTilt, hide_figure_window=False)
    
    # Plot the spout speed and jerk over time.
    # If you want to add this trial to a plot with all trials, set fig to the previous output of this function.
    #   If you want to plot a single trial, set fig=None.
    print(' Plotting the spout dynamics')
    previous_plot_handles_spoutDynamics = plot_spout_dynamics(
      feature_matrix,
      output_filepath=os.path.join(output_dir, 'spout_dynamics_%s.jpg' % subject_id_toProcess) if is_last_trial_to_parse else None,
      shade_pouring_region=False,
      fig=previous_plot_handles_spoutDynamics, hide_figure_window=False)
    
# Show the plot windows.
matplotlib.use(default_matplotlib_backend)
plt.show(block=True)





