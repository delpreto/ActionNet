
import h5py
import os

import matplotlib
default_matplotlib_backend = matplotlib.rcParams['backend']
matplotlib.rcParams['figure.max_open_warning'] = 35 # default 20
import matplotlib.pyplot as plt

from helpers import *

##################################################################
# Configuration.
##################################################################

# Specify the subject to process and the directory with HDF5 files.
data_dir = 'C:/Users/jdelp/Desktop/ActionSense/results/learning_trajectories'

# Specify the output feature matrices to evaluate.
# For example, may have entries for each subject
#  and may have an entries for model outputs.
feature_matrices_filepaths = {
  'S00': os.path.join(data_dir, 'pouring_training_data_S00.hdf5'),
  'S10': os.path.join(data_dir, 'pouring_training_data_S10.hdf5'),
  'S11': os.path.join(data_dir, 'pouring_training_data_S11.hdf5'),
}
referenceObjects_filepaths = {
  'S00': os.path.join(data_dir, 'pouring_training_referenceObject_positions_S00.hdf5'),
  'S10': os.path.join(data_dir, 'pouring_training_referenceObject_positions_S10.hdf5'),
  'S11': os.path.join(data_dir, 'pouring_training_referenceObject_positions_S11.hdf5'),
}

# Specify which outputs to process.
# Animations.
plot_example_timestep = False
interactively_animate_trajectories_exampleType = None # interactive - can move around scene and press enter to step through time # None to not animate
save_trajectory_animations_eachType = False
save_trajectory_animations_compositeTypes = False
# Plots (mostly time series).
plot_spout_tilt = True
plot_spout_pouring_projection = True
plot_spout_height = True
plot_spout_speedJerk = True
plot_body_speedJerk = True
plot_joint_angles = True
# Plots and comparisons of distributions.
plot_compare_distribution_body_speedJerk = True
plot_compare_distribution_spout_speedJerk = True
plot_compare_distribution_joint_angles = True
plot_compare_distribution_spout_projection = True
plot_compare_distribution_spout_height = True
plot_compare_distribution_spout_tilt = True

# Specify whether to show figure windows or process them in the background.
# Either way, plots will be saved as images if output_dir is specified below.
keep_plots_open = True

# Specify where outputs should be saved.
# Can be None to not save any outputs.
output_dir = os.path.join(data_dir, 'output_plots')

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
  times_s = feature_matrix[:, -1]

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
      time_index=round(len(times_s)/2), times_s=times_s,
      referenceObject_position_m=referenceObject_position_m,
      previous_handles=previous_plot_handles_trajectoryTimestep,
      pause_after_plotting=False)


##################################################################
# Results: Trajectory Visualizations
##################################################################

print()
print('='*70)

# Animate the trajectory and optionally save it as a video.
#  The pitcher will be a gray box.
#  The human arm will be two connected black lines, with gray circles at the joints.
#  A black circle will be at the origin.
#  The light blue circle represents the glass, with a dark circle at its center.
#  The magenta circle represents the projection of the spout onto the table.
#  The red line from the magenta circle represents the pouring angle of the pitcher.
# Will create a subplot for each example type in the dictionaries provided.
if save_trajectory_animations_compositeTypes or save_trajectory_animations_eachType or (interactively_animate_trajectories_exampleType is not None):
  # Loop through each trial.
  max_num_trials = max([len(x) for x in feature_matrices_byType.values()])
  for trial_index in range(max_num_trials):
    # Get the data for this trial for each type of example provided.
    feature_matrices_byType_forTrial = {}
    times_s_byType_forTrial = {}
    durations_s_byType_forTrial = {}
    referenceObject_positions_m_byType_forTrial = {}
    have_trial_for_all_types = True
    for example_type in example_types:
      if trial_index >= len(feature_matrices_byType[example_type]):
        feature_matrices_byType_forTrial[example_type] = None
        times_s_byType_forTrial[example_type] = None
        durations_s_byType_forTrial[example_type] = None
        referenceObject_positions_m_byType_forTrial[example_type] = None
        have_trial_for_all_types = False
      else:
        feature_matrices_byType_forTrial[example_type] = np.squeeze(feature_matrices_byType[example_type][trial_index])
        times_s_byType_forTrial[example_type] = feature_matrices_byType_forTrial[example_type][:, -1]
        durations_s_byType_forTrial[example_type] = times_s_byType_forTrial[example_type][-1]
        referenceObject_positions_m_byType_forTrial[example_type] = np.squeeze(referenceObject_positions_m_byType[example_type][trial_index])
    
    # Animate the trajectory; press enter to advance to the next timestep.
    if interactively_animate_trajectories_exampleType is not None:
      if feature_matrices_byType_forTrial[interactively_animate_trajectories_exampleType] is not None:
        print('Animating the trajectory; press enter on the figure window to advance time')
        plot_trajectory(
          feature_matrices_byType_forTrial[interactively_animate_trajectories_exampleType], durations_s_byType_forTrial[interactively_animate_trajectories_exampleType],
          referenceObject_positions_m_byType_forTrial[interactively_animate_trajectories_exampleType],
          pause_after_timesteps=True)
    
    # Save the animation as a video, with a subplot for each example type.
    if save_trajectory_animations_compositeTypes:
      if have_trial_for_all_types:
        print('Saving trajectory animations as a video for trial index %02d all types' % trial_index)
        save_trajectory_animation(
          feature_matrices_byType_forTrial, durations_s_byType_forTrial, referenceObject_positions_m_byType_forTrial,
          subject_id='', trial_index=trial_index,
          output_filepath=os.path.join(output_dir, 'trajectory_animations', 'trajectory_animation_allTypes_trial%02d.mp4' % (trial_index)) if output_dir is not None else None
      )
    if save_trajectory_animations_eachType:
      for example_type in example_types:
        if feature_matrices_byType_forTrial[example_type] is not None:
          print('Saving trajectory animations as a video for trial index %02d type %s' % (trial_index, example_type))
          save_trajectory_animation(
            feature_matrices_byType_forTrial[example_type], durations_s_byType_forTrial[example_type], referenceObject_positions_m_byType_forTrial[example_type],
            subject_id=example_type, trial_index=trial_index,
            output_filepath=os.path.join(output_dir, 'trajectory_animations', 'trajectory_animation_%s_trial%02d.mp4' % (example_type, trial_index)) if output_dir is not None else None
          )
  
##################################################################
# Results: Plots
##################################################################

# Plot the spout tilt over time.
if plot_spout_tilt:
  print('Plotting the spout tilt, relative to the table plane')
  # Plot each trial as a trace on the plot, with a separate plot for each example type.
  for example_type in example_types:
    plot_pour_tilting(
      feature_matrices_byType[example_type],
      plot_mean=True, plot_std_shading=False, plot_all_trials=True,
      subtitle=example_type.title(), label=None,
      output_filepath=os.path.join(output_dir, 'spout_tilt_angle_%s.jpg' % (example_type)) if output_dir is not None else None,
      shade_pouring_region=False,
      fig=None, hide_figure_window=not keep_plots_open)
  # Plot mean and standard deviation shading for each example type, on the same plot.
  previous_plot_handles = None
  for example_type in example_types:
    is_last_type = example_type == example_types[-1]
    previous_plot_handles = plot_pour_tilting(
      feature_matrices_byType[example_type],
      plot_mean=True, plot_std_shading=True, plot_all_trials=False,
      subtitle=None, label=example_type.title(),
      output_filepath=(os.path.join(output_dir, 'spout_tilt_angle_allTypes.jpg') if is_last_type else None) if output_dir is not None else None,
      shade_pouring_region=False,
      fig=previous_plot_handles, hide_figure_window=not keep_plots_open)

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
      subtitle=example_type.title(), label=None,
      output_filepath=os.path.join(output_dir, 'spout_projection_relativeToGlass_%s.jpg' % (example_type)) if output_dir is not None else None,
      fig=None, hide_figure_window=not keep_plots_open)
  # Plot mean and standard deviation shading for each example type, on the same plot.
  previous_plot_handles = None
  for (example_index, example_type) in enumerate(example_types):
    is_last_type = example_type == example_types[-1]
    previous_plot_handles = plot_pour_relativePosition(
      feature_matrices_byType[example_type], referenceObject_positions_m_byType[example_type],
      plot_mean=True, plot_std_shading=True, plot_all_trials=False,
      subtitle=None, label=example_type.title(),
      color=plt.rcParams['axes.prop_cycle'].by_key()['color'][example_index],
      output_filepath=(os.path.join(output_dir, 'spout_projection_relativeToGlass_allTypes.jpg') if is_last_type else None) if output_dir is not None else None,
      fig=previous_plot_handles, hide_figure_window=not keep_plots_open)

if plot_spout_height:
  # Plot the spout height over time, relative to the glass height.
  print('Plotting the spout height relative to the glass')
  # Plot each trial as a trace on the plot, with a separate plot for each example type.
  for example_type in example_types:
    plot_pour_relativeHeight(
      feature_matrices_byType[example_type], referenceObject_positions_m_byType[example_type],
      plot_mean=True, plot_std_shading=False, plot_all_trials=True,
      subtitle=example_type.title(), label=None,
      output_filepath=os.path.join(output_dir, 'spout_height_%s.jpg' % (example_type)) if output_dir is not None else None,
      shade_pouring_region=False,
      fig=None, hide_figure_window=not keep_plots_open)
  # Plot mean and standard deviation shading for each example type, on the same plot.
  previous_plot_handles = None
  for example_type in example_types:
    is_last_type = example_type == example_types[-1]
    previous_plot_handles = plot_pour_relativeHeight(
      feature_matrices_byType[example_type], referenceObject_positions_m_byType[example_type],
      plot_mean=True, plot_std_shading=True, plot_all_trials=False,
      subtitle=None, label=example_type.title(),
      output_filepath=(os.path.join(output_dir, 'spout_height_allTypes.jpg') if is_last_type else None) if output_dir is not None else None,
      shade_pouring_region=False,
      fig=previous_plot_handles, hide_figure_window=not keep_plots_open)

if plot_body_speedJerk:
  # Plot the body speed and jerk over time.
  print('Plotting body segment dynamics')
  # Plot each trial as a trace on the plot, with a separate plot for each example type.
  for example_type in example_types:
    plot_body_dynamics(
      feature_matrices_byType[example_type],
      plot_mean=True, plot_std_shading=False, plot_all_trials=True,
      subtitle=example_type.title(), label=None,
      output_filepath=os.path.join(output_dir, 'body_dynamics_%s.jpg' % (example_type)) if output_dir is not None else None,
      shade_pouring_region=False,
      fig=None, hide_figure_window=not keep_plots_open)
  # Plot mean and standard deviation shading for each example type, on the same plot.
  previous_plot_handles = None
  for example_type in example_types:
    is_last_type = example_type == example_types[-1]
    previous_plot_handles =  plot_body_dynamics(
      feature_matrices_byType[example_type],
      plot_mean=True, plot_std_shading=True, plot_all_trials=False,
      subtitle=None, label=example_type.title(),
      output_filepath=(os.path.join(output_dir, 'body_dynamics_allTypes.jpg') if is_last_type else None) if output_dir is not None else None,
      shade_pouring_region=False,
      fig=previous_plot_handles, hide_figure_window=not keep_plots_open)

if plot_spout_speedJerk:
  # Plot the spout speed and jerk over time.
  print('Plotting spout dynamics')
  # Plot each trial as a trace on the plot, with a separate plot for each example type.
  for example_type in example_types:
    plot_spout_dynamics(
      feature_matrices_byType[example_type],
      plot_mean=True, plot_std_shading=False, plot_all_trials=True,
      subtitle=example_type.title(), label=None,
      output_filepath=os.path.join(output_dir, 'spout_dynamics_%s.jpg' % (example_type)) if output_dir is not None else None,
      shade_pouring_region=False,
      fig=None, hide_figure_window=not keep_plots_open)
  # Plot mean and standard deviation shading for each example type, on the same plot.
  previous_plot_handles = None
  for example_type in example_types:
    is_last_type = example_type == example_types[-1]
    previous_plot_handles =  plot_spout_dynamics(
      feature_matrices_byType[example_type],
      plot_mean=True, plot_std_shading=True, plot_all_trials=False,
      subtitle=None, label=example_type.title(),
      output_filepath=(os.path.join(output_dir, 'spout_dynamics_allTypes.jpg') if is_last_type else None) if output_dir is not None else None,
      shade_pouring_region=False,
      fig=previous_plot_handles, hide_figure_window=not keep_plots_open)

if plot_joint_angles:
  # Plot the body joint angles over time.
  print('Plotting joint angles')
  # Plot each trial as a trace on the plot, with a separate plot for each example type.
  for example_type in example_types:
    plot_body_joint_angles(
      feature_matrices_byType[example_type],
      plot_mean=True, plot_std_shading=False, plot_all_trials=True,
      subtitle=example_type.title(), label=None,
      output_filepath=os.path.join(output_dir, 'joint_angles_%s.jpg' % (example_type)) if output_dir is not None else None,
      shade_pouring_region=False,
      fig=None, hide_figure_window=not keep_plots_open)
  # Plot mean and standard deviation shading for each example type, on the same plot.
  previous_plot_handles = None
  for example_type in example_types:
    is_last_type = example_type == example_types[-1]
    print('example_type.title()', example_type.title())
    previous_plot_handles = plot_body_joint_angles(
      feature_matrices_byType[example_type],
      plot_mean=True, plot_std_shading=True, plot_all_trials=False,
      subtitle=None, label=example_type.title(),
      output_filepath=(os.path.join(output_dir, 'joint_angles_allTypes.jpg') if is_last_type else None) if output_dir is not None else None,
      shade_pouring_region=False,
      fig=previous_plot_handles, hide_figure_window=not keep_plots_open)

print()

##################################################################
# Results: Distributions (plots and statistical tests)
##################################################################

if plot_compare_distribution_body_speedJerk:
  # Plot and compare the body segment speed and jerk distributions.
  print()
  print('='*70)
  print('Plotting and comparing distributions of body dynamics')
  plot_compare_distributions_body_dynamics(
    feature_matrices_byType,
    output_filepath=os.path.join(output_dir, 'body_dynamics_distributions.jpg') if output_dir is not None else None,
    region=None, # 'pre_pouring', 'pouring', 'post_pouring', None for all
    print_comparison_results=True,
    plot_distributions=True,
    num_histogram_bins=25, histogram_range_quantiles=(0, 0.95),
    fig=None, hide_figure_window=not keep_plots_open)

if plot_compare_distribution_spout_speedJerk:
  # Plot and compare the spout speed and jerk distributions.
  print()
  print('='*70)
  print('Plotting and comparing distributions of spout dynamics')
  plot_compare_distributions_spout_dynamics(
    feature_matrices_byType,
    output_filepath=os.path.join(output_dir, 'spout_dynamics_distributions.jpg') if output_dir is not None else None,
    region=None, # 'pre_pouring', 'pouring', 'post_pouring', None for all
    print_comparison_results=True,
    plot_distributions=True,
    num_histogram_bins=25, histogram_range_quantiles=(0, 0.95),
    fig=None, hide_figure_window=not keep_plots_open)
  
if plot_compare_distribution_joint_angles:
  # Plot and compare the joint angle distributions.
  print()
  print('='*70)
  print('Plotting and comparing distributions of joint angles')
  plot_compare_distributions_joint_angles(
    feature_matrices_byType,
    output_filepath=os.path.join(output_dir, 'joint_angles_distributions.jpg') if output_dir is not None else None,
    region=None, # 'pre_pouring', 'pouring', 'post_pouring', None for all
    print_comparison_results=True,
    plot_distributions=True,
    num_histogram_bins=25, histogram_range_quantiles=(0, 0.95),
    fig=None, hide_figure_window=not keep_plots_open)

if plot_compare_distribution_spout_projection:
  # Plot and compare the spout projection distributions.
  print()
  print('='*70)
  print('Plotting and comparing distributions of spout pouring projections')
  plot_compare_distributions_spout_projections(
    feature_matrices_byType, referenceObject_positions_m_byType,
    output_filepath=os.path.join(output_dir, 'joint_angles_distributions.jpg') if output_dir is not None else None,
    print_comparison_results=True,
    plot_distributions=True,
    fig=None, hide_figure_window=not keep_plots_open)
  
if plot_compare_distribution_spout_height:
  # Plot and compare the spout heights.
  print()
  print('='*70)
  print('Plotting and comparing distributions of spout heights during pouring')
  plot_compare_distributions_spout_relativeHeights(
    feature_matrices_byType, referenceObject_positions_m_byType,
    output_filepath=os.path.join(output_dir, 'spout_height_distributions.jpg') if output_dir is not None else None,
    region='pouring', # 'pre_pouring', 'pouring', 'post_pouring', None for all
    print_comparison_results=True,
    plot_distributions=True,
    num_histogram_bins=25, histogram_range_quantiles=(0, 0.95),
    fig=None, hide_figure_window=not keep_plots_open)

if plot_compare_distribution_spout_tilt:
  # Plot and compare the spout tilts.
  print()
  print('='*70)
  print('Plotting and comparing distributions of spout tilts during pouring')
  plot_compare_distributions_spout_tilts(
    feature_matrices_byType,
    output_filepath=os.path.join(output_dir, 'spout_tilt_distributions.jpg') if output_dir is not None else None,
    region='pouring', # 'pre_pouring', 'pouring', 'post_pouring', None for all
    print_comparison_results=True,
    plot_distributions=True,
    num_histogram_bins=25, histogram_range_quantiles=(0, 0.95),
    fig=None, hide_figure_window=not keep_plots_open)
  

##################################################################
# Cleanup
##################################################################

print()
print()

# Show the plot windows.
if keep_plots_open:
  print('Close all plot windows to exit')
  # matplotlib.use(default_matplotlib_backend)
  plt.show(block=True)

print()
print('Done!')
print()








