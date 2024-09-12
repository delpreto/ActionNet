
import h5py
import numpy as np
import json
import sys
import os
script_dir = os.path.dirname(os.path.realpath(__file__))

import matplotlib
default_matplotlib_backend = matplotlib.rcParams['backend']
matplotlib.rcParams['figure.max_open_warning'] = 35 # default 20
import matplotlib.pyplot as plt

from helpers.parse_process_feature_data import *
from helpers.plot_animations import *
from helpers.plot_metrics_timeseries import *
from helpers.plot_metrics_distributions import *

##################################################################
# Configuration.
##################################################################

if len(sys.argv) == 1:
  actionsense_root_dir = script_dir
  while os.path.split(actionsense_root_dir)[-1] != 'ActionSense':
    actionsense_root_dir = os.path.realpath(os.path.join(actionsense_root_dir, '..'))
    
  # Specify the directory with HDF5 files.
  data_dir = os.path.realpath(os.path.join(actionsense_root_dir, 'results', 'learning_trajectories'))
  data_dir_humans = os.path.realpath(os.path.join(data_dir, 'humans'))
  data_dir_model = os.path.join(data_dir, 'models', 'state-space',
                                # '2024-09-10_17-10'
                                '2024-09-12_15-08'
                                )
  
  # Specify the feature data to evaluate.
  # For example, may have entries for each subject
  #  and may have an entries for model outputs.
  feature_data_filepaths_byType = {
    'S00': os.path.join(data_dir_humans, 'pouring_trainingData_S00.hdf5'),
    # 'S10': os.path.join(data_dir_humans, 'pouring_trainingData_S10.hdf5'),
    'S11': os.path.join(data_dir_humans, 'pouring_trainingData_S11.hdf5'),
    # 'model': os.path.join(data_dir_model, 'pouring_modelData.hdf5'),
  }
  # Specify where outputs should be saved.
  # Can be None to not save any outputs.
  # output_dir = None
  output_dir = os.path.join(data_dir_humans, 'S00-S11')
  # output_dir = os.path.join(data_dir_model, 'only model')
  
  # Specify which outputs to process.
  # Animations.
  interactively_animate_trajectories_exampleType = None # 'S00' # interactive - can move around scene and press enter to step through time # None to not animate
  save_trajectory_animations_eachType = True
  save_trajectory_animations_compositeTypes = False
  # Plots (mostly time series).
  plot_all_trajectories_singlePlot = True
  plot_all_startingConditions_singlePlot = True
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
  plot_distributions_hand_to_pitcher_angles = True
  
  # Specify whether to show figure windows or process them in the background.
  # Either way, plots will be saved as images if output_dir is specified below.
  keep_plots_open = False
else:
  evaluation_config = json.loads(sys.argv[1])
  feature_data_filepaths_byType = evaluation_config['feature_data_filepaths_byType']
  output_dir = evaluation_config['output_dir']
  interactively_animate_trajectories_exampleType = evaluation_config['interactively_animate_trajectories_exampleType']
  save_trajectory_animations_eachType = evaluation_config['save_trajectory_animations_eachType']
  save_trajectory_animations_compositeTypes = evaluation_config['save_trajectory_animations_compositeTypes']
  plot_all_trajectories_singlePlot = evaluation_config['plot_all_trajectories_singlePlot']
  plot_all_startingConditions_singlePlot = evaluation_config['plot_all_startingConditions_singlePlot']
  plot_spout_tilt = evaluation_config['plot_spout_tilt']
  plot_spout_pouring_projection = evaluation_config['plot_spout_pouring_projection']
  plot_spout_height = evaluation_config['plot_spout_height']
  plot_spout_speedJerk = evaluation_config['plot_spout_speedJerk']
  plot_body_speedJerk = evaluation_config['plot_body_speedJerk']
  plot_joint_angles = evaluation_config['plot_joint_angles']
  plot_compare_distribution_body_speedJerk = evaluation_config['plot_compare_distribution_body_speedJerk']
  plot_compare_distribution_spout_speedJerk = evaluation_config['plot_compare_distribution_spout_speedJerk']
  plot_compare_distribution_joint_angles = evaluation_config['plot_compare_distribution_joint_angles']
  plot_compare_distribution_spout_projection = evaluation_config['plot_compare_distribution_spout_projection']
  plot_compare_distribution_spout_height = evaluation_config['plot_compare_distribution_spout_height']
  plot_compare_distribution_spout_tilt = evaluation_config['plot_compare_distribution_spout_tilt']
  plot_distributions_hand_to_pitcher_angles = evaluation_config['plot_distributions_hand_to_pitcher_angles']
  keep_plots_open = evaluation_config['keep_plots_open']
  
print()

##################################################################
# Load the data.
##################################################################

# Determine the types of examples provided, and the ones used for comparisons to the primary.
example_types = list(feature_data_filepaths_byType.keys())

# Load the feature datas.
feature_data_byType = {}
truth_data_byType = {}
for (example_type, feature_data_filepath) in feature_data_filepaths_byType.items():
  if feature_data_filepath is not None:
    feature_data_byType[example_type] = {}
    truth_data_byType[example_type] = {}
    labels = None
    # Load the feature data.
    feature_data_file = h5py.File(feature_data_filepath, 'r')
    for key in feature_data_file:
      if key == 'labels':
        labels = [x.decode('utf-8') for x in np.array(feature_data_file[key])]
      if isinstance(feature_data_file[key], h5py._hl.dataset.Dataset): # check that it is not a group
        feature_data_byType[example_type][key] = np.array(feature_data_file[key])
    
    # Load truth data if it is provided.
    if 'truth' in feature_data_file:
      for key in feature_data_file['truth']:
        truth_data_byType[example_type][key] = np.array(feature_data_file['truth'][key])
    
    feature_data_file.close()
    
    # If labels are provided, only use the human examples.
    if labels is not None:
      example_indices_toUse = [i for (i, label) in enumerate(labels) if label == 'human']
      for key in feature_data_byType[example_type]:
        feature_data_byType[example_type][key] = feature_data_byType[example_type][key][example_indices_toUse]
      for key in truth_data_byType[example_type]:
        truth_data_byType[example_type][key] = truth_data_byType[example_type][key][example_indices_toUse]
    
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
  try:
    num_trials_byType = dict([(example_type, len(x['time_s'])) for (example_type, x) in feature_data_byType.items()])
  except:
    num_trials_byType = dict([(example_type, x['feature_matrices'].shape[0]) for (example_type, x) in feature_data_byType.items()])
  max_num_trials = max(num_trials_byType.values())
  for trial_index in range(max_num_trials):
    # Get the data for this trial for each type of example provided.
    feature_data_byType_forTrial = {}
    durations_s_byType_forTrial = {}
    have_trial_for_all_types = True
    for example_type in example_types:
      if trial_index >= num_trials_byType[example_type]:
        feature_data_byType_forTrial[example_type] = None
        durations_s_byType_forTrial[example_type] = None
        have_trial_for_all_types = False
      else:
        feature_data_byType_forTrial[example_type] = {}
        for key in feature_data_byType[example_type]:
          feature_data_byType_forTrial[example_type][key] = np.squeeze(feature_data_byType[example_type][key][trial_index])
        feature_data_byType_forTrial[example_type] = parse_feature_data(feature_data_byType_forTrial[example_type])
        
    # Animate the trajectory; press enter to advance to the next timestep.
    if interactively_animate_trajectories_exampleType is not None:
      if feature_data_byType_forTrial[interactively_animate_trajectories_exampleType] is not None:
        print('Animating the trajectory; press enter on the figure window to advance time')
        animate_trajectory(
          feature_data=feature_data_byType_forTrial[interactively_animate_trajectories_exampleType],
          subject_id=interactively_animate_trajectories_exampleType, trial_index=trial_index,
          wait_for_user_after_timesteps=True)
    
    # Save the animation as a video, with a subplot for each example type.
    if save_trajectory_animations_compositeTypes:
      if have_trial_for_all_types:
        print('Saving trajectory animations as a video for trial index %02d all types' % trial_index)
        save_trajectory_animation(
          feature_data_byType_forTrial,
          subject_id='', trial_index=trial_index,
          output_filepath=os.path.join(output_dir, 'trajectory_animations', 'trajectory_animation_allTypes_trial%02d.mp4' % (trial_index)) if output_dir is not None else None
      )
    if save_trajectory_animations_eachType:
      for example_type in example_types:
        if feature_data_byType_forTrial[example_type] is not None:
          print('Saving trajectory animations as a video for trial index %02d type %s' % (trial_index, example_type))
          save_trajectory_animation(
            {example_type: feature_data_byType_forTrial[example_type]},
            subject_id='', trial_index=trial_index,
            output_filepath=os.path.join(output_dir, 'trajectory_animations', 'trajectory_animation_%s_trial%02d.mp4' % (example_type, trial_index)) if output_dir is not None else None
          )

##################################################################
# Results: Plots
##################################################################

# Plot all trajectories on a single graph.
if plot_all_trajectories_singlePlot:
  print('Plotting all trajectories on a single graph')
  for example_type in example_types:
    plot_all_trajectories(
      feature_data_byType[example_type],
      subject_id=example_type,
      output_filepath=os.path.join(output_dir, 'all_trajectories_%s.jpg' % (example_type)) if output_dir is not None else None,
      hide_figure_window=not keep_plots_open)

# Plot all starting conditions on a single graph.
if plot_all_startingConditions_singlePlot:
  print('Plotting all starting conditions on a single graph')
  plot_all_startingConditions(
    feature_data_byType, truth_data_byType=truth_data_byType,
    model_timestep_index=1,
    output_filepath=os.path.join(output_dir, 'all_startingConditions.jpg') if output_dir is not None else None,
    hide_figure_window=not keep_plots_open)

# Plot the spout tilt over time.
if plot_spout_tilt:
  print('Plotting the spout tilt, relative to the table plane')
  # Plot each trial as a trace on the plot, with a separate plot for each example type.
  for example_type in example_types:
    plot_pour_tilting(
      feature_data_byType[example_type],
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
      feature_data_byType[example_type],
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
      feature_data_byType[example_type],
      plot_mean=False, plot_std_shading=False, plot_all_trials=True,
      subtitle=example_type.title(), label=None,
      output_filepath=os.path.join(output_dir, 'spout_projection_relativeToGlass_%s.jpg' % (example_type)) if output_dir is not None else None,
      fig=None, hide_figure_window=not keep_plots_open)
  # Plot mean and standard deviation shading for each example type, on the same plot.
  previous_plot_handles = None
  for (example_index, example_type) in enumerate(example_types):
    is_last_type = example_type == example_types[-1]
    previous_plot_handles = plot_pour_relativePosition(
      feature_data_byType[example_type],
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
      feature_data_byType[example_type],
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
      feature_data_byType[example_type],
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
      feature_data_byType[example_type],
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
      feature_data_byType[example_type],
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
      feature_data_byType[example_type],
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
      feature_data_byType[example_type],
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
      feature_data_byType[example_type],
      plot_mean=True, plot_std_shading=False, plot_all_trials=True,
      subtitle=example_type.title(), label=None,
      output_filepath=os.path.join(output_dir, 'joint_angles_%s.jpg' % (example_type)) if output_dir is not None else None,
      shade_pouring_region=False,
      fig=None, hide_figure_window=not keep_plots_open)
  # Plot mean and standard deviation shading for each example type, on the same plot.
  previous_plot_handles = None
  for example_type in example_types:
    is_last_type = example_type == example_types[-1]
    previous_plot_handles = plot_body_joint_angles(
      feature_data_byType[example_type],
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
    feature_data_byType,
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
    feature_data_byType,
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
    feature_data_byType,
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
    feature_data_byType,
    output_filepath=os.path.join(output_dir, 'spout_projections_distributions.jpg') if output_dir is not None else None,
    print_comparison_results=True,
    plot_distributions=True,
    fig=None, hide_figure_window=not keep_plots_open)

if plot_distributions_hand_to_pitcher_angles:
  # Plot the estimated hand-to-pitcher holding angles.
  print()
  print('='*70)
  print('Plotting distributions of pitcher holding angles')
  plot_distribution_hand_to_pitcher_angles(
    feature_data_byType,
    output_filepath=os.path.join(output_dir, 'hand_to_pitcher_angles_distributions.jpg') if output_dir is not None else None,
    hide_figure_window=not keep_plots_open)
  
if plot_compare_distribution_spout_height:
  # Plot and compare the spout heights.
  print()
  print('='*70)
  print('Plotting and comparing distributions of spout heights during pouring')
  plot_compare_distributions_spout_relativeHeights(
    feature_data_byType,
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
    feature_data_byType,
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








