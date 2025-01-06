
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
  # activity_type = 'pouring'
  # activity_type = 'scooping'
  # activity_type = 'scoopingPowder'
  activity_type = 'stirring'
  
  actionsense_root_dir = script_dir
  while os.path.split(actionsense_root_dir)[-1] != 'ActionSense':
    actionsense_root_dir = os.path.realpath(os.path.join(actionsense_root_dir, '..'))
    
  # Specify the directory with HDF5 files.
  data_dir = os.path.realpath(os.path.join(actionsense_root_dir, 'results', 'learning_trajectories'))
  data_dir_humans = os.path.realpath(os.path.join(data_dir, 'humans'))
  data_dir_model = None
  # data_dir_model = os.path.join(data_dir, 'models', 'state-space',
  #                               # '2024-09-10_17-10'
  #                               # '2024-09-12_15-08'
  #                               # '2024-09-13_08-56'
  #                               '2024-09-13_18-15_forSubmission'
  #                               )
  
  # Specify the feature data to evaluate.
  # For example, may have entries for each subject
  #  and may have an entries for model outputs.
  feature_data_filepaths_byType = {
    # 'S00': os.path.join(data_dir_humans, '%s_trainingData_S00.hdf5' % activity_type),
    # 'S10': os.path.join(data_dir_humans, '%s_trainingData_S10.hdf5' % activity_type),
    # 'S11': os.path.join(data_dir_humans, '%s_trainingData_S11.hdf5' % activity_type),
    'S14': os.path.join(data_dir_humans, '%s_trainingData_S14.hdf5' % activity_type),
    'S15': os.path.join(data_dir_humans, '%s_trainingData_S15.hdf5' % activity_type),
    # 'Model': os.path.join(data_dir_model, 'pouring_modelData.hdf5'),
  }
  # Specify where outputs should be saved.
  # Can be None to not save any outputs.
  # output_dir = None
  output_dir = os.path.join(data_dir_humans, '%s_evaluation_outputs' % activity_type)
  # output_dir = os.path.join(data_dir_model, 'only model - new animation limits')
  
  plot_exports_extension = 'png' # jpg png pdf
  
  # Specify which outputs to process.
  # Animations.
  interactively_animate_trajectories_exampleType = None # 'S00' # interactive - can move around scene and press enter to step through time # None to not animate
  save_trajectory_animations_eachType = False
  save_trajectory_animations_compositeTypes = False
  # Plots (mostly time series).
  plot_all_trajectories_singlePlot = True
  plot_all_startingConditions_singlePlot = True
  plot_motionObject_tilt = True
  plot_motionObjectKeypoint_projection = True
  plot_motionObjectKeypoint_height = True
  plot_motionObjectKeypoint_speedJerk = True
  plot_body_speedJerk = True
  plot_joint_angles = True
  # Plots and comparisons of distributions.
  plot_compare_distribution_body_speedJerk = True
  plot_compare_distribution_motionObjectKeypoint_speedJerk = True # includes Wasserstein distances
  plot_compare_distribution_joint_angles = True
  plot_compare_distribution_motionObjectKeypoint_projection = True
  plot_compare_distribution_motionObjectKeypoint_height = True
  plot_compare_distribution_motionObject_tilt = True
  plot_distributions_hand_to_pitcher_angles = False
  
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
  plot_motionObject_tilt = evaluation_config['plot_motionObject_tilt']
  plot_motionObjectKeypoint_projection = evaluation_config['plot_motionObjectKeypoint_pouring_projection']
  plot_motionObjectKeypoint_height = evaluation_config['plot_motionObjectKeypoint_height']
  plot_motionObjectKeypoint_speedJerk = evaluation_config['plot_motionObjectKeypoint_speedJerk']
  plot_body_speedJerk = evaluation_config['plot_body_speedJerk']
  plot_joint_angles = evaluation_config['plot_joint_angles']
  plot_compare_distribution_body_speedJerk = evaluation_config['plot_compare_distribution_body_speedJerk']
  plot_compare_distribution_motionObjectKeypoint_speedJerk = evaluation_config['plot_compare_distribution_motionObjectKeypoint_speedJerk']
  plot_compare_distribution_joint_angles = evaluation_config['plot_compare_distribution_joint_angles']
  plot_compare_distribution_motionObjectKeypoint_projection = evaluation_config['plot_compare_distribution_motionObjectKeypoint_projection']
  plot_compare_distribution_motionObjectKeypoint_height = evaluation_config['plot_compare_distribution_motionObjectKeypoint_height']
  plot_compare_distribution_motionObject_tilt = evaluation_config['plot_compare_distribution_motionObject_tilt']
  plot_distributions_hand_to_pitcher_angles = evaluation_config['plot_distributions_hand_to_pitcher_angles']
  keep_plots_open = evaluation_config['keep_plots_open']
  plot_exports_extension = evaluation_config['plot_exports_extension']
  activity_type = evaluation_config['activity_type'] if 'activity_type' in evaluation_config else 'pouring'
  
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
#  The motion object will be a gray box.
#  The human arm will be two connected black lines, with gray circles at the joints.
#  A black circle will be at the origin.
#  The light blue circle represents the reference object, with a dark circle at its center.
#  The magenta circle represents the projection of the motion object tip onto the table.
#  The red line from the magenta circle represents the angle of the motion object (water direction if the activity is pouring).
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
        animate_trajectory(activity_type,
          feature_data=feature_data_byType_forTrial[interactively_animate_trajectories_exampleType],
          subject_id=interactively_animate_trajectories_exampleType, trial_index=trial_index,
          wait_for_user_after_timesteps=True)
    
    # Save the animation as a video, with a subplot for each example type.
    if save_trajectory_animations_compositeTypes:
      if have_trial_for_all_types:
        print('Saving trajectory animations as a video for trial index %02d all types' % trial_index)
        save_trajectory_animation(activity_type,
          feature_data_byType_forTrial,
          subject_id='', trial_index=trial_index,
          output_filepath=os.path.join(output_dir, 'trajectory_animations', 'trajectory_animation_allTypes_trial%02d.mp4' % (trial_index)) if output_dir is not None else None
      )
    if save_trajectory_animations_eachType:
      for example_type in example_types:
        if feature_data_byType_forTrial[example_type] is not None:
          print('Saving trajectory animations as a video for trial index %02d type %s' % (trial_index, example_type))
          save_trajectory_animation(activity_type,
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
      activity_type,
      subject_id=example_type,
      output_filepath=os.path.join(output_dir, 'all_trajectories_%s.%s' % (example_type, plot_exports_extension)) if output_dir is not None else None,
      hide_figure_window=not keep_plots_open)

# Plot all starting conditions on a single graph.
if plot_all_startingConditions_singlePlot:
  print('Plotting all starting conditions on a single graph')
  plot_all_startingConditions(
    feature_data_byType, activity_type, truth_data_byType=truth_data_byType,
    model_timestep_index=1,
    output_filepath=os.path.join(output_dir, 'all_startingConditions.%s' % plot_exports_extension) if output_dir is not None else None,
    hide_figure_window=not keep_plots_open)

# Plot the motion object keypoint tilt over time.
if plot_motionObject_tilt:
  print('Plotting the %s tilt, relative to the table plane' % motionObject_name[activity_type])
  # Plot each trial as a trace on the plot, with a separate plot for each example type.
  for example_type in example_types:
    plot_motionObject_tilting(
      feature_data_byType[example_type], activity_type,
      plot_mean=True, plot_std_shading=False, plot_all_trials=True,
      subtitle=example_type.title(), label=None,
      output_filepath=os.path.join(output_dir, '%s_tilt_angle_%s.%s' % (motionObject_name[activity_type], example_type, plot_exports_extension)) if output_dir is not None else None,
      shade_stationary_region=False,
      fig=None, hide_figure_window=not keep_plots_open)
  # Plot mean and standard deviation shading for each example type, on the same plot.
  previous_plot_handles = None
  for example_type in example_types:
    is_last_type = example_type == example_types[-1]
    previous_plot_handles = plot_motionObject_tilting(
      feature_data_byType[example_type], activity_type,
      plot_mean=True, plot_std_shading=True, plot_all_trials=False,
      subtitle=None, label=example_type.title(),
      output_filepath=(os.path.join(output_dir, '%s_tilt_angle_allTypes.%s' % (motionObject_name[activity_type], plot_exports_extension)) if is_last_type else None) if output_dir is not None else None,
      shade_stationary_region=False,
      fig=previous_plot_handles, hide_figure_window=not keep_plots_open)

if plot_motionObjectKeypoint_projection:
  # Plot the object keypoint projection onto the table relative to the reference object.
  #   The blue shaded circle represents the reference object.
  #   A point will be plotted at the projection onto the table, relative to the reference object.
  #   If the activity is pouring:
  #     The y-axis is the pouring direction, as inferred from the yaw of the pitcher in each trial.
  #     So the water would pour upward on the plot from the plotted spout position.
  print('Plotting the %s pouring position relative to the %s' % (motionObjectKeypoint_name[activity_type], referenceObject_name[activity_type]))
  # Plot each trial as a separate point, with a separate plot for each example type.
  for example_type in example_types:
    plot_motionObjectKeypoint_relativePosition(
      feature_data_byType[example_type], activity_type,
      plot_mean=False, plot_std_shading=False, plot_all_trials=True,
      subtitle=example_type.title(), label=None,
      output_filepath=os.path.join(output_dir, '%s_projection_relativeTo%s_%s.%s' % (motionObjectKeypoint_name[activity_type], referenceObject_name[activity_type], example_type, plot_exports_extension)) if output_dir is not None else None,
      fig=None, hide_figure_window=not keep_plots_open)
  # Plot mean and standard deviation shading for each example type, on the same plot.
  previous_plot_handles = None
  for (example_index, example_type) in enumerate(example_types):
    is_last_type = example_type == example_types[-1]
    previous_plot_handles = plot_motionObjectKeypoint_relativePosition(
      feature_data_byType[example_type], activity_type,
      plot_mean=True, plot_std_shading=True, plot_all_trials=False,
      subtitle=None, label=example_type.title(),
      color=plt.rcParams['axes.prop_cycle'].by_key()['color'][example_index],
      output_filepath=(os.path.join(output_dir, '%s_projection_relativeTo%s_allTypes.%s' % (motionObjectKeypoint_name[activity_type], referenceObject_name[activity_type], plot_exports_extension)) if is_last_type else None) if output_dir is not None else None,
      fig=previous_plot_handles, hide_figure_window=not keep_plots_open)

if plot_motionObjectKeypoint_height:
  # Plot the object keypoint height over time, relative to the reference object height.
  print('Plotting the %s height relative to the %s' % (motionObjectKeypoint_name[activity_type], referenceObject_name[activity_type]))
  # Plot each trial as a trace on the plot, with a separate plot for each example type.
  for example_type in example_types:
    plot_motionObjectKeypoint_relativeHeight(
      feature_data_byType[example_type], activity_type,
      plot_mean=True, plot_std_shading=False, plot_all_trials=True,
      subtitle=example_type.title(), label=None,
      output_filepath=os.path.join(output_dir, '%s_height_%s.%s' % (motionObjectKeypoint_name[activity_type], example_type, plot_exports_extension)) if output_dir is not None else None,
      shade_stationary_region=False,
      fig=None, hide_figure_window=not keep_plots_open)
  # Plot mean and standard deviation shading for each example type, on the same plot.
  previous_plot_handles = None
  for example_type in example_types:
    is_last_type = example_type == example_types[-1]
    previous_plot_handles = plot_motionObjectKeypoint_relativeHeight(
      feature_data_byType[example_type], activity_type,
      plot_mean=True, plot_std_shading=True, plot_all_trials=False,
      subtitle=None, label=example_type.title(),
      output_filepath=(os.path.join(output_dir, '%s_height_allTypes.%s' % (motionObjectKeypoint_name[activity_type], plot_exports_extension)) if is_last_type else None) if output_dir is not None else None,
      shade_stationary_region=False,
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
      output_filepath=os.path.join(output_dir, 'body_dynamics_%s.%s' % (example_type, plot_exports_extension)) if output_dir is not None else None,
      shade_stationary_region=False,
      fig=None, hide_figure_window=not keep_plots_open)
  # Plot mean and standard deviation shading for each example type, on the same plot.
  previous_plot_handles = None
  for example_type in example_types:
    is_last_type = example_type == example_types[-1]
    previous_plot_handles =  plot_body_dynamics(
      feature_data_byType[example_type],
      plot_mean=True, plot_std_shading=True, plot_all_trials=False,
      subtitle=None, label=example_type.title(),
      output_filepath=(os.path.join(output_dir, 'body_dynamics_allTypes.%s' % plot_exports_extension) if is_last_type else None) if output_dir is not None else None,
      shade_stationary_region=False,
      fig=previous_plot_handles, hide_figure_window=not keep_plots_open)

if plot_motionObjectKeypoint_speedJerk:
  # Plot the motion object speed and jerk over time.
  print('Plotting %s dynamics' % motionObjectKeypoint_name[activity_type])
  # Plot each trial as a trace on the plot, with a separate plot for each example type.
  for example_type in example_types:
    plot_motionObjectKeypoint_dynamics(
      feature_data_byType[example_type], activity_type,
      plot_mean=True, plot_std_shading=False, plot_all_trials=True,
      subtitle=example_type.title(), label=None,
      output_filepath=os.path.join(output_dir, '%s_dynamics_%s.%s' % (motionObjectKeypoint_name[activity_type], example_type, plot_exports_extension)) if output_dir is not None else None,
      shade_stationary_region=False,
      fig=None, hide_figure_window=not keep_plots_open)
  # Plot mean and standard deviation shading for each example type, on the same plot.
  previous_plot_handles = None
  for example_type in example_types:
    is_last_type = example_type == example_types[-1]
    previous_plot_handles =  plot_motionObjectKeypoint_dynamics(
      feature_data_byType[example_type], activity_type,
      plot_mean=True, plot_std_shading=True, plot_all_trials=False,
      subtitle=None, label=example_type.title(),
      output_filepath=(os.path.join(output_dir, '%s_dynamics_allTypes.%s' % (motionObjectKeypoint_name[activity_type], plot_exports_extension)) if is_last_type else None) if output_dir is not None else None,
      shade_stationary_region=False,
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
      output_filepath=os.path.join(output_dir, 'joint_angles_%s.%s' % (example_type, plot_exports_extension)) if output_dir is not None else None,
      shade_stationary_region=False,
      fig=None, hide_figure_window=not keep_plots_open)
  # Plot mean and standard deviation shading for each example type, on the same plot.
  previous_plot_handles = None
  for example_type in example_types:
    is_last_type = example_type == example_types[-1]
    previous_plot_handles = plot_body_joint_angles(
      feature_data_byType[example_type],
      plot_mean=True, plot_std_shading=True, plot_all_trials=False,
      subtitle=None, label=example_type.title(),
      output_filepath=(os.path.join(output_dir, 'joint_angles_allTypes.%s' % plot_exports_extension) if is_last_type else None) if output_dir is not None else None,
      shade_stationary_region=False,
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
    feature_data_byType, activity_type,
    output_filepath=os.path.join(output_dir, 'body_dynamics_distributions.%s' % plot_exports_extension) if output_dir is not None else None,
    region=None, # 'pre_stationary', 'stationary', 'post_stationary', None for all
    print_comparison_results=True,
    plot_distributions=True,
    num_histogram_bins=25, histogram_range_quantiles=(0, 0.95),
    fig=None, hide_figure_window=not keep_plots_open)

if plot_compare_distribution_motionObjectKeypoint_speedJerk:
  # Plot and compare the motion object keypoint speed and jerk distributions.
  print()
  print('='*70)
  print('Plotting and comparing distributions of %s dynamics' % motionObjectKeypoint_name[activity_type])
  plot_compare_distributions_motionObjectKeypoint_dynamics(
    feature_data_byType, activity_type,
    output_filepath=os.path.join(output_dir, '%s_dynamics_distributions.%s' % (motionObjectKeypoint_name[activity_type], plot_exports_extension)) if output_dir is not None else None,
    region=None, # 'pre_stationary', 'stationary', 'post_stationary', None for all
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
    feature_data_byType, activity_type,
    output_filepath=os.path.join(output_dir, 'joint_angles_distributions.%s' % plot_exports_extension) if output_dir is not None else None,
    region=None, # 'pre_stationary', 'stationary', 'post_stationary', None for all
    print_comparison_results=True,
    plot_distributions=True,
    num_histogram_bins=25, histogram_range_quantiles=(0, 0.95),
    fig=None, hide_figure_window=not keep_plots_open)

if plot_compare_distribution_motionObjectKeypoint_projection:
  # Plot and compare the motion object keypoint projection distributions.
  print()
  print('='*70)
  print('Plotting and comparing distributions of %s pouring projections' % motionObjectKeypoint_name[activity_type])
  plot_compare_distributions_motionObjectKeypoint_projections(
    feature_data_byType, activity_type,
    output_filepath=os.path.join(output_dir, '%s_projections_distributions.%s' % (motionObjectKeypoint_name[activity_type], plot_exports_extension)) if output_dir is not None else None,
    print_comparison_results=True,
    plot_distributions=True,
    fig=None, hide_figure_window=not keep_plots_open)

if plot_distributions_hand_to_pitcher_angles:
  if activity_type != 'pouring':
    raise AssertionError('Evaluating hand to pitcher angles is only available for the pouring activity')
  # Plot the estimated hand-to-pitcher holding angles.
  print()
  print('='*70)
  print('Plotting distributions of pitcher holding angles')
  plot_distribution_hand_to_pitcher_angles(
    feature_data_byType,
    output_filepath=os.path.join(output_dir, 'hand_to_pitcher_angles_distributions.%s' % plot_exports_extension) if output_dir is not None else None,
    hide_figure_window=not keep_plots_open)
  
if plot_compare_distribution_motionObjectKeypoint_height:
  # Plot and compare the motion object keypoint heights.
  print()
  print('='*70)
  print('Plotting and comparing distributions of %s heights during pouring' % motionObjectKeypoint_name[activity_type])
  plot_compare_distributions_motionObjectKeypoint_relativeHeights(
    feature_data_byType, activity_type,
    output_filepath=os.path.join(output_dir, '%s_height_distributions.%s' % (motionObjectKeypoint_name[activity_type], plot_exports_extension)) if output_dir is not None else None,
    region='stationary', # 'pre_stationary', 'stationary', 'post_stationary', None for all
    print_comparison_results=True,
    plot_distributions=True,
    num_histogram_bins=25, histogram_range_quantiles=(0, 0.95),
    fig=None, hide_figure_window=not keep_plots_open)

if plot_compare_distribution_motionObject_tilt:
  # Plot and compare the motion object keypoint tilts.
  print()
  print('='*70)
  print('Plotting and comparing distributions of %s tilts during the stationary period' % motionObjectKeypoint_name[activity_type])
  plot_compare_distributions_motionObject_tilts(
    feature_data_byType, activity_type,
    output_filepath=os.path.join(output_dir, '%s_tilt_distributions.%s' % (motionObject_name[activity_type], plot_exports_extension)) if output_dir is not None else None,
    region='stationary', # 'pre_stationary', 'stationary', 'post_stationary', None for all
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








