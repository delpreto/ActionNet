############
#
# Copyright (c) 2024 MIT CSAIL and Joseph DelPreto
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
# Created 2021-2024 for the MIT ActionSense project by Joseph DelPreto [https://josephdelpreto.com].
#
############

from learning_trajectories.helpers.configuration import *
from learning_trajectories.helpers.transformation import *
from learning_trajectories.helpers.parse_process_feature_data import *
from learning_trajectories.helpers.numpy_scipy_utils import *
from learning_trajectories.helpers.plot_animations import plt_wait_for_keyboard_press

from scipy import stats
import cv2
import os
import distinctipy

import matplotlib
# Avoid type 3 fonts (which are not accepted by PaperPlaza)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# Store the default backend, for switching between visible/hidden plotting.
default_matplotlib_backend = matplotlib.rcParams['backend']

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from mpl_toolkits.mplot3d import art3d


# ================================================================
# Plot the spout projection onto the table relative to the glass.
#   The blue shaded circle represents the glass.
#   The y-axis is the pouring direction, as inferred from the yaw of the pitcher in each trial.
#   A point will be plotted at the spout's projection onto the table, relative to the glass.
#   So the water would pour upward on the plot from the plotted spout position.
def plot_pour_relativePosition(feature_data_byTrial,
                               plot_all_trials=True, plot_std_shading=False, plot_mean=False,
                               subtitle=None, label=None,
                               fig=None, hide_figure_window=False,
                               color=None,
                               output_filepath=None):
  if output_filepath is not None:
    os.makedirs(os.path.split(output_filepath)[0], exist_ok=True)
  if not isinstance(feature_data_byTrial, dict):
    raise AssertionError('feature_data_allTrials should map keys to matrices whose first dimension is the trial index')
  if not (isinstance(feature_data_byTrial['time_s'], np.ndarray) and feature_data_byTrial['time_s'].ndim == 3):
    raise AssertionError('feature_data_allTrials should map keys to matrices whose first dimension is the trial index')
  if isinstance(fig, (list, tuple)):
    fig = fig[0]
  if fig is None:
    if hide_figure_window:
      try:
        matplotlib.use('Agg')
      except:
        pass
    else:
      matplotlib.use(default_matplotlib_backend)
    fig = plt.figure(figsize=(9, 7))
    # if not hide_figure_window:
    #   figManager = plt.get_current_fig_manager()
    #   figManager.window.showMaximized()
    #   plt_wait_for_keyboard_press(0.3)
    plt.ion()
    fig.add_subplot(1, 1, 1)
    ax = fig.get_axes()[0]
  else:
    ax = fig.get_axes()[0]
  
  num_trials = feature_data_byTrial['time_s'].shape[0]
  num_timesteps = feature_data_byTrial['time_s'].shape[1]
  
  spout_relativeOffsets_cm = np.zeros((num_trials, 2))
  for trial_index in range(num_trials):
    feature_data = get_feature_data_for_trial(feature_data_byTrial, trial_index)
    # Get the pouring time.
    pouring_inference = infer_pour_pose(feature_data)
    pour_index = pouring_inference['time_index']
    # Get the pouring relative offset.
    spout_relativeOffsets_cm[trial_index, :] = infer_spout_relativeOffset_cm(
      feature_data,
      referenceObject_position_m=None, # will use the one in feature_data
      hand_to_pitcher_rotation_toUse=None, # will use the one in feature_data
      time_index=pour_index)
  
  # Plot a standard deviation shaded region if desired.
  if plot_std_shading:
    if label.lower() in example_types_to_offset:
      spout_relativeOffsets_cm += np.array([1, 2])
      
    # Plot the ellipse for the spout positions.
    plot_confidence_ellipse(x=spout_relativeOffsets_cm[:,0], y=spout_relativeOffsets_cm[:,1], ax=ax,
                            n_std=1.0, facecolor='none', alpha=1, edgecolor=color, linewidth=3)
    plot_confidence_ellipse(x=spout_relativeOffsets_cm[:,0], y=spout_relativeOffsets_cm[:,1], ax=ax,
                            n_std=2.0, facecolor='none', alpha=0.2, edgecolor=color, linewidth=3,)
    plot_confidence_ellipse(x=spout_relativeOffsets_cm[:,0], y=spout_relativeOffsets_cm[:,1], ax=ax,
                            n_std=3.0, facecolor='none', alpha=0.2, edgecolor=color, linewidth=3)
    plot_confidence_ellipse(x=spout_relativeOffsets_cm[:,0], y=spout_relativeOffsets_cm[:,1], ax=ax,
                            n_std=3.0, facecolor=color, alpha=0.2,
                            label=('%s: StdDevs' % label) if label is not None else 'StdDevs')
    ax.legend()
    
  # Plot all trial results if desired.
  if plot_all_trials:
    for trial_index in range(num_trials):
      ax.scatter(*spout_relativeOffsets_cm[trial_index,:], c=color, s=125)
  
  # Plot the mean if desired.
  if plot_mean:
    ax.scatter(*np.mean(spout_relativeOffsets_cm, axis=0), c=color, s=40)
  
  # Plot the reference object and the origin.
  referenceObject_circle = mpatches.Circle(
      (0, 0),
      radius=referenceObject_diameter_cm/2, ec=None, color=[0.8,1,1],
      linewidth=3, alpha=0.13)
  ax.add_patch(referenceObject_circle)
  referenceObject_circle = mpatches.Circle(
      (0, 0),
      radius=referenceObject_diameter_cm/2, ec=[0.4,1,1], facecolor='none',
      linewidth=3, alpha=1)
  ax.add_patch(referenceObject_circle)
  # ax.scatter(0, 0, s=10, c='k')
  
  # Plot formatting.
  axis_fontsize = 24
  title_fontsize = 24
  ax.set_ylim([-5, 5])
  ax.set_xlim([-5, 5])
  ax.tick_params(axis='x', labelsize=18)  # Set x-axis tick font size
  ax.tick_params(axis='y', labelsize=18)  # Set y-axis tick font size
  ax.set_aspect('equal')
  ax.set_xlabel('Perpendicular to Pouring Direction [cm]', fontsize=axis_fontsize)
  ax.set_ylabel('Along Pouring Direction [cm]', fontsize=axis_fontsize)
  # plt.title('Pouring Position and Direction%s' % ((': %s' % subtitle) if subtitle is not None else ''), fontsize=title_fontsize)
  plt.title('Pouring Relative to Glass', fontsize=title_fontsize)
  ax.grid(True, color='lightgray')
  ax.set_axisbelow(True)
  
  # Show the plot.
  plt.draw()
  
  # Save the plot if desired.
  if output_filepath is not None:
    fig.savefig(output_filepath, dpi=300)
  
  return (fig, spout_relativeOffsets_cm)

# ================================================================
# Plot and compare distributions of the spout position and orientation projected onto the table
#  during the inferred pouring window.
#  The projection is such that:
#   The y-axis is the pouring direction, as inferred from the yaw of the pitcher in each trial.
#   A point will be plotted at the spout's projection onto the table, relative to the glass.
#   So the water would pour upward on the plot from the plotted spout position.
# feature_matrices_byType and referenceObject_positions_m_byType are
#   dictionaries mapping distribution category to matrices for each trial.
def plot_compare_distributions_spout_projections(feature_data_byType,
                                                 output_filepath=None,
                                                 print_comparison_results=True,
                                                 plot_distributions=True,
                                                 fig=None, hide_figure_window=False):
  
  # Plot mean and standard deviation shading for each example type, on the same plot.
  spout_relativeOffsets_cm_byType = {}
  example_types = list(feature_data_byType.keys())
  previous_results = (fig, None)
  for (example_index, example_type) in enumerate(example_types):
    is_last_type = example_type == example_types[-1]
    previous_results = plot_pour_relativePosition(
      feature_data_byType[example_type],
      plot_mean=True, plot_std_shading=True, plot_all_trials=False,
      subtitle=None, label=example_type.title(),
      color=plt.rcParams['axes.prop_cycle'].by_key()['color'][example_index],
      output_filepath=output_filepath,
      fig=previous_results, hide_figure_window=hide_figure_window or (not plot_distributions))
    spout_relativeOffsets_cm_byType[example_type] = previous_results[1]
  
  # Statistical tests.
  results_relativeOffsets = dict([(example_type, dict([(key, {}) for key in example_types])) for example_type in example_types])
  
  # Statistical tests to compare the distributions.
  for example_type_1 in example_types:
    for example_type_2 in example_types:
      for (axis_index, axis) in enumerate(['x', 'y']):
        spout_relativeOffsets_cm_1 = spout_relativeOffsets_cm_byType[example_type_1][:, axis_index]
        spout_relativeOffsets_cm_2 = spout_relativeOffsets_cm_byType[example_type_2][:, axis_index]
        results_relativeOffsets[example_type_1][example_type_2][axis] = \
          stats.kstest(spout_relativeOffsets_cm_1, spout_relativeOffsets_cm_2,
                       alternative='two-sided', # 'two-sided', 'less', 'greater'
                       method='auto', # ‘auto’, ‘exact’, ‘approx’, ‘asymp’
                       )
  
  # Print statistical test results.
  if print_comparison_results:
    for axis in ['x', 'y']:
      print(' Statistical comparison results for spout projection, %s axis:' % axis)
      for example_type_1 in example_types:
        for example_type_2 in example_types:
          print('  Comparing [%s] to [%s]: ' % (example_type_1, example_type_2), end='')
          results = results_relativeOffsets[example_type_1][example_type_2][axis]
          p = results.pvalue
          print('Different? %s (p = %0.4f)' % ('yes' if p < 0.05 else 'no', p))
  
  return previous_results

# ================================================================
# Plot distributions of the hand-to-pitcher holding angles
#  during the inferred pouring window.
def plot_distribution_hand_to_pitcher_angles(feature_data_byType,
                                              output_filepath=None,
                                              hide_figure_window=False):
  if hide_figure_window:
    try:
      matplotlib.use('Agg')
    except:
      pass
  else:
    matplotlib.use(default_matplotlib_backend)
  fig, axs = plt.subplots(nrows=1, ncols=3,
                          squeeze=False, # if False, always return 2D array of axes
                          sharex=False, sharey=False,
                          subplot_kw={'frame_on': True},
                          figsize=(13, 7),
                          )
  if 'hand_to_pitcher_angles_rad' not in list(feature_data_byType.values())[0]:
    return None
  
  if not hide_figure_window:
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt_wait_for_keyboard_press(0.3)
  # Plot box plots for each example type, for each angle axis.
  axis_names = ['X (left/right)', 'Y (down/up)', 'Z (inward/outward)']
  median_color = [0, 0, 0.75]
  example_type_colors = distinctipy.get_colors(len(feature_data_byType), exclude_colors=[median_color])
  for (axis_index, axis_name) in enumerate(axis_names):
    ax = axs[0][axis_index]
    data = []
    for (example_type, feature_data) in feature_data_byType.items():
      hand_to_pitcher_angles_rad = np.squeeze(feature_data['hand_to_pitcher_angles_rad'])
      data.append(np.degrees(hand_to_pitcher_angles_rad[:, axis_index]))
    boxplot_dict = ax.boxplot(data, labels=list(feature_data_byType.keys()), patch_artist=True)
    for b, c in zip(boxplot_dict['boxes'], example_type_colors):
      b.set_alpha(0.6)
      b.set_edgecolor('k') # or try 'black'
      b.set_facecolor(c)
      b.set_linewidth(1)
    for median in boxplot_dict['medians']:
      median.set_color(median_color)
      median.set_linewidth(2)
    ax.set_title('%s' % axis_name)
    # ax.set_xlabel('Data Source')
    ax.set_ylabel('Angle [deg]')
    ax.grid(True, color='lightgray')
  # Save the plot if desired.
  if output_filepath is not None:
    fig.savefig(output_filepath, dpi=300)
  return None

# ================================================================
# Plot and compare distributions of the spout speed and jerk.
# feature_matrices_allTypes is a dictionary mapping distribution category to matrices for each trial.
# If region is provided, will only consider timesteps during that region for each trial.
#   Can be 'pre_pouring', 'pouring', 'post_pouring', or None for all.
def plot_compare_distributions_spout_dynamics(feature_data_byType,
                                              subtitle=None,
                                              output_filepath=None,
                                              region=None,  # 'pre_pouring', 'pouring', 'post_pouring', None for all
                                              print_comparison_results=True,
                                              plot_distributions=True,
                                              num_histogram_bins=100, histogram_range_quantiles=(0,1),
                                              fig=None, hide_figure_window=False):
    
  if output_filepath is not None:
    os.makedirs(os.path.split(output_filepath)[0], exist_ok=True)
  if plot_distributions:
    if fig is None:
      if hide_figure_window:
        try:
          matplotlib.use('Agg')
        except:
          pass
      else:
        matplotlib.use(default_matplotlib_backend)
      fig, axs = plt.subplots(nrows=2, ncols=1,
                                 squeeze=False, # if False, always return 2D array of axes
                                 sharex=False, sharey=False,
                                 subplot_kw={'frame_on': True},
                                 figsize=(13, 7),
                                 )
      if not hide_figure_window:
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt_wait_for_keyboard_press(0.3)
      plt.ion()
    else:
      (fig, axs) = fig
  
  # Initialize results/outputs.
  example_types = list(feature_data_byType.keys())
  distributions_speed_m_s = dict.fromkeys(example_types, None)
  distributions_jerk_m_s_s_s = dict.fromkeys(example_types, None)
  results_speed_m_s = dict([(example_type, dict.fromkeys(example_types, None)) for example_type in example_types])
  results_jerk_m_s_s_s = dict([(example_type, dict.fromkeys(example_types, None)) for example_type in example_types])
  
  # Process each example type (each distribution category).
  for (example_type, feature_data_allTrials) in feature_data_byType.items():
    num_trials = feature_data_allTrials['time_s'].shape[0]
    num_timesteps = feature_data_allTrials['time_s'].shape[1]
    # Get the spout dynamics for each timestep.
    speeds_m_s = [None]*num_trials
    speeds_m_s_pouring = [None]*num_trials
    jerks_m_s_s_s = [None]*num_trials
    for trial_index in range(num_trials):
      feature_data = get_feature_data_for_trial(feature_data_allTrials, trial_index)
      speeds_m_s[trial_index] = infer_spout_speed_m_s(feature_data)
      jerks_m_s_s_s[trial_index] = infer_spout_jerk_m_s_s_s(feature_data)
    
    # If desired, only extract a certain window of time relative to the inferred pouring time.
    for trial_index in range(num_trials):
      feature_data = get_feature_data_for_trial(feature_data_allTrials, trial_index)
      pouring_inference = infer_pour_pose(feature_data)
      pour_start_index = pouring_inference['start_time_index']
      pour_end_index = pouring_inference['end_time_index']
      if region == 'pouring':
        speeds_m_s[trial_index] = speeds_m_s[trial_index][pour_start_index:pour_end_index]
        jerks_m_s_s_s[trial_index] = jerks_m_s_s_s[trial_index][pour_start_index:pour_end_index]
      if region == 'pre_pouring':
        speeds_m_s[trial_index] = speeds_m_s[trial_index][0:pour_start_index]
        jerks_m_s_s_s[trial_index] = jerks_m_s_s_s[trial_index][0:pour_start_index]
      if region == 'post_pouring':
        speeds_m_s[trial_index] = speeds_m_s[trial_index][pour_end_index:-1]
        jerks_m_s_s_s[trial_index] = jerks_m_s_s_s[trial_index][pour_end_index:-1]
      
    # Store results.
    distributions_speed_m_s[example_type] = np.abs(np.stack(np.concatenate(speeds_m_s)))
    distributions_speed_m_s[example_type] = distributions_speed_m_s[example_type][~np.isnan(distributions_speed_m_s[example_type])]
    distributions_jerk_m_s_s_s[example_type] = np.abs(np.stack(np.concatenate(jerks_m_s_s_s)))
    distributions_jerk_m_s_s_s[example_type] = distributions_jerk_m_s_s_s[example_type][~np.isnan(distributions_jerk_m_s_s_s[example_type])]
    distributions_speed_m_s['%s_half1' % example_type] = distributions_speed_m_s[example_type][0:distributions_speed_m_s[example_type].shape[0]//2]
    distributions_speed_m_s['%s_half2' % example_type] = distributions_speed_m_s[example_type][distributions_speed_m_s[example_type].shape[0]//2:]
    distributions_jerk_m_s_s_s['%s_half1' % example_type] = distributions_jerk_m_s_s_s[example_type][0:distributions_jerk_m_s_s_s[example_type].shape[0]//2]
    distributions_jerk_m_s_s_s['%s_half2' % example_type] = distributions_jerk_m_s_s_s[example_type][distributions_jerk_m_s_s_s[example_type].shape[0]//2:]
  
  # Create a category for all non-model example types.
  distributions_jerk_m_s_s_s_nonModel = []
  distributions_speed_m_s_nonModel = []
  combined_type = ''
  for example_type in example_types:
    if 'model' not in example_type:
      combined_type += example_type
      distributions_jerk_m_s_s_s_nonModel.append(distributions_jerk_m_s_s_s[example_type])
      distributions_speed_m_s_nonModel.append(distributions_speed_m_s[example_type])
  distributions_jerk_m_s_s_s[combined_type] = np.concatenate(distributions_jerk_m_s_s_s_nonModel)
  distributions_speed_m_s[combined_type] = np.concatenate(distributions_speed_m_s_nonModel)
  
  # # Print the speed and jerk for copying into Matlab.
  # print('distributions_jerk_m_s_s_s')
  # for k, v in distributions_jerk_m_s_s_s.items():
  #   print('%s = %s\';' % (k, list(v)))
  # print('distributions_speed_m_s')
  # for k, v in distributions_speed_m_s.items():
  #   print(k)
  #   print(np.mean(v), np.std(v))
  #   # print('%s = %s\';' % (k, list(v)))
  # # print(distributions_jerk_m_s_s_s.keys())
  
  # Print summaries of the speed and jerk distributions.
  print()
  print('Jerk [cm/s/s/s]')
  for k, v in distributions_jerk_m_s_s_s.items():
    print('  %s: mean %g stdev %g' % (k, np.mean(100*v), np.std(100*v)))
  print()
  print('Speed [cm/s]')
  for k, v in distributions_speed_m_s.items():
    print('  %s: mean %g stdev %g' % (k, np.mean(100*v), np.std(100*v)))
  
  # Statistical tests to compare the distributions.
  print()
  print('Wasserstein Distances for Jerk [using cm/s/s/s]')
  max_str_len = max([len(x) for x in distributions_jerk_m_s_s_s])
  for example_type_1 in distributions_jerk_m_s_s_s:
    for example_type_2 in distributions_jerk_m_s_s_s:
      jerks_m_s_s_s_1 = distributions_jerk_m_s_s_s[example_type_1]
      jerks_m_s_s_s_2 = distributions_jerk_m_s_s_s[example_type_2]
      emd = stats.wasserstein_distance(100*jerks_m_s_s_s_1, 100*jerks_m_s_s_s_2)
      print('  %s <> %s: %g' % (example_type_1.ljust(max_str_len), example_type_2.ljust(max_str_len), emd))
  print()
  print('Wasserstein Distances for Speed [using cm/s]')
  max_str_len = max([len(x) for x in distributions_speed_m_s])
  for example_type_1 in distributions_speed_m_s:
    for example_type_2 in distributions_speed_m_s:
      speeds_m_s_1 = distributions_speed_m_s[example_type_1]
      speeds_m_s_2 = distributions_speed_m_s[example_type_2]
      emd = stats.wasserstein_distance(100*speeds_m_s_1, 100*speeds_m_s_2)
      print('  %s <> %s: %g' % (example_type_1.ljust(max_str_len), example_type_2.ljust(max_str_len), emd))
  print()
  
  # Plot distributions.
  if plot_distributions:
    ax_speed = axs[0][0]
    ax_jerk = axs[1][0]
    for example_type in example_types:
      speeds_m_s = distributions_speed_m_s[example_type]
      jerks_m_s_s_s = distributions_jerk_m_s_s_s[example_type]
      ax_speed.hist(speeds_m_s, bins=num_histogram_bins,
                    histtype='stepfilled', # 'bar', 'barstacked', 'step', 'stepfilled'
                    log=False, density=True,
                    range=(np.nanquantile(speeds_m_s, histogram_range_quantiles[0]), np.nanquantile(speeds_m_s, histogram_range_quantiles[1])),
                    alpha=0.5, label=example_type.title())
      ax_jerk.hist(jerks_m_s_s_s, bins=num_histogram_bins,
                   histtype='stepfilled', # 'bar', 'barstacked', 'step', 'stepfilled'
                   log=False, density=True,
                   range=(np.nanquantile(jerks_m_s_s_s, histogram_range_quantiles[0]), np.nanquantile(jerks_m_s_s_s, histogram_range_quantiles[1])),
                   alpha=0.5, label=example_type.title())
    
    # Plot formatting.
    # ax_speed.set_xlabel('Speed [m/s]')
    # ax_jerk.set_xlabel('Jerk [m/s/s/s]')
    ax_speed.set_ylabel('Density')
    ax_jerk.set_ylabel('Density')
    ax_speed.grid(True, color='lightgray')
    ax_jerk.grid(True, color='lightgray')
    speed_title_str = 'Spout Speed [m/s]%s' % ((': %s' % subtitle) if subtitle is not None else '')
    jerk_title_str = 'Spout Jerk [m/s/s/s]%s' % ((': %s' % subtitle) if subtitle is not None else '')
    if len(example_types) == 2:
      stats_results = results_speed_m_s[example_types[0]][example_types[1]]
      p = stats_results.pvalue
      speed_title_str += ' | Distributions are different? %s (p = %0.4f)' % ('yes' if p < 0.05 else 'no', p)
      stats_results = results_jerk_m_s_s_s[example_types[0]][example_types[1]]
      p = stats_results.pvalue
      jerk_title_str += ' | Distributions are different? %s (p = %0.4f)' % ('yes' if p < 0.05 else 'no', p)
    ax_speed.set_title(speed_title_str)
    ax_jerk.title.set_text(jerk_title_str)
    ax_speed.legend()
    ax_jerk.legend()
    if region is not None:
      fig.suptitle('During %s' % region.replace('_', '-').title())
    
    # Show the plot.
    plt.draw()
    
    # Save the plot if desired.
    if output_filepath is not None:
      fig.savefig(output_filepath, dpi=300)

  # # Print statistical test results.
  # if print_comparison_results:
  #   print(' Statistical comparison results for spout speed:')
  #   for example_type_1 in example_types:
  #     for example_type_2 in example_types:
  #       print('  Comparing [%s] to [%s]: ' % (example_type_1, example_type_2), end='')
  #       results = results_speed_m_s[example_type_1][example_type_2]
  #       p = results.pvalue
  #       print('Different? %s (p = %0.4f)' % ('yes' if p < 0.05 else 'no', p))
  #   print(' Statistical comparison results for spout jerk:')
  #   for example_type_1 in example_types:
  #     for example_type_2 in example_types:
  #       print('  Comparing [%s] to [%s]: ' % (example_type_1, example_type_2), end='')
  #       results = results_jerk_m_s_s_s[example_type_1][example_type_2]
  #       p = results.pvalue
  #       print('Different? %s (p = %0.4f)' % ('yes' if p < 0.05 else 'no', p))
  
  return (fig, axs)

# ================================================================
# Plot and compare distributions of the hand, elbow, and shoulder speed and jerk.
# feature_matrices_allTypes is a dictionary mapping distribution category to matrices for each trial.
# If region is provided, will only consider timesteps during that region for each trial.
#   Can be 'pre_pouring', 'pouring', 'post_pouring', or None for all.
def plot_compare_distributions_body_dynamics(feature_data_byType,
                                             subtitle=None,
                                             output_filepath=None,
                                             region=None,  # 'pre_pouring', 'pouring', 'post_pouring', None for all
                                             print_comparison_results=True,
                                             plot_distributions=True,
                                             num_histogram_bins=100, histogram_range_quantiles=(0,1),
                                             fig=None, hide_figure_window=False):
    
  if output_filepath is not None:
    os.makedirs(os.path.split(output_filepath)[0], exist_ok=True)
  if plot_distributions:
    if fig is None:
      if hide_figure_window:
        try:
          matplotlib.use('Agg')
        except:
          pass
      else:
        matplotlib.use(default_matplotlib_backend)
      fig, axs = plt.subplots(nrows=2, ncols=3,
                                 squeeze=False, # if False, always return 2D array of axes
                                 sharex=False, sharey=False,
                                 subplot_kw={'frame_on': True},
                                 figsize=(13, 7),
                                 )
      if not hide_figure_window:
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt_wait_for_keyboard_press(0.3)
      plt.ion()
    else:
      (fig, axs) = fig
  
  # Initialize results/outputs.
  example_types = list(feature_data_byType.keys())
  distributions_speed_m_s = dict([(key, {}) for key in example_types])
  distributions_jerk_m_s_s_s = dict([(key, {}) for key in example_types])
  results_speed_m_s = dict([(example_type, dict([(key, {}) for key in example_types])) for example_type in example_types])
  results_jerk_m_s_s_s = dict([(example_type, dict([(key, {}) for key in example_types])) for example_type in example_types])
  
  # Process each example type (each distribution category).
  for (example_type, feature_data_allTrials) in feature_data_byType.items():
    num_trials = feature_data_allTrials['time_s'].shape[0]
    num_timesteps = feature_data_allTrials['time_s'].shape[1]
    # Get the body dynamics for each timestep.
    speeds_m_s = [None]*num_trials
    jerks_m_s_s_s = [None]*num_trials
    for trial_index in range(num_trials):
      feature_data = get_feature_data_for_trial(feature_data_allTrials, trial_index)
      speeds_m_s[trial_index] = get_body_speed_m_s(feature_data)
      jerks_m_s_s_s[trial_index] = get_body_jerk_m_s_s_s(feature_data)
      body_keys = list(speeds_m_s[trial_index].keys())
  
    # If desired, only extract a certain window of time relative to the inferred pouring time.
    for trial_index in range(num_trials):
      feature_data = get_feature_data_for_trial(feature_data_allTrials, trial_index)
      pouring_inference = infer_pour_pose(feature_data)
      pour_start_index = pouring_inference['start_time_index']
      pour_end_index = pouring_inference['end_time_index']
      if region == 'pouring':
        speeds_m_s[trial_index] = dict([(body_key, speeds_m_s[trial_index][body_key][pour_start_index:pour_end_index]) for body_key in body_keys])
        jerks_m_s_s_s[trial_index] = dict([(body_key, jerks_m_s_s_s[trial_index][body_key][pour_start_index:pour_end_index]) for body_key in body_keys])
      if region == 'pre_pouring':
        speeds_m_s[trial_index] = dict([(body_key, speeds_m_s[trial_index][body_key][0:pour_start_index]) for body_key in body_keys])
        jerks_m_s_s_s[trial_index] = dict([(body_key, jerks_m_s_s_s[trial_index][body_key][0:pour_start_index]) for body_key in body_keys])
      if region == 'post_pouring':
        speeds_m_s[trial_index] = dict([(body_key, speeds_m_s[trial_index][body_key][pour_end_index:-1]) for body_key in body_keys])
        jerks_m_s_s_s[trial_index] = dict([(body_key, jerks_m_s_s_s[trial_index][body_key][pour_end_index:-1]) for body_key in body_keys])
    
    # Store results.
    for body_key in body_keys:
      speeds_m_s_allTrials = [speed_m_s[body_key] for speed_m_s in speeds_m_s]
      jerks_m_s_s_s_allTrials = [jerk_m_s_s_s[body_key] for jerk_m_s_s_s in jerks_m_s_s_s]
      distributions_speed_m_s[example_type][body_key] = np.abs(np.stack(np.concatenate(speeds_m_s_allTrials)))
      distributions_jerk_m_s_s_s[example_type][body_key] = np.abs(np.stack(np.concatenate(jerks_m_s_s_s_allTrials)))
  
  # Statistical tests to compare the distributions.
  for example_type_1 in example_types:
    for example_type_2 in example_types:
      results_speed_m_s[example_type_1][example_type_2] = dict.fromkeys(body_keys, None)
      for body_key in body_keys:
        speeds_m_s_1 = distributions_speed_m_s[example_type_1][body_key]
        speeds_m_s_2 = distributions_speed_m_s[example_type_2][body_key]
        jerks_m_s_s_s_1 = distributions_jerk_m_s_s_s[example_type_1][body_key]
        jerks_m_s_s_s_2 = distributions_jerk_m_s_s_s[example_type_2][body_key]
        results_speed_m_s[example_type_1][example_type_2][body_key] = \
          stats.kstest(speeds_m_s_1, speeds_m_s_2,
                       alternative='two-sided', # 'two-sided', 'less', 'greater'
                       method='auto', # ‘auto’, ‘exact’, ‘approx’, ‘asymp’
                       )
        results_jerk_m_s_s_s[example_type_1][example_type_2][body_key] = \
          stats.kstest(jerks_m_s_s_s_1, jerks_m_s_s_s_2,
                       alternative='two-sided', # 'two-sided', 'less', 'greater'
                       method='auto', # ‘auto’, ‘exact’, ‘approx’, ‘asymp’
                       )
  
  # Plot distributions.
  if plot_distributions:
    for (body_index, body_key) in enumerate(body_keys):
      ax_speed = axs[0][body_index]
      ax_jerk = axs[1][body_index]
      for example_type in example_types:
        speeds_m_s = distributions_speed_m_s[example_type][body_key]
        jerks_m_s_s_s = distributions_jerk_m_s_s_s[example_type][body_key]
        ax_speed.hist(speeds_m_s, bins=num_histogram_bins,
                      histtype='stepfilled', # 'bar', 'barstacked', 'step', 'stepfilled'
                      log=False, density=True,
                      range=(np.nanquantile(speeds_m_s, histogram_range_quantiles[0]), np.nanquantile(speeds_m_s, histogram_range_quantiles[1])),
                      alpha=0.5, label=example_type.title())
        ax_jerk.hist(jerks_m_s_s_s, bins=num_histogram_bins,
                     histtype='stepfilled', # 'bar', 'barstacked', 'step', 'stepfilled'
                     log=False, density=True,
                     range=(np.nanquantile(jerks_m_s_s_s, histogram_range_quantiles[0]), np.nanquantile(jerks_m_s_s_s, histogram_range_quantiles[1])),
                     alpha=0.5, label=example_type.title())
      
      # Plot formatting.
      # ax_speed.set_xlabel('Speed [m/s]')
      # ax_jerk.set_xlabel('Jerk [m/s/s/s]')
      ax_speed.set_ylabel('Density')
      ax_jerk.set_ylabel('Density')
      ax_speed.grid(True, color='lightgray')
      ax_jerk.grid(True, color='lightgray')
      speed_title_str = '%s Speed [m/s]%s' % (body_key.title(), (': %s' % subtitle) if subtitle is not None else '')
      jerk_title_str = '%s Jerk [m/s/s/s]%s' % (body_key.title(), (': %s' % subtitle) if subtitle is not None else '')
      if len(example_types) == 2:
        stats_results = results_speed_m_s[example_types[0]][example_types[1]][body_key]
        p = stats_results.pvalue
        speed_title_str += ' | Diff? %s (p = %0.2f)' % ('yes' if p < 0.05 else 'no', p)
        stats_results = results_jerk_m_s_s_s[example_types[0]][example_types[1]][body_key]
        p = stats_results.pvalue
        jerk_title_str += ' | Diff? %s (p = %0.2f)' % ('yes' if p < 0.05 else 'no', p)
      ax_speed.set_title(speed_title_str)
      ax_jerk.title.set_text(jerk_title_str)
      ax_speed.legend()
      ax_jerk.legend()
      if region is not None:
        fig.suptitle('During %s' % region.replace('_', '-').title())
    
    # Show the plot.
    plt.draw()
    
    # Save the plot if desired.
    if output_filepath is not None:
      fig.savefig(output_filepath, dpi=300)

  # Print statistical test results.
  if print_comparison_results:
    for body_key in body_keys:
      print(' Statistical comparison results for %s speed:' % body_key)
      for example_type_1 in example_types:
        for example_type_2 in example_types:
          print('  Comparing [%s] to [%s]: ' % (example_type_1, example_type_2), end='')
          results = results_speed_m_s[example_type_1][example_type_2][body_key]
          p = results.pvalue
          print('Different? %s (p = %0.4f)' % ('yes' if p < 0.05 else 'no', p))
      print(' Statistical comparison results for %s jerk:' % body_key)
      for example_type_1 in example_types:
        for example_type_2 in example_types:
          print('  Comparing [%s] to [%s]: ' % (example_type_1, example_type_2), end='')
          results = results_jerk_m_s_s_s[example_type_1][example_type_2][body_key]
          p = results.pvalue
          print('Different? %s (p = %0.4f)' % ('yes' if p < 0.05 else 'no', p))
  
  return (fig, axs)

# ================================================================
# Plot and compare distributions of the hand, elbow, and shoulder joint angles.
# feature_matrices_allTypes is a dictionary mapping distribution category to matrices for each trial.
# If region is provided, will only consider timesteps during that region for each trial.
#   Can be 'pre_pouring', 'pouring', 'post_pouring', or None for all.
def plot_compare_distributions_joint_angles(feature_data_byType,
                                            subtitle=None,
                                            output_filepath=None,
                                            region=None,  # 'pre_pouring', 'pouring', 'post_pouring', None for all
                                            print_comparison_results=True,
                                            plot_distributions=True,
                                            num_histogram_bins=100, histogram_range_quantiles=(0,1),
                                            fig=None, hide_figure_window=False):
    
  if output_filepath is not None:
    os.makedirs(os.path.split(output_filepath)[0], exist_ok=True)
  if plot_distributions:
    if fig is None:
      if hide_figure_window:
        try:
          matplotlib.use('Agg')
        except:
          pass
      else:
        matplotlib.use(default_matplotlib_backend)
      fig, axs = plt.subplots(nrows=3, ncols=3,
                                 squeeze=False, # if False, always return 2D array of axes
                                 sharex=False, sharey=False,
                                 subplot_kw={'frame_on': True},
                                 figsize=(13, 7),
                                 )
      if not hide_figure_window:
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt_wait_for_keyboard_press(0.3)
      plt.ion()
    else:
      (fig, axs) = fig
  
  # Initialize results/outputs.
  example_types = list(feature_data_byType.keys())
  distributions_joint_angles_rad = dict([(key, {}) for key in example_types])
  results_joint_angles_rad = dict([(example_type, dict([(key, {}) for key in example_types])) for example_type in example_types])
  joint_axes = ['x', 'y', 'z']
  
  # Process each example type (each distribution category).
  for (example_type, feature_data_allTrials) in feature_data_byType.items():
    num_trials = feature_data_allTrials['time_s'].shape[0]
    num_timesteps = feature_data_allTrials['time_s'].shape[1]
    # Get the body joint angles for each timestep.
    joint_angles_rad = [None]*num_trials
    for trial_index in range(num_trials):
      feature_data = get_feature_data_for_trial(feature_data_allTrials, trial_index)
      joint_angles_rad[trial_index] = get_body_joint_angles_rad(feature_data)
      body_keys = list(joint_angles_rad[trial_index].keys())
  
    # If desired, only extract a certain window of time relative to the inferred pouring time.
    for trial_index in range(num_trials):
      feature_data = get_feature_data_for_trial(feature_data_allTrials, trial_index)
      pouring_inference = infer_pour_pose(feature_data)
      pour_start_index = pouring_inference['start_time_index']
      pour_end_index = pouring_inference['end_time_index']
      if region == 'pouring':
        joint_angles_rad[trial_index] = dict([(body_key, joint_angles_rad[trial_index][body_key][pour_start_index:pour_end_index, :]) for body_key in body_keys])
      if region == 'pre_pouring':
        joint_angles_rad[trial_index] = dict([(body_key, joint_angles_rad[trial_index][body_key][0:pour_start_index, :]) for body_key in body_keys])
      if region == 'post_pouring':
        joint_angles_rad[trial_index] = dict([(body_key, joint_angles_rad[trial_index][body_key][pour_end_index:-1, :]) for body_key in body_keys])
    
    # Store results.
    for body_key in body_keys:
      joint_angles_rad_allTrials = [joint_angle_rad[body_key] for joint_angle_rad in joint_angles_rad]
      distributions_joint_angles_rad[example_type][body_key] = np.vstack(joint_angles_rad_allTrials)
  
  # Statistical tests to compare the distributions.
  for example_type_1 in example_types:
    for example_type_2 in example_types:
      results_joint_angles_rad[example_type_1][example_type_2] = dict.fromkeys(body_keys, None)
      for body_key in body_keys:
        results_joint_angles_rad[example_type_1][example_type_2][body_key] = dict.fromkeys(joint_axes, None)
        for joint_axis_index, joint_axis in enumerate(joint_axes):
          joint_angles_rad_1 = np.reshape(distributions_joint_angles_rad[example_type_1][body_key][:,joint_axis_index], (-1,))
          joint_angles_rad_2 = np.reshape(distributions_joint_angles_rad[example_type_2][body_key][:,joint_axis_index], (-1,))
          results_joint_angles_rad[example_type_1][example_type_2][body_key][joint_axis] = \
            stats.kstest(joint_angles_rad_1, joint_angles_rad_2,
                         alternative='two-sided', # 'two-sided', 'less', 'greater'
                         method='auto', # ‘auto’, ‘exact’, ‘approx’, ‘asymp’
                         )
  
  # Plot distributions.
  if plot_distributions:
    for (body_index, body_key) in enumerate(body_keys):
      for joint_axis_index, joint_axis in enumerate(['x', 'y', 'z']):
        ax = axs[joint_axis_index][body_index]
        for example_type in example_types:
          joint_angles_deg = np.degrees(np.reshape(distributions_joint_angles_rad[example_type][body_key][:,joint_axis_index], (-1,)))
          ax.hist(joint_angles_deg, bins=num_histogram_bins,
                        histtype='stepfilled', # 'bar', 'barstacked', 'step', 'stepfilled'
                        log=False, density=True,
                        range=(np.quantile(joint_angles_deg, histogram_range_quantiles[0]), np.quantile(joint_angles_deg, histogram_range_quantiles[1])),
                        alpha=0.5, label=example_type.title())
        
        # Plot formatting.
        # ax.set_xlabel('Joint Angle [deg]')
        ax.set_ylabel('Density')
        ax.grid(True, color='lightgray')
        title_str = '%s %s Angle [deg]%s' % (body_key.title(), joint_axis, (': %s' % subtitle) if subtitle is not None else '')
        if len(example_types) == 2:
          stats_results = results_joint_angles_rad[example_types[0]][example_types[1]][body_key][joint_axis]
          p = stats_results.pvalue
          title_str += ' | Diff? %s (p = %0.2f)' % ('yes' if p < 0.05 else 'no', p)
        ax.set_title(title_str)
        ax.legend()
    if region is not None:
      fig.suptitle('During %s' % region.replace('_', '-').title())
    
    # Show the plot.
    plt.draw()
    
    # Save the plot if desired.
    if output_filepath is not None:
      fig.savefig(output_filepath, dpi=300)

  # Print statistical test results.
  if print_comparison_results:
    for body_key in body_keys:
      for joint_axis in joint_axes:
        print(' Statistical comparison results for %s %s:' % (body_key, joint_axis))
        for example_type_1 in example_types:
          for example_type_2 in example_types:
            print('  Comparing [%s] to [%s]: ' % (example_type_1, example_type_2), end='')
            results = results_joint_angles_rad[example_type_1][example_type_2][body_key][joint_axis]
            p = results.pvalue
            print('Different? %s (p = %0.4f)' % ('yes' if p < 0.05 else 'no', p))
  
  return (fig, axs)

# ================================================================
# Plot and compare distributions of the spout height relative to the top of the glass.
# feature_matrices_allTypes is a dictionary mapping distribution category to matrices for each trial.
# If region is provided, will only consider timesteps during that region for each trial.
#   Can be 'pre_pouring', 'pouring', 'post_pouring', or None for all.
def plot_compare_distributions_spout_relativeHeights(feature_data_byType,
                                                     subtitle=None,
                                                     output_filepath=None,
                                                     region=None,  # 'pre_pouring', 'pouring', 'post_pouring', None for all
                                                     print_comparison_results=True,
                                                     plot_distributions=True,
                                                     num_histogram_bins=100, histogram_range_quantiles=(0,1),
                                                     fig=None, hide_figure_window=False):
    
  if output_filepath is not None:
    os.makedirs(os.path.split(output_filepath)[0], exist_ok=True)
  if plot_distributions:
    if fig is None:
      if hide_figure_window:
        try:
          matplotlib.use('Agg')
        except:
          pass
      else:
        matplotlib.use(default_matplotlib_backend)
      fig, axs = plt.subplots(nrows=1, ncols=1,
                                 squeeze=False, # if False, always return 2D array of axes
                                 sharex=False, sharey=False,
                                 subplot_kw={'frame_on': True},
                                 figsize=(13, 7),
                                 )
      if not hide_figure_window:
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt_wait_for_keyboard_press(0.3)
      plt.ion()
    else:
      (fig, axs) = fig
  
  # Initialize results/outputs.
  example_types = list(feature_data_byType.keys())
  distributions_relativeHeights_cm = dict.fromkeys(example_types, None)
  results_relativeHeights_cm = dict([(example_type, dict.fromkeys(example_types, None)) for example_type in example_types])
  
  # Process each example type (each distribution category).
  for (example_type, feature_data_allTrials) in feature_data_byType.items():
    num_trials = feature_data_allTrials['time_s'].shape[0]
    num_timesteps = feature_data_allTrials['time_s'].shape[1]
    # Get the spout heights for each timestep of each trial.
    spout_relativeHeights_cm = []
    for trial_index in range(num_trials):
      feature_data = get_feature_data_for_trial(feature_data_allTrials, trial_index)
      referenceObject_position_m = np.squeeze(feature_data['referenceObject_position_m'])
      # Get the spout height at each timestep.
      spout_relativeHeight_cm = np.zeros(shape=(num_timesteps,))
      for time_index in range(num_timesteps):
        spout_position_cm = 100*infer_spout_position_m(feature_data, time_index)
        spout_relativeHeight_cm[time_index] = spout_position_cm[2] - 100*referenceObject_position_m[2]
      spout_relativeHeights_cm.append(spout_relativeHeight_cm)
      
    # If desired, only extract a certain window of time relative to the inferred pouring time.
    for trial_index in range(num_trials):
      feature_data = get_feature_data_for_trial(feature_data_allTrials, trial_index)
      pouring_inference = infer_pour_pose(feature_data)
      pour_start_index = pouring_inference['start_time_index']
      pour_end_index = pouring_inference['end_time_index']
      if region == 'pouring':
        spout_relativeHeights_cm[trial_index] = spout_relativeHeights_cm[trial_index][pour_start_index:pour_end_index]
      if region == 'pre_pouring':
        spout_relativeHeights_cm[trial_index] = spout_relativeHeights_cm[trial_index][0:pour_start_index]
      if region == 'post_pouring':
        spout_relativeHeights_cm[trial_index] = spout_relativeHeights_cm[trial_index][pour_end_index:-1]
    
    # Store results.
    distributions_relativeHeights_cm[example_type] = np.stack(np.concatenate(spout_relativeHeights_cm))
  
  # Statistical tests to compare the distributions.
  for example_type_1 in example_types:
    for example_type_2 in example_types:
      relativeHeights_cm_1 = distributions_relativeHeights_cm[example_type_1]
      relativeHeights_cm_2 = distributions_relativeHeights_cm[example_type_2]
      results_relativeHeights_cm[example_type_1][example_type_2] = \
        stats.kstest(relativeHeights_cm_1, relativeHeights_cm_2,
                     alternative='two-sided', # 'two-sided', 'less', 'greater'
                     method='auto', # ‘auto’, ‘exact’, ‘approx’, ‘asymp’
                     )
  
  # Plot distributions.
  if plot_distributions:
    ax = axs[0][0]
    for example_type in example_types:
      relativeHeights_cm = distributions_relativeHeights_cm[example_type]
      ax.hist(relativeHeights_cm, bins=num_histogram_bins,
                    histtype='stepfilled', # 'bar', 'barstacked', 'step', 'stepfilled'
                    log=False, density=True,
                    range=(np.quantile(relativeHeights_cm, histogram_range_quantiles[0]), np.quantile(relativeHeights_cm, histogram_range_quantiles[1])),
                    alpha=0.5, label=example_type.title())
    
    # Plot formatting.
    # ax.set_xlabel('Speed [m/s]')
    ax.set_ylabel('Density')
    ax.grid(True, color='lightgray')
    title_str = 'Spout Height Above Glass [cm]%s' % ((': %s' % subtitle) if subtitle is not None else '')
    if len(example_types) == 2:
      stats_results = results_relativeHeights_cm[example_types[0]][example_types[1]]
      p = stats_results.pvalue
      title_str += ' | Distributions are different? %s (p = %0.4f)' % ('yes' if p < 0.05 else 'no', p)
    ax.set_title(title_str)
    ax.legend()
    if region is not None:
      fig.suptitle('During %s' % region.replace('_', '-').title())
    
    # Show the plot.
    plt.draw()
    
    # Save the plot if desired.
    if output_filepath is not None:
      fig.savefig(output_filepath, dpi=300)

  # Print statistical test results.
  if print_comparison_results:
    print(' Statistical comparison results for spout height above glass:')
    for example_type_1 in example_types:
      for example_type_2 in example_types:
        print('  Comparing [%s] to [%s]: ' % (example_type_1, example_type_2), end='')
        results = results_relativeHeights_cm[example_type_1][example_type_2]
        p = results.pvalue
        print('Different? %s (p = %0.4f)' % ('yes' if p < 0.05 else 'no', p))
  
  return (fig, axs)

# ================================================================
# Plot and compare distributions of the spout tilt angles.
# feature_matrices_allTypes is a dictionary mapping distribution category to matrices for each trial.
# If region is provided, will only consider timesteps during that region for each trial.
#   Can be 'pre_pouring', 'pouring', 'post_pouring', or None for all.
def plot_compare_distributions_spout_tilts(feature_data_byType,
                                           subtitle=None,
                                           output_filepath=None,
                                           region=None,  # 'pre_pouring', 'pouring', 'post_pouring', None for all
                                           print_comparison_results=True,
                                           plot_distributions=True,
                                           num_histogram_bins=100, histogram_range_quantiles=(0,1),
                                           fig=None, hide_figure_window=False):
    
  if output_filepath is not None:
    os.makedirs(os.path.split(output_filepath)[0], exist_ok=True)
  if plot_distributions:
    if fig is None:
      if hide_figure_window:
        try:
          matplotlib.use('Agg')
        except:
          pass
      else:
        matplotlib.use(default_matplotlib_backend)
      fig, axs = plt.subplots(nrows=1, ncols=1,
                                 squeeze=False, # if False, always return 2D array of axes
                                 sharex=False, sharey=False,
                                 subplot_kw={'frame_on': True},
                                 figsize=(13, 7),
                                 )
      if not hide_figure_window:
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt_wait_for_keyboard_press(0.3)
      plt.ion()
    else:
      (fig, axs) = fig
  
  # Initialize results/outputs.
  example_types = list(feature_data_byType.keys())
  distributions_tilts_rad = dict.fromkeys(example_types, None)
  results_tilts_rad = dict([(example_type, dict.fromkeys(example_types, None)) for example_type in example_types])
  
  # Process each example type (each distribution category).
  for (example_type, feature_data_allTrials) in feature_data_byType.items():
    num_trials = feature_data_allTrials['time_s'].shape[0]
    num_timesteps = feature_data_allTrials['time_s'].shape[1]
    # Get the spout tilt for each timestep of each trial.
    spout_tilts_deg = []
    for trial_index in range(num_trials):
      feature_data = get_feature_data_for_trial(feature_data_allTrials, trial_index)
      spout_tilt_rad = np.zeros(shape=(num_timesteps,))
      for time_index in range(num_timesteps):
        spout_tilt_rad[time_index] = infer_spout_tilting(feature_data, time_index)
      spout_tilts_deg.append(spout_tilt_rad)
      
    # If desired, only extract a certain window of time relative to the inferred pouring time.
    for trial_index in range(num_trials):
      feature_data = get_feature_data_for_trial(feature_data_allTrials, trial_index)
      pouring_inference = infer_pour_pose(feature_data)
      pour_start_index = pouring_inference['start_time_index']
      pour_end_index = pouring_inference['end_time_index']
      if region == 'pouring':
        spout_tilts_deg[trial_index] = spout_tilts_deg[trial_index][pour_start_index:pour_end_index]
      if region == 'pre_pouring':
        spout_tilts_deg[trial_index] = spout_tilts_deg[trial_index][0:pour_start_index]
      if region == 'post_pouring':
        spout_tilts_deg[trial_index] = spout_tilts_deg[trial_index][pour_end_index:-1]
    
    # Store results.
    distributions_tilts_rad[example_type] = np.stack(np.concatenate(spout_tilts_deg))
  
  # Statistical tests to compare the distributions.
  for example_type_1 in example_types:
    for example_type_2 in example_types:
      tilts_rad_1 = distributions_tilts_rad[example_type_1]
      tilts_rad_2 = distributions_tilts_rad[example_type_2]
      results_tilts_rad[example_type_1][example_type_2] = \
        stats.kstest(tilts_rad_1, tilts_rad_2,
                     alternative='two-sided', # 'two-sided', 'less', 'greater'
                     method='auto', # ‘auto’, ‘exact’, ‘approx’, ‘asymp’
                     )
  
  # Plot distributions.
  if plot_distributions:
    ax = axs[0][0]
    for example_type in example_types:
      spout_tilts_deg = np.degrees(distributions_tilts_rad[example_type])
      ax.hist(spout_tilts_deg, bins=num_histogram_bins,
                    histtype='stepfilled', # 'bar', 'barstacked', 'step', 'stepfilled'
                    log=False, density=True,
                    range=(np.quantile(spout_tilts_deg, histogram_range_quantiles[0]), np.quantile(spout_tilts_deg, histogram_range_quantiles[1])),
                    alpha=0.5, label=example_type.title())
    
    # Plot formatting.
    # ax.set_xlabel('Speed [m/s]')
    ax.set_ylabel('Density')
    ax.grid(True, color='lightgray')
    title_str = 'Spout Tilt Angle [deg]%s' % ((': %s' % subtitle) if subtitle is not None else '')
    if len(example_types) == 2:
      stats_results = results_tilts_rad[example_types[0]][example_types[1]]
      p = stats_results.pvalue
      title_str += ' | Distributions are different? %s (p = %0.4f)' % ('yes' if p < 0.05 else 'no', p)
    ax.set_title(title_str)
    ax.legend()
    if region is not None:
      fig.suptitle('During %s' % region.replace('_', '-').title())
    
    # Show the plot.
    plt.draw()
    
    # Save the plot if desired.
    if output_filepath is not None:
      fig.savefig(output_filepath, dpi=300)

  # Print statistical test results.
  if print_comparison_results:
    print(' Statistical comparison results for spout tilt angle:')
    for example_type_1 in example_types:
      for example_type_2 in example_types:
        print('  Comparing [%s] to [%s]: ' % (example_type_1, example_type_2), end='')
        results = results_tilts_rad[example_type_1][example_type_2]
        p = results.pvalue
        print('Different? %s (p = %0.4f)' % ('yes' if p < 0.05 else 'no', p))
  
  return (fig, axs)



##################################################################
# Various helpers
##################################################################

# Helper function from https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
def plot_confidence_ellipse(x, y, ax, n_std=3.0, facecolor=None, **kwargs):
  """
  Create a plot of the covariance confidence ellipse of *x* and *y*.
  Parameters
  ----------
  x, y : array-like, shape (n, )
      Input data.
  ax : matplotlib.axes.Axes
      The axes object to draw the ellipse into.
  n_std : float
      The number of standard deviations to determine the ellipse's radiuses.
  **kwargs
      Forwarded to `~matplotlib.patches.Ellipse`
  Returns
  -------
  matplotlib.patches.Ellipse
  """
  if x.size != y.size:
      raise ValueError("x and y must be the same size")

  cov = np.cov(x, y)
  pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
  # Using a special case to obtain the eigenvalues of this
  # two-dimensional dataset.
  ell_radius_x = np.sqrt(1 + pearson)
  ell_radius_y = np.sqrt(1 - pearson)
  ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                    facecolor=facecolor, **kwargs)

  # Calculating the standard deviation of x from
  # the squareroot of the variance and multiplying
  # with the given number of standard deviations.
  scale_x = np.sqrt(cov[0, 0]) * n_std
  mean_x = np.mean(x)

  # calculating the standard deviation of y ...
  scale_y = np.sqrt(cov[1, 1]) * n_std
  mean_y = np.mean(y)

  transf = transforms.Affine2D() \
      .rotate_deg(45) \
      .scale(scale_x, scale_y) \
      .translate(mean_x, mean_y)

  ellipse.set_transform(transf + ax.transData)
  return ax.add_patch(ellipse)



















