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

import cv2
import os

import matplotlib
default_matplotlib_backend = matplotlib.rcParams['backend']
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from mpl_toolkits.mplot3d import art3d



# ================================================================
# Plot the pitcher tilt angle over time.
def plot_pour_tilting(feature_data_allTrials, shade_pouring_region=False,
                      plot_all_trials=True, plot_std_shading=False, plot_mean=False,
                      label=None, subtitle=None,
                      fig=None, hide_figure_window=False, output_filepath=None):
  if output_filepath is not None:
    os.makedirs(os.path.split(output_filepath)[0], exist_ok=True)
  if not isinstance(feature_data_allTrials, dict):
    raise AssertionError('feature_data_allTrials should map keys to matrices whose first dimension is the trial index')
  if not (isinstance(feature_data_allTrials['time_s'], np.ndarray) and feature_data_allTrials['time_s'].ndim == 3):
    raise AssertionError('feature_data_allTrials should map keys to matrices whose first dimension is the trial index')
  if fig is None:
    if hide_figure_window:
      try:
        matplotlib.use('Agg')
      except:
        pass
    else:
      matplotlib.use(default_matplotlib_backend)
    fig = plt.figure(figsize=(13, 7))
    if not hide_figure_window:
      figManager = plt.get_current_fig_manager()
      figManager.window.showMaximized()
      plt_wait_for_keyboard_press(0.3)
    plt.ion()
    fig.add_subplot(1, 1, 1)
  ax = fig.get_axes()[0]
  
  num_trials = feature_data_allTrials['time_s'].shape[0]
  num_timesteps = feature_data_allTrials['time_s'].shape[1]
  
  # Get the tilt angle for each timestep of each example.
  angles_toXY_rad = np.zeros((num_trials, num_timesteps))
  for trial_index in range(num_trials):
    feature_data = get_feature_data_for_trial(feature_data_allTrials, trial_index)
    angle_toXY_rad = np.zeros(shape=(num_timesteps,))
    for time_index in range(num_timesteps):
      angle_toXY_rad[time_index] = infer_spout_tilting(feature_data, time_index)
    angles_toXY_rad[trial_index, :] = angle_toXY_rad
  
  # Get the pouring times.
  if shade_pouring_region:
    pour_start_indexes = np.zeros((num_trials, 1))
    pour_end_indexes = np.zeros((num_trials, 1))
    for trial_index in range(num_trials):
      feature_data = get_feature_data_for_trial(feature_data_allTrials, trial_index)
      pouring_inference = infer_pour_pose(feature_data)
      pour_start_indexes[trial_index] = pouring_inference['start_time_index']
      pour_end_indexes[trial_index] = pouring_inference['end_time_index']
  
  # Plot standard deviation shading if desired.
  if plot_std_shading:
    if label.lower() in example_types_to_offset:
      angles_toXY_rad += np.radians(10)
    x = np.linspace(start=0, stop=angles_toXY_rad.shape[1], num=angles_toXY_rad.shape[1])
    mean = np.mean(angles_toXY_rad, axis=0)
    std = np.std(angles_toXY_rad, axis=0)
    ax.fill_between(x, np.degrees(mean-std), np.degrees(mean+std), alpha=0.4,
                    label=('%s: 1 StdDev' % label) if label is not None else '1 StdDev')
    ax.legend()
  
  # Plot all traces if desired.
  if plot_all_trials:
    for trial_index in range(num_trials):
      ax.plot(np.degrees(angles_toXY_rad[trial_index, :]), linewidth=1)
    
  # Plot the mean if desired.
  if plot_mean:
    mean_label = None
    if not plot_std_shading:
      mean_label = ('%s: Mean' % label) if label is not None else 'Mean'
    ax.plot(np.mean(np.degrees(angles_toXY_rad), axis=0),
            color='k' if plot_all_trials else None, linewidth=3,
            label=mean_label)
    if mean_label is not None:
      ax.legend()
    
  # Shade the pouring regions if desired.
  if shade_pouring_region:
    for trial_index in range(num_trials):
      ax.axvspan(pour_start_indexes[trial_index], pour_end_indexes[trial_index], alpha=0.5, color='gray')
  
  # Plot formatting.
  ax.set_ylim([-90, 15])
  ax.set_xlabel('Time Index')
  ax.set_ylabel('Tilt Angle to XY Plane [deg]')
  ax.grid(True, color='lightgray')
  plt.title('Tilt angle of the pitcher%s' % ((': %s' % subtitle) if subtitle is not None else ''))
  
  # Show the plot.
  plt.draw()
  
  # Save the plot if desired.
  if output_filepath is not None:
    fig.savefig(output_filepath, dpi=300)
  
  return fig

# ================================================================
# Plot the height of the spout relative to the top of the glass over time.
def plot_pour_relativeHeight(feature_data_allTrials,
                             shade_pouring_region=False,
                             plot_all_trials=True, plot_std_shading=False, plot_mean=False,
                             subtitle=None, label=None,
                             fig=None, hide_figure_window=False,
                             color=None,
                             output_filepath=None):
  if output_filepath is not None:
    os.makedirs(os.path.split(output_filepath)[0], exist_ok=True)
  if not isinstance(feature_data_allTrials, dict):
    raise AssertionError('feature_data_allTrials should map keys to matrices whose first dimension is the trial index')
  if not (isinstance(feature_data_allTrials['time_s'], np.ndarray) and feature_data_allTrials['time_s'].ndim == 3):
    raise AssertionError('feature_data_allTrials should map keys to matrices whose first dimension is the trial index')
  if fig is None:
    if hide_figure_window:
      try:
        matplotlib.use('Agg')
      except:
        pass
    else:
      matplotlib.use(default_matplotlib_backend)
    fig = plt.figure(figsize=(13, 7))
    if not hide_figure_window:
      figManager = plt.get_current_fig_manager()
      figManager.window.showMaximized()
      plt_wait_for_keyboard_press(0.3)
    plt.ion()
    fig.add_subplot(1, 1, 1)
    ax = fig.get_axes()[0]
  else:
    ax = fig.get_axes()[0]
  
  num_trials = feature_data_allTrials['time_s'].shape[0]
  num_timesteps = feature_data_allTrials['time_s'].shape[1]
  
  # Get the spout heights for each trial.
  spout_heights_cm = np.zeros((num_trials, num_timesteps))
  for trial_index in range(num_trials):
    feature_data = get_feature_data_for_trial(feature_data_allTrials, trial_index)
    referenceObject_position_m = np.squeeze(feature_data['referenceObject_position_m'])
    # Get the spout height at each timestep.
    spout_relativeHeight_cm = np.zeros(shape=(num_timesteps,))
    for time_index in range(num_timesteps):
      spout_position_cm = 100*infer_spout_position_m(feature_data, time_index)
      spout_relativeHeight_cm[time_index] = spout_position_cm[2] - 100*referenceObject_position_m[2]
    spout_heights_cm[trial_index, :] = spout_relativeHeight_cm
  
  # Get the pouring times.
  if shade_pouring_region:
    pour_start_indexes = np.zeros((num_trials, 1))
    pour_end_indexes = np.zeros((num_trials, 1))
    for trial_index in range(num_trials):
      feature_data = get_feature_data_for_trial(feature_data_allTrials, trial_index)
      pouring_inference = infer_pour_pose(feature_data)
      pour_start_indexes[trial_index] = pouring_inference['start_time_index']
      pour_end_indexes[trial_index] = pouring_inference['end_time_index']
  
  # Plot shading if desired.
  if plot_std_shading:
    if label.lower() in example_types_to_offset:
      spout_heights_cm += 2
    x = np.linspace(start=0, stop=spout_heights_cm.shape[1], num=spout_heights_cm.shape[1])
    mean = np.mean(spout_heights_cm, axis=0)
    std = np.std(spout_heights_cm, axis=0)
    ax.fill_between(x, mean-std, mean+std, alpha=0.4,
                    label=('%s: 1 StdDev' % label) if label is not None else '1 StdDev')
    ax.legend()
  
  # Plot all traces if desired.
  if plot_all_trials:
    for trial_index in range(num_trials):
      ax.plot(spout_heights_cm[trial_index, :], linewidth=1)
    
  # Plot the mean if desired.
  if plot_mean:
    mean_label = None
    if not plot_std_shading:
      mean_label = ('%s: Mean' % label) if label is not None else 'Mean'
    ax.plot(np.mean(spout_heights_cm, axis=0),
            color='k' if plot_all_trials else None, linewidth=3,
            label=mean_label)
    if mean_label is not None:
      ax.legend()
    
  # Shade the pouring regions if desired.
  if shade_pouring_region:
    for trial_index in range(num_trials):
      ax.axvspan(pour_start_indexes[trial_index], pour_end_indexes[trial_index], alpha=0.5, color='gray')
  
  # Plot the glass height.
  ax.axhline(y=0, color='k', linestyle='--')
  
  # Plot formatting.
  ylim = ax.get_ylim()
  ax.set_ylim([min(-5, min(ylim)), max(ylim)])
  ax.set_xlabel('Time Index')
  ax.set_ylabel('Spout Height Above Glass [cm]')
  ax.grid(True, color='lightgray')
  plt.title('Spout height relative to glass%s' % ((': %s' % subtitle) if subtitle is not None else ''))
  
  # Show the plot.
  plt.draw()
  
  # Save the plot if desired.
  if output_filepath is not None:
    fig.savefig(output_filepath, dpi=300)
  
  return fig

# ================================================================
# Plot the speed and jerk of the spout over time.
def plot_spout_dynamics(feature_data_allTrials,
                        plot_all_trials=True, plot_mean=False, plot_std_shading=False,
                        subtitle=None, label=None,
                        output_filepath=None,
                        shade_pouring_region=False,
                        fig=None, hide_figure_window=False):
    
  if output_filepath is not None:
    os.makedirs(os.path.split(output_filepath)[0], exist_ok=True)
  if not isinstance(feature_data_allTrials, dict):
    raise AssertionError('feature_data_allTrials should map keys to matrices whose first dimension is the trial index')
  if not (isinstance(feature_data_allTrials['time_s'], np.ndarray) and feature_data_allTrials['time_s'].ndim == 3):
    raise AssertionError('feature_data_allTrials should map keys to matrices whose first dimension is the trial index')
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
                               sharex=True, sharey=False,
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
  
  num_trials = feature_data_allTrials['time_s'].shape[0]
  num_timesteps = feature_data_allTrials['time_s'].shape[1]
  
  # Get the spout dynamics for each timestep.
  speeds_m_s = [None]*num_trials
  jerks_m_s_s_s = [None]*num_trials
  for trial_index in range(num_trials):
    feature_data = get_feature_data_for_trial(feature_data_allTrials, trial_index)
    speeds_m_s[trial_index] = infer_spout_speed_m_s(feature_data)
    jerks_m_s_s_s[trial_index] = infer_spout_jerk_m_s_s_s(feature_data)
  
  # Get the pouring times.
  if shade_pouring_region:
    pour_start_indexes = np.zeros((num_trials, 1))
    pour_end_indexes = np.zeros((num_trials, 1))
    for trial_index in range(num_trials):
      feature_data = get_feature_data_for_trial(feature_data_allTrials, trial_index)
      pouring_inference = infer_pour_pose(feature_data)
      pour_start_indexes[trial_index] = pouring_inference['start_time_index']
      pour_end_indexes[trial_index] = pouring_inference['end_time_index']
  
  # Plot.
  ax_speed = axs[0][0]
  ax_jerk = axs[1][0]
  speeds_m_s_toPlot = np.array(speeds_m_s)
  jerks_m_s_s_s_toPlot = np.array(jerks_m_s_s_s)
  x = np.linspace(start=0, stop=speeds_m_s_toPlot.shape[1], num=speeds_m_s_toPlot.shape[1])
  # Plot shading if desired.
  if plot_std_shading:
    # Shading for speed
    mean = np.mean(speeds_m_s_toPlot, axis=0)
    std = np.std(speeds_m_s_toPlot, axis=0)
    ax_speed.fill_between(x, mean-std, mean+std, alpha=0.4,
                          label=('%s: 1 StdDev' % label) if label is not None else '1 StdDev')
    ax_speed.legend()
    # Shading for jerk
    mean = np.mean(jerks_m_s_s_s_toPlot, axis=0)
    std = np.std(jerks_m_s_s_s_toPlot, axis=0)
    ax_jerk.fill_between(x, mean-std, mean+std, alpha=0.4,
                         label=('%s: 1 StdDev' % label) if label is not None else '1 StdDev')
    ax_jerk.legend()
  # Plot all traces if desired.
  if plot_all_trials:
    for trial_index in range(num_trials):
      ax_speed.plot(speeds_m_s_toPlot[trial_index, :], linewidth=1)
      ax_jerk.plot(jerks_m_s_s_s_toPlot[trial_index, :], linewidth=1)
  # Plot the mean if desired.
  if plot_mean:
    mean_label = None
    if not plot_std_shading:
      mean_label=('%s: Mean' % label) if label is not None else 'Mean'
    ax_speed.plot(np.mean(speeds_m_s_toPlot, axis=0),
                  color='k' if plot_all_trials else None, linewidth=3,
                  label=mean_label)
    if mean_label is not None:
      ax_speed.legend()
    mean_label = None
    if not plot_std_shading:
      mean_label=('%s: Mean' % label) if label is not None else 'Mean'
    ax_jerk.plot(np.mean(jerks_m_s_s_s_toPlot, axis=0),
                  color='k' if plot_all_trials else None, linewidth=3,
                  label=mean_label)
    if mean_label is not None:
      ax_jerk.legend()
  # Shade the pouring regions if desired.
  if shade_pouring_region:
    for trial_index in range(num_trials):
      ax_speed.axvspan(pour_start_indexes[trial_index], pour_end_indexes[trial_index], alpha=0.5, color='gray')
      ax_jerk.axvspan(pour_start_indexes[trial_index], pour_end_indexes[trial_index], alpha=0.5, color='gray')
  
  # Plot formatting.
  ax_speed.set_ylabel('Speed [m/s]')
  ax_jerk.set_ylabel('Jerk [m/s/s/s]')
  ax_speed.grid(True, color='lightgray')
  ax_jerk.grid(True, color='lightgray')
  ax_speed.title.set_text('Spout Speed%s' % ((': %s' % subtitle) if subtitle is not None else ''))
  ax_jerk.title.set_text('Spout Jerk%s' % ((': %s' % subtitle) if subtitle is not None else ''))
  ax_jerk.set_xlabel('Time Index')
  
  # Show the plot.
  plt.draw()
  
  # Save the plot if desired.
  if output_filepath is not None:
    fig.savefig(output_filepath, dpi=300)

  return (fig, axs)

# ================================================================
# Plot the speed and jerk of the hand, elbow, and shoulder over time.
def plot_body_dynamics(feature_data_allTrials,
                       plot_all_trials=True, plot_mean=False, plot_std_shading=False,
                       subtitle=None, label=None,
                       output_filepath=None,
                       shade_pouring_region=False,
                       fig=None, hide_figure_window=False):
    
  if output_filepath is not None:
    os.makedirs(os.path.split(output_filepath)[0], exist_ok=True)
  if not isinstance(feature_data_allTrials, dict):
    raise AssertionError('feature_data_allTrials should map keys to matrices whose first dimension is the trial index')
  if not (isinstance(feature_data_allTrials['time_s'], np.ndarray) and feature_data_allTrials['time_s'].ndim == 3):
    raise AssertionError('feature_data_allTrials should map keys to matrices whose first dimension is the trial index')
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
                               sharex=True, sharey=False,
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
  
  num_trials = feature_data_allTrials['time_s'].shape[0]
  num_timesteps = feature_data_allTrials['time_s'].shape[1]
  
  # Get the body dynamics for each timestep.
  speeds_m_s = [None]*num_trials
  jerks_m_s_s_s = [None]*num_trials
  for trial_index in range(num_trials):
    feature_data = get_feature_data_for_trial(feature_data_allTrials, trial_index)
    speeds_m_s[trial_index] = get_body_speed_m_s(feature_data)
    jerks_m_s_s_s[trial_index] = get_body_jerk_m_s_s_s(feature_data)
  
  # Get the pouring times.
  if shade_pouring_region:
    pour_start_indexes = np.zeros((num_trials, 1))
    pour_end_indexes = np.zeros((num_trials, 1))
    for trial_index in range(num_trials):
      feature_data = get_feature_data_for_trial(feature_data_allTrials, trial_index)
      pouring_inference = infer_pour_pose(feature_data)
      pour_start_indexes[trial_index] = pouring_inference['start_time_index']
      pour_end_indexes[trial_index] = pouring_inference['end_time_index']
  
  # Plot.
  for (body_index, body_key) in enumerate(list(speeds_m_s[0].keys())):
    ax_speed = axs[0][body_index]
    ax_jerk = axs[1][body_index]
    speeds_m_s_toPlot = np.array([speed_m_s[body_key] for speed_m_s in speeds_m_s])
    jerks_m_s_s_s_toPlot = np.array([jerk_m_s_s_s[body_key] for jerk_m_s_s_s in jerks_m_s_s_s])
    x = np.linspace(start=0, stop=speeds_m_s_toPlot.shape[1], num=speeds_m_s_toPlot.shape[1])
    # Plot shading if desired.
    if plot_std_shading:
      # Shading for speed
      mean = np.mean(speeds_m_s_toPlot, axis=0)
      std = np.std(speeds_m_s_toPlot, axis=0)
      ax_speed.fill_between(x, mean-std, mean+std, alpha=0.4,
                            label=('%s: 1 StdDev' % label) if label is not None else '1 StdDev')
      if body_index == len(speeds_m_s[0].keys())-1:
        ax_speed.legend()
      # Shading for jerk
      mean = np.mean(jerks_m_s_s_s_toPlot, axis=0)
      std = np.std(jerks_m_s_s_s_toPlot, axis=0)
      ax_jerk.fill_between(x, mean-std, mean+std, alpha=0.4,
                           label=('%s: 1 StdDev' % label) if label is not None else '1 StdDev')
      if body_index == len(speeds_m_s[0].keys())-1:
        ax_jerk.legend()
    
    # Plot all traces if desired.
    if plot_all_trials:
      for trial_index in range(num_trials):
        ax_speed.plot(speeds_m_s_toPlot[trial_index, :], linewidth=1)
        ax_jerk.plot(jerks_m_s_s_s_toPlot[trial_index, :], linewidth=1)
    
    # Plot the mean if desired.
    if plot_mean:
      # Mean for speed
      mean_label = None
      if (body_index == len(speeds_m_s[0].keys())-1) and (not plot_std_shading):
        mean_label = ('%s: Mean' % label) if label is not None else 'Mean'
      ax_speed.plot(np.mean(speeds_m_s_toPlot, axis=0),
                    color='k' if plot_all_trials else None, linewidth=3,
                    label=mean_label)
      if mean_label is not None:
        ax_speed.legend()
      # Mean for jerk
      mean_label = None
      if (body_index == len(speeds_m_s[0].keys())-1) and (not plot_std_shading):
        mean_label = ('%s: Mean' % label) if label is not None else 'Mean'
      ax_jerk.plot(np.mean(jerks_m_s_s_s_toPlot, axis=0),
                    color='k' if plot_all_trials else None, linewidth=3,
                    label=mean_label)
      if mean_label is not None:
        ax_jerk.legend()
    
    # Shade the pouring regions if desired.
    if shade_pouring_region:
      for trial_index in range(num_trials):
        ax_speed.axvspan(pour_start_indexes[trial_index], pour_end_indexes[trial_index], alpha=0.5, color='gray')
        ax_jerk.axvspan(pour_start_indexes[trial_index], pour_end_indexes[trial_index], alpha=0.5, color='gray')
    
    # Plot formatting.
    if body_index == 0:
      ax_speed.set_ylabel('Speed [m/s]')
      ax_jerk.set_ylabel('Jerk [m/s/s/s]')
    ax_speed.grid(True, color='lightgray')
    ax_jerk.grid(True, color='lightgray')
    ax_speed.title.set_text('%s Speed%s' % (body_key.title(), ': %s' % subtitle if subtitle is not None else ''))
    ax_jerk.title.set_text('%s Jerk%s' % (body_key.title(), ': %s' % subtitle if subtitle is not None else ''))
    ax_jerk.set_xlabel('Time Index')
  
  # Show the plot.
  plt.draw()
  
  # Save the plot if desired.
  if output_filepath is not None:
    fig.savefig(output_filepath, dpi=300)

  return (fig, axs)

# ================================================================
# Plot the joint angles of the hand, elbow, and shoulder over time.
def plot_body_joint_angles(feature_data_allTrials,
                           plot_all_trials=True, plot_mean=False, plot_std_shading=False,
                           subtitle=None, label=None,
                           output_filepath=None,
                           shade_pouring_region=False,
                           fig=None, hide_figure_window=False):
    
  if output_filepath is not None:
    os.makedirs(os.path.split(output_filepath)[0], exist_ok=True)
  if not isinstance(feature_data_allTrials, dict):
    raise AssertionError('feature_data_allTrials should map keys to matrices whose first dimension is the trial index')
  if not (isinstance(feature_data_allTrials['time_s'], np.ndarray) and feature_data_allTrials['time_s'].ndim == 3):
    raise AssertionError('feature_data_allTrials should map keys to matrices whose first dimension is the trial index')
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
                               sharex=True, sharey=False,
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
  
  num_trials = feature_data_allTrials['time_s'].shape[0]
  num_timesteps = feature_data_allTrials['time_s'].shape[1]
  
  # Get the body dynamics for each timestep.
  joint_angles_rad = [None]*num_trials
  for trial_index in range(num_trials):
    feature_data = get_feature_data_for_trial(feature_data_allTrials, trial_index)
    joint_angles_rad[trial_index] = get_body_joint_angles_rad(feature_data)
  
  # Get the pouring times.
  if shade_pouring_region:
    pour_start_indexes = np.zeros((num_trials, 1))
    pour_end_indexes = np.zeros((num_trials, 1))
    for trial_index in range(num_trials):
      feature_data = get_feature_data_for_trial(feature_data_allTrials, trial_index)
      pouring_inference = infer_pour_pose(feature_data)
      pour_start_indexes[trial_index] = pouring_inference['start_time_index']
      pour_end_indexes[trial_index] = pouring_inference['end_time_index']
  
  # Plot.
  for (body_index, body_key) in enumerate(list(joint_angles_rad[0].keys())):
    for joint_axis_index, joint_axis in enumerate(['x', 'y', 'z']):
      ax_joint_angles = axs[joint_axis_index][body_index]
      joint_angles_deg_toPlot = np.degrees(np.array([joint_angle_rad[body_key] for joint_angle_rad in joint_angles_rad]))
      x = np.linspace(start=0, stop=joint_angles_deg_toPlot.shape[1], num=joint_angles_deg_toPlot.shape[1])
      # Take the desired angle for each joint.
      joint_angles_deg_toPlot = np.squeeze(joint_angles_deg_toPlot[:, :, joint_axis_index])
      
      # Plot shading if desired.
      if plot_std_shading:
        mean = np.mean(joint_angles_deg_toPlot, axis=0)
        std = np.std(joint_angles_deg_toPlot, axis=0)
        ax_joint_angles.fill_between(x, mean-std, mean+std, alpha=0.4,
                                      label=('%s: 1 StdDev' % label) if label is not None else '1 StdDev')
        if body_index == len(joint_angles_rad[0].keys())-1:
          ax_joint_angles.legend()
      
      # Plot all traces if desired.
      if plot_all_trials:
        for trial_index in range(num_trials):
          ax_joint_angles.plot(joint_angles_deg_toPlot[trial_index, :], linewidth=1)
      
      # Plot the mean if desired.
      if plot_mean:
        mean_label = None
        if body_index == len(joint_angles_rad[0].keys())-1 and not plot_std_shading:
          mean_label=('%s: Mean' % label) if label is not None else 'Mean'
        ax_joint_angles.plot(np.mean(joint_angles_deg_toPlot, axis=0),
                              color='k' if plot_all_trials else None, linewidth=3,
                              label=mean_label)
        if mean_label is not None:
          ax_joint_angles.legend()
      
      # Shade the pouring regions if desired.
      if shade_pouring_region:
        for trial_index in range(num_trials):
          ax_joint_angles.axvspan(pour_start_indexes[trial_index], pour_end_indexes[trial_index], alpha=0.5, color='gray')
      
      # Plot formatting.
      if body_index == 0:
        ax_joint_angles.set_ylabel('Joint Angle [deg]')
      ax_joint_angles.grid(True, color='lightgray')
      ax_joint_angles.title.set_text('%s %s Angle%s' % (body_key.title(), joint_axis, ': %s' % subtitle if subtitle is not None else ''))
      if joint_axis_index == len(joint_angles_rad[0].keys())-1:
        ax_joint_angles.set_xlabel('Time Index')
  
  # Show the plot.
  plt.draw()
  
  # Save the plot if desired.
  if output_filepath is not None:
    fig.savefig(output_filepath, dpi=300)

  return (fig, axs)





























