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

import cv2
import os
import time
import copy

import matplotlib
default_matplotlib_backend = matplotlib.rcParams['backend']
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from mpl_toolkits.mplot3d import art3d



# ================================================================
# Wait for the user to press a key while a plot window is active.
def plt_wait_for_keyboard_press(timeout_s=-1.0):
  keyboardClick=False
  while keyboardClick == False:
    keyboardClick = plt.waitforbuttonpress(timeout=timeout_s)



##################################################################
# Helpers for 3D plotting
##################################################################

# ================================================================
# Plot a 3D box with a given position and orientation.
# If center_preRotation_cm is provided, will translate box to that location before applying the given quaternion.
def plot_3d_box(ax, quaternion_localToGlobal_wijk, center_cm, center_preRotation_cm, box_dimensions_cm, color, alpha=0.8): # function created using ChatGPT
  
  # Rotate the box.
  (corners, faces) = rotate_3d_box(quaternion_localToGlobal_wijk, center_preRotation_cm, box_dimensions_cm)
  
  # Translate the box.
  corners = corners + center_cm
  
  # Plot the box.
  # ax.set_box_aspect([np.ptp(corners[:,dim]) for dim in range(3)])
  # ax.set_xlim3d(corners[:,0].min(), corners[:,0].max())
  # ax.set_ylim3d(corners[:,1].min(), corners[:,1].max())
  # ax.set_zlim3d(corners[:,2].min(), corners[:,2].max())
  box = art3d.Poly3DCollection([corners[face] for face in faces],
                               alpha=alpha,
                               facecolor=color,
                               edgecolor=0.4*color)
  ax.add_collection3d(box)
  return box

# ================================================================
# Plot a box representing the hand pose.
def plot_hand_box(ax, hand_quaternion_localToGlobal_wijk, hand_center_cm):
  return plot_3d_box(ax, hand_quaternion_localToGlobal_wijk,
                     hand_center_cm, np.array([0, 0, 0]),
                     hand_box_dimensions_cm, hand_box_color)

# ================================================================
# Plot a box representing the motion object pose.
def plot_motionObject_box(ax, hand_quaternion_localToGlobal_wijk, hand_center_cm, activity_type, hand_to_motionObject_rotation_toUse=None):
  if hand_to_motionObject_rotation_toUse is None:
    hand_to_motionObject_rotation_toUse = hand_to_motionObject_rotation[activity_type]
  hand_rotation = Rotation.from_quat(hand_quaternion_localToGlobal_wijk[[1,2,3,0]])
  motionObject_rotation = hand_rotation * hand_to_motionObject_rotation_toUse
  motionObject_quaternion_localToGlobal_ijkw = motionObject_rotation.as_quat()
  return plot_3d_box(ax, motionObject_quaternion_localToGlobal_ijkw[[3,0,1,2]],
                     hand_center_cm, hand_to_motionObject_offset_cm[activity_type],
                     motionObject_shape_dimensions_cm[activity_type], motionObject_box_color)



##################################################################
# Trajectory animation
##################################################################

# ================================================================
# Plot the scene for a single timestep.
def plot_timestep(time_s, time_index, activity_type,
                  feature_data=None, bodyPath_data=None,  # supply one of these
                  referenceObject_position_m=None,  # if using bodyPath_data or not in feature_data
                  subject_id=-1, num_total_subjects=1, subplot_index=0,
                  trial_index=-1, trial_start_index_offset_forTitle=0,
                  previous_handles=None, clear_previous_timestep=True, redraw_trajectory_each_timestep=False,
                  include_skeleton=True, include_motionObject=True, include_hand=True,
                  include_motionObjectKeypoint_projection=True, include_referenceObject=True,
                  animation_view_angle_toUse=None,
                  wait_for_user_after_plotting=False, hide_figure_window=False):
  # Parse the feature data if needed.
  if feature_data is not None:
    feature_data = parse_feature_data(feature_data)
    include_pelvis = False
    if 'elbow' not in feature_data['position_m']:
      include_skeleton = False
    if 'referenceObject_position_m' in feature_data:
      referenceObject_position_m = feature_data['referenceObject_position_m']
  # Parse the body path data if needed.
  else:
    feature_data = {'position_m': {}, 'quaternion_wijk': {}}
    include_pelvis = True
    for data_type in feature_data.keys():
      feature_data[data_type]['hand'] = bodyPath_data[data_type]['RightHand']
      feature_data[data_type]['elbow'] = bodyPath_data[data_type]['RightForeArm']
      feature_data[data_type]['shoulder'] = bodyPath_data[data_type]['RightUpperArm']
      feature_data[data_type]['pelvis'] = bodyPath_data[data_type]['Pelvis']
  
  # Infer the stationary position.
  (_, stationary_pose) = infer_stationary_poses(time_s, feature_data,
                                                use_variance=stationary_position_use_variance[activity_type],
                                                hand_segment_key='hand',
                                                stationary_position_hardcoded_time_fraction=stationary_position_hardcoded_time_fraction[activity_type])
  
  # Initialize figure state.
  if previous_handles is None:
    fig = None
    h_chains = []
    h_scatters = []
    h_hand = None
    h_motionObject = None
  else:
    (fig, h_chains, h_scatters, h_hand, h_motionObject) = previous_handles
    
  # Create a figure if needed.
  if fig is None:
    if hide_figure_window:
      matplotlib.use('Agg')
    else:
      matplotlib.use(default_matplotlib_backend)
    fig = plt.figure(figsize=(13, 7))
    if not hide_figure_window:
      figManager = plt.get_current_fig_manager()
      figManager.window.showMaximized()
      plt_wait_for_keyboard_press(0.3)
    plt.ion()
    num_rows = int(np.sqrt(num_total_subjects))
    num_cols = int(np.ceil(num_total_subjects/num_rows))
    for s in range(num_total_subjects):
      fig.add_subplot(num_rows, num_cols, s+1, projection='3d')
  ax = fig.get_axes()[subplot_index]
  
  # Get the table height.
  if referenceObject_position_m is not None:
    table_z_cm = 100*referenceObject_position_m[2] - referenceObject_height_cm[activity_type]
  else:
    table_z_cm = None
  
  # Draw items that remain the same across frames.
  if previous_handles is None or redraw_trajectory_each_timestep:
    if clear_previous_timestep:
      ax.clear()
    
    # Add labels and titles.
    ax.set_xlabel('X [cm]')
    ax.set_ylabel('Y [cm]')
    ax.set_zlabel('Z [cm]')
    
    # Set the view angle
    if animation_view_angle_toUse is None:
      animation_view_angle_toUse = animation_view_angle
    ax.view_init(*animation_view_angle_toUse)
    
    # Plot trajectories of the right arm and pelvis.
    hand_position_cm = 100*feature_data['position_m']['hand']
    h_hand_path = ax.plot3D(hand_position_cm[:, 0], hand_position_cm[:, 1], hand_position_cm[:, 2], alpha=1)
    if include_skeleton:
      forearm_position_cm = 100*feature_data['position_m']['elbow']
      upperArm_position_cm = 100*feature_data['position_m']['shoulder']
      ax.plot3D(forearm_position_cm[:, 0], forearm_position_cm[:, 1], forearm_position_cm[:, 2], alpha=0.3)
      ax.plot3D(upperArm_position_cm[:, 0], upperArm_position_cm[:, 1], upperArm_position_cm[:, 2], alpha=0.3)
    if include_pelvis:
      pelvis_position_cm = 100*feature_data['position_m']['pelvis']
      ax.plot3D(pelvis_position_cm[:, 0], pelvis_position_cm[:, 1], pelvis_position_cm[:, 2], alpha=0.3)
      
    # Plot origin and start/end/stationary hand positions.
    ax.scatter(0, 0, 0, s=25, color=[0, 0, 0])
    ax.scatter(hand_position_cm[0, 0], hand_position_cm[0, 1], hand_position_cm[0, 2], s=25, color='g')
    ax.scatter(hand_position_cm[-1, 0], hand_position_cm[-1, 1], hand_position_cm[-1, 2], s=25, color='r')
    ax.scatter(100*stationary_pose['position_m']['hand'][0],
               100*stationary_pose['position_m']['hand'][1],
               100*stationary_pose['position_m']['hand'][2],
               s=25, color=h_hand_path[0].get_color(), edgecolor='k')
    
    if referenceObject_position_m is not None and include_referenceObject:
      # Plot the reference object projection onto the table.
      ax.scatter(100*referenceObject_position_m[0],
                 100*referenceObject_position_m[1],
                 table_z_cm,
                 s=25, color=h_hand_path[0].get_color(), edgecolor='c')
      referenceObject_circle = mpatches.Circle((100*referenceObject_position_m[0], 100*referenceObject_position_m[1]),
        radius=referenceObject_diameter_cm[activity_type]/2, ec=[0.4,1,1], color=[0.8,1,1])
      ax.add_patch(referenceObject_circle)
      art3d.patch_2d_to_3d(referenceObject_circle, z=table_z_cm, zdir='z')
    
      # Plot the reference object.
      color_hex = h_hand_path[0].get_color().strip('#')
      color_rgb = tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4))
      color_rgb = np.array(color_rgb)/255
      plot_3d_box(ax, [1, 0, 0, 0], 100*referenceObject_position_m, [0, 0, -referenceObject_height_cm[activity_type]/2],
                  [referenceObject_diameter_cm[activity_type], referenceObject_diameter_cm[activity_type], referenceObject_height_cm[activity_type]],
                  color_rgb, alpha=0.2)
  
  # Clear plots from the previous timestep.
  if clear_previous_timestep:
    for i in range(len(h_chains)):
      h_chains[i][0].remove()
    for i in range(len(h_scatters)):
      h_scatters[i].remove()
    if h_hand is not None:
      h_hand.remove()
    if h_motionObject is not None:
      h_motionObject.remove()
    h_chains = []
    h_scatters = []
    h_hand = None
    h_motionObject = None

  # Animate the whole skeleton.
  if include_skeleton:
    # Draw the skeleton chains
    x = []
    y = []
    z = []
    for segment_name in ['hand', 'elbow', 'shoulder']:
      if segment_name not in feature_data['position_m']:
        continue
      position_cm = 100*feature_data['position_m'][segment_name]
      x.append(position_cm[time_index, 0])
      y.append(position_cm[time_index, 1])
      z.append(position_cm[time_index, 2])
    h_chains.append(ax.plot3D(x, y, z, color='k'))
    h_scatters.append(ax.scatter(x, y, z, s=25, color=[0.5, 0.5, 0.5]))
    
  if include_motionObjectKeypoint_projection:
    if table_z_cm is not None:
      motionObjectKeypoint_projection_z_cm = table_z_cm
    else:
      motionObjectKeypoint_projection_z_cm = 0
    # Draw the motion object keypoint projection onto the table.
    position_cm = 100*eval(infer_motionObjectKeypoint_position_m_fn[activity_type])(
      feature_data=feature_data, activity_type=activity_type, time_index=time_index)
    
    h_scatters.append(ax.scatter(position_cm[0], position_cm[1],
                                 motionObjectKeypoint_projection_z_cm,
                                 s=30, color='m'))
    # Draw an indicator of the keypoint direction on the table.
    motionObject_yawvector = infer_motionObject_yawvector(feature_data=feature_data, activity_type=activity_type, time_index=time_index)
    motionObject_yawvector = motionObject_yawvector * referenceObject_diameter_cm[activity_type]/2
    motionObject_yawsegment = np.array([[0,0,0], list(motionObject_yawvector)])
    motionObject_yawsegment = motionObject_yawsegment + position_cm
    h_chains.append(ax.plot3D(motionObject_yawsegment[:,0], motionObject_yawsegment[:,1],
                              (motionObjectKeypoint_projection_z_cm+0.1)*np.array([1,1]),
                              color='r', linewidth=2))
  
  # ax.set_xlim([-60, 0])
  # ax.set_ylim([-40, 40])
  # ax.set_zlim([0, 40])
  
  x_lim = ax.get_xlim()
  y_lim = ax.get_ylim()
  z_lim = ax.get_zlim()
  
  if include_hand:
    # Visualize a box as the hand.
    # hand_dimensions_cm = [1, 3, 5]
    # hand_rotation_matrix = Rotation.from_quat(bodySegment_data['quaternion']['RightHand'])
    # print(hand_rotation_matrix.apply(hand_dimensions_cm))
    # hand_box_data = np.ones(hand_dimensions_cm, dtype=bool)
    # hand_colors = np.empty(hand_dimensions_cm + [4], dtype=np.float32)
    # hand_colors[:] = [1, 0, 0, 0.8]
    # h_hand = ax.voxels(hand_box_data, facecolors=hand_colors)
    h_hand = plot_hand_box(ax, hand_center_cm=100*feature_data['position_m']['hand'][time_index, :],
                               hand_quaternion_localToGlobal_wijk=feature_data['quaternion_wijk']['hand'][time_index, :])
  if include_motionObject:
    if 'hand_to_motionObject_angles_rad' in feature_data:
      hand_to_motionObject_angles_rad = np.squeeze(feature_data['hand_to_motionObject_angles_rad'])
      hand_to_motionObject_rotation_toUse = Rotation.from_rotvec(hand_to_motionObject_angles_rad)
    else:
      hand_to_motionObject_rotation_toUse = None
    h_motionObject = plot_motionObject_box(ax, hand_center_cm=100*feature_data['position_m']['hand'][time_index, :],
                                      hand_quaternion_localToGlobal_wijk=feature_data['quaternion_wijk']['hand'][time_index, :],
                                      hand_to_motionObject_rotation_toUse=hand_to_motionObject_rotation_toUse,
                                      activity_type=activity_type)
  
  # Set the aspect ratio
  ax.set_xlim(x_lim)
  ax.set_ylim(y_lim)
  ax.set_zlim(z_lim)
  ax.set_box_aspect([ub - lb for lb, ub in (x_lim, y_lim, z_lim)])
  
  # Set the title.
  trial_str = ''
  if trial_index >= 0:
    trial_str = 'Trial %02d\n' % (trial_index+trial_start_index_offset_forTitle)
  subject_str = ''
  if subject_id != -1:
    subject_str = 'Subject %s ' % str(subject_id)
  time_str = 't=%0.2fs' % ((time_s[time_index] - time_s[0]))
  ax.set_title('%s%s%s' % (subject_str, trial_str, time_str))
  
  # Show the plot.
  plt.draw()
  
  # Save the previous handles.
  previous_handles = (fig, h_chains, h_scatters, h_hand, h_motionObject)
  
  # plt_wait_for_keyboard_press()
  # print('view elev/azim:', ax.elev, ax.azim)
  
  if wait_for_user_after_plotting and not hide_figure_window:
    plt_wait_for_keyboard_press(timeout_s=-1)#spf)
    
  return previous_handles

# ================================================================
# Animate a whole trajectory.
# Supply a feature_matrix and duration_s OR a bodyPath_data and time_s
def animate_trajectory(activity_type, feature_data=None, duration_s=None,
                       bodyPath_data=None, time_s=None,
                       referenceObject_position_m=None,
                       subject_id=-1, num_total_subjects=1, subplot_index=0,
                       trial_index=-1, trial_start_index_offset_forTitle=0,
                       include_skeleton=True, delay_s_between_timesteps=None,
                       wait_for_user_after_timesteps=False, hide_figure_window=False,
                       timestep_interval=1):
  if duration_s is not None:
    time_s = np.linspace(start=0, stop=duration_s, num=feature_data['time_s'].shape[0])
  if time_s is None:
    time_s = feature_data['time_s']
  previous_handles = None
  for time_index in range(0, len(time_s), timestep_interval):
    start_plotting_s = time.time()
    previous_handles = plot_timestep(
      time_s, time_index, activity_type,
      feature_data=feature_data, bodyPath_data=bodyPath_data,
      referenceObject_position_m=referenceObject_position_m,
      subject_id=subject_id, num_total_subjects=num_total_subjects, subplot_index=subplot_index,
      trial_index=trial_index, trial_start_index_offset_forTitle=trial_start_index_offset_forTitle,
      previous_handles=previous_handles, include_skeleton=include_skeleton,
      wait_for_user_after_plotting=wait_for_user_after_timesteps, hide_figure_window=hide_figure_window)
    if delay_s_between_timesteps is not None and time_index+1 < len(time_s):
      if delay_s_between_timesteps == 'realtime':
        delay_s = time_s[time_index+1] - time_s[time_index]
      else:
        delay_s = delay_s_between_timesteps - (time.time() - start_plotting_s)
      delay_s -= (time.time() - start_plotting_s)
      if delay_s > 0:
        time.sleep(delay_s)
  return previous_handles[0] if previous_handles is not None else None
  
# ================================================================
# Animate a whole trajectory, and save it as a video.
# feature_data_byType is a dictionary mapping example type to feature data.
def save_trajectory_animation(activity_type, feature_data_byType,
                              output_filepath,
                              referenceObject_position_m_byType=None,
                              subject_id='', trial_index=-1):
  example_types = list(feature_data_byType.keys())
  # Determine the times and rates of each provided example.
  times_s_byType = {}
  fps_byType = {}
  for example_type in example_types:
    feature_data = parse_feature_data(feature_data_byType[example_type])
    times_s_byType[example_type] = np.linspace(start=0, stop=feature_data['time_s'][-1],
                                        num=feature_data['time_s'].shape[0])
    fps_byType[example_type] = int(round((len(times_s_byType[example_type])-1)/(times_s_byType[example_type][-1] - times_s_byType[example_type][0])))
  min_time_s = min([t[0] for t in list(times_s_byType.values())])
  max_time_s = max([t[-1] for t in list(times_s_byType.values())])
  max_fps = max(list(fps_byType.values()))
  duration_s = max_time_s - min_time_s
  times_s_forVideo = np.arange(start=0, stop=duration_s, step=1/max_fps)
  
  os.makedirs(os.path.split(output_filepath)[0], exist_ok=True)
  
  video_writer = None
  previous_handles_byExampleType = dict([(example_type, None) for example_type in example_types])
  for (frame_index, time_s) in enumerate(times_s_forVideo):
    plot_imgs = []
    for example_type in example_types:
      times_s_forExample = times_s_byType[example_type]
      feature_data = feature_data_byType[example_type]
      if referenceObject_position_m_byType is not None:
        referenceObject_position_m = referenceObject_position_m_byType[example_type]
      else:
        referenceObject_position_m = None
      if time_s >= times_s_forExample[0] and time_s <= times_s_forExample[-1]:
        time_index = times_s_forExample.searchsorted(time_s)
        previous_handles_byExampleType[example_type] = plot_timestep(time_s=times_s_forExample, time_index=time_index, activity_type=activity_type,
                                                                     feature_data=feature_data,
                                                                     referenceObject_position_m=referenceObject_position_m,
                                                                     subject_id='%s%s' % ((str(example_type) if len(str(example_type).strip()) > 0 else ''),
                                                                              (str(subject_id) if len(str(subject_id).strip()) > 0 else '')),
                                                                     trial_index=trial_index,
                                                                     previous_handles=None,
                                                                     wait_for_user_after_plotting=False,
                                                                     hide_figure_window=True)
        fig = previous_handles_byExampleType[example_type][0]
        plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)
        plt.close(fig)
      else:
        plot_img = None
      plot_imgs.append(plot_img)
    sample_plot_img = [plot_img for plot_img in plot_imgs if plot_img is not None][0]
    for i in range(len(plot_imgs)):
      if plot_imgs[i] is None:
        plot_imgs[i] = np.zeros(sample_plot_img.shape, dtype=np.uint8)
    frame_img = cv2.hconcat(plot_imgs)
    if video_writer is None:
      video_writer = cv2.VideoWriter(output_filepath,
                        cv2.VideoWriter_fourcc(*'mp4v'), # for AVI: cv2.VideoWriter_fourcc(*'MJPG'),
                        max_fps,
                        (frame_img.shape[1], frame_img.shape[0])
                        )
    video_writer.write(frame_img)
  video_writer.release()


# ================================================================
# Plot all trajectories.
def plot_all_trajectories(feature_data_allTrials, activity_type, subject_id=None, output_filepath=None, hide_figure_window=False):
  num_trials = feature_data_allTrials['time_s'].shape[0]
  handles_allPaths = None
  for trial_index in range(num_trials):
    feature_data = get_feature_data_for_trial(feature_data_allTrials, trial_index)
    handles_allPaths = plot_timestep(feature_data['time_s'], time_index=0, activity_type=activity_type,
                                     feature_data=feature_data, bodyPath_data=None,  # supply one of these
                                     subject_id=subject_id,
                                     previous_handles=handles_allPaths, clear_previous_timestep=False,
                                     redraw_trajectory_each_timestep=True,
                                     include_skeleton=False, include_motionObject=False, include_hand=False,
                                     include_motionObjectKeypoint_projection=False, include_referenceObject=False,
                                     animation_view_angle_toUse=animation_view_angle_forAllTrajectories,
                                     wait_for_user_after_plotting=False, hide_figure_window=hide_figure_window)
  if output_filepath is not None:
    os.makedirs(os.path.split(output_filepath)[0], exist_ok=True)
    plt.savefig(output_filepath, dpi=300)
  return handles_allPaths

# ================================================================
# Plot all starting conditions.
def plot_all_startingConditions(feature_data_allTrials_byType, activity_type, truth_data_byType=None,
                                trial_indexes_to_exclude_byType=None,
                                model_timestep_index=0,
                                output_filepath=None, hide_figure_window=False):
  if trial_indexes_to_exclude_byType is None:
    trial_indexes_to_exclude_byType = {}
  fig = plt.figure(figsize=(13, 7))
  fig.add_subplot(1, 2, 1)
  fig.add_subplot(1, 2, 2)
  ax_xy = fig.get_axes()[0]
  ax_z = fig.get_axes()[1]
  if not hide_figure_window:
    plt.get_current_fig_manager().window.showMaximized()
    plt_wait_for_keyboard_press(2)
  colors = ['m', 'r', 'g', 'b', 'k']
  starting_position_offsets_cm = []
  for (example_index, example_type) in enumerate(list(feature_data_allTrials_byType.keys())):
    feature_data_allTrials = feature_data_allTrials_byType[example_type]
    color = colors[example_index % len(colors)]
    num_trials = feature_data_allTrials['time_s'].shape[0]
    for trial_index in range(num_trials):
      if example_type in trial_indexes_to_exclude_byType and trial_index in trial_indexes_to_exclude_byType[example_type]:
        continue
      feature_data = get_feature_data_for_trial(feature_data_allTrials, trial_index)
      feature_data = parse_feature_data(feature_data)
      referenceObject_position_cm = 100*feature_data['referenceObject_position_m']
      hand_position_cm = 100*feature_data['position_m']['hand']
      ax_xy.plot(referenceObject_position_cm[1], referenceObject_position_cm[0], 'd',
                 markersize=10, color=color,
                 label=('%s: %s' % (referenceObject_name[activity_type], example_type)) if trial_index == 0 else None)
      if example_type == 'model':
        truth_starting_hand_position_cm = 100*truth_data_byType[example_type]['starting_hand_position_m'][trial_index]
        starting_hand_position_cm = hand_position_cm[model_timestep_index, :]
        ax_xy.plot(truth_starting_hand_position_cm[1], truth_starting_hand_position_cm[0],
                   '.', markersize=20, color='c',
                   label=('Target Hand: %s' % example_type) if trial_index == 0 else None)
        starting_position_offsets_cm.append(np.array(truth_starting_hand_position_cm) - np.array(starting_hand_position_cm))
      else:
        starting_hand_position_cm = [hand_position_cm[0, 0], hand_position_cm[0, 1]]
      ax_xy.plot(starting_hand_position_cm[1], starting_hand_position_cm[0], '.',
                 markersize=10, color=color,
                 label=('Hand: %s' % example_type) if trial_index == 0 else None)
      if example_type == 'model':
        ax_xy.plot([truth_starting_hand_position_cm[1], starting_hand_position_cm[1]],
                   [truth_starting_hand_position_cm[0], starting_hand_position_cm[0]],
                   '-', color='k', label=None)
      else:
        ax_xy.plot([0, 0], [0, 0], '-', color='k', alpha=0, label=' ' if trial_index == 0 else None)
      ax_z.plot(trial_index, hand_position_cm[0, 2], '.', markersize=20, color=color,
                label=('Hand: %s' % example_type) if trial_index == 0 else None)
  ax_xy.grid(True, color='lightgray')
  ax_xy.set_xlabel('Y [cm]')
  ax_xy.set_ylabel('X [cm]')
  ax_xy.set_title('Projections of %s and Starting Hand Position' % referenceObject_name[activity_type])
  ax_xy.set_aspect('equal')
  ax_z.grid(True, color='lightgray')
  ax_z.set_xlabel('Trial Index')
  ax_z.set_ylabel('Z [cm]')
  ax_z.set_ylim([min(ax_z.get_ylim()[0], 10), max(ax_z.get_ylim()[1], 20)])
  ax_z.set_title('Starting Hand Height')
  def add_legend_below(ax, legend_y):
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.15,
                     box.width, box.height * 0.85])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, legend_y),
              fancybox=True, shadow=True, ncol=len(feature_data_allTrials_byType))
  add_legend_below(ax_xy, -0.2)
  add_legend_below(ax_z, -0.1)
  
  if output_filepath is not None:
    os.makedirs(os.path.split(output_filepath)[0], exist_ok=True)
    plt.savefig(output_filepath, dpi=300)
  # Print information about the starting hand position offsets if available.
  if len(starting_position_offsets_cm) > 0:
    print('Starting position offsets [cm]')
    starting_position_offsets_cm = np.array(starting_position_offsets_cm)
    for (axis_index, axis_name) in enumerate(['x', 'y', 'z']):
      axis_offsets_cm = starting_position_offsets_cm[:, axis_index]
      print('  %s axis  : min %5.2f | max %5.2f | mean %5.2f | stdev %5.2f' % (
        axis_name, np.min(axis_offsets_cm), np.max(axis_offsets_cm), np.mean(axis_offsets_cm), np.std(axis_offsets_cm)
      ))
    distances_cm = np.linalg.norm(starting_position_offsets_cm, axis=1)
    print('  distance: min %5.2f | max %5.2f | mean %5.2f | stdev %5.2f' % (
        np.min(distances_cm), np.max(distances_cm), np.mean(distances_cm), np.std(distances_cm)
      ))
  return starting_position_offsets_cm












