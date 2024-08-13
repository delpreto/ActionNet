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
from learning_trajectories.helpers.parse_process_feature_matrices import *

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
# Plot a box representing the pitcher pose.
def plot_pitcher_box(ax, hand_quaternion_localToGlobal_wijk, hand_center_cm):
  hand_rotation = Rotation.from_quat(hand_quaternion_localToGlobal_wijk[[1,2,3,0]])
  pitcher_rotation = hand_rotation * hand_to_pitcher_rotation
  pitcher_quaternion_localToGlobal_ijkw = pitcher_rotation.as_quat()
  return plot_3d_box(ax, pitcher_quaternion_localToGlobal_ijkw[[3,0,1,2]],
                     hand_center_cm, hand_to_pitcher_offset_cm,
                     pitcher_box_dimensions_cm, pitcher_box_color)



##################################################################
# Trajectory animation
##################################################################

# ================================================================
# Plot the scene for a single timestep.
def plot_timestep(time_s, time_index,
                  referenceObject_position_m=None,
                  feature_matrix=None, bodyPath_data=None,  # supply one of these
                  subject_id=-1, num_total_subjects=1, subplot_index=0,
                  trial_index=-1, trial_start_index_offset_forTitle=0,
                  previous_handles=None, clear_previous_timestep=True, redraw_trajectory_each_timestep=False,
                  include_skeleton=True, include_pitcher=True, include_hand=True,
                  include_spout_projection=True, include_referenceObject=True,
                  wait_for_user_after_plotting=False, hide_figure_window=False):
  # Parse the feature matrix if needed.
  if feature_matrix is not None:
    body_data = parse_feature_matrix(feature_matrix)
    include_pelvis = False
    if 'elbow' not in body_data['position_m']:
      include_skeleton = False
  # Parse the body path data if needed.
  else:
    body_data = {'position_m': {}, 'quaternion_wijk': {}}
    include_pelvis = True
    for data_type in body_data.keys():
      body_data[data_type]['hand'] = bodyPath_data[data_type]['RightHand']
      body_data[data_type]['elbow'] = bodyPath_data[data_type]['RightForeArm']
      body_data[data_type]['shoulder'] = bodyPath_data[data_type]['RightUpperArm']
      body_data[data_type]['pelvis'] = bodyPath_data[data_type]['Pelvis']
  
  # Infer the pouring position.
  (_, stationary_pose) = infer_stationary_poses(time_s, body_data,
                                                use_variance=True, hand_segment_key='hand')
  
  # Initialize figure state.
  if previous_handles is None:
    fig = None
    h_chains = []
    h_scatters = []
    h_hand = None
    h_pitcher = None
  else:
    (fig, h_chains, h_scatters, h_hand, h_pitcher) = previous_handles
    
  # Create a figure if needed.
  if fig is None:
    if hide_figure_window:
      matplotlib.use('Agg')
    else:
      matplotlib.use(default_matplotlib_backend)
    fig = plt.figure()
    if not hide_figure_window:
      figManager = plt.get_current_fig_manager()
      figManager.window.showMaximized()
    plt.ion()
    num_rows = int(np.sqrt(num_total_subjects))
    num_cols = int(np.ceil(num_total_subjects/num_rows))
    for s in range(num_total_subjects):
      fig.add_subplot(num_rows, num_cols, s+1, projection='3d')
  ax = fig.get_axes()[subplot_index]
  
  # Get the table height.
  if referenceObject_position_m is not None:
    table_z_cm = 100*referenceObject_position_m[2] - referenceObject_height_cm
  
  # Draw items that remain the same across frames.
  if previous_handles is None or redraw_trajectory_each_timestep:
    if clear_previous_timestep:
      ax.clear()
    
    # Add labels and titles.
    ax.set_xlabel('X [cm]')
    ax.set_ylabel('Y [cm]')
    ax.set_zlabel('Z [cm]')
    
    # Set the view angle
    ax.view_init(*animation_view_angle)
    
    # Plot trajectories of the right arm and pelvis.
    hand_position_cm = 100*body_data['position_m']['hand']
    h_hand_path = ax.plot3D(hand_position_cm[:, 0], hand_position_cm[:, 1], hand_position_cm[:, 2], alpha=1)
    if include_skeleton:
      forearm_position_cm = 100*body_data['position_m']['elbow']
      upperArm_position_cm = 100*body_data['position_m']['shoulder']
      ax.plot3D(forearm_position_cm[:, 0], forearm_position_cm[:, 1], forearm_position_cm[:, 2], alpha=0.3)
      ax.plot3D(upperArm_position_cm[:, 0], upperArm_position_cm[:, 1], upperArm_position_cm[:, 2], alpha=0.3)
    if include_pelvis:
      pelvis_position_cm = 100*body_data['position_m']['pelvis']
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
        radius=referenceObject_diameter_cm/2, ec=[0.4,1,1], color=[0.8,1,1])
      ax.add_patch(referenceObject_circle)
      art3d.patch_2d_to_3d(referenceObject_circle, z=table_z_cm, zdir='z')
    
      # Plot the reference object.
      color_hex = h_hand_path[0].get_color().strip('#')
      color_rgb = tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4))
      color_rgb = np.array(color_rgb)/255
      plot_3d_box(ax, [1, 0, 0, 0], 100*referenceObject_position_m, [0, 0, -referenceObject_height_cm/2],
                  [referenceObject_diameter_cm, referenceObject_diameter_cm, referenceObject_height_cm],
                  color_rgb, alpha=0.2)
  
  # Clear plots from the previous timestep.
  if clear_previous_timestep:
    for i in range(len(h_chains)):
      h_chains[i][0].remove()
    for i in range(len(h_scatters)):
      h_scatters[i].remove()
    if h_hand is not None:
      h_hand.remove()
    if h_pitcher is not None:
      h_pitcher.remove()
    h_chains = []
    h_scatters = []
    h_hand = None
    h_pitcher = None

  # Animate the whole skeleton.
  if include_skeleton:
    # Draw the skeleton chains
    x = []
    y = []
    z = []
    for segment_name in ['hand', 'elbow', 'shoulder']:
      if segment_name not in body_data['position_m']:
        continue
      position_cm = 100*body_data['position_m'][segment_name]
      x.append(position_cm[time_index, 0])
      y.append(position_cm[time_index, 1])
      z.append(position_cm[time_index, 2])
    h_chains.append(ax.plot3D(x, y, z, color='k'))
    h_scatters.append(ax.scatter(x, y, z, s=25, color=[0.5, 0.5, 0.5]))
    
  if include_spout_projection:
    # Draw the pitcher tip projection onto the table.
    position_cm = 100*infer_spout_position_m(parsed_data=body_data, time_index=time_index)
    h_scatters.append(ax.scatter(position_cm[0], position_cm[1],
                                 table_z_cm,
                                 s=30, color='m'))
    # Draw an indicator of the spout direction on the table.
    spout_yawvector = infer_spout_yawvector(parsed_data=body_data, time_index=time_index)
    spout_yawvector = spout_yawvector * referenceObject_diameter_cm/2
    spout_yawsegment = np.array([[0,0,0], list(spout_yawvector)])
    spout_yawsegment = spout_yawsegment + position_cm
    h_chains.append(ax.plot3D(spout_yawsegment[:,0], spout_yawsegment[:,1],
                              (table_z_cm+0.1)*np.array([1,1]),
                              color='r', linewidth=2))
  
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
    h_hand = plot_hand_box(ax, hand_center_cm=100*body_data['position_m']['hand'][time_index, :],
                               hand_quaternion_localToGlobal_wijk=body_data['quaternion_wijk']['hand'][time_index, :])
  if include_pitcher:
    h_pitcher = plot_pitcher_box(ax, hand_center_cm=100*body_data['position_m']['hand'][time_index, :],
                                     hand_quaternion_localToGlobal_wijk=body_data['quaternion_wijk']['hand'][time_index, :])
  
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
  if subject_id >= 0:
    subject_str = 'Subject %s ' % str(subject_id)
  time_str = 't=%0.2fs' % ((time_s[time_index] - time_s[0]))
  ax.set_title('%s%s%s' % (subject_str, trial_str, time_str))
  
  # Show the plot.
  plt.draw()
  
  # Save the previous handles.
  previous_handles = (fig, h_chains, h_scatters, h_hand, h_pitcher)
  
  # plt_wait_for_keyboard_press()
  # print('view elev/azim:', ax.elev, ax.azim)
  
  if wait_for_user_after_plotting and not hide_figure_window:
    plt_wait_for_keyboard_press(timeout_s=-1)#spf)
    
  return previous_handles

# ================================================================
# Animate a whole trajectory.
# Supply a feature_matrix and duration_s OR a bodyPath_data and time_s
def animate_trajectory(feature_matrix=None, duration_s=None,
                       bodyPath_data=None, time_s=None,
                       referenceObject_position_m=None,
                       subject_id=-1, num_total_subjects=None, subplot_index=0,
                       trial_index=-1, trial_start_index_offset_forTitle=0,
                       include_skeleton=True, delay_s_between_timesteps=None,
                       wait_for_user_after_timesteps=False, hide_figure_window=False,
                       timestep_interval=1):
  if duration_s is not None:
    time_s = np.linspace(start=0, stop=duration_s, num=feature_matrix.shape[0])
  previous_handles = None
  for time_index in range(0, len(time_s), timestep_interval):
    start_plotting_s = time.time()
    previous_handles = plot_timestep(
      time_s, time_index,
      referenceObject_position_m=referenceObject_position_m,
      feature_matrix=feature_matrix, bodyPath_data=bodyPath_data,
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
# If feature_matrices, durations_s, and referenceObject_positions_m are dictionaries,
#  will assume each key is an example type and each value is the data for that type.
#  Will make a subplot for each example type, so they animate together.
def save_trajectory_animation(feature_matrices, durations_s, referenceObject_positions_m,
                              output_filepath, subject_id='', trial_index=-1):
  if not isinstance(feature_matrices, dict):
    feature_matrices = {'': feature_matrices}
    durations_s = {'': durations_s}
    referenceObject_positions_m = {'': referenceObject_positions_m}
  example_types = list(feature_matrices.keys())
  # Determine the times and rates of each provided example.
  times_s = {}
  fps = {}
  for example_type in example_types:
    times_s[example_type] = np.linspace(start=0, stop=durations_s[example_type],
                                        num=feature_matrices[example_type].shape[0])
    fps[example_type] = int(round((len(times_s[example_type])-1)/(times_s[example_type][-1] - times_s[example_type][0])))
  min_time_s = min([t[0] for t in list(times_s.values())])
  max_time_s = max([t[-1] for t in list(times_s.values())])
  max_fps = max(list(fps.values()))
  duration_s = max_time_s - min_time_s
  times_s_forVideo = np.arange(start=0, stop=duration_s, step=1/max_fps)
  
  os.makedirs(os.path.split(output_filepath)[0], exist_ok=True)
  
  video_writer = None
  previous_handles_byExampleType = dict([(example_type, None) for example_type in example_types])
  for (frame_index, time_s) in enumerate(times_s_forVideo):
    plot_imgs = []
    for example_type in example_types:
      times_s_forExample = times_s[example_type]
      feature_matrix = feature_matrices[example_type]
      if time_s >= times_s_forExample[0] and time_s <= times_s_forExample[-1]:
        time_index = times_s_forExample.searchsorted(time_s)
        previous_handles_byExampleType[example_type] = plot_timestep(feature_matrix,
                                                                     time_index=time_index, time_s=times_s_forExample,
                                                                     subject_id='%s%s' % ((str(example_type) if len(str(example_type).strip()) > 0 else ''),
                                                                              (str(subject_id) if len(str(subject_id).strip()) > 0 else '')),
                                                                     trial_index=trial_index,
                                                                     referenceObject_position_m=referenceObject_positions_m[example_type],
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











