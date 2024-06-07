import numpy as np
from scipy.spatial.transform import Rotation
from scipy import stats

import matplotlib
default_matplotlib_backend = matplotlib.rcParams['backend']
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from mpl_toolkits.mplot3d import art3d
import os

from utils.numpy_scipy_utils import *

import cv2

##################################################################
# Configuration
##################################################################

stationary_position_buffer_length = 15
stationary_position_minIndex = 10
stationary_position_maxIndex = 90
hand_box_dimensions_cm = np.array([4.8, 3, 1.3])
hand_box_color = 0.8*np.array([1, 0.6, 0])
pitcher_box_dimensions_cm = np.array([23, 23, 10.8]) # [height, top length, width]
pitcher_box_color = 0.8*np.array([1, 1, 1])
hand_to_pitcher_rotation = Rotation.from_rotvec(np.array([np.radians(90),
                                                          np.radians(0),
                                                          np.radians(-5)]))
hand_to_pitcherTop_cm = 8
hand_to_pitcher_offset_cm = np.array([hand_to_pitcherTop_cm - pitcher_box_dimensions_cm[0]/2,
                                      -(0+pitcher_box_dimensions_cm[1]/2),
                                      2])

referenceObject_diameter_cm = 7.3 # glass top 7.3 bottom 6.3
referenceObject_height_cm = 15.8
hand_to_referenceObject_bottom_cm = 6
hand_to_referenceObject_top_cm = referenceObject_height_cm - hand_to_referenceObject_bottom_cm

animation_view_angle_backLeft = (16, -28)
animation_view_angle_backRight = (16, 44)
animation_view_angle_forBaxter = (30, -179.9)
# animation_view_angle = animation_view_angle_backLeft
# animation_view_angle = animation_view_angle_backRight
animation_view_angle = animation_view_angle_forBaxter

# Used to artificially shift distributions for demonstration purposes.
example_types_to_offset = []#['model']
if len(example_types_to_offset) > 0:
  print()
  print('*'*50)
  print('*'*50)
  print('NOTE THAT DISTRIBUTIONS ARE BEING ARTIFICALLY')
  print('SHIFTED FOR DEMONSTRATION PURPOSES FOR')
  print('THE FOLLOWING EXAMPLE TYPES')
  print(example_types_to_offset)
  print('*'*50)
  print('*'*50)
  input('Press Enter to confirm and continue')
  print()
  print()
  
##################################################################
# Helpers to parse feature matrices
##################################################################

# ================================================================
# Feature_matrices should be Tx30, where
#   T is the number of timesteps in each trial
#   30 is the concatenation of:
#     xyz position for hand > elbow > shoulder
#     wijk quaternion for hand > lower arm > upper arm
#     xzy joint angle for wrist > elbow > shoulder
def parse_feature_matrix(feature_matrix):
  if feature_matrix.shape[-1] == 31: # human demonstrations
    return {
      'position_m' : {
        'hand':     feature_matrix[:, 0:3],
        'elbow':    feature_matrix[:, 3:6],
        'shoulder': feature_matrix[:, 6:9],
      },
      'quaternion_wijk': {
        'hand':     feature_matrix[:, 9:13],
        'elbow':    feature_matrix[:, 13:17],
        'shoulder': feature_matrix[:, 17:21],
      },
      'joint_angle_rad': {
        'hand':     feature_matrix[:, 21:24],
        'elbow':    feature_matrix[:, 24:27],
        'shoulder': feature_matrix[:, 27:30],
      },
      'time_s': feature_matrix[:, 30]
    }
  elif feature_matrix.shape[-1] == 16: # model outputs
    return {
      'position_m' : {
        'hand':     feature_matrix[:, 0:3],
        # 'elbow':    None,
        # 'shoulder': None,
      },
      'quaternion_wijk': {
        'hand':     feature_matrix[:, 3:7],
        # 'elbow':    None,
        # 'shoulder': None,
      },
      'joint_angle_rad': {
        'hand':     feature_matrix[:, 7:10],
        'elbow':    feature_matrix[:, 10:13],
        'shoulder': feature_matrix[:, 13:16],
      },
      'time_s': np.linspace(0, 10, feature_matrix.shape[0]),
    }
    

# ================================================================
# Get the 3D positions of body segments.
# Will return a dict with 'hand', 'elbow', and 'shoulder' each of which has xyz positions.
def get_body_position_m(feature_matrix):
  parsed_data = parse_feature_matrix(feature_matrix)
  return parsed_data['position_m']

# ================================================================
# Get the 3D rotation angles of each joint.
# Will return a dict with 'hand', 'elbow', and 'shoulder' each of which has xyz rotations.
def get_body_joint_angles_rad(feature_matrix):
  parsed_data = parse_feature_matrix(feature_matrix)
  return parsed_data['joint_angle_rad']

# ================================================================
# Get the 3D speeds of body segments.
# Will return a dict with 'hand', 'elbow', and 'shoulder' each of which has a speed vector.
def get_body_speed_m_s(feature_matrix):
  positions_m = get_body_position_m(feature_matrix)
  times_s = feature_matrix[:,-1]
  dt = np.reshape(np.diff(times_s, axis=0), (-1, 1))
  speeds_m_s = dict.fromkeys(positions_m.keys())
  for (body_key, position_m) in positions_m.items():
    if position_m is None:
      continue
    # Infer the speed.
    dxdydz = np.diff(position_m, axis=0)
    speed_m_s = np.hstack([np.squeeze([0]), np.linalg.norm(dxdydz, axis=1)/np.squeeze(dt)])
    speeds_m_s[body_key] = speed_m_s
  return speeds_m_s
  
# ================================================================
# Get the 3D accelerations of body segments.
# Will return a dict with 'hand', 'elbow', and 'shoulder' each of which has an acceleration vector.
def get_body_acceleration_m_s_s(feature_matrix):
  speeds_m_s = get_body_speed_m_s(feature_matrix)
  times_s = feature_matrix[:,-1]
  dt = np.reshape(np.diff(times_s, axis=0), (-1, 1))
  accelerations_m_s_s = dict.fromkeys(speeds_m_s.keys())
  for (body_key, speed_m_s) in speeds_m_s.items():
    if speed_m_s is None:
      continue
    # Infer the acceleration.
    dv = np.diff(speed_m_s, axis=0)
    dt = np.reshape(np.diff(times_s, axis=0), (-1, 1))
    acceleration_m_s_s = np.hstack([np.squeeze([0]), dv/np.squeeze(dt)])
    accelerations_m_s_s[body_key] = acceleration_m_s_s
  return accelerations_m_s_s

# ================================================================
# Get the 3D jerks of body segments.
# Will return a dict with 'hand', 'elbow', and 'shoulder' each of which has a jerk vector.
def get_body_jerk_m_s_s_s(feature_matrix):
  accelerations_m_s_s = get_body_acceleration_m_s_s(feature_matrix)
  times_s = feature_matrix[:,-1]
  dt = np.reshape(np.diff(times_s, axis=0), (-1, 1))
  jerks_m_s_s_s = dict.fromkeys(accelerations_m_s_s.keys())
  for (body_key, acceleration_m_s_s) in accelerations_m_s_s.items():
    if acceleration_m_s_s is None:
      continue
    # Infer the jerk.
    da = np.diff(acceleration_m_s_s, axis=0)
    dt = np.reshape(np.diff(times_s, axis=0), (-1, 1))
    jerk_m_s_s_s = np.hstack([np.squeeze([0]), da/np.squeeze(dt)])
    jerks_m_s_s_s[body_key] = jerk_m_s_s_s
  return jerks_m_s_s_s

##################################################################
# Helpers for 3D plotting
##################################################################

# ================================================================
# Rotate a box in 3D for visualization purposes.
# If center_preRotation_cm is provided, will translate box to that location before applying the given quaternion.
def rotate_3d_box(quaternion_localToGlobal_wijk, center_preRotation_cm, box_dimensions_cm):
  # Define vertices of a unit box in the global frame
  corners = np.array([
    [-1, -1, -1],
    [-1, -1, 1],
    [-1, 1, -1],
    [-1, 1, 1],
    [1, -1, -1],
    [1, -1, 1],
    [1, 1, -1],
    [1, 1, 1]
  ]) * 0.5
  # Define faces of the box in the global frame, using corner indexes
  faces = np.array([
    [0, 1, 3, 2], # bottom face
    [0, 2, 6, 4],
    [0, 1, 5, 4],
    [4, 5, 7, 6], # top face
    [1, 3, 7, 5],
    [2, 3, 7, 6], # hand-side face
  ])
  # Scale the box
  corners = corners * box_dimensions_cm
  
  # Translate the box
  corners = corners + center_preRotation_cm
  
  # Invert quaternion.
  quaternion_globalToLocal_ijkw = [
    -quaternion_localToGlobal_wijk[1],
    -quaternion_localToGlobal_wijk[2],
    -quaternion_localToGlobal_wijk[3],
    quaternion_localToGlobal_wijk[0],
    ]
  # Rotate the box using the quaternion,
  rot = Rotation.from_quat(quaternion_globalToLocal_ijkw).as_matrix()
  corners = np.dot(corners, rot)
  
  return (corners, faces)

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

# ================================================================
# Wait for the user to press a key while a plot window is active.
def plt_wait_for_keyboard_press(timeout_s=-1.0):
  keyboardClick=False
  while keyboardClick == False:
    keyboardClick = plt.waitforbuttonpress(timeout=timeout_s)

##################################################################
# Trajectory animation
##################################################################

# ================================================================
# Plot the scene for a single timestep.
def plot_timestep(feature_matrix, times_s, time_index,
                  referenceObject_position_m,
                  subject_id=-1, trial_index=-1,
                  previous_handles=None, include_skeleton=True, spf=None,
                  pause_after_plotting=False, hide_figure_window=False):
  # Parse the feature matrix.
  parsed_data = parse_feature_matrix(feature_matrix)
  if 'elbow' not in parsed_data['position_m']:
    include_skeleton = False
  # Infer the pouring position.
  stationary_pose = infer_pour_pose(feature_matrix)
  
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
    fig.add_subplot(1, 1, 1, projection='3d')
  ax = fig.get_axes()[0]
  
  # Get the table height.
  table_z_cm = 100*referenceObject_position_m[2] - referenceObject_height_cm
  
  # Draw items that remain the same across frames.
  if previous_handles is None:
    ax.clear()
    
    # Add labels and titles.
    ax.set_xlabel('X [cm]')
    ax.set_ylabel('Y [cm]')
    ax.set_zlabel('Z [cm]')
    
    # Set the view angle
    ax.view_init(*animation_view_angle)
    
    # Plot trajectories of the right arm and pelvis.
    hand_position_cm = 100*parsed_data['position_m']['hand']
    h_hand_path = ax.plot3D(hand_position_cm[:, 0], hand_position_cm[:, 1], hand_position_cm[:, 2], alpha=1)
    if include_skeleton:
      forearm_position_cm = 100*parsed_data['position_m']['elbow']
      upperArm_position_cm = 100*parsed_data['position_m']['shoulder']
      ax.plot3D(forearm_position_cm[:, 0], forearm_position_cm[:, 1], forearm_position_cm[:, 2], alpha=0.3)
      ax.plot3D(upperArm_position_cm[:, 0], upperArm_position_cm[:, 1], upperArm_position_cm[:, 2], alpha=0.3)
    
    # Plot origin and start/end/stationary hand positions.
    ax.scatter(0, 0, 0, s=25, color=[0, 0, 0])
    ax.scatter(hand_position_cm[0, 0], hand_position_cm[0, 1], hand_position_cm[0, 2], s=25, color='g')
    ax.scatter(hand_position_cm[-1, 0], hand_position_cm[-1, 1], hand_position_cm[-1, 2], s=25, color='r')
    ax.scatter(100*stationary_pose['position_m']['hand'][0],
               100*stationary_pose['position_m']['hand'][1],
               100*stationary_pose['position_m']['hand'][2],
               s=25, color=h_hand_path[0].get_color(), edgecolor='k')
    
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

  if include_skeleton:
    # Animate the whole skeleton.
    sampling_rate_hz = (times_s.shape[0] - 1) / (times_s[-1] - times_s[0])
    spf = spf or 1/sampling_rate_hz
    timestep_interval = max([1, int(sampling_rate_hz*spf)])
    
    # Draw the skeleton chains
    x = []
    y = []
    z = []
    for segment_name in ['hand', 'elbow', 'shoulder']:
      position_cm = 100*parsed_data['position_m'][segment_name]
      x.append(position_cm[time_index, 0])
      y.append(position_cm[time_index, 1])
      z.append(position_cm[time_index, 2])
    h_chains.append(ax.plot3D(x, y, z, color='k'))
    h_scatters.append(ax.scatter(x, y, z, s=25, color=[0.5, 0.5, 0.5]))
    
  # Draw the pitcher tip projection onto the table.
  position_cm = 100*infer_spout_position_m(feature_matrix, time_index)
  h_scatters.append(ax.scatter(position_cm[0], position_cm[1],
                               table_z_cm,
                               s=30, color='m'))
  # Draw an indicator of the spout direction on the table.
  spout_yawvector = infer_spout_yawvector(feature_matrix, time_index)
  spout_yawvector = spout_yawvector * referenceObject_diameter_cm/2
  spout_yawsegment = np.array([[0,0,0], list(spout_yawvector)])
  spout_yawsegment = spout_yawsegment + position_cm
  h_chains.append(ax.plot3D(spout_yawsegment[:,0], spout_yawsegment[:,1],
                            table_z_cm*np.array([1,1]),
                            color='r', linewidth=2))
  
  x_lim = ax.get_xlim()
  y_lim = ax.get_ylim()
  z_lim = ax.get_zlim()
  
  # Visualize a box as the hand.
  # hand_dimensions_cm = [1, 3, 5]
  # hand_rotation_matrix = Rotation.from_quat(bodySegment_data['quaternion']['RightHand'])
  # print(hand_rotation_matrix.apply(hand_dimensions_cm))
  # hand_box_data = np.ones(hand_dimensions_cm, dtype=bool)
  # hand_colors = np.empty(hand_dimensions_cm + [4], dtype=np.float32)
  # hand_colors[:] = [1, 0, 0, 0.8]
  # h_hand = ax.voxels(hand_box_data, facecolors=hand_colors)
  h_hand = plot_hand_box(ax, hand_center_cm=100 * parsed_data['position_m']['hand'][time_index, :],
                             hand_quaternion_localToGlobal_wijk=parsed_data['quaternion_wijk']['hand'][time_index, :])
  h_pitcher = plot_pitcher_box(ax, hand_center_cm=100 * parsed_data['position_m']['hand'][time_index, :],
                                   hand_quaternion_localToGlobal_wijk=parsed_data['quaternion_wijk']['hand'][time_index, :])
  
  # Set the aspect ratio
  ax.set_xlim(x_lim)
  ax.set_ylim(y_lim)
  ax.set_zlim(z_lim)
  ax.set_box_aspect([ub - lb for lb, ub in (x_lim, y_lim, z_lim)])
  
  # Set the title.
  ax.set_title('%s Trial %02d\nt=%0.2fs' % (str(subject_id), trial_index, times_s[time_index] - times_s[0]))
  
  # Show the plot.
  plt.draw()
  
  # Save the previous handles.
  previous_handles = (fig, h_chains, h_scatters, h_hand, h_pitcher)
  
  # plt_wait_for_keyboard_press()
  # print('view elev/azim:', ax.elev, ax.azim)
  
  if pause_after_plotting and not hide_figure_window:
    plt_wait_for_keyboard_press(timeout_s=-1)#spf)
    
  return previous_handles

# ================================================================
# Animate a whole trajectory.
def plot_trajectory(feature_matrix, duration_s, referenceObject_position_m, pause_after_timesteps):
  times_s = np.linspace(start=0, stop=duration_s, num=feature_matrix.shape[0])
  previous_handles = None
  for (time_index, time_s) in enumerate(times_s):
    previous_handles = plot_timestep(feature_matrix,
                                     time_index=time_index, times_s=times_s,
                                     referenceObject_position_m=referenceObject_position_m,
                                     previous_handles=previous_handles,
                                     pause_after_plotting=pause_after_timesteps)

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
                                                         time_index=time_index, times_s=times_s_forExample,
                                                         subject_id='%s%s' % ((str(example_type) if len(str(example_type).strip()) > 0 else ''),
                                                                              (str(subject_id) if len(str(subject_id).strip()) > 0 else '')),
                                                         trial_index=trial_index,
                                                         referenceObject_position_m=referenceObject_positions_m[example_type],
                                                         previous_handles=None,
                                                         pause_after_plotting=False,
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

##################################################################
# Plot time series or summary metrics about the trajectories.
##################################################################

# ================================================================
# Plot the pitcher tilt angle over time.
def plot_pour_tilting(feature_matrices, shade_pouring_region=False,
                      plot_all_trials=True, plot_std_shading=False, plot_mean=False,
                      label=None, subtitle=None,
                      fig=None, hide_figure_window=False, output_filepath=None):
  if output_filepath is not None:
    os.makedirs(os.path.split(output_filepath)[0], exist_ok=True)
  if not isinstance(feature_matrices, (list, tuple)) and not (isinstance(feature_matrices, np.ndarray) and feature_matrices.ndim == 3):
    feature_matrices = [feature_matrices]
  if fig is None:
    if hide_figure_window:
      try:
        matplotlib.use('Agg')
      except:
        pass
    else:
      matplotlib.use(default_matplotlib_backend)
    fig = plt.figure()
    if not hide_figure_window:
      figManager = plt.get_current_fig_manager()
      figManager.window.showMaximized()
      plt_wait_for_keyboard_press(0.5)
    plt.ion()
    fig.add_subplot(1, 1, 1)
  ax = fig.get_axes()[0]
  
  # Get the tilt angle for each timestep of each example.
  angles_toXY_rad = np.zeros((len(feature_matrices), feature_matrices[0].shape[0]))
  for trial_index, feature_matrix in enumerate(feature_matrices):
    angle_toXY_rad = np.zeros(shape=(feature_matrix.shape[0],))
    for time_index in range(feature_matrix.shape[0]):
      angle_toXY_rad[time_index] = infer_spout_tilting(feature_matrix, time_index)
    angles_toXY_rad[trial_index, :] = angle_toXY_rad
  
  # Get the pouring times.
  if shade_pouring_region:
    pour_start_indexes = np.zeros((len(feature_matrices), 1))
    pour_end_indexes = np.zeros((len(feature_matrices), 1))
    for trial_index, feature_matrix in enumerate(feature_matrices):
      pouring_inference = infer_pour_pose(feature_matrix)
      pour_start_indexes[trial_index] = pouring_inference['start_time_index']
      pour_end_indexes[trial_index] = pouring_inference['end_time_index']
  
  # Plot shading if desired.
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
    for trial_index in range(len(feature_matrices)):
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
    for trial_index in range(len(feature_matrices)):
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
# Plot the spout projection onto the table relative to the glass.
#   The blue shaded circle represents the glass.
#   The y-axis is the pouring direction, as inferred from the yaw of the pitcher in each trial.
#   A point will be plotted at the spout's projection onto the table, relative to the glass.
#   So the water would pour upward on the plot from the plotted spout position.
def plot_pour_relativePosition(feature_matrices, referenceObject_positions_m,
                               plot_all_trials=True, plot_std_shading=False, plot_mean=False,
                               subtitle=None, label=None,
                               fig=None, hide_figure_window=False,
                               color=None,
                               output_filepath=None):
  if output_filepath is not None:
    os.makedirs(os.path.split(output_filepath)[0], exist_ok=True)
  if not isinstance(feature_matrices, (list, tuple)) and not (isinstance(feature_matrices, np.ndarray) and feature_matrices.ndim == 3):
    feature_matrices = [feature_matrices]
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
    fig = plt.figure()
    if not hide_figure_window:
      figManager = plt.get_current_fig_manager()
      figManager.window.showMaximized()
      plt_wait_for_keyboard_press(0.5)
    plt.ion()
    fig.add_subplot(1, 1, 1)
    ax = fig.get_axes()[0]
  else:
    ax = fig.get_axes()[0]
  
  spout_relativeOffsets_cm = np.zeros((len(feature_matrices), 2))
  for trial_index in range(len(feature_matrices)):
    feature_matrix = feature_matrices[trial_index]
    referenceObject_position_m = referenceObject_positions_m[trial_index]
    # Get the pouring time.
    pouring_inference = infer_pour_pose(feature_matrix)
    pour_index = pouring_inference['time_index']
    # Get the spout projection and yaw.
    spout_position_cm = 100*infer_spout_position_m(feature_matrix, pour_index)
    spout_yawvector = infer_spout_yawvector(feature_matrix, pour_index)
    # Project everything to the XY plane.
    spout_position_cm = spout_position_cm[0:2]
    spout_yawvector = spout_yawvector[0:2]
    referenceObject_position_cm = 100*referenceObject_position_m[0:2]
    # Use the spout projection as the origin.
    referenceObject_position_cm = referenceObject_position_cm - spout_position_cm
    # Rotate so the yaw vector is the new y-axis.
    yaw_rotation_matrix = rotation_matrix_from_vectors([spout_yawvector[0], spout_yawvector[1], 0],
                                                       [0, 1, 0])
    referenceObject_position_cm = yaw_rotation_matrix.dot(np.array([referenceObject_position_cm[0], referenceObject_position_cm[1], 0]))
    referenceObject_position_cm = referenceObject_position_cm[0:2]
    # Move the origin to the reference object.
    spout_relativeOffset_cm = -referenceObject_position_cm
    # Store the result.
    spout_relativeOffsets_cm[trial_index, :] = spout_relativeOffset_cm
  
  # Plot a standard deviation shaded region if desired.
  if plot_std_shading:
    if label.lower() in example_types_to_offset:
      spout_relativeOffsets_cm += np.array([1, 2])
      
    # Helper function from https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
    def confidence_ellipse(x, y, ax, n_std=3.0, facecolor=None, **kwargs):
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
    
    # Plot the ellipse for the spout positions.
    confidence_ellipse(x=spout_relativeOffsets_cm[:,0], y=spout_relativeOffsets_cm[:,1], ax=ax,
                       n_std=1.0, facecolor='none', alpha=1, edgecolor=color, linewidth=3)
    confidence_ellipse(x=spout_relativeOffsets_cm[:,0], y=spout_relativeOffsets_cm[:,1], ax=ax,
                       n_std=2.0, facecolor='none', alpha=0.2, edgecolor=color, linewidth=3,)
    confidence_ellipse(x=spout_relativeOffsets_cm[:,0], y=spout_relativeOffsets_cm[:,1], ax=ax,
                       n_std=3.0, facecolor='none', alpha=0.2, edgecolor=color, linewidth=3)
    confidence_ellipse(x=spout_relativeOffsets_cm[:,0], y=spout_relativeOffsets_cm[:,1], ax=ax,
                       n_std=3.0, facecolor=color, alpha=0.2,
                       label=('%s: StdDevs' % label) if label is not None else 'StdDevs')
    ax.legend()
    
  # Plot all trial results if desired.
  if plot_all_trials:
    for trial_index in range(len(feature_matrices)):
      ax.scatter(*spout_relativeOffsets_cm[trial_index,:], c=color, s=25)
  
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
  ax.scatter(0, 0, s=10, c='k')
  
  # Plot formatting.
  ax.set_aspect('equal')
  ax.set_xlabel('Horizontal Relative to Pouring Direction [cm]')
  ax.set_ylabel('Vertical Relative to Pouring Direction [cm]')
  ax.grid(True, color='lightgray')
  ax.set_axisbelow(True)
  plt.title('Spout projected onto table, along pouring axis%s' % ((': %s' % subtitle) if subtitle is not None else ''))
  
  # Show the plot.
  plt.draw()
  
  # Save the plot if desired.
  if output_filepath is not None:
    fig.savefig(output_filepath, dpi=300)
  
  return (fig, spout_relativeOffsets_cm)

# ================================================================
# Plot the height of the spout relative to the top of the glass over time.
def plot_pour_relativeHeight(feature_matrices, referenceObject_positions_m,
                             shade_pouring_region=False,
                             plot_all_trials=True, plot_std_shading=False, plot_mean=False,
                             subtitle=None, label=None,
                             fig=None, hide_figure_window=False,
                             color=None,
                             output_filepath=None):
  if output_filepath is not None:
    os.makedirs(os.path.split(output_filepath)[0], exist_ok=True)
  if not isinstance(feature_matrices, (list, tuple)) and not (isinstance(feature_matrices, np.ndarray) and feature_matrices.ndim == 3):
    feature_matrices = [feature_matrices]
  if not isinstance(referenceObject_positions_m, (list, tuple)) and not (isinstance(referenceObject_positions_m, np.ndarray) and referenceObject_positions_m.ndim == 2 and referenceObject_positions_m.shape[0] == len(feature_matrices)):
    referenceObject_positions_m = [referenceObject_positions_m]
  if fig is None:
    if hide_figure_window:
      try:
        matplotlib.use('Agg')
      except:
        pass
    else:
      matplotlib.use(default_matplotlib_backend)
    fig = plt.figure()
    if not hide_figure_window:
      figManager = plt.get_current_fig_manager()
      figManager.window.showMaximized()
      plt_wait_for_keyboard_press(0.5)
    plt.ion()
    fig.add_subplot(1, 1, 1)
    ax = fig.get_axes()[0]
  else:
    ax = fig.get_axes()[0]
  
  # Get the spout heights for each trial.
  spout_heights_cm = np.zeros((len(feature_matrices), feature_matrices[0].shape[0]))
  for trial_index in range(len(feature_matrices)):
    feature_matrix = feature_matrices[trial_index]
    referenceObject_position_m = referenceObject_positions_m[trial_index]
    # Get the spout height at each timestep.
    spout_relativeHeight_cm = np.zeros(shape=(feature_matrix.shape[0],))
    for time_index in range(feature_matrix.shape[0]):
      spout_position_cm = 100*infer_spout_position_m(feature_matrix, time_index)
      spout_relativeHeight_cm[time_index] = spout_position_cm[2] - 100*referenceObject_position_m[2]
    spout_heights_cm[trial_index, :] = spout_relativeHeight_cm
  
  # Get the pouring times.
  if shade_pouring_region:
    pour_start_indexes = np.zeros((len(feature_matrices), 1))
    pour_end_indexes = np.zeros((len(feature_matrices), 1))
    for trial_index, feature_matrix in enumerate(feature_matrices):
      pouring_inference = infer_pour_pose(feature_matrix)
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
    for trial_index in range(len(feature_matrices)):
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
    for trial_index in range(len(feature_matrices)):
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
def plot_spout_dynamics(feature_matrices,
                        plot_all_trials=True, plot_mean=False, plot_std_shading=False,
                        subtitle=None, label=None,
                        output_filepath=None,
                        shade_pouring_region=False,
                        fig=None, hide_figure_window=False):
    
  if output_filepath is not None:
    os.makedirs(os.path.split(output_filepath)[0], exist_ok=True)
  if not isinstance(feature_matrices, (list, tuple)) and not (isinstance(feature_matrices, np.ndarray) and feature_matrices.ndim == 3):
    feature_matrices = [feature_matrices]
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
                               )
    if not hide_figure_window:
      figManager = plt.get_current_fig_manager()
      figManager.window.showMaximized()
      plt_wait_for_keyboard_press(0.5)
    plt.ion()
  else:
    (fig, axs) = fig
  
  # Get the spout dynamics for each timestep.
  speeds_m_s = [None]*len(feature_matrices)
  jerks_m_s_s_s = [None]*len(feature_matrices)
  for trial_index, feature_matrix in enumerate(feature_matrices):
    speeds_m_s[trial_index] = infer_spout_speed_m_s(feature_matrix)
    jerks_m_s_s_s[trial_index] = infer_spout_jerk_m_s_s_s(feature_matrix)
  
  # Get the pouring times.
  if shade_pouring_region:
    pour_start_indexes = np.zeros((len(feature_matrices), 1))
    pour_end_indexes = np.zeros((len(feature_matrices), 1))
    for trial_index, feature_matrix in enumerate(feature_matrices):
      pouring_inference = infer_pour_pose(feature_matrix)
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
    for trial_index in range(len(feature_matrices)):
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
    for trial_index in range(len(feature_matrices)):
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
def plot_body_dynamics(feature_matrices,
                        plot_all_trials=True, plot_mean=False, plot_std_shading=False,
                        subtitle=None, label=None,
                        output_filepath=None,
                        shade_pouring_region=False,
                        fig=None, hide_figure_window=False):
    
  if output_filepath is not None:
    os.makedirs(os.path.split(output_filepath)[0], exist_ok=True)
  if not isinstance(feature_matrices, (list, tuple)) and not (isinstance(feature_matrices, np.ndarray) and feature_matrices.ndim == 3):
    feature_matrices = [feature_matrices]
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
                               )
    if not hide_figure_window:
      figManager = plt.get_current_fig_manager()
      figManager.window.showMaximized()
      plt_wait_for_keyboard_press(0.5)
    plt.ion()
  else:
    (fig, axs) = fig
  
  # Get the body dynamics for each timestep.
  speeds_m_s = [None]*len(feature_matrices)
  jerks_m_s_s_s = [None]*len(feature_matrices)
  for trial_index, feature_matrix in enumerate(feature_matrices):
    speeds_m_s[trial_index] = get_body_speed_m_s(feature_matrix)
    jerks_m_s_s_s[trial_index] = get_body_jerk_m_s_s_s(feature_matrix)
  
  # Get the pouring times.
  if shade_pouring_region:
    pour_start_indexes = np.zeros((len(feature_matrices), 1))
    pour_end_indexes = np.zeros((len(feature_matrices), 1))
    for trial_index, feature_matrix in enumerate(feature_matrices):
      pouring_inference = infer_pour_pose(feature_matrix)
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
      for trial_index in range(len(feature_matrices)):
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
      for trial_index in range(len(feature_matrices)):
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
def plot_body_joint_angles(feature_matrices,
                            plot_all_trials=True, plot_mean=False, plot_std_shading=False,
                            subtitle=None, label=None,
                            output_filepath=None,
                            shade_pouring_region=False,
                            fig=None, hide_figure_window=False):
    
  if output_filepath is not None:
    os.makedirs(os.path.split(output_filepath)[0], exist_ok=True)
  if not isinstance(feature_matrices, (list, tuple)) and not (isinstance(feature_matrices, np.ndarray) and feature_matrices.ndim == 3):
    feature_matrices = [feature_matrices]
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
                               )
    if not hide_figure_window:
      figManager = plt.get_current_fig_manager()
      figManager.window.showMaximized()
      plt_wait_for_keyboard_press(0.5)
    plt.ion()
  else:
    (fig, axs) = fig
  
  # Get the body dynamics for each timestep.
  joint_angles_rad = [None]*len(feature_matrices)
  for trial_index, feature_matrix in enumerate(feature_matrices):
    joint_angles_rad[trial_index] = get_body_joint_angles_rad(feature_matrix)
  
  # Get the pouring times.
  if shade_pouring_region:
    pour_start_indexes = np.zeros((len(feature_matrices), 1))
    pour_end_indexes = np.zeros((len(feature_matrices), 1))
    for trial_index, feature_matrix in enumerate(feature_matrices):
      pouring_inference = infer_pour_pose(feature_matrix)
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
        for trial_index in range(len(feature_matrices)):
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
        for trial_index in range(len(feature_matrices)):
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



##################################################################
# Plot and compare distributions of trajectory metrics.
##################################################################

# ================================================================
# Plot and compare distributions of the spout speed and jerk.
# feature_matrices_allTypes is a dictionary mapping distribution category to matrices for each trial.
# If region is provided, will only consider timesteps during that region for each trial.
#   Can be 'pre_pouring', 'pouring', 'post_pouring', or None for all.
def plot_compare_distributions_spout_dynamics(feature_matrices_allTypes,
                        subtitle=None,
                        output_filepath=None,
                        region=None, # 'pre_pouring', 'pouring', 'post_pouring', None for all
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
                                 )
      if not hide_figure_window:
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt_wait_for_keyboard_press(0.5)
      plt.ion()
    else:
      (fig, axs) = fig
  
  # Initialize results/outputs.
  example_types = list(feature_matrices_allTypes.keys())
  distributions_speed_m_s = dict.fromkeys(example_types, None)
  distributions_jerk_m_s_s_s = dict.fromkeys(example_types, None)
  results_speed_m_s = dict([(example_type, dict.fromkeys(example_types, None)) for example_type in example_types])
  results_jerk_m_s_s_s = dict([(example_type, dict.fromkeys(example_types, None)) for example_type in example_types])
  
  # Process each example type (each distribution category).
  for (example_type, feature_matrices) in feature_matrices_allTypes.items():
    # Get the spout dynamics for each timestep.
    speeds_m_s = [None]*len(feature_matrices)
    jerks_m_s_s_s = [None]*len(feature_matrices)
    for trial_index, feature_matrix in enumerate(feature_matrices):
      speeds_m_s[trial_index] = infer_spout_speed_m_s(feature_matrix)
      jerks_m_s_s_s[trial_index] = infer_spout_jerk_m_s_s_s(feature_matrix)
  
    # If desired, only extract a certain window of time relative to the inferred pouring time.
    for trial_index, feature_matrix in enumerate(feature_matrices):
      pouring_inference = infer_pour_pose(feature_matrix)
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
    distributions_jerk_m_s_s_s[example_type] = np.abs(np.stack(np.concatenate(jerks_m_s_s_s)))
  
  # Statistical tests to compare the distributions.
  for example_type_1 in example_types:
    for example_type_2 in example_types:
      speeds_m_s_1 = distributions_speed_m_s[example_type_1]
      speeds_m_s_2 = distributions_speed_m_s[example_type_2]
      jerks_m_s_s_s_1 = distributions_jerk_m_s_s_s[example_type_1]
      jerks_m_s_s_s_2 = distributions_jerk_m_s_s_s[example_type_2]
      results_speed_m_s[example_type_1][example_type_2] = \
        stats.kstest(speeds_m_s_1, speeds_m_s_2,
                     alternative='two-sided', # 'two-sided', 'less', 'greater'
                     method='auto', # ‘auto’, ‘exact’, ‘approx’, ‘asymp’
                     )
      results_jerk_m_s_s_s[example_type_1][example_type_2] = \
        stats.kstest(jerks_m_s_s_s_1, jerks_m_s_s_s_2,
                     alternative='two-sided', # 'two-sided', 'less', 'greater'
                     method='auto', # ‘auto’, ‘exact’, ‘approx’, ‘asymp’
                     )
  
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
                    range=(np.quantile(speeds_m_s, histogram_range_quantiles[0]), np.quantile(speeds_m_s, histogram_range_quantiles[1])),
                    alpha=0.5, label=example_type.title())
      ax_jerk.hist(jerks_m_s_s_s, bins=num_histogram_bins,
                   histtype='stepfilled', # 'bar', 'barstacked', 'step', 'stepfilled'
                   log=False, density=True,
                   range=(np.quantile(jerks_m_s_s_s, histogram_range_quantiles[0]), np.quantile(jerks_m_s_s_s, histogram_range_quantiles[1])),
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

  # Print statistical test results.
  if print_comparison_results:
    print(' Statistical comparison results for spout speed:')
    for example_type_1 in example_types:
      for example_type_2 in example_types:
        print('  Comparing [%s] to [%s]: ' % (example_type_1, example_type_2), end='')
        results = results_speed_m_s[example_type_1][example_type_2]
        p = results.pvalue
        print('Different? %s (p = %0.4f)' % ('yes' if p < 0.05 else 'no', p))
    print(' Statistical comparison results for spout jerk:')
    for example_type_1 in example_types:
      for example_type_2 in example_types:
        print('  Comparing [%s] to [%s]: ' % (example_type_1, example_type_2), end='')
        results = results_jerk_m_s_s_s[example_type_1][example_type_2]
        p = results.pvalue
        print('Different? %s (p = %0.4f)' % ('yes' if p < 0.05 else 'no', p))
  
  return (fig, axs)

# ================================================================
# Plot and compare distributions of the hand, elbow, and shoulder speed and jerk.
# feature_matrices_allTypes is a dictionary mapping distribution category to matrices for each trial.
# If region is provided, will only consider timesteps during that region for each trial.
#   Can be 'pre_pouring', 'pouring', 'post_pouring', or None for all.
def plot_compare_distributions_body_dynamics(feature_matrices_allTypes,
                        subtitle=None,
                        output_filepath=None,
                        region=None, # 'pre_pouring', 'pouring', 'post_pouring', None for all
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
                                 )
      if not hide_figure_window:
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt_wait_for_keyboard_press(0.5)
      plt.ion()
    else:
      (fig, axs) = fig
  
  # Initialize results/outputs.
  example_types = list(feature_matrices_allTypes.keys())
  distributions_speed_m_s = dict([(key, {}) for key in example_types])
  distributions_jerk_m_s_s_s = dict([(key, {}) for key in example_types])
  results_speed_m_s = dict([(example_type, dict([(key, {}) for key in example_types])) for example_type in example_types])
  results_jerk_m_s_s_s = dict([(example_type, dict([(key, {}) for key in example_types])) for example_type in example_types])
  
  # Process each example type (each distribution category).
  for (example_type, feature_matrices) in feature_matrices_allTypes.items():
    # Get the body dynamics for each timestep.
    speeds_m_s = [None]*len(feature_matrices)
    jerks_m_s_s_s = [None]*len(feature_matrices)
    for trial_index, feature_matrix in enumerate(feature_matrices):
      speeds_m_s[trial_index] = get_body_speed_m_s(feature_matrix)
      jerks_m_s_s_s[trial_index] = get_body_jerk_m_s_s_s(feature_matrix)
      body_keys = list(speeds_m_s[trial_index].keys())
  
    # If desired, only extract a certain window of time relative to the inferred pouring time.
    for trial_index, feature_matrix in enumerate(feature_matrices):
      pouring_inference = infer_pour_pose(feature_matrix)
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
                      range=(np.quantile(speeds_m_s, histogram_range_quantiles[0]), np.quantile(speeds_m_s, histogram_range_quantiles[1])),
                      alpha=0.5, label=example_type.title())
        ax_jerk.hist(jerks_m_s_s_s, bins=num_histogram_bins,
                     histtype='stepfilled', # 'bar', 'barstacked', 'step', 'stepfilled'
                     log=False, density=True,
                     range=(np.quantile(jerks_m_s_s_s, histogram_range_quantiles[0]), np.quantile(jerks_m_s_s_s, histogram_range_quantiles[1])),
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
def plot_compare_distributions_joint_angles(feature_matrices_allTypes,
                        subtitle=None,
                        output_filepath=None,
                        region=None, # 'pre_pouring', 'pouring', 'post_pouring', None for all
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
                                 )
      if not hide_figure_window:
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt_wait_for_keyboard_press(0.5)
      plt.ion()
    else:
      (fig, axs) = fig
  
  # Initialize results/outputs.
  example_types = list(feature_matrices_allTypes.keys())
  distributions_joint_angles_rad = dict([(key, {}) for key in example_types])
  results_joint_angles_rad = dict([(example_type, dict([(key, {}) for key in example_types])) for example_type in example_types])
  joint_axes = ['x', 'y', 'z']
  
  # Process each example type (each distribution category).
  for (example_type, feature_matrices) in feature_matrices_allTypes.items():
    # Get the body joint angles for each timestep.
    joint_angles_rad = [None]*len(feature_matrices)
    for trial_index, feature_matrix in enumerate(feature_matrices):
      joint_angles_rad[trial_index] = get_body_joint_angles_rad(feature_matrix)
      body_keys = list(joint_angles_rad[trial_index].keys())
  
    # If desired, only extract a certain window of time relative to the inferred pouring time.
    for trial_index, feature_matrix in enumerate(feature_matrices):
      pouring_inference = infer_pour_pose(feature_matrix)
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
def plot_compare_distributions_spout_relativeHeights(feature_matrices_allTypes, referenceObject_positions_m_allTypes,
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
                                 )
      if not hide_figure_window:
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt_wait_for_keyboard_press(0.5)
      plt.ion()
    else:
      (fig, axs) = fig
  
  # Initialize results/outputs.
  example_types = list(feature_matrices_allTypes.keys())
  distributions_relativeHeights_cm = dict.fromkeys(example_types, None)
  results_relativeHeights_cm = dict([(example_type, dict.fromkeys(example_types, None)) for example_type in example_types])
  
  # Process each example type (each distribution category).
  for (example_type, feature_matrices) in feature_matrices_allTypes.items():
    referenceObject_positions_m = referenceObject_positions_m_allTypes[example_type]
    # Get the spout heights for each timestep of each trial.
    spout_relativeHeights_cm = []
    for trial_index in range(len(feature_matrices)):
      feature_matrix = feature_matrices[trial_index]
      referenceObject_position_m = referenceObject_positions_m[trial_index]
      # Get the spout height at each timestep.
      spout_relativeHeight_cm = np.zeros(shape=(feature_matrix.shape[0],))
      for time_index in range(feature_matrix.shape[0]):
        spout_position_cm = 100*infer_spout_position_m(feature_matrix, time_index)
        spout_relativeHeight_cm[time_index] = spout_position_cm[2] - 100*referenceObject_position_m[2]
      spout_relativeHeights_cm.append(spout_relativeHeight_cm)
      
    # If desired, only extract a certain window of time relative to the inferred pouring time.
    for trial_index, feature_matrix in enumerate(feature_matrices):
      pouring_inference = infer_pour_pose(feature_matrix)
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
def plot_compare_distributions_spout_tilts(feature_matrices_allTypes,
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
                                 )
      if not hide_figure_window:
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt_wait_for_keyboard_press(0.5)
      plt.ion()
    else:
      (fig, axs) = fig
  
  # Initialize results/outputs.
  example_types = list(feature_matrices_allTypes.keys())
  distributions_tilts_rad = dict.fromkeys(example_types, None)
  results_tilts_rad = dict([(example_type, dict.fromkeys(example_types, None)) for example_type in example_types])
  
  # Process each example type (each distribution category).
  for (example_type, feature_matrices) in feature_matrices_allTypes.items():
    # Get the spout tilt for each timestep of each trial.
    spout_tilts_deg = []
    for trial_index in range(len(feature_matrices)):
      feature_matrix = feature_matrices[trial_index]
      spout_tilt_rad = np.zeros(shape=(feature_matrix.shape[0],))
      for time_index in range(feature_matrix.shape[0]):
        spout_tilt_rad[time_index] = infer_spout_tilting(feature_matrix, time_index)
      spout_tilts_deg.append(spout_tilt_rad)
      
    # If desired, only extract a certain window of time relative to the inferred pouring time.
    for trial_index, feature_matrix in enumerate(feature_matrices):
      pouring_inference = infer_pour_pose(feature_matrix)
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

# ================================================================
# Plot and compare distributions of the spout position and orientation projected onto the table
#  during the inferred pouring window.
#  The projection is such that:
#   The y-axis is the pouring direction, as inferred from the yaw of the pitcher in each trial.
#   A point will be plotted at the spout's projection onto the table, relative to the glass.
#   So the water would pour upward on the plot from the plotted spout position.
# feature_matrices_byType and referenceObject_positions_m_byType are
#   dictionaries mapping distribution category to matrices for each trial.
def plot_compare_distributions_spout_projections(feature_matrices_byType, referenceObject_positions_m_byType,
                                                  output_filepath=None,
                                                  print_comparison_results=True,
                                                  plot_distributions=True,
                                                  fig=None, hide_figure_window=False):
  
  # Plot mean and standard deviation shading for each example type, on the same plot.
  spout_relativeOffsets_cm_byType = {}
  example_types = list(feature_matrices_byType.keys())
  previous_plot_handles = (fig, None)
  for (example_index, example_type) in enumerate(example_types):
    is_last_type = example_type == example_types[-1]
    previous_plot_handles = plot_pour_relativePosition(
      feature_matrices_byType[example_type], referenceObject_positions_m_byType[example_type],
      plot_mean=True, plot_std_shading=True, plot_all_trials=False,
      subtitle=None, label=example_type.title(),
      color=plt.rcParams['axes.prop_cycle'].by_key()['color'][example_index],
      output_filepath=output_filepath,
      fig=previous_plot_handles, hide_figure_window=hide_figure_window or (not plot_distributions))
    spout_relativeOffsets_cm_byType[example_type] = previous_plot_handles[1]
  
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
  
  return previous_plot_handles






##################################################################
# Infer metrics or other quantities about trajectories.
##################################################################

# ================================================================
# Get the body position and orientation during an inferred pouring window.
# Will infer the pouring window by finding a region that is the most stationary.
def infer_pour_pose(feature_matrix):
  # Parse the feature matrix.
  parsed_data = parse_feature_matrix(feature_matrix)
  
  # Initialize state.
  body_position_m = parsed_data['position_m']
  body_quaternion_wijk = parsed_data['quaternion_wijk']
  num_timesteps = feature_matrix.shape[0]
  min_average_distance_m = None
  min_average_distance_buffer_start_index = None
  min_average_distance_buffer_end_index = None
  body_position_stationary_buffer_m = None
  body_quaternion_wijk_stationary_buffer = None
  # Find the most stationary buffer.
  for buffer_start_index in range(num_timesteps):
    buffer_end_index = buffer_start_index + stationary_position_buffer_length
    if buffer_end_index >= num_timesteps:
      break
    if buffer_start_index < stationary_position_minIndex:
      continue
    if buffer_end_index > stationary_position_maxIndex:
      continue
    body_position_buffers_m = dict([(name, position_m[buffer_start_index:buffer_end_index, :]) for (name, position_m) in body_position_m.items()])
    body_quaternion_wijk_buffers = dict([(name, quaternion_wijk[buffer_start_index:buffer_end_index, :]) for (name, quaternion_wijk) in body_quaternion_wijk.items()])
    median_hand_position_m = np.median(body_position_buffers_m['hand'], axis=0)
    distances_m = np.linalg.norm(body_position_buffers_m['hand'] - median_hand_position_m, axis=1)
    average_distance_m = np.mean(distances_m, axis=0)
    if min_average_distance_m is None or average_distance_m < min_average_distance_m:
      min_average_distance_m = average_distance_m
      min_average_distance_buffer_start_index = buffer_start_index
      min_average_distance_buffer_end_index = buffer_end_index
      body_position_stationary_buffer_m = body_position_buffers_m
      body_quaternion_wijk_stationary_buffer = body_quaternion_wijk_buffers
  
  # Return the position and orientation during the most stationary window, and the window bounds.
  return {
    'position_m':
      dict([(name, np.median(position_m, axis=0)) for (name, position_m) in body_position_stationary_buffer_m.items()]),
    'quaternion_wijk':
      dict([(name, quaternion_wijk[int(quaternion_wijk.shape[0]/2),:]) for (name, quaternion_wijk) in body_quaternion_wijk_stationary_buffer.items()]),
    'time_index':
      int(np.mean([min_average_distance_buffer_start_index, min_average_distance_buffer_end_index])),
    'start_time_index': min_average_distance_buffer_start_index,
    'end_time_index': min_average_distance_buffer_end_index,
    }

# ================================================================
# Get the tilt angle of the spout at a specific time index or during the entire trial.
def infer_spout_tilting(feature_matrix, time_index=None):
  # Get tilt for all time if desired
  if time_index is None:
    spout_tilts = []
    for time_index in range(feature_matrix.shape[0]):
      spout_tilts.append(infer_spout_tilting(feature_matrix, time_index=time_index))
    return np.array(spout_tilts)
  
  # Parse the feature matrix.
  parsed_data = parse_feature_matrix(feature_matrix)
  # Rotate a box for the pitcher according to the hand quaternion.
  hand_center_cm = 100*parsed_data['position_m']['hand'][time_index, :]
  hand_quaternion_localToGlobal_wijk = parsed_data['quaternion_wijk']['hand'][time_index, :]
  hand_rotation = Rotation.from_quat(hand_quaternion_localToGlobal_wijk[[1,2,3,0]])
  pitcher_rotation = hand_rotation * hand_to_pitcher_rotation
  pitcher_quaternion_localToGlobal_ijkw = pitcher_rotation.as_quat()
  (corners, faces) = rotate_3d_box(pitcher_quaternion_localToGlobal_ijkw[[3,0,1,2]], hand_to_pitcher_offset_cm, pitcher_box_dimensions_cm)
  # Get a line segment along the long axis of the top of the pitcher, and move it to the origin.
  pitcher_topFace_line = corners[4,:] - corners[6,:]
  # Compute the angle between the pitcher top and the xy plane.
  angle_toZ_rad = np.arccos(np.dot(pitcher_topFace_line, [0, 0, 1]) / (np.linalg.norm(pitcher_topFace_line)*1))
  angle_toXY_rad = (np.pi/2) - angle_toZ_rad
  return angle_toXY_rad

# ================================================================
# Get the 3D spout position at a specific time index or during the entire trial.
def infer_spout_position_m(feature_matrix, time_index=None):
  # Get position for all time if desired
  if time_index is None:
    spout_position_m = []
    for time_index in range(feature_matrix.shape[0]):
      spout_position_m.append(infer_spout_position_m(feature_matrix, time_index=time_index))
    return np.array(spout_position_m)
  
  # Parse the feature matrix.
  parsed_data = parse_feature_matrix(feature_matrix)
  # Rotate a box for the pitcher according to the hand quaternion.
  hand_center_cm = 100*parsed_data['position_m']['hand'][time_index, :]
  hand_quaternion_localToGlobal_wijk = parsed_data['quaternion_wijk']['hand'][time_index, :]
  hand_rotation = Rotation.from_quat(hand_quaternion_localToGlobal_wijk[[1,2,3,0]])
  pitcher_rotation = hand_rotation * hand_to_pitcher_rotation
  pitcher_quaternion_localToGlobal_ijkw = pitcher_rotation.as_quat()
  (corners, faces) = rotate_3d_box(pitcher_quaternion_localToGlobal_ijkw[[3,0,1,2]],
                                   hand_to_pitcher_offset_cm, pitcher_box_dimensions_cm)
  corners = corners + hand_center_cm
  corners = corners/100
  faces = faces/100
  
  # Average two points at the front of the pitcher to get the spout position.
  return np.mean(corners[[4,5],:], axis=0)

# ================================================================
# Get the spout speed at a specific time index or during the entire trial.
def infer_spout_speed_m_s(feature_matrix, time_index=None):
  # Get the spout position.
  spout_position_m = infer_spout_position_m(feature_matrix, time_index=None)
  times_s = feature_matrix[:,-1]
  # Infer the speed.
  dxdydz = np.diff(spout_position_m, axis=0)
  dt = np.reshape(np.diff(times_s, axis=0), (-1, 1))
  spout_speed_m_s = np.hstack([np.squeeze([0]), np.linalg.norm(dxdydz, axis=1)/np.squeeze(dt)])
  if time_index is None:
    return spout_speed_m_s
  else:
    return spout_speed_m_s[time_index]

# ================================================================
# Get the spout acceleration at a specific time index or during the entire trial.
def infer_spout_acceleration_m_s_s(feature_matrix, time_index=None):
  # Get the spout speed.
  spout_speed_m_s = infer_spout_speed_m_s(feature_matrix, time_index=None)
  times_s = feature_matrix[:,-1]
  # Infer the acceleration.
  dv = np.diff(spout_speed_m_s, axis=0)
  dt = np.reshape(np.diff(times_s, axis=0), (-1, 1))
  spout_acceleration_m_s_s = np.hstack([np.squeeze([0]), dv/np.squeeze(dt)])
  if time_index is None:
    return spout_acceleration_m_s_s
  else:
    return spout_acceleration_m_s_s[time_index]

# ================================================================
# Get the spout jerk at a specific time index or during the entire trial.
def infer_spout_jerk_m_s_s_s(feature_matrix, time_index=None):
  # Get the spout speed.
  spout_acceleration_m_s_s = infer_spout_acceleration_m_s_s(feature_matrix, time_index=None)
  times_s = feature_matrix[:,-1]
  # Infer the jerk.
  da = np.diff(spout_acceleration_m_s_s, axis=0)
  dt = np.reshape(np.diff(times_s, axis=0), (-1, 1))
  spout_jerk_m_s_s_s = np.hstack([np.squeeze([0]), da/np.squeeze(dt)])
  if time_index is None:
    return spout_jerk_m_s_s_s
  else:
    return spout_jerk_m_s_s_s[time_index]

# ================================================================
# Get the spout yaw vector at a specific time index or during the entire trial.
def infer_spout_yawvector(feature_matrix, time_index=None):
  # Get vector for all time indexes if desired.
  if time_index is None:
    spout_yawvectors = []
    for time_index in range(feature_matrix.shape[0]):
      spout_yawvectors.append(infer_spout_yawvector(feature_matrix, time_index=time_index))
    return np.array(spout_yawvectors)
  
  # Parse the feature matrix.
  parsed_data = parse_feature_matrix(feature_matrix)
  # Rotate a box for the pitcher according to the hand quaternion.
  hand_center_cm = 100*parsed_data['position_m']['hand'][time_index, :]
  hand_quaternion_localToGlobal_wijk = parsed_data['quaternion_wijk']['hand'][time_index, :]
  hand_rotation = Rotation.from_quat(hand_quaternion_localToGlobal_wijk[[1,2,3,0]])
  pitcher_rotation = hand_rotation * hand_to_pitcher_rotation
  pitcher_quaternion_localToGlobal_ijkw = pitcher_rotation.as_quat()
  (corners, faces) = rotate_3d_box(pitcher_quaternion_localToGlobal_ijkw[[3,0,1,2]],
                                   hand_to_pitcher_offset_cm, pitcher_box_dimensions_cm)
  corners = corners + hand_center_cm
  corners = corners/100
  faces = faces/100
  
  # Get a line segment along the long axis of the top of the pitcher.
  handside_point = corners[6,:]
  spoutside_point = corners[4,:]
  # Project it, move it to the origin and normalize.
  handside_point[2] = 0
  spoutside_point[2] = 0
  yawvector = spoutside_point - handside_point
  return yawvector/np.linalg.norm(yawvector)



  






# ##################################################################
# # Testing
# ##################################################################
#
# if __name__ == '__main__':
#
#   import os
#   import h5py
#
#   duration_s = 10
#
#   data_dir = 'C:/Users/jdelp/Desktop/ActionSense/results/learning_trajectories/S00'
#   training_data_filepath = os.path.join(data_dir, 'pouring_training_data.hdf5')
#   referenceObjects_filepath = os.path.join(data_dir, 'pouring_training_referenceObject_positions.hdf5')
#
#   output_dir = 'C:/Users/jdelp/Desktop/ActionSense/results/learning_trajectories/testing_plots'
#
#   training_data_file = h5py.File(training_data_filepath, 'r')
#   feature_matrices = np.array(training_data_file['feature_matrices'])
#   labels = [x.decode('utf-8') for x in np.array(training_data_file['labels'])]
#   training_data_file.close()
#   referenceObjects_file = h5py.File(referenceObjects_filepath, 'r')
#   referenceObject_positions_m = np.array(referenceObjects_file['position_m'])
#   referenceObjects_file.close()
#
#   fig = None
#   for (trial_index, label) in enumerate(labels):
#     # # if not (trial_index > 94 and trial_index < 100):
#     # #   continue
#     # if trial_index != 94:
#     #   continue
#     if label == 'human':
#       print('Main loop plotting trial_index %d' % trial_index)
#       feature_matrix = np.squeeze(feature_matrices[trial_index])
#       referenceObject_position_m = np.squeeze(referenceObject_positions_m[trial_index])
#       plot_trajectory(feature_matrix, duration_s, referenceObject_position_m,
#                       pause_after_timesteps=True)
#       # save_trajectory_animation(feature_matrix, duration_s, referenceObject_position_m,
#       #                           output_filepath=os.path.join(output_dir, 'trajectory_animation_trial%02d.mp4' % trial_index))
#       # fig = plot_pour_tilting(feature_matrix, fig=fig, shade_pouring_region=False,
#       #                         output_filepath=os.path.join(output_dir, 'tilts_all.jpg') if trial_index >= len(labels)-2 else None)
#       # plot_pour_tilting(feature_matrix,
#       #                   output_filepath=os.path.join(output_dir, 'tilt_individualTrials', 'tilt_trial%02d.jpg' % trial_index),
#       #                   shade_pouring_region=True, hide_figure_window=True)
#       # fig = plot_pour_relativePosition(feature_matrix, referenceObject_position_m,
#       #                                  fig=fig, hide_figure_window=False,
#       #                                  output_filepath=os.path.join(output_dir, 'glassOffsets_all.jpg') if trial_index >= len(labels)-2 else None)
#       # infer_spout_speed_m_s(feature_matrix, time_index=None)
#       # infer_spout_acceleration_m_s_s(feature_matrix, time_index=None)
#       # infer_spout_jerk_m_s_s_s(feature_matrix, time_index=None)
#       # fig = plot_spout_dynamics(feature_matrix,
#       #                            output_filepath=os.path.join(output_dir, 'spout_dynamics_individualTrials', 'tilt_trial%02d.jpg' % trial_index),
#       #                            shade_pouring_region=False, fig=fig, hide_figure_window=False)
#       # fig = plot_body_dynamics(feature_matrix,
#       #                            output_filepath=os.path.join(output_dir, 'body_jerk_individualTrials', 'body_jerk_trial%02d.jpg' % trial_index),
#       #                            shade_pouring_region=False, fig=fig, hide_figure_window=False)
#   plt.show(block=True)
