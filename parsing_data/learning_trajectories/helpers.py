
import numpy as np
from scipy.spatial.transform import Rotation

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import art3d

from utils.numpy_scipy_utils import *

import cv2

#############################################

stationary_position_buffer_length = 15
stationary_position_minIndex = 10
stationary_position_maxIndex = 90
hand_box_dimensions_cm = np.array([4.8, 3, 1.3])
hand_box_color = 0.8*np.array([1, 0.6, 0])
pitcher_box_dimensions_cm = np.array([23, 23, 10.8]) # [height, top length, width]
pitcher_box_color = 0.8*np.array([1, 1, 1])
hand_to_pitcher_rotation = Rotation.from_rotvec(np.pi/2 * np.array([1, 0, 0]))
hand_to_pitcher_offset_cm = np.array([0, -(7-5+pitcher_box_dimensions_cm[1]/2), 0])

referenceObject_diameter_cm = 7.3 # glass top 7.3 bottom 6.3


#############################################

# Feature_matrices should be Tx30, where
#   T is the number of timesteps in each trial
#   30 is the concatenation of:
#     xyz position for hand > elbow > shoulder
#     wijk quaternion for hand > lower arm > upper arm
#     xzy joint angle for wrist > elbow > shoulder
def parse_feature_matrix(feature_matrix):
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
    }
  }

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

def plot_3d_box(ax, quaternion_localToGlobal_wijk, center_cm, center_preRotation_cm, box_dimensions_cm, color): # function created using ChatGPT
  
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
                               alpha=0.8,
                               facecolor=color,
                               edgecolor=0.4*color)
  ax.add_collection3d(box)
  return box

def plot_hand_box(ax, hand_quaternion_localToGlobal_wijk, hand_center_cm):
  return plot_3d_box(ax, hand_quaternion_localToGlobal_wijk,
                     hand_center_cm, np.array([0, 0, 0]),
                     hand_box_dimensions_cm, hand_box_color)

def plot_pitcher_box(ax, hand_quaternion_localToGlobal_wijk, hand_center_cm):
  hand_rotation = Rotation.from_quat(hand_quaternion_localToGlobal_wijk[[1,2,3,0]])
  pitcher_rotation = hand_rotation * hand_to_pitcher_rotation
  pitcher_quaternion_localToGlobal_ijkw = pitcher_rotation.as_quat()
  return plot_3d_box(ax, pitcher_quaternion_localToGlobal_ijkw[[3,0,1,2]],
                     hand_center_cm, hand_to_pitcher_offset_cm,
                     pitcher_box_dimensions_cm, pitcher_box_color)



def plt_wait_for_keyboard_press(timeout_s=-1.0):
  keyboardClick=False
  while keyboardClick == False:
    keyboardClick = plt.waitforbuttonpress(timeout=timeout_s)

def plot_timestep(feature_matrix, times_s, time_index,
                  referenceObject_position_m,
                  subject_id=-1, trial_index=-1,
                  previous_handles=None, include_skeleton=True, spf=None,
                  pause_after_plotting=False, hide_figure_window=False):
  # Parse the feature matrix.
  parsed_data = parse_feature_matrix(feature_matrix)
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
    fig = plt.figure()
    if not hide_figure_window:
      figManager = plt.get_current_fig_manager()
      figManager.window.showMaximized()
    plt.ion()
    fig.add_subplot(1, 1, 1, projection='3d')
  ax = fig.get_axes()[0]
  
  # Draw items that remain the same across frames.
  if previous_handles is None:
    ax.clear()
    
    # Add labels and titles.
    ax.set_xlabel('X [cm]')
    ax.set_ylabel('Y [cm]')
    ax.set_zlabel('Z [cm]')
    
    # Set the view angle
    ax.view_init(16, -28)
    # ax.view_init(16, 44)
    # ax.view_init(90, 0)
    
    # Plot trajectories of the right arm and pelvis.
    hand_position_cm = 100*parsed_data['position_m']['hand']
    forearm_position_cm = 100*parsed_data['position_m']['elbow']
    upperArm_position_cm = 100*parsed_data['position_m']['shoulder']
    h_hand_path = ax.plot3D(hand_position_cm[:, 0], hand_position_cm[:, 1], hand_position_cm[:, 2], alpha=1)
    if include_skeleton:
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
               100*referenceObject_position_m[2],
               s=25, color=h_hand_path[0].get_color(), edgecolor='c')
    referenceObject_circle = mpatches.Circle(
      (100*referenceObject_position_m[0], 100*referenceObject_position_m[1]),
      radius=referenceObject_diameter_cm/2, ec=[0.4,1,1], color=[0.8,1,1])
    ax.add_patch(referenceObject_circle)
    art3d.patch_2d_to_3d(referenceObject_circle, z=100*referenceObject_position_m[2], zdir='z')
  
  if include_skeleton:
    # Animate the whole skeleton.
    sampling_rate_hz = (times_s.shape[0] - 1) / (times_s[-1] - times_s[0])
    spf = spf or 1/sampling_rate_hz
    timestep_interval = max([1, int(sampling_rate_hz*spf)])
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
                                 100*referenceObject_position_m[2],
                                 s=30, color='m'))
    # Draw an indicator of the spout direction on the table.
    spout_yawvector = infer_spout_yawvector(feature_matrix, time_index)
    spout_yawvector = spout_yawvector * referenceObject_diameter_cm/2
    spout_yawsegment = np.array([[0,0,0], list(spout_yawvector)])
    spout_yawsegment = spout_yawsegment + position_cm
    h_chains.append(ax.plot3D(spout_yawsegment[:,0], spout_yawsegment[:,1],
                              100*referenceObject_position_m[2]*np.array([1,1]),
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
    ax.set_title('Subject %02d Trial %02d\nt=%0.2fs' % (subject_id, trial_index, times_s[time_index] - times_s[0]))
    
    # Show the plot.
    plt.draw()
    
    # Save the previous handles.
    previous_handles = (fig, h_chains, h_scatters, h_hand, h_pitcher)
  else:
    # Set the aspect ratio
    ax.set_box_aspect([ub - lb for lb, ub in (ax.get_xlim(), ax.get_ylim(), ax.get_zlim())])
    # Set the title.
    ax.set_title('Subject %02d' % (subject_id))
    # Show the plot.
    plt.draw()
    # Save the previous handles.
    previous_handles = (fig, None, None, None)
  
  # plt_wait_for_keyboard_press()
  # print('view elev/azim:', ax.elev, ax.azim)
  
  if pause_after_plotting and not hide_figure_window:
    plt_wait_for_keyboard_press(timeout_s=-1)#spf)
    
  return previous_handles

def plot_trajectory(feature_matrix, duration_s, referenceObject_position_m, pause_after_timesteps):
  times_s = np.linspace(start=0, stop=duration_s, num=feature_matrix.shape[0])
  previous_handles = None
  for (time_index, time_s) in enumerate(times_s):
    previous_handles = plot_timestep(feature_matrix,
                                     time_index=time_index, times_s=times_s,
                                     referenceObject_position_m=referenceObject_position_m,
                                     previous_handles=previous_handles,
                                     pause_after_plotting=pause_after_timesteps)

def save_trajectory_animation(feature_matrix, duration_s, referenceObject_position_m,
                              output_filepath):
  times_s = np.linspace(start=0, stop=duration_s, num=feature_matrix.shape[0])
  fps = int(round((len(times_s)-1)/(times_s[-1] - times_s[0])))
  video_writer = None
  previous_handles = None
  for (time_index, time_s) in enumerate(times_s):
    previous_handles = plot_timestep(feature_matrix,
                                     time_index=time_index, times_s=times_s,
                                     referenceObject_position_m=referenceObject_position_m,
                                     previous_handles=previous_handles,
                                     pause_after_plotting=False,
                                     hide_figure_window=True)
    fig = previous_handles[0]
    plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)
    if video_writer is None:
      video_writer = cv2.VideoWriter(output_filepath,
                        cv2.VideoWriter_fourcc(*'mp4v'), # for AVI: cv2.VideoWriter_fourcc(*'MJPG'),
                        fps,
                        (plot_img.shape[1], plot_img.shape[0])
                        )
    video_writer.write(plot_img)
  video_writer.release()
  
def plot_pour_tilting(feature_matrix, shade_pouring_region=False,
                      fig=None, hide_figure_window=False, output_filepath=None):
  if fig is None:
    if hide_figure_window:
      try:
        matplotlib.use('Agg')
      except:
        pass
    fig = plt.figure()
    if not hide_figure_window:
      figManager = plt.get_current_fig_manager()
      figManager.window.showMaximized()
    plt.ion()
    fig.add_subplot(1, 1, 1)
  ax = fig.get_axes()[0]
  
  # Get the tilt angle for each timestep.
  angle_toXY_rad = np.zeros(shape=(feature_matrix.shape[0], 1))
  for time_index in range(feature_matrix.shape[0]):
    angle_toXY_rad[time_index] = infer_spout_tilting(feature_matrix, time_index)
  
  # Get the pouring times.
  if shade_pouring_region:
    pouring_inference = infer_pour_pose(feature_matrix)
    pour_start_index = pouring_inference['start_time_index']
    pour_end_index = pouring_inference['end_time_index']
  
  # Plot.
  ax.plot(np.degrees(angle_toXY_rad))
  if shade_pouring_region:
    ax.axvspan(pour_start_index, pour_end_index, alpha=0.5, color='gray')
  
  # Plot formatting.
  ax.set_ylim([-90, 15])
  ax.set_xlabel('Time Index')
  ax.set_ylabel('Tilt Angle to XY Plane [deg]')
  ax.grid(True, color='lightgray')
  
  # Show the plot.
  plt.draw()
  
  # Save the plot if desired.
  if output_filepath is not None:
    fig.savefig(output_filepath, dpi=300)
  
  return fig

def plot_pour_relativePosition(feature_matrix, referenceObject_position_m,
                               fig=None, hide_figure_window=False,
                               output_filepath=None):
  if fig is None:
    if hide_figure_window:
      try:
        matplotlib.use('Agg')
      except:
        pass
    fig = plt.figure()
    if not hide_figure_window:
      figManager = plt.get_current_fig_manager()
      figManager.window.showMaximized()
    plt.ion()
    fig.add_subplot(1, 1, 1)
    ax = fig.get_axes()[0]
    # Plot the reference object and the origin.
    referenceObject_circle = mpatches.Circle(
        (0, 0),
        radius=referenceObject_diameter_cm/2, ec=[0.4,1,1], color=[0.8,1,1],
        alpha=0.5)
    ax.add_patch(referenceObject_circle)
    ax.scatter(0, 0, s=10, c='k')
  else:
    ax = fig.get_axes()[0]
  
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
  
  # Plot.
  ax.scatter(*spout_relativeOffset_cm, s=25)
  
  # Plot formatting.
  ax.set_aspect('equal')
  ax.set_xlabel('Horizontal Relative to Pouring Direction [cm]')
  ax.set_ylabel('Vertical Relative to Pouring Direction [cm]')
  ax.grid(True, color='lightgray')
  ax.set_axisbelow(True)
  
  # Show the plot.
  plt.draw()
  
  # Save the plot if desired.
  if output_filepath is not None:
    fig.savefig(output_filepath, dpi=300)
  
  return fig

def infer_pour_position(feature_matrix):
  pass

def infer_pour_pose(feature_matrix):
  # Parse the feature matrix.
  parsed_data = parse_feature_matrix(feature_matrix)
  
  body_position_m = parsed_data['position_m']
  body_quaternion_wijk = parsed_data['quaternion_wijk']
  num_timesteps = feature_matrix.shape[0]
  min_average_distance_m = None
  min_average_distance_buffer_start_index = None
  min_average_distance_buffer_end_index = None
  body_position_stationary_buffer_m = None
  body_quaternion_wijk_stationary_buffer = None
  for buffer_start_index in range(num_timesteps):
    buffer_end_index = buffer_start_index + stationary_position_buffer_length
    if buffer_end_index >= num_timesteps:
      break
    if buffer_start_index < stationary_position_minIndex:
      continue
    if buffer_end_index> stationary_position_maxIndex:
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

def infer_spout_tilting(feature_matrix, time_index):
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

def infer_spout_position_m(feature_matrix, time_index):
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

def infer_spout_yawvector(feature_matrix, time_index):
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







if __name__ == '__main__':
  
  import os
  import h5py
  
  duration_s = 10
  
  data_dir = 'C:/Users/jdelp/Desktop/ActionSense/results/learning_trajectories/S00'
  training_data_filepath = os.path.join(data_dir, 'pouring_training_data.hdf5')
  referenceObjects_filepath = os.path.join(data_dir, 'pouring_training_referenceObject_positions.hdf5')
  
  output_dir = 'C:/Users/jdelp/Desktop/ActionSense/results/learning_trajectories/testing_plots'
  
  training_data_file = h5py.File(training_data_filepath, 'r')
  feature_matrices = np.array(training_data_file['feature_matrices'])
  labels = [x.decode('utf-8') for x in np.array(training_data_file['labels'])]
  training_data_file.close()
  referenceObjects_file = h5py.File(referenceObjects_filepath, 'r')
  referenceObject_positions_m = np.array(referenceObjects_file['position_m'])
  referenceObjects_file.close()
  
  fig = None
  for (trial_index, label) in enumerate(labels):
    # # if not (trial_index > 94 and trial_index < 100):
    # #   continue
    # if trial_index != 94:
    #   continue
    if label == 'human':
      print('Main loop plotting trial_index %d' % trial_index)
      feature_matrix = np.squeeze(feature_matrices[trial_index])
      referenceObject_position_m = np.squeeze(referenceObject_positions_m[trial_index])
      # plot_trajectory(feature_matrix, duration_s, referenceObject_position_m,
      #                 pause_after_timesteps=True)
      # save_trajectory_animation(feature_matrix, duration_s, referenceObject_position_m,
      #                           output_filepath=os.path.join(output_dir, 'trajectory_animation_trial%02d.mp4' % trial_index))
      fig = plot_pour_tilting(feature_matrix, fig=fig, shade_pouring_region=False,
                              output_filepath=os.path.join(output_dir, 'tilts_all.jpg') if trial_index >= len(labels)-2 else None)
      # plot_pour_tilting(feature_matrix,
      #                   output_filepath=os.path.join(output_dir, 'tilt_individualTrials', 'tilt_trial%02d.jpg' % trial_index),
      #                   shade_pouring_region=True, hide_figure_window=True)
      # fig = plot_pour_relativePosition(feature_matrix, referenceObject_position_m,
      #                                  fig=fig, hide_figure_window=False,
      #                                  output_filepath=os.path.join(output_dir, 'glassOffsets_all.jpg') if trial_index >= len(labels)-2 else None)
  plt.show(block=True)
