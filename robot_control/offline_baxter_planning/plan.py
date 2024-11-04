import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import h5py

from ActionNet.robot_control.offline_baxter_planning.BaxterPlanner import BaxterPlanner, wrap_joint_angle, unwrap_joint_angle
from ActionNet.robot_control.offline_baxter_planning.analyze_trajectory import flat_R_to_quat


def load_traj_hdf5(filename):
  with h5py.File(filename, "r") as f:
    data = f["trajectory"][:]
    columns = f["trajectory"].attrs["columns"]
    return data, columns


def add_angles_hdf5(filename, data, id):
  """For saving joint angle trajectories"""
  with h5py.File(filename, 'a') as f:
    f.create_dataset(f'trajectory_{id}', data=data, dtype='float64')
    dataset = f[f'trajectory_{id}']
    dataset.attrs['columns'] = ['time', 'q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6']


def moving_average(data, window_size):
  padded = np.apply_along_axis(lambda col: np.pad(col, pad_width=window_size//2, mode='reflect'), axis=0, arr=data)
  return np.apply_along_axis(lambda col: np.convolve(col, np.ones(window_size) / window_size, mode='valid'), axis=0, arr=padded)


if __name__ == '__main__':
  # Parse config
  parser = argparse.ArgumentParser()
  parser.add_argument('--config_file', type=str, help='relative path to config file (.json)', default='default.json')
  args = parser.parse_args()
  if '/' in args.config_file:
    raise argparse.ArgumentError("file should be a relative path from config/")
  parent_dir = os.path.dirname(__file__)
  config_file = os.path.join(parent_dir, 'config', args.config_file)

  with open(config_file, 'r') as file:
    config = json.load(file)

  urdf_file = os.path.join(parent_dir, config['urdf_xml_file'])
  limb_name = config['limb_name']
  joint_names = config['joint_names']
  joint_names = ['%s_%s' % (limb_name, jn) for jn in joint_names]
  nominal_joint_angle = wrap_joint_angle(joint_names, config['nominal_joint_angle'])
  start_joint_angle = wrap_joint_angle(config['start_joint_angle']) if config['start_joint_angle'] else None
  freq = config['output_frequency']
  input_traj_file = os.path.join(parent_dir, config['input_trajectory_file'])
  output_traj_file = os.path.join(parent_dir, config['output_trajectory_file'])
  generate_plots = config['generate_plots']

  # Load trajectory
  data, columns = load_traj_hdf5(input_traj_file)

  # Planner init
  baxter_planner = BaxterPlanner(urdf_file, limb_name, joint_names, nominal_joint_angle)

  # Bookkeeping
  if generate_plots:
    input_plot_dir = ('/').join(input_traj_file.split('/')[:-1] + ['figures'])
    output_plot_dir = ('/').join(output_traj_file.split('/')[:-1] + ['figures'])
    os.makedirs(input_plot_dir, exist_ok=True)
    os.makedirs(output_plot_dir, exist_ok=True)
  
  # Process trajectories
  for id, in_traj in enumerate(data):
    t = in_traj[:,0]

    # Convert quaternions / rotation matrices
    if len(columns) > 10 or 'r11' in columns:
      pos = in_traj[:,1:4]
      quat = flat_R_to_quat(in_traj[:,4:])
      pose = np.concatenate([pos, quat], axis=1)
    else:
      pose = in_traj[:,1:8]

    # Plan
    time, joint_angle, gripper_pose = baxter_planner.plan(t, pose, freq, start_joint_angle)
    angles = np.array([unwrap_joint_angle(q) for q in joint_angle])
    out_traj = np.concatenate([time.reshape((-1,1)), angles], axis=1)
    add_angles_hdf5(output_traj_file, out_traj, id)

    if generate_plots:
      # Plot Input Trajectory
      fig,ax = plt.subplots(2)
      ax[0].plot(t, pose[:, :3])
      ax[0].set_title('Position')
      ax[1].plot(t, pose[:, 3:])
      ax[1].set_title('Quaternion')
      fig.suptitle('Input Trajectory')
      fig.savefig(input_plot_dir + f'/traj_{id}.png')
      plt.close()

      # Plot Output Trajectory
      fig, ax = plt.subplots(7,1, figsize=(8,8))
      for i in range(angles.shape[1]):
        ax[i].plot(time, angles[:,i], color='blue')
      fig.suptitle('Output Joint Angle Trajectory')
      fig.savefig(output_plot_dir + f'/traj_{id}.png')
      plt.close()