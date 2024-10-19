import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from ActionNet.robot_control.offline_baxter_planning.BaxterPlanner import BaxterPlanner, wrap_joint_angle, unwrap_joint_angle

if __name__ == '__main__':
  # Parse config
  parser = argparse.ArgumentParser()
  parser.add_argument('config_file', type=str, help='relative path to config file (.json)')
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

  # Load trajectory
  input_traj_file = os.path.join(parent_dir, config['input_trajectory_file'])
  input_traj = np.load(input_traj_file)
  t = input_traj[:,0]
  pose = input_traj[:,1:8]

  # Plot Input Trajectory
  fig,ax = plt.subplots(2,1)
  ax[0].plot(t, pose[:, :3])
  ax[0].set_title('Position')
  ax[1].plot(t, pose[:, 3:])
  ax[1].set_title('Quaternion')
  fig.suptitle('Input Trajectory')
  plt.show()

  # Plan
  baxter_planner = BaxterPlanner(urdf_file, limb_name, joint_names, nominal_joint_angle)
  time, joint_angle, gripper_pose = baxter_planner.plan(t, pose, freq, start_joint_angle)

  # Plot Planned Trajectory
  fig, ax = plt.subplots(7,1, figsize=(8,8))
  angles = np.array([unwrap_joint_angle(q) for q in joint_angle])
  for i in range(angles.shape[1]):
    ax[i].plot(time, angles[:,i], color='blue')
  fig.suptitle('Output Joint Angle Trajectory')
  plt.show()

  # Output
  out = np.concatenate([time.reshape((-1,1)), angles], axis=1)
  np.save(os.path.join(parent_dir, config['output_trajectory_file']), out)