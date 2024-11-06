"""
Offline baxter planning from gripper trajectory to joint angles
Ingests a HDF5 file of multiple hand pose trajectories
	for example, generated from process_scooping_trajectories.py
Outputs similarly formatted HDF5 file with multiple baxter joint angle trajectories
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py

from BaxterPlanner import BaxterPlanner, wrap_joint_angle, unwrap_joint_angle


# - Constants - #

URDF_XML_FILE = "baxter_urdf.xml"
LIMB_NAME = "right"
ANGLE_NAMES = ["s0", "s1", "e0", "e1", "w0", "w1", "w2"]
NOMINAL_JOINT_ANGLE = [0.80, -0.25, 0.00, 1.50, 0.00, -1.30, -1.5]


# - Main - #

def plan(input_trajectory_file,
         output_trajectory_file,
         frequency,
         start_joint_angle=None,
         generate_plots=True,
         ):
    # Load input/output trajectory files
    with h5py.File(input_trajectory_file, 'r') as f_in:
        dataset_name = f_in['name']

        with h5py.File(output_trajectory_file, 'w') as f_out:
            f_out.attrs['name'] = dataset_name
            
            # Planner init
            urdf_file = os.path.join(os.path.dirname(__file__), URDF_XML_FILE)
            joint_names = ['%s_%s' % (LIMB_NAME, angle_name) for angle_name in ANGLE_NAMES]
            baxter_planner = BaxterPlanner(urdf_file, 
                                          LIMB_NAME, 
                                          joint_names, 
                                          NOMINAL_JOINT_ANGLE,
                                          )
            
            # Process & plan over all trajectories
            for id, trajectory_key in enumerate(f_in.keys()):
      
                # Load relevant fields
                traj = f_in[trajectory_key]
                time = np.array(traj['data']['time'])
                pos_world_to_hand_W = np.array(traj['data']['pos_world_to_hand_W'])
                quat_world_to_hand_ijkw = np.array(traj['data']['quat_world_to_hand_ijkw'])

                # Plan
                pose = np.concatenate([pos_world_to_hand_W, quat_world_to_hand_ijkw], axis=1)
                time_resampled, joint_angle, gripper_pose = baxter_planner.plan(time, 
                                                                                pose, 
                                                                                frequency, 
                                                                                start_joint_angle)
                angles = np.array([unwrap_joint_angle(q) for q in joint_angle])
                
                # Store output
                traj_group = f_out.creat_group(trajectory_key)
                traj_group['description'] = 'Hand trajectory in joint angles and reference positions'
                data_group = traj_group.create_group('data')
                data_group.attrs['description'] = 'Hand trajectory in joint angles'
                data_group.create_dataset('time', data=time_resampled)
                data_group.create_dataset('joint_angle', data=joint_angle)
                # Copy references from input trajectory
                ref_group = traj_group.create_group('reference')
                ref_group.create_dataset('pos_world_to_pan_W', data=traj['reference']['pos_world_to_pan_W'])
                ref_group.create_dataset('pos_world_to_plate_W', data=traj['reference']['pos_world_to_plate_W'])
                
                if generate_plots:
                    plot_dir = os.path.join(os.path.dirname(__file__), 'figures')
                    os.makedirs(plot_dir, exist_ok=True)
                    
					# Plot Input Trajectory
                    fig,ax = plt.subplots(2)
                    ax[0].plot(time, pose[:, :3])
                    ax[0].set_title('Position')
                    ax[1].plot(time, pose[:, 3:])
                    ax[1].set_title('Quaternion')
                    fig.suptitle('Input Pose Trajectory')
                    fig.savefig(plot_dir + f'/{dataset_name}/input_traj_{id}.png')
                    plt.close()
                    
					# Plot Output Trajectory
                    fig, ax = plt.subplots(7,1, figsize=(8,8))
                    for i in range(angles.shape[1]):
                        ax[i].plot(time_resampled, angles[:,i], color='blue')
                    fig.suptitle('Output Joint Angle Trajectory')
                    fig.savefig(plot_dir + f'/{dataset_name}/output_traj_{id}.png')
                    plt.close()
                


if __name__ == '__main__':
  # Script inputs
  input_trajectory_file = os.path.expanduser("~/data/scooping/scooping_processed_S00.h5")
  output_trajectory_file = os.path.expanduser("~/data/scooping/scooping_angles_S00.h5")
  output_frequency = 20.0
  start_joint_angle = None
  generate_plots = True

  # Run script
  plan(input_trajectory_file,
       output_trajectory_file,
       output_frequency,
       start_joint_angle=start_joint_angle,
       generate_plots=generate_plots,
       )