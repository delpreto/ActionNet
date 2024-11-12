"""
Offline baxter planning from gripper trajectory to joint angles
Ingests a HDF5 file of multiple hand pose trajectories
	for example, generated from process_scooping_trajectories.py
Outputs similarly formatted HDF5 file with multiple baxter joint angle trajectories
Generates figures for input / output trajectories

TODO: There are still some bugs in this code
- control output is often very choppy and ugly, unsure why
- turning kp, kd gains on (including auxiliary objective) makes output worse
- this is true for both linoss model output and human data
- more work needs to be done to characterize where exactly the discontinuities arise from
    - it's possible the method of differentiation is causing small problematic discontinuities
    - my gut is telling me the jacobian matrices provided by the kinematics library are 
        weird for some reason. near-singular? numerical errors?
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.spatial.transform as tf

from BaxterPlanner import BaxterPlanner, wrap_joint_angle, unwrap_joint_angle


# - Constants - #

URDF_XML_FILE = "baxter_urdf.xml"
LIMB_NAME = "right"
ANGLE_NAMES = ["s0", "s1", "e0", "e1", "w0", "w1", "w2"]
NOMINAL_JOINT_ANGLE = [0.80, -0.25, 0.00, 1.50, 0.00, -1.30, -1.5]


# - Helpers - #

def rot_matrix_to_quat_ijkw(R):
    rot = tf.Rotation.from_matrix(R)
    quat_ijkw = rot.as_quat()

    return quat_ijkw


# - Main - #

def plan_trajectories(
    input_trajectory_file,
    output_trajectory_file,
    frequency,
    start_joint_angle=None,
    generate_plots=True,
    output_figure_directory='',
):
    # Load input/output trajectory files
    with h5py.File(input_trajectory_file, 'r') as f_in:
        with h5py.File(output_trajectory_file, 'w') as f_out:
            
            # Planner init
            urdf_file = os.path.join(os.path.dirname(__file__), URDF_XML_FILE)
            joint_names = ['%s_%s' % (LIMB_NAME, angle_name) for angle_name in ANGLE_NAMES]
            nominal_joint_angle = wrap_joint_angle(joint_names, NOMINAL_JOINT_ANGLE)
            baxter_planner = BaxterPlanner(urdf_file, LIMB_NAME, joint_names, nominal_joint_angle)
            
            # Process & plan over all trajectories
            for id, trajectory_key in enumerate(f_in.keys()):
                print(f'Planning over {trajectory_key}')
      
                # Load relevant fields
                traj = f_in[trajectory_key]
                dataset_name = traj.attrs['name']
                data = traj['data']
                time = np.array(data['time'])
                pos_world_to_hand_W = np.array(data['pos_world_to_hand_W'])
                if 'quat_world_to_hand_ijkw' in data.keys():
                    quat_world_to_hand_ijkw = np.array(data['quat_world_to_hand_ijkw'])
                elif 'rot_world_to_hand' in data.keys():
                    R = np.array(data['rot_world_to_hand'])
                    quat_world_to_hand_ijkw = np.array([rot_matrix_to_quat_ijkw(r) for r in R])
                else:
                    raise KeyError('Neither quaternion nor rotation matrix available in input trajectory')

                # Plan
                pose = np.concatenate([pos_world_to_hand_W, quat_world_to_hand_ijkw], axis=1)
                time_resampled, joint_angle, gripper_pose = baxter_planner.plan(time, pose, frequency, start_joint_angle)

                # Solving error: trash trajectory
                if joint_angle is None:
                    print('Skipping trajectory...')
                    continue

                time_resampled = np.array(time_resampled)
                angles = np.array([unwrap_joint_angle(q) for q in joint_angle])
                gripper_pose = np.array(gripper_pose)
                
                # Store output
                traj_group = f_out.create_group(trajectory_key)
                traj_group.attrs['name'] = dataset_name
                traj_group['description'] = 'Hand trajectory in joint angles and reference positions'
                data_group = traj_group.create_group('data')
                data_group.attrs['description'] = 'Hand trajectory in joint angles'
                data_group.create_dataset('time', data=time_resampled)
                data_group.create_dataset('joint_angle', data=angles)
                # Copy references from input trajectory
                ref_group = traj_group.create_group('reference')
                ref_group.create_dataset('pos_world_to_pan_W', data=traj['reference']['pos_world_to_pan_W'])
                ref_group.create_dataset('pos_world_to_plate_W', data=traj['reference']['pos_world_to_plate_W'])
                
                if generate_plots and output_figure_directory != '':
                    print('Generating figures..')

                    plot_dir = output_figure_directory + dataset_name
                    os.makedirs(plot_dir, exist_ok=True)
                    
					# Plot Input Trajectory
                    fig,ax = plt.subplots(2)
                    ax[0].plot(time, pose[:, :3])
                    ax[0].set_title('Position')
                    ax[1].plot(time, pose[:, 3:])
                    ax[1].set_title('Quaternion')
                    fig.suptitle('Input Pose Trajectory')
                    fig.savefig(plot_dir + f'/input_pose_traj_{id}.png')
                    plt.close()
                    
					# Plot Output Trajectory -- Angle
                    fig, ax = plt.subplots(7,1, figsize=(8,8))
                    for i in range(angles.shape[1]):
                        ax[i].plot(time_resampled, angles[:,i], color='blue')
                    fig.suptitle('Output Joint Angle Trajectory')
                    fig.savefig(plot_dir + f'/output_angle_traj_{id}.png')
                    plt.close()

                    # Plot Output Trajectory -- Pose
                    fig,ax = plt.subplots(2)
                    ax[0].plot(time_resampled, gripper_pose[:, :3])
                    ax[0].set_title('Position')
                    ax[1].plot(time_resampled, gripper_pose[:, 3:])
                    ax[1].set_title('Quaternion')
                    fig.suptitle('Output Pose Trajectory')
                    fig.savefig(plot_dir + f'/output_pose_traj_{id}.png')
                    plt.close()
                


if __name__ == '__main__':
    # Script inputs
    input_trajectory_file = os.path.expanduser("~/data/scooping/scooping_processed_S00.hdf5")
    output_trajectory_file = os.path.expanduser("~/data/scooping/scooping_processed_S00_angles.hdf5")
    output_frequency = 100.0
    start_joint_angle = None
    generate_plots = True
    output_figure_directory = os.path.expanduser("~/data/scooping/figures/planning/")

    # Run script
    plan_trajectories(
        input_trajectory_file,
        output_trajectory_file,
        output_frequency,
        start_joint_angle=start_joint_angle,
        generate_plots=generate_plots,
        output_figure_directory=output_figure_directory,
    )