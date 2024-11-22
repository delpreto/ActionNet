"""
Offline baxter planning from gripper trajectory to joint angles
Generates figures for input / output trajectories
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from BaxterPlanner import BaxterPlanner, wrap_joint_angle, unwrap_joint_angle


# - Constants - #

URDF_XML_FILE = "baxter_urdf.xml"
LIMB_NAME = "right"
ANGLE_NAMES = ["s0", "s1", "e0", "e1", "w0", "w1", "w2"]
NOMINAL_JOINT_ANGLE = [0.80, -0.25, 0.00, 1.50, 0.00, -1.30, 0]


# - Main - #

def plan_trajectories(
    input_trajectory_directory,
    output_trajectory_directory,
    frequency,
    start_joint_angle=None,
    generate_plots=True,
):
    # Load trajectory
    time = np.load(input_trajectory_directory + 'time.npy')
    pos_world_to_hand_W = np.load(input_trajectory_directory + 'pos_world_to_hand_W.npy')
    quat_world_to_hand_wijk = np.load(input_trajectory_directory + 'quat_world_to_hand_wijk.npy')
    quat_world_to_hand_ijkw = np.roll(quat_world_to_hand_wijk, shift=-1, axis=1)
            
    # Planner init
    urdf_file = os.path.join(os.path.dirname(__file__), URDF_XML_FILE)
    joint_names = ['%s_%s' % (LIMB_NAME, angle_name) for angle_name in ANGLE_NAMES]
    nominal_joint_angle = wrap_joint_angle(joint_names, NOMINAL_JOINT_ANGLE)
    baxter_planner = BaxterPlanner(urdf_file, LIMB_NAME, joint_names, nominal_joint_angle)

    # Plan
    pose = np.concatenate([pos_world_to_hand_W, quat_world_to_hand_ijkw], axis=1)
    time_resampled, joint_angle, gripper_pose = baxter_planner.plan(time, pose, frequency, start_joint_angle)

    # Solving error: trash trajectory
    if joint_angle is None:
        raise RuntimeError('Planning failure')

    time_resampled = np.array(time_resampled)
    angles = np.array([unwrap_joint_angle(q) for q in joint_angle])
    gripper_pose = np.array(gripper_pose)

    # Save outputs
    os.makedirs(output_trajectory_directory, exist_ok=True)
    np.save(output_trajectory_directory + 'time.npy', arr=time_resampled)
    np.save(output_trajectory_directory + 'angles.npy', arr=angles)
    
    if generate_plots:
        print('Generating figures..')
        
        # Plot Input Trajectory
        fig,ax = plt.subplots(2)
        ax[0].plot(time, pose[:, :3])
        ax[0].set_title('Position')
        ax[1].plot(time, pose[:, 3:], label=['i','j','k','w'])
        ax[1].set_title('Quaternion')
        ax[1].legend()
        fig.suptitle('Input Pose Trajectory')
        fig.savefig(output_trajectory_directory + f'/input_pose.png')
        plt.close()
        
        # Plot Output Trajectory -- Angle
        fig, ax = plt.subplots(7,1, figsize=(8,8))
        for i in range(angles.shape[1]):
            ax[i].plot(time_resampled, angles[:,i], color='blue')
        fig.suptitle('Output Joint Angle Trajectory')
        fig.savefig(output_trajectory_directory + f'/output_angle.png')
        plt.close()

        # Plot Output Trajectory -- Pose
        fig,ax = plt.subplots(2)
        ax[0].plot(time_resampled, gripper_pose[:, :3])
        ax[0].set_title('Position')
        ax[1].plot(time_resampled, gripper_pose[:, 3:], label=['i','j','k','w'])
        ax[1].set_title('Quaternion')
        ax[1].legend()
        fig.suptitle('Output Pose Trajectory')
        fig.savefig(output_trajectory_directory + f'/output_pose.png')
        plt.close()
                

if __name__ == '__main__':
    # Script inputs
    input_trajectory_directory = os.path.expanduser("~/data/scooping/controller/inference_LinOSS_train_scooping_5678/trajectory_001/")
    output_trajectory_directory = os.path.expanduser("~/data/scooping/controller/LinOSS_train_scooping_5678_angles/trajectory_001/")
    output_frequency = 100.0
    start_joint_angle = None
    generate_plots = True

    # Run script
    plan_trajectories(
        input_trajectory_directory,
        output_trajectory_directory,
        output_frequency,
        start_joint_angle=start_joint_angle,
        generate_plots=generate_plots,
    )