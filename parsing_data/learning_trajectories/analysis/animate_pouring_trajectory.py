"""
Generates an animation given a pre-processed HDF5 file for the pouring task (and a trajectory id within such file)
"""
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

from constants import *
import utils


def generate_pouring_animation(
    time, 
    pos_world_to_hand_W, 
    rot_world_to_hand, 
    pos_world_to_glass_rim_W,
):
    """
    Args:
        - time (np.ndarray) : (n,) array of timesteps (seconds)
        - pos_world_to_hand_W (np.ndarray) : (n,3) array of hand position in world frame
        - rot_world_to_hand (np.ndarray) : (n,3,3) rotation matrices of world to hand frame
        - pos_world_to_glass_W (np.ndarray) : (3) array of glass position in world frame
    """
    # Initialize figure for animation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=30, azim=180)
    
    # Create animation
    def update(frame):
        ax.clear()

        # Plot reference objects
        pos_world_to_glass_middle_W = pos_world_to_glass_rim_W - np.array([0, 0, GLASS_HEIGHT/2])
        utils.plot_cylinder(ax, GLASS_RADIUS, GLASS_HEIGHT, position=pos_world_to_glass_middle_W, color='pink')

        # Plot hand
        utils.plot_box(ax, *HAND_BOX, position=pos_world_to_hand_W[frame], rotation=rot_world_to_hand[frame], color='yellow')

        # Plot jug
        rot_world_to_jug = rot_world_to_hand[frame] @ ROT_HAND_TO_JUG
        pos_world_to_jug_W = pos_world_to_hand_W[frame] + rot_world_to_jug @ POS_HAND_TO_JUG_J
        utils.plot_box(ax, *JUG_BOX, position=pos_world_to_jug_W, rotation=rot_world_to_jug, lengthwise=True)

        # Timestamp & bounds
        ax.set_title(f't = {time[frame]:.3f}s')
        ax.set_xlim([-0.75,0.25])
        ax.set_ylim([-0.5,0.5])
        ax.set_zlim([-0.25,0.75])

    interval_s = (time[-1] - time[0]) / (len(time) + 1)
    ani = FuncAnimation(fig, update, frames=len(time), interval=1000*interval_s, blit=False)

    return ani


# - Main - #

def animate_pouring_trajectory(
    input_trajectory_file,
    trajectory_id,
    output_directory,
):
    # Read trajectory file
    with h5py.File(input_trajectory_file, 'r') as f:
        traj = f[f'trajectory_{trajectory_id:03d}']
        dataset_name = traj.attrs['name']

        # Hand trajectory
        data = traj['data']
        time = np.array(data['time'])
        pos_world_to_hand_W = np.array(data['pos_world_to_hand_W'])
        rot_world_to_hand = np.array(data['rot_world_to_hand'])

        # Reference objects
        ref = traj['reference']
        pos_world_to_glass_rim_W = np.array(ref['pos_world_to_glass_rim_W'])

    # Animate
    ani = generate_pouring_animation(
        time, 
        pos_world_to_hand_W, 
        rot_world_to_hand, 
        pos_world_to_glass_rim_W
    )
    ani.save(output_directory + f'{dataset_name}_{trajectory_id}.gif')


if __name__ == '__main__':
    # Script inputs
    input_trajectory_file = os.path.expanduser(f'~/data/pouring/pouring_processed_S00.hdf5')
    trajectory_id = 1
    output_directory = os.path.expanduser('~/data/pouring/')
    
    # Run script
    animate_pouring_trajectory(
        input_trajectory_file, 
        trajectory_id,
        output_directory,
    )