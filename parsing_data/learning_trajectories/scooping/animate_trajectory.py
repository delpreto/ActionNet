"""
Generates an animation given a pre-processed HDF5 file for the scooping task (and a trajectory id within such file)
"""
import os
import numpy as np
import h5py
import scipy.spatial.transform as tf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from constants import *


def plot_box(
    ax, 
    length, 
    width, 
    height, 
    position=np.array([0,0,0]), 
    rotation=np.eye(3), 
    lengthwise=False,
):
    """lengthwise=True means box extends from 0 to length, -width/2 to width/2, -height/2 to height/2"""
    # Form box
    l, w, h = length/2, width/2, height/2
    if lengthwise:
        points = np.array([
            [0, -w, -h], [0, -w, h],
            [0, w, -h], [0, w, h],
            [2*l, -w, -h], [2*l, -w, h],
            [2*l, w, -h], [2*l, w, h],
        ])
    else:
        points = np.array([
            [-l, -w, -h], [-l, -w, h],
            [-l, w, -h], [-l, w, h],
            [l, -w, -h], [l, -w, h],
            [l, w, -h], [l, w, h],
        ])

    # Transform box coordinates
    tf_points = (rotation @ points.T + position.reshape(3,1)).T
    
    # Create box faces
    faces = np.array([
        [tf_points[j] for j in [0, 1, 3, 2]],
        [tf_points[j] for j in [4, 5, 7, 6]],
        [tf_points[j] for j in [0, 1, 5, 4]],
        [tf_points[j] for j in [2, 3, 7, 6]],
        [tf_points[j] for j in [1, 3, 7, 5]],
        [tf_points[j] for j in [4, 6, 2, 0]],
    ])
    
    # Plot surface
    for face in faces:
        verts = [face]
        face = Poly3DCollection(verts, color="cyan", alpha=0.5, edgecolor="k")
        ax.add_collection3d(face)


def plot_cylinder(
    ax, 
    radius, 
    height, 
    position=np.array([0,0,0]), 
    rotation=np.eye(3),
):
    """assumes ax is a matplotlib 3D projection subplot"""
    # Form cylinder with polar coordinates
    m = 20
    theta = np.linspace(0, 2*np.pi, m)
    z = np.array([-height/2, height/2])
    theta, z = np.meshgrid(theta, z)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    # Transform cylinder coordinates
    points = np.stack([x,y,z])
    tf_points = rotation @ points.reshape(3,-1) + position.reshape(3,1)
    x, y, z = tf_points.reshape(3, 2, m)

    # Plot surface
    cylinder = ax.plot_surface(x, y, z, alpha=0.8)

    return cylinder


def generate_scooping_animation(
    time, 
    pos_world_to_hand_W, 
    rot_world_to_hand, 
    pos_world_to_plate_W, 
    pos_world_to_pan_W,
):
    """
    Args:
        - time (np.ndarray) : (n,) array of timesteps (seconds)
        - pos_world_to_hand_W (np.ndarray) : (n,3) array of hand position in world frame
        - rot_world_to_hand (np.ndarray) : (n,3,3) rotation matrices of world to hand frame
        - pos_world_to_plate_W (np.ndarray) : (3) array of plate position in world frame
        - pos_world_to_pan_W (np.ndarray) : (3) array of pan position in world frame
    """
    # Initialize figure for animation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-0.75,0.25])
    ax.set_ylim([-0.5,0.5])
    ax.set_zlim([-0.25,0.75])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=40, azim=180)
    
    # Create animation
    def update(frame):
        ax.clear()

        # Plot hand
        plot_box(ax, *HAND_BOX, position=pos_world_to_hand_W[frame], rotation=rot_world_to_hand[frame])

        # Plot spoon
        scoop_rotation = rot_world_to_hand[frame] @ ROT_HAND_TO_SPOON
        plot_box(ax, *SPOON_BOX, position=pos_world_to_hand_W[frame], rotation=scoop_rotation, lengthwise=True)

        # Plot reference objects
        plot_cylinder(ax, PAN_RADIUS, PAN_HEIGHT, position=pos_world_to_pan_W)
        plot_cylinder(ax, PLATE_RADIUS, PLATE_HEIGHT, position=pos_world_to_plate_W)

        # Timestamp
        ax.set_title(f't = {time[frame]:.3f}s')

    interval_s = (time[-1] - time[0]) / (len(time) + 1)
    ani = FuncAnimation(fig, update, frames=len(time), interval=1000*interval_s, blit=False)

    return ani


# - Main - #

def animate_trajectory(
    input_trajectory_file, 
    trajectory_id,
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
        pos_world_to_plate_W = np.array(ref['pos_world_to_plate_W'])
        pos_world_to_pan_W = np.array(ref['pos_world_to_pan_W'])

    # Animate
    ani = generate_scooping_animation(time, pos_world_to_hand_W, rot_world_to_hand, pos_world_to_plate_W, pos_world_to_pan_W)
    ani.save(os.path.expanduser(f'~/data/scooping/{dataset_name}_{trajectory_id}.gif'))


if __name__ == '__main__':
    # Script inputs
    input_trajectory_file = os.path.expanduser(f'~/data/scooping/scooping_processed_S11.hdf5')
    trajectory_id = 1
    
    # Run script
    animate_trajectory(
        input_trajectory_file, 
        trajectory_id,
    )