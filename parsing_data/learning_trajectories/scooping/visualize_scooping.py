"""
This script takes in a pre-processed HDF5 file for the scooping task and generates plots / animations
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

def plot_box(ax, length, width, height, position=np.array([0,0,0]), rotation=np.eye(3), lengthwise=False):
    """lengthwise=True means box extends from 0 to length, -width/2 to width/2, -height/2 to height/2"""
    # Form box
    l, w, h = length/2, width/2, height/2
    if lengthwise:
        points = np.array([
            [0, -w, -h],
            [0, -w, h],
            [0, w, -h],
            [0, w, h],
            [2*l, -w, -h],
            [2*l, -w, h],
            [2*l, w, -h],
            [2*l, w, h],
        ])
    else:
        points = np.array([
            [-l, -w, -h],
            [-l, -w, h],
            [-l, w, -h],
            [-l, w, h],
            [l, -w, -h],
            [l, -w, h],
            [l, w, -h],
            [l, w, h],
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


def plot_cylinder(ax, radius, height, position=np.array([0,0,0]), rotation=np.eye(3)):
    """assumes ax is a matplotlib 3D projection subplot"""
    # Form cylinder with polar coordinates
    n = 10
    m = 25
    z = np.linspace(0, height, n)
    theta = np.linspace(0, 2*np.pi, m)
    theta, z = np.meshgrid(theta, z)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    # Transform cylinder coordinates
    points = np.stack([x,y,z])
    tf_points = rotation @ points.reshape(3,-1) + position.reshape(3,1)
    x, y, z = tf_points.reshape(3, n, m)

    # Plot surface
    cylinder = ax.plot_surface(x, y, z, alpha=0.8)

    return cylinder


def generate_scooping_animation(scooping_file, traj_num):
    # Read trajectory file
    with h5py.File(scooping_file, 'r') as f:
        data = f[f'trajectory_{traj_num:03d}']['data']
        time = np.array(data['time']).flatten()
        pos_world_to_hand_W = np.array(data['position'])
        rot_hand_to_world = np.array(data['rotation'])
        ref = f[f'trajectory_{traj_num:03d}']['reference']
        pos_world_to_plate_W = np.array(ref['plate_position'])
        pos_world_to_pan_W = np.array(ref['pan_position'])

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
        plot_box(ax, *hand_box, position=pos_world_to_hand_W[frame], rotation=rot_hand_to_world[frame])

        # Plot spoon
        scoop_rotation = rot_hand_to_world[frame] @ rot_spoon_to_hand
        plot_box(ax, *spoon_box, position=pos_world_to_hand_W[frame], rotation=scoop_rotation, lengthwise=True)

        # Plot reference objects
        plot_cylinder(ax, pan_radius, pan_height, position=pos_world_to_pan_W)
        plot_cylinder(ax, plate_radius, plate_height, position=pos_world_to_plate_W)

        # Timestamp
        ax.set_title(f't = {time[frame]:.3f}s')

    interval_s = (time[-1] - time[0]) / (len(time) + 1)
    ani = FuncAnimation(fig, update, frames=len(time), interval=1000*interval_s, blit=False)

    return ani


if __name__ == '__main__':
    file = os.path.expanduser(f'~/data/scooping/scooping_processed_S11.hdf5')
    num = 1
    ani = generate_scooping_animation(file, num)
    ani.save(os.path.expanduser(f'~/data/scooping/scooping_S11_{num}.gif'))