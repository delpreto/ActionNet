import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.spatial.transform as tf


# - File I/O utilities - #

def save_pickle(obj, filename):
    """Saves a pickle object."""
    with open(filename, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


# - TF utilities - #

def quat_to_rot_matrix(quat, scalar_first=False):
    if scalar_first:
        quat_ijkw = np.array([
            quat[1],
            quat[2],
            quat[3],
            quat[0],
        ])
    else:
        quat_ijkw = quat
    
    rot = tf.Rotation.from_quat(quat_ijkw)
    R = rot.as_matrix()
    
    return R

def rot_matrix_to_quat(R, scalar_first=False):
    rot = tf.Rotation.from_matrix(R)
    quat_ijkw = rot.as_quat() # no scalar_first argument in py3.12

    if scalar_first:
        quat_wijk = np.array([
            quat_ijkw[3],
            quat_ijkw[0],
            quat_ijkw[1],
            quat_ijkw[2],
        ])
        return quat_wijk
    else:
        return quat_ijkw


# - Animation utilities - #

def plot_box(
    ax, 
    length, 
    width, 
    height, 
    position=np.array([0,0,0]), 
    rotation=np.eye(3), 
    lengthwise=False,
    color='cyan',
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
        face = Poly3DCollection(verts, color=color, alpha=0.5, edgecolor="k")
        ax.add_collection3d(face)


def plot_cylinder(
    ax, 
    radius, 
    height, 
    position=np.array([0,0,0]), 
    rotation=np.eye(3),
    color='cyan',
):
    """assumes ax is a matplotlib 3D projection subplot"""
    # Form points of top and bottom circles
    m = 20
    theta = np.linspace(0, 2*np.pi, m)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z_top = np.full(m, height/2)
    z_bottom = np.full(m, -height/2)
    pts_top = np.stack([x, y, z_top])
    pts_bottom = np.stack([x, y, z_bottom])

    # Transform points
    tf_pts_top = rotation @ pts_top + position.reshape(3,1)
    tf_pts_bottom = rotation @ pts_bottom + position.reshape(3,1)

    # Create faces
    faces = []
    for i in range(m - 1):
        face = [tf_pts_top[:,i], tf_pts_top[:,i+1], 
                tf_pts_bottom[:,i+1], tf_pts_bottom[:,i]]
        faces.append(face)
    faces.append(tf_pts_top.T)
    faces.append(tf_pts_bottom.T)

    # Plot surface
    shape = Poly3DCollection(faces, color=color, alpha=0.5, edgecolor="k")
    ax.add_collection3d(shape)