"""
This script evaluates some performance metrics given a pre-processed HDF5 file for the scooping task
Namely, it calculates: 
    - Maximum of volumetric intersection between spoon and plate (how much does it scoop?)
    - Minimum of distance between plate center and spoon tip (how accurate does it place?)
These metrics are of course rough approximations and don't necessarily imply the scooping task was completed successfully
"""
import os
import h5py
import numpy as np
import trimesh

from constants import *


def create_box_mesh(
    length, 
    width, 
    height, 
    position=np.array([0,0,0]), 
    rotation=np.eye(3), 
) -> trimesh.Trimesh:
    """Creates box mesh given size, position, and orientation
    """
    # Form (4,4) transformation matrix
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = position
    
    # Create transformed box
    box_mesh = trimesh.creation.box(
        extents=[length, width, height],
        transform=T,
    )

    return box_mesh


def create_cylinder_mesh(
    radius, 
    height, 
    position=np.array([0,0,0]), 
    rotation=np.eye(3),
) -> trimesh.Trimesh:
    """Creates cylinder mesh given size, position, and orientation
    """
    # Form (4,4) transformation matrix
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = position

    # Create transformed cylinder
    cylinder_mesh = trimesh.creation.cylinder(
        radius=radius,
        height=height,
        transform=T,
    )

    return cylinder_mesh


def calculate_volumetric_intersection(
    mesh1, 
    mesh2,
) -> float:
    """Given two 3D pymesh.Mesh objects, 
    calculates their volumetric intersection
    """
    # Intersection of meshes
    intersection_mesh = mesh1.intersection(mesh2)

    # Calculate volume
    intersection_volume = intersection_mesh.volume

    return intersection_volume


# - Main - # 

def evaluate_scooping_performance(
    input_trajectory_file, 
    trajectory_id,
):
    """Loads a specific trajectory from an HDF5 file
    and calculates scooping metrics.
    """
    # Read trajectory file
    with h5py.File(input_trajectory_file, 'r') as f:
        traj = f[f'trajectory_{trajectory_id:03d}']
        dataset_name = traj.attrs['name']

        # Hand trajectory
        data = traj['data']
        t = np.array(data['time'])
        pos_world_to_hand_W = np.array(data['pos_world_to_hand_W'])
        rot_world_to_hand = np.array(data['rot_world_to_hand'])
        n = pos_world_to_hand_W.shape[0]

        # Reference objects
        ref = traj['reference']
        pos_world_to_plate_W = np.array(ref['pos_world_to_plate_W'])
        pos_world_to_pan_W = np.array(ref['pos_world_to_pan_W'])

    # Create initial meshes
    spoon_mesh = create_box_mesh(*SPOON_BOX)
    pan_mesh = create_cylinder_mesh(
        PAN_RADIUS * 1.2, 
        PAN_HEIGHT * 25,
        position=pos_world_to_pan_W,
    )

    # Transform meshes and calculate volumetric intersections
    volumes = []
    for i in range(n):

        # Transform spoon mesh
        T = np.eye(4)
        T[:3, :3] = rot_world_to_hand[i] @ ROT_HAND_TO_SPOON
        T[:3, 3] = pos_world_to_hand_W[i]
        spoon_mesh = spoon_mesh.apply_transform(T)

        # Calculate volumetric intersection
        volume = calculate_volumetric_intersection(spoon_mesh, pan_mesh)
        volumes.append(volume)

        # Un-transform spoon mesh
        Tinv = np.linalg.inv(T)
        spoon_mesh = spoon_mesh.apply_transform(Tinv)

    # Print the maximum value
    print('Maximum volumetric intersection of spoon and pan (m^3): ', np.max(volumes))
        
    # Calculate spoon tip positions (vectorized)
    pos_hand_to_spoon_W = rot_world_to_hand @ ROT_HAND_TO_SPOON @ POS_HAND_TO_SPOON_S
    pos_world_to_spoon_W = pos_world_to_hand_W + pos_hand_to_spoon_W

    # Minimum distance from spoon tip to plate
    pos_plate_to_spoon_W = pos_world_to_spoon_W - pos_world_to_plate_W
    distance_plate_to_spoon = np.min(np.linalg.norm(pos_plate_to_spoon_W, axis=1))
    print('Minimum distance from plate to spoon (m): ', distance_plate_to_spoon)


if __name__ == '__main__':
    # Script inputs
    input_trajectory_file = os.path.expanduser(f'~/data/scooping/scooping_processed_S00.hdf5')
    trajectory_id = 1

    # Main
    evaluate_scooping_performance(
        input_trajectory_file, 
        trajectory_id,
    )