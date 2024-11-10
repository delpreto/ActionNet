"""
This script evaluates some performance metrics given a pre-processed HDF5 file for the scooping task
Namely, it calculates: 
    - Maximum of volumetric intersection between spoon and plate (how much does it scoop?)
    - Maximum of volumetric intersection between spoon and pan (how much does it place?)
    - Maximum of volumetric intersection between spoon and table (does it collide with obstacles?)
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
    lengthwise=False,
) -> trimesh.Trimesh:
    """Creates box mesh given size, position, and orientation
    lengthwise=True means the box will extend from 0 to length,
        otherwise it will extend from -length/2 to length/2
    """
    # Pre-transform shift
    T = np.eye(4,4)
    if lengthwise:
        T[:3, 3] = np.array([length/2, 0, 0])
    
    # Create transformed box
    box_mesh = trimesh.creation.box(
        extents=[length, width, height],
        transform=T,
    )

    # Form (4,4) transformation matrix
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = position
    
    # Post-transform
    box_mesh = box_mesh.apply_transform(T)

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
    mesh1: trimesh.Trimesh, 
    mesh2: trimesh.Trimesh,
) -> float:
    """Given two trimesh.Trimesh objects, 
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
    # Spoon mesh
    spoon_mesh = create_box_mesh(*SPOON_BOX, lengthwise=True)

    # Scooping region
    pan_region_offset = np.array([0, 0, PAN_HEIGHT/2 + SCOOPING_HEIGHT/2])
    pan_region_mesh = create_cylinder_mesh(
        PAN_RADIUS, 
        SCOOPING_HEIGHT,
        position=pos_world_to_pan_W + pan_region_offset,
    )

    # Placing region
    plate_region_offset = np.array([0, 0, PLATE_HEIGHT/2 + SCOOPING_HEIGHT/2])
    plate_region_mesh = create_cylinder_mesh(
        PLATE_RADIUS, 
        SCOOPING_HEIGHT,
        position=pos_world_to_plate_W + plate_region_offset,
    )

    # Table (for obstacle checking)
    table_mesh = create_box_mesh(
        *TABLE_BOX,
        position=TABLE_ORIGIN
    )

    # Transform meshes and calculate volumetric intersections
    pan_volumes = []
    plate_volumes = []
    table_volumes = []
    for i in range(n):

        # Transform spoon mesh
        T = np.eye(4)
        T[:3, :3] = rot_world_to_hand[i] @ ROT_HAND_TO_SPOON
        T[:3, 3] = pos_world_to_hand_W[i]
        spoon_mesh = spoon_mesh.apply_transform(T)

        # Calculate volumetric intersections
        pan_volume = calculate_volumetric_intersection(spoon_mesh, pan_region_mesh)
        pan_volumes.append(pan_volume)

        plate_volume = calculate_volumetric_intersection(spoon_mesh, plate_region_mesh)
        plate_volumes.append(plate_volume)

        table_volume = calculate_volumetric_intersection(spoon_mesh, table_mesh)
        table_volumes.append(table_volume)

        # Un-transform spoon mesh
        Tinv = np.linalg.inv(T)
        spoon_mesh = spoon_mesh.apply_transform(Tinv)

    # Print the maximum volumetric intersections
    print('Maximum volumetric intersection of spoon and pan (m^3): ', np.max(pan_volumes))
    print('Maximum volumetric intersection of spoon and plate (m^3): ', np.max(plate_volumes))
    print('Spoon intersects with table: ', np.max(table_volumes) > 1e-8)


if __name__ == '__main__':
    # Script inputs
    input_trajectory_file = os.path.expanduser(f'~/data/scooping/scooping_processed_S00.hdf5')
    trajectory_id = 1

    # Main
    evaluate_scooping_performance(
        input_trajectory_file, 
        trajectory_id,
    )