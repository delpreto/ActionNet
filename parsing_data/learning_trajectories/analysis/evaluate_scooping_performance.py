"""
This script evaluates some performance metrics given pre-processed HDF5 files for the scooping task
Namely, it calculates: 
    - Maximum of volumetric intersection between spoon and plate (how much does it scoop?)
    - Maximum of volumetric intersection between spoon and pan (how much does it place?)
    - Maximum of volumetric intersection between spoon and table (does it collide with obstacles?)
These metrics are of course rough approximations and don't necessarily imply the scooping task was completed successfully
Histograms are generated for these metrics
"""
import os
import h5py
import numpy as np
import trimesh
import matplotlib.pyplot as plt

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
    input_trajectory_files, 
    output_figure_directory,
):
    """Loads HDF5 trajectory files
    and calculates scooping metrics.
    """
    pan_volumes = []
    plate_volumes = []
    table_volumes = []

    for input_trajectory_file in input_trajectory_files:

        # Read trajectory file
        with h5py.File(input_trajectory_file, 'r') as f:

            for trajectory_key, traj in f.items():
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

                # Pickup region
                pan_region_offset = np.array([0, 0, PAN_HEIGHT/2 + PICKUP_HEIGHT/2])
                pan_region_mesh = create_cylinder_mesh(
                    PAN_RADIUS, 
                    PICKUP_HEIGHT,
                    position=pos_world_to_pan_W + pan_region_offset,
                )

                # Dropoff region
                plate_region_offset = np.array([0, 0, PLATE_HEIGHT/2 + DROPOFF_HEIGHT/2])
                plate_region_mesh = create_cylinder_mesh(
                    PLATE_RADIUS, 
                    DROPOFF_HEIGHT,
                    position=pos_world_to_plate_W + plate_region_offset,
                )

                # Table (for obstacle checking)
                table_mesh = create_box_mesh(
                    *TABLE_BOX,
                    position=TABLE_ORIGIN
                )

                # Transform meshes and calculate volumetric intersections
                pan_volumes_i = []
                plate_volumes_i = []
                table_volumes_i = []
                for i in range(n):

                    # Transform spoon mesh
                    T = np.eye(4)
                    T[:3, :3] = rot_world_to_hand[i] @ ROT_HAND_TO_SPOON
                    T[:3, 3] = pos_world_to_hand_W[i]
                    spoon_mesh = spoon_mesh.apply_transform(T)

                    # Calculate volumetric intersections
                    pan_volume = calculate_volumetric_intersection(spoon_mesh, pan_region_mesh)
                    pan_volumes_i.append(pan_volume)

                    plate_volume = calculate_volumetric_intersection(spoon_mesh, plate_region_mesh)
                    plate_volumes_i.append(plate_volume)

                    table_volume = calculate_volumetric_intersection(spoon_mesh, table_mesh)
                    table_volumes_i.append(table_volume)

                    # Un-transform spoon mesh
                    Tinv = np.linalg.inv(T)
                    spoon_mesh = spoon_mesh.apply_transform(Tinv)

                # Print the maximum volumetric intersections
                print('Maximum volumetric intersection of spoon and pan (m^3): ', np.max(pan_volumes_i))
                print('Maximum volumetric intersection of spoon and plate (m^3): ', np.max(plate_volumes_i))
                print('Spoon intersects with table: ', np.max(table_volumes_i) > 1e-8)

                pan_volumes.append(np.max(pan_volumes_i))
                plate_volumes.append(np.max(plate_volumes_i))
                table_volumes.append(int(np.max(table_volumes_i) > 1e-8))

    # Create histograms
    os.makedirs(output_figure_directory, exist_ok=True)
    
    fig, ax = plt.subplots(1)
    ax.hist(np.array(pan_volumes) * 100**3, alpha=0.5, edgecolor='black')
    ax.set_xlabel('Volume of mesh intersection (cm^3)')
    ax.set_ylabel('Count')
    ax.set_title('Volumetric intersection between spoon and pan pickup region')
    fig.savefig(output_figure_directory + 'pan_intersection.png')

    fig, ax = plt.subplots(1)
    ax.hist(np.array(plate_volumes) * 100**3, alpha=0.5, edgecolor='black')
    ax.set_xlabel('Volume of mesh intersection (cm^3)')
    ax.set_ylabel('Count')
    ax.set_title('Volumetric intersection between spoon and plate dropoff region')
    fig.savefig(output_figure_directory + 'plate_intersection.png')

    fig, ax = plt.subplots(1)
    ax.hist(np.array(table_volumes), bins=2, alpha=0.5, edgecolor='black')
    ax.set_xlabel('Collision boolean')
    ax.set_ylabel('Count')
    ax.set_title('Collision between spoon and table')
    fig.savefig(output_figure_directory + 'collision.png')


if __name__ == '__main__':
    # Script inputs
    input_trajectory_files = [
        os.path.expanduser(f'~/data/scooping/inference_LinOSS_scooping_5678.hdf5'),
    ]
    output_figure_directory = os.path.expanduser(f'~/data/scooping/figures/inference_LinOSS_scooping_5678/')

    # Main
    evaluate_scooping_performance(
        input_trajectory_files,
        output_figure_directory,
    )