"""
This script evaluates some performance metrics given pre-processed HDF5 files for the scooping task
Namely, it calculates: 
    - Maximum of volumetric intersection between spoon and plate (how much does it scoop?)
    - Maximum of volumetric intersection between spoon and pan (how much does it place?)
    - Maximum of volumetric intersection between spoon and table (does it collide with obstacles?)
These metrics are of course rough approximations and don't necessarily imply the scooping task was completed successfully
Histograms are generated for these metrics
Additionally, generates some aggregate figures for scooping trajectories:
    - Scoop height
    - Scoop tilt angle
    - Pickup location
    - Dropoff location
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

def evaluate_scooping(
    input_trajectory_files, 
    output_figure_directory,
):
    """Loads HDF5 trajectory files
    and calculates scooping metrics.
    """
    pan_volumes = []
    plate_volumes = []
    table_volumes = []
    pickup_squared_errors = []
    dropoff_squared_errors = []

    z_fig, z_ax = plt.subplots(1,figsize=(9,9))
    tilt_fig, tilt_ax = plt.subplots(1,figsize=(9,9))
    loc_fig, loc_ax = plt.subplots(1,2,figsize=(9,9))
    ref_fig, ref_ax = plt.subplots(1,figsize=(9,9))

    for input_trajectory_file in input_trajectory_files:

        # Read trajectory file
        with h5py.File(input_trajectory_file, 'r') as f:

            for trajectory_key, traj in f.items():
                dataset_name = traj.attrs['name']

                # Hand trajectory
                data = traj['data']
                time = np.array(data['time'])
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

                pan_volumes.append(np.max(pan_volumes_i))
                plate_volumes.append(np.max(plate_volumes_i))
                table_volumes.append(int(np.max(table_volumes_i) > 1e-8))

                # Height
                rot_world_to_spoon = rot_world_to_hand @ ROT_HAND_TO_SPOON
                pos_hand_to_spoon_W = rot_world_to_spoon @ POS_HAND_TO_SPOON_S
                pos_world_to_spoon_W = pos_world_to_hand_W + pos_hand_to_spoon_W
                z_ax.plot(time, pos_world_to_spoon_W[:,2], color='darkblue', alpha=0.5)

                # Tilt
                z_world_to_spoon = rot_world_to_spoon[:,2]
                tilt_angle = np.arccos(np.dot(z_world_to_spoon / np.linalg.norm(z_world_to_spoon), np.array([0, 0, 1])))
                tilt_ax.plot(time, tilt_angle, color='darkblue', alpha=0.5)

                # # Pickup location -- maximum tilt angle closest to pan
                # hover_idx = np.linalg.norm(pos_world_to_spoon_W[:,:2] - pos_world_to_pan_W[:2], axis=1) \
                #     < np.linalg.norm(pos_world_to_spoon_W[:,:2] - pos_world_to_plate_W[:2], axis=1)
                # pos_world_to_pickup_W = pos_world_to_spoon_W[hover_idx][np.argmax(tilt_angle[hover_idx])]
                # d_pos_world_to_pickup_W = pos_world_to_pickup_W - pos_world_to_pan_W
                # pickup_squared_errors.append(np.linalg.norm(d_pos_world_to_pickup_W[0:2]) ** 2)
                # loc_ax[0].scatter(d_pos_world_to_pickup_W[0], d_pos_world_to_pickup_W[1], color='darkblue')

                # # Dropoff location -- maximum tilt angle closest to plate
                # hover_idx = ~hover_idx # invert pickup indices -> dropoff indices
                # pos_world_to_dropoff_W = pos_world_to_spoon_W[hover_idx][np.argmax(tilt_angle[hover_idx])]
                # d_pos_world_to_dropoff_W = pos_world_to_dropoff_W - pos_world_to_plate_W
                # dropoff_squared_errors.append(np.linalg.norm(d_pos_world_to_dropoff_W[0:2]) ** 2)
                # loc_ax[1].scatter(d_pos_world_to_dropoff_W[0], d_pos_world_to_dropoff_W[1], color='darkred')

                # Reference locations
                ref_ax.scatter(*pos_world_to_pan_W[[0,1]], color='darkblue', label='pickup')
                ref_ax.scatter(*pos_world_to_plate_W[[0,1]], color='darkred', label='dropoff')

    # Create figures
    os.makedirs(output_figure_directory, exist_ok=True)
    
    # Pickup intersections
    fig, ax = plt.subplots(1)
    ax.hist(np.array(pan_volumes) * 100**3, alpha=0.5, edgecolor='black')
    ax.set_xlabel('Volume of mesh intersection (cm^3)')
    ax.set_ylabel('Count')
    ax.set_title('Volumetric intersection between spoon and pan pickup region')
    plt.tight_layout()
    fig.savefig(output_figure_directory + 'pan_intersection.png')

    # Dropoff intersections
    fig, ax = plt.subplots(1)
    ax.hist(np.array(plate_volumes) * 100**3, alpha=0.5, edgecolor='black')
    ax.set_xlabel('Volume of mesh intersection (cm^3)')
    ax.set_ylabel('Count')
    ax.set_title('Volumetric intersection between spoon and plate dropoff region')
    plt.tight_layout()
    fig.savefig(output_figure_directory + 'plate_intersection.png')

    # Collision
    fig, ax = plt.subplots(1)
    ax.hist(np.array(table_volumes), bins=2, alpha=0.5, edgecolor='black')
    ax.set_xlabel('Collision boolean')
    ax.set_ylabel('Count')
    ax.set_title('Collision between spoon and table')
    plt.tight_layout()
    fig.savefig(output_figure_directory + 'collision.png')

    # Height
    z_fig.suptitle('Spoon height')
    z_ax.set_ylabel('Height (m)')
    z_ax.set_xlabel('Time (s)')
    plt.tight_layout()
    z_fig.savefig(output_figure_directory + f'spoon_height.png')

    # Tilt
    tilt_fig.suptitle('Spoon tilt angle')
    tilt_ax.set_ylabel('Tilt (rad)')
    tilt_ax.set_xlabel('Time (s)')
    plt.tight_layout()
    tilt_fig.savefig(output_figure_directory + f'spoon_tilt.png')

    # Pickup
    loc_fig.suptitle('Estimated pickup and dropoff locations')
    theta = np.linspace(0, 2*np.pi, 50)
    x = PAN_RADIUS * np.cos(theta)
    y = PAN_RADIUS * np.sin(theta)
    pickup_mse = np.mean(pickup_squared_errors)
    loc_ax[0].plot(x, y, color='darkblue')
    loc_ax[0].set_ylabel('Y (m)')
    loc_ax[0].set_xlabel('X (m)')
    loc_ax[0].set_title(f'Pickup location: MSE = {pickup_mse:05f} (m^2)')
    loc_ax[0].set_aspect('equal')

    # Dropoff
    x = PLATE_RADIUS * np.cos(theta)
    y = PLATE_RADIUS * np.sin(theta)
    dropoff_mse = np.mean(dropoff_squared_errors)
    loc_ax[1].plot(x, y, color='darkred')
    loc_ax[1].set_ylabel('Y (m)')
    loc_ax[1].set_xlabel('X (m)')
    loc_ax[1].set_title(f'Dropoff location: MSE = {dropoff_mse:05f} (m^2)')
    loc_ax[1].set_aspect('equal')
    plt.tight_layout()
    loc_fig.savefig(output_figure_directory + f'pickup_dropoff.png')

    # Reference object locations
    ref_fig.suptitle('Pan and plate locations')
    ref_ax.set_xlabel('X (m)')
    ref_ax.set_ylabel('Y (m)')
    ref_ax.set_aspect('equal')
    ref_ax.legend()
    plt.tight_layout()
    ref_fig.savefig(output_figure_directory + f'reference.png')

    plt.close()


if __name__ == '__main__':
    # Script inputs
    input_trajectory_files = [
        os.path.expanduser(f'~/data/scooping/scooping_processed.hdf5'),
    ]
    output_figure_directory = os.path.expanduser(f'~/data/scooping/figures/human/')

    # Main
    evaluate_scooping(
        input_trajectory_files,
        output_figure_directory,
    )