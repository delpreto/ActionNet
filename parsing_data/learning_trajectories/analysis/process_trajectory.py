"""
This script processes the (already pre-processed/parsed) '[task_name]_trainingData_S00.h5'-like data 

Rotation matrices are added for each trajectory, and quaternions are changed from wijk -> ijkw
The output HDF5 is reformatted to into separate groups for each trajectory

For the scooping task, pan position is estimated based on initial conditions, and the reference object provided is the plate position
For the pouring task, the reference object provided is the glass rim position

A dataset identifying label is also given to the HDF5 file, it should contain either 'pouring' or 'scooping' depending on the task
"""
import os
import h5py
import numpy as np
import scipy.spatial.transform as tf

from constants import *


# - Main - #

def process_trajectory(
    input_trajectory_file, 
    output_trajectory_file, 
    dataset_name='',
):
    # Read raw HDF5 and pull relevant trajectory fields
    with h5py.File(input_trajectory_file, 'r') as f_in:
        time = np.array(f_in['time_s'])
        hand_position_m = np.array(f_in['hand_position_m'])
        hand_quaternion_wijk = np.array(f_in['hand_quaternion_wijk'])
        reference_object_position_m = np.array(f_in['referenceObject_position_m'])
        n = hand_position_m.shape[0] # num trials

     # Output HDF5
    with h5py.File(output_trajectory_file, 'w') as f_out:

        # Process each trajectory
        for i in range(n):
            if i % 10 == 0:
                print(f'Processing trajectory {i}')
            
            t = time[i].squeeze()
            pos_world_to_hand_W = hand_position_m[i]
            quat_world_to_hand_ijkw = np.vstack([
                hand_quaternion_wijk[i][:,1],
                hand_quaternion_wijk[i][:,2],
                hand_quaternion_wijk[i][:,3],
                hand_quaternion_wijk[i][:,0],
            ]).T
            rot_world_to_hand = np.array([tf.Rotation.from_quat(quat).as_matrix() for quat in quat_world_to_hand_ijkw])
            pos_world_to_reference_W = reference_object_position_m[i].flatten()

            # Add trajectory to HDF5
            traj_group = f_out.create_group(f'trajectory_{i+1:03d}')
            traj_group.attrs['name'] = dataset_name
            traj_group.attrs['description'] = f'Hand pose trajectory {i+1} and reference positions'

            # Store datasets
            data_group = traj_group.create_group('data')
            data_group.attrs['description'] = f'Hand pose trajectory'
            data_group.create_dataset('time', data=t)
            data_group.create_dataset('pos_world_to_hand_W', data=pos_world_to_hand_W)
            data_group.create_dataset('quat_world_to_hand_ijkw', data=quat_world_to_hand_ijkw)
            data_group.create_dataset('rot_world_to_hand', data=rot_world_to_hand)

            # Reference objects for pouring task
            if 'pouring' in dataset_name:
                # Store references
                ref_group = traj_group.create_group('reference')
                ref_group.create_dataset('pos_world_to_glass_rim_W', data=pos_world_to_reference_W)

            # Reference objects for scooping task
            elif 'scooping' in dataset_name:
                # Transforms between world frame and spoon end (scoop)
                rot_world_to_spoon = rot_world_to_hand[0] @ ROT_HAND_TO_SPOON
                pos_world_to_spoon_W = rot_world_to_spoon @ POS_HAND_TO_SPOON_S + pos_world_to_hand_W[0]
                
                # Combine positions to get scoop position at timestep 0
                pos_world_to_pan_W = np.array([pos_world_to_spoon_W[0], pos_world_to_spoon_W[1], PAN_Z_OFFSET])

                # Store references
                ref_group = traj_group.create_group('reference')
                ref_group.create_dataset('pos_world_to_pan_W', data=pos_world_to_pan_W)
                ref_group.create_dataset('pos_world_to_plate_W', data=pos_world_to_reference_W)

            # Neither scooping nor pouring dataset?
            else:
                raise ValueError('Either "pouring" or "scooping" should be in the dataset name')

if __name__ == '__main__':
    # Script inputs
    input_trajectory_file = os.path.expanduser(f'~/data/pouring/pouring_trainingData_S10.hdf5')
    output_trajectory_file= os.path.expanduser(f'~/data/pouring/pouring_processed_S10.hdf5')
    dataset_name = 'pouring_S10_human'

    # Run script
    process_trajectory(
        input_trajectory_file,
        output_trajectory_file,
        dataset_name=dataset_name,
    )