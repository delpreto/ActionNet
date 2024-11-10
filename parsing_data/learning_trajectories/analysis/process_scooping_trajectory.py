"""
This script processes the (already pre-processed/parsed) 'scooping_trainingData_S00.h5'-like scooping data 
Reference object positions (pan and plate) are estimated based on initial conditions
Rotation matrices are added for each trajectory, and quaternions are changed from wijk -> ijkw
The output HDF5 is reformatted to into separate groups for each trajectory
"""
import os
import h5py
import numpy as np
import scipy.spatial.transform as tf

from constants import *


# - Main - #

def process_scooping_trajectory(
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
            pos_world_to_plate_W = reference_object_position_m[i].flatten()
            rot_world_to_hand = np.array([tf.Rotation.from_quat(quat).as_matrix() for quat in quat_world_to_hand_ijkw])

            # Transforms between world frame and spoon end (scoop)
            rot_world_to_spoon = rot_world_to_hand[0] @ ROT_HAND_TO_SPOON
            pos_world_to_spoon_W = rot_world_to_spoon @ POS_HAND_TO_SPOON_S + pos_world_to_hand_W[0]
            
            # Combine positions to get scoop position at timestep 0
            pos_world_to_pan_W = np.array([pos_world_to_spoon_W[0], pos_world_to_spoon_W[1], PAN_Z_OFFSET])

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

            # Store references
            ref_group = traj_group.create_group('reference')
            ref_group.create_dataset('pos_world_to_pan_W', data=pos_world_to_pan_W)
            ref_group.create_dataset('pos_world_to_plate_W', data=pos_world_to_plate_W)

if __name__ == '__main__':
    # Script inputs
    input_trajectory_file = os.path.expanduser(f'~/data/scooping/scooping_trainingData_S00.hdf5')
    output_trajectory_file= os.path.expanduser(f'~/data/scooping/scooping_processed_S00.hdf5')
    dataset_name = 'scooping_S00_human'

    # Run script
    process_scooping_trajectory(
        input_trajectory_file,
        output_trajectory_file,
        dataset_name=dataset_name,
    )