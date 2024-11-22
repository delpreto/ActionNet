"""
Given an input HDF5 file of hand pose trajectories
Transforms the hand trajectories by a given static rotation (with respect to hand frame)
Outputs a similar HDF5 file
"""
import os
import numpy as np
import h5py

import constants


def rotate_trajectory(
    input_trajectory_file,
    output_trajectory_file,
    rot_hand_to_hand_transformed,
):
    with h5py.File(input_trajectory_file, 'r') as in_file, \
        h5py.File(output_trajectory_file, 'w') as out_file:

        # Iterate through input trajectories
        for in_traj_key, in_traj_group in in_file.items():

            # Create output trajectory
            out_traj_group = out_file.create_group(in_traj_key)
            out_traj_group.attrs['name'] = in_traj_group.attrs['name']
            out_traj_group.attrs['description'] = in_traj_group.attrs['description'] + ', transformed'

            # Copy reference group
            out_traj_group.copy(in_traj_group['reference'], 'reference')
            
            # Copy timestamps and hand position
            out_data_group = out_traj_group.create_group('data')
            out_data_group.attrs['description'] = in_traj_group['data'].attrs['description'] + ', transformed'
            out_data_group.copy(in_traj_group['data']['time'], 'time')
            out_data_group.copy(in_traj_group['data']['pos_world_to_hand_W'], 'pos_world_to_hand_W')

            # Transform hand rotation
            rot_world_to_hand = np.array(in_traj_group['data']['rot_world_to_hand'])
            rot_world_to_hand_transformed = rot_world_to_hand @ rot_hand_to_hand_transformed
            out_data_group.create_dataset('rot_world_to_hand', data=rot_world_to_hand_transformed)


if __name__ == '__main__':
    # Script inputs
    input_trajectory_file = os.path.expanduser('~/data/scooping/inference_LinOSS_train_scooping_5678.hdf5')
    output_trajectory_file = os.path.expanduser('~/data/scooping/inference_LinOSS_train_scooping_5678_straight.hdf5')
    
    # Side spoon
    # rot_spoon_to_baxter_spoon = ROT_HAND_TO_SPOON.T @ ROT_HAND_TO_SIDE_SPOON
    # rot_hand_to_hand_transformed = rot_spoon_to_baxter_spoon.T

    # Straight spoon
    rot_hand_to_hand_transformed = constants.ROT_HAND_TO_SPOON @ constants._rot_hand_to_prespoon.T

    # Main
    rotate_trajectory(
        input_trajectory_file,
        output_trajectory_file,
        rot_hand_to_hand_transformed,
    )