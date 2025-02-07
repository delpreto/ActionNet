import os
import h5py
import numpy as np
import scipy.spatial.transform as tf
from typing import List

import utils


# - Main - #

def create_training_data(
    input_trajectory_files: List[str],
    output_directory: str,
):
    """Assumes all trajectories are same length
    """
    # Load all trajectories
    time = []
    pos_world_to_hand_W = []
    flat_rot_world_to_hand = []
    pos_world_to_references_W_combined = []

    for input_trajectory_file in input_trajectory_files:
        with h5py.File(input_trajectory_file, 'r') as f:
            for traj_name, traj in f.items():

                # Load hand trajectory
                data = traj['data']
                dataset_name = traj.attrs['name']
                time.append(np.array(data['time']))
                pos_world_to_hand_W.append(np.array(data['pos_world_to_hand_W']))
                flat_rot_world_to_hand.append(np.array(data['rot_world_to_hand']).reshape((-1,9)))

                # Load reference object positions
                ref = traj['reference']

                # For the pouring task
                if 'pouring' in dataset_name:
                    references_combined = np.array(ref['pos_world_to_glass_rim_W'])

                # For the scooping task
                elif 'scoopingPepper' in dataset_name:
                    references_combined = np.concatenate([
                        np.array(ref['pos_world_to_pan_W']), 
                        np.array(ref['pos_world_to_plate_W']),
                    ])

                # Neither task specified?
                else: 
                    raise ValueError('Either "pouring" or "scooping" should be in HDF5 dataset label')
                
                pos_world_to_references_W_combined.append(references_combined)

    # Stack values into one block
    # (n,m,j) = (# of trajectories, # of timesteps, # of dimensions)
    time = np.stack(time)
    pos_world_to_hand_W = np.stack(pos_world_to_hand_W)
    flat_rot_world_to_hand = np.stack(flat_rot_world_to_hand)
    pos_world_to_references_W_combined = np.stack(pos_world_to_references_W_combined)

    # Construct trajectory matrix
    # (n,m,k) =(# of trajectories, # of timestamps, # of data dimensions)
    trajectories = np.concatenate([
        pos_world_to_hand_W,
        flat_rot_world_to_hand,
    ], axis=2) # combine along data dimension axis

    # Construct reference vectors
    # (n,l) = (# of trajectories, # of label dimensions)
    references = np.concatenate([
        pos_world_to_hand_W[:,0,:], # Initial hand position
        flat_rot_world_to_hand[:,0,:], # Initial hand rotation
        pos_world_to_references_W_combined, # Reference object positions
        pos_world_to_hand_W[:,-1,:], # Terminal hand position
    ], axis=1) # combine along label dimension axis
    references = np.repeat(references[:,np.newaxis,:], repeats=trajectories.shape[1], axis=1) # Promote dimension to (n,m,l)

    # Export training dataset as .pkl
    os.makedirs(output_directory, exist_ok=True)
    utils.save_pickle(time, output_directory + 'time.pkl')
    utils.save_pickle(references, output_directory + 'data.pkl') # Reference vectors are inputs
    utils.save_pickle(trajectories, output_directory + 'labels.pkl') # Trajectories are outputs
    

if __name__ == '__main__':
    # Script inputs
    input_trajectory_files = [
        os.path.expanduser('~/data/pouring/pouring_processed.hdf5'),
    ]
    output_directory = os.path.expanduser('~/drl/linoss/data_dir/processed/pouring/')

    # Main
    create_training_data(
        input_trajectory_files,
        output_directory,
    )