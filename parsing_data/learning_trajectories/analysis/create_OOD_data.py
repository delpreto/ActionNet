"""
This script creates input sequences that are out of distribution from those seen in pouring and scooping tasks
(Does not generate labels)

Starting rotation matrix and time vector are randomly sampled from existing data

For the scooping task:
    - Pan and plate location are randomly sampled uniformly in the entire table bounding box
    - Starting hand position and ending hand position are chosen to be slightly above pan location (each with added noise)

For the pouring task:
    - Glass location is randomly sampled uniformly in the entire table bounding box
    - Hand position is randomly sampled uniformly in the entire table bounding box
        - Starting and ending are perturbed by a small amount of noise

TODO:
    - OOD model performance across almost all models is very poor
    - is the data being generated wrong?
    - are the model parameters super overfit?
        - but, testing data works great?
"""
import os
import h5py
import numpy as np

import utils
from constants import *


# - Main - #

def create_OOD_data(
    input_trajectory_files,
    num_samples,
    output_directory,
):
    """Assumes all trajectories are same length
    """
    # Load all trajectories
    time = []
    flat_rot_world_to_hand = []
    scooping = False
    pouring = False

    for input_trajectory_file in input_trajectory_files:
        with h5py.File(input_trajectory_file, 'r') as f:
            for traj_name, traj in f.items():

                # Load hand trajectory
                data = traj['data']
                dataset_name = traj.attrs['name']
                time.append(np.array(data['time']))
                flat_rot_world_to_hand.append(np.array(data['rot_world_to_hand']).reshape((-1,9)))

                if 'pouring' in dataset_name:
                    pouring = True
                elif 'scooping' in dataset_name:
                    scooping = True
                else:
                    raise ValueError('Either "scooping" or "pouring" should be in dataset name')

    # Stack values into one block
    # (n,m,j) = (# of trajectories, # of timesteps, # of dimensions)
    time = np.stack(time)
    flat_rot_world_to_hand = np.stack(flat_rot_world_to_hand)
    n = time.shape[0]

    # Randomly sample time vector and starting rotation matrix
    time_sampled = time[np.random.randint(n, size=num_samples),:]
    flat_rot_world_to_hand_sampled = flat_rot_world_to_hand[np.random.randint(n, size=num_samples),0,:]

    # Generate reference positions
    if scooping:
        print('Generating scooping data points')
        pos_world_to_references_W = np.concatenate([
            np.random.uniform(*TABLE_BBOX, size=(num_samples,2)),
            PAN_Z_OFFSET * np.ones((num_samples,1)),
            np.random.uniform(*TABLE_BBOX, size=(num_samples,2)),
            PLATE_Z_OFFSET * np.ones((num_samples,1)),
        ], axis=1)
        pos_world_to_hand_start_W = pos_world_to_references_W[:,[0,1,2]] \
            + np.random.normal(loc=[0, 0, 0], scale=XYZ_NOISE_STD_DEV, size=(num_samples,3)) \
            + np.array([[0, 0, HAND_Z_OFFSET - PAN_Z_OFFSET]])
        pos_world_to_hand_end_W = pos_world_to_references_W[:,[0,1,2]] \
            + np.random.normal(loc=[0, 0, 0], scale=XYZ_NOISE_STD_DEV, size=(num_samples,3)) \
            + np.array([[0, 0, HAND_Z_OFFSET - PLATE_Z_OFFSET]])
    elif pouring:
        print('Generating pouring data points')
        pos_world_to_references_W = np.concatenate([
            np.random.uniform(*TABLE_BBOX, size=(num_samples,2)),
            GLASS_Z_OFFSET * np.ones((num_samples,1)),
        ], axis=1)
        pos_world_to_hand_W = np.concatenate([
            np.random.uniform(*TABLE_BBOX, size=(num_samples,2)),
            HAND_Z_OFFSET * np.ones((num_samples,1)),
        ], axis=1)
        pos_world_to_hand_start_W = pos_world_to_hand_W \
            + np.random.normal(loc=[0,0,0], scale=XYZ_NOISE_STD_DEV, size=(num_samples,3))
        pos_world_to_hand_end_W = pos_world_to_hand_W \
            + np.random.normal(loc=[0,0,0], scale=XYZ_NOISE_STD_DEV, size=(num_samples,3))
    else:
        raise ValueError('Neither pouring nor scooping mode')

    # Construct reference vectors
    # (n,l) = (# of trajectories, # of label dimensions)
    references = np.concatenate([
        pos_world_to_hand_start_W, # Initial hand position
        flat_rot_world_to_hand_sampled, # Initial hand rotation
        pos_world_to_references_W, # Reference object positions
        pos_world_to_hand_end_W, # Terminal hand position
    ], axis=1) # combine along label dimension axis
    references = np.repeat(references[:,np.newaxis,:], repeats=time.shape[1], axis=1) # Promote dimension to (n,m,l)

    # Export training dataset as .pkl
    os.makedirs(output_directory, exist_ok=True)
    utils.save_pickle(time_sampled, output_directory + 'time.pkl')
    utils.save_pickle(references, output_directory + 'data.pkl') # Reference vectors are inputs
    

if __name__ == '__main__':
    # Script inputs
    input_trajectory_files = [
        os.path.expanduser('~/data/scooping/scooping_processed.hdf5'),
    ]
    num_samples = 10
    output_directory = os.path.expanduser('~/drl/linoss/data_dir/processed/scooping_OOD/')

    # Main
    create_OOD_data(
        input_trajectory_files,
        num_samples,
        output_directory,
    )