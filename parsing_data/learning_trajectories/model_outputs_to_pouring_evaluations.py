"""
Converts the LinOSS model outputs to the inputs required for the evaluate_trajectories.py script, then executes that script
Adapted from previous version of 'modelOutputs_to_evaluations.py' script, for new model output hdf5 format
This script also generates figures if you put in human trajectories
"""
import os
import h5py
import json
import numpy as np
import subprocess

import analysis.utils as utils


# - Main - #

def model_outputs_to_evaluations(
    input_trajectory_files,
    output_directory,
):
    """Assumes all model output trajectories have the same length
    """
    time = []
    pos_world_to_hand_W_inferenced = []
    quat_world_to_hand_wijk_inferenced = []
    pos_world_to_hand_W_truth = []
    quat_world_to_hand_wijk_truth = []
    pos_world_to_glass_rim_W = []
    truth = False

    for input_trajectory_file in input_trajectory_files:

        # Load input trajectory
        with h5py.File(input_trajectory_file, 'r') as f:
            
            for trajectory_key in f.keys():
                traj = f[trajectory_key]
                dataset_name = traj.attrs['name']

                # Inferenced hand trajectory
                data = traj['data']
                time.append(np.array(data['time'])[:,np.newaxis])
                pos_world_to_hand_W_inferenced.append(np.array(data['pos_world_to_hand_W']))
                R = np.array(data['rot_world_to_hand'])
                quat_wijk = np.array([utils.rot_matrix_to_quat(r, scalar_first=True) for r in R])
                quat_world_to_hand_wijk_inferenced.append(quat_wijk)

                # Reference objects
                ref = traj['reference']
                pos_world_to_glass_rim_W.append(np.array(ref['pos_world_to_glass_rim_W']))

                # Truth hand trajectory
                if 'truth' in traj.keys():
                    truth = True
                    truth = traj['truth']
                    pos_world_to_hand_W_truth.append(np.array(truth['pos_world_to_hand_W']))
                    R = np.array(truth['rot_world_to_hand'])
                    quat_wijk = np.array([utils.rot_matrix_to_quat(r, scalar_first=True) for r in R])
                    quat_world_to_hand_wijk_truth.append(quat_wijk)

    # Stack into one trajectory block
    time = np.stack(time)
    pos_world_to_hand_W_inferenced = np.stack(pos_world_to_hand_W_inferenced)
    quat_world_to_hand_wijk_inferenced = np.stack(quat_world_to_hand_wijk_inferenced)
    pos_world_to_glass_rim_W = np.stack(pos_world_to_glass_rim_W)
    if truth:
        pos_world_to_hand_W_truth = np.stack(pos_world_to_hand_W_truth)
        quat_world_to_hand_wijk_truth = np.stack(quat_world_to_hand_wijk_truth)
        pos_world_to_hand_W_initial = pos_world_to_hand_W_truth[:, 0, :]
    else:
        pos_world_to_hand_W_truth = np.empty(pos_world_to_hand_W_inferenced.shape)
        quat_world_to_hand_wijk_truth = np.empty(quat_world_to_hand_wijk_inferenced.shape)
        pos_world_to_hand_W_initial = np.empty((pos_world_to_hand_W_inferenced.shape[0], pos_world_to_hand_W_inferenced.shape[2]))

    # Save HDF5 trajectory as pre-evaluation format
    # Copied this format from previous "modelOutputs_to_evaluations.py" script
    os.makedirs(output_directory, exist_ok=True)
    output_trajectory_file = output_directory + 'evaluation_trajectory.hdf5'
    with h5py.File(output_trajectory_file, 'w') as f:
        f.create_dataset('hand_position_m', data=pos_world_to_hand_W_inferenced)
        f.create_dataset('hand_quaternion_wijk', data=quat_world_to_hand_wijk_inferenced)
        f.create_dataset('referenceObject_position_m', data=pos_world_to_glass_rim_W)
        f.create_dataset('time_s', data=time)
        truth_group = f.create_group('truth')
        truth_group.create_dataset('referenceObject_position_m', data=pos_world_to_glass_rim_W)
        truth_group.create_dataset('starting_hand_position_m', data=pos_world_to_hand_W_initial)

    # Create evaluation script configuration
    evaluation_config = {
        'feature_data_filepaths_byType': {
            'model': os.path.realpath(output_trajectory_file),
        },
        'output_dir': os.path.realpath(output_directory), # None to not save the plots
        'plot_exports_extension': 'png',
        
        'plot_motionObjectKeypoint_speedJerk':  True,
        'plot_motionObject_tilt':  True,
        'plot_motionObjectKeypoint_height':  True,
        'plot_motionObjectKeypoint_pouring_projection':  True,
        
        'plot_all_trajectories_singlePlot':  False,
        'plot_all_startingConditions_singlePlot':  False,
        
        'interactively_animate_trajectories_exampleType': None,
        'save_trajectory_animations_eachType': False,
        'save_trajectory_animations_compositeTypes': False,
        
        'plot_body_speedJerk':  False,
        'plot_joint_angles':  False,
        'plot_compare_distribution_body_speedJerk':  False,
        'plot_compare_distribution_motionObjectKeypoint_speedJerk':  True, # includes Wasserstein distances
        'plot_compare_distribution_joint_angles':  False,
        'plot_compare_distribution_motionObjectKeypoint_projection':  False,
        'plot_compare_distribution_motionObjectKeypoint_height':  False,
        'plot_compare_distribution_motionObject_tilt':  False,
        'plot_distributions_hand_to_pitcher_angles':  False,
        
        'keep_plots_open':  False,
    }

    # Call evaluation script
    evaluation_config_json = json.dumps(evaluation_config)
    # evaluation_config_json = evaluation_config_json.replace('"', '\\"')
    working_directory = os.path.dirname(os.path.realpath(__file__))
    subprocess.run(['python', f'{working_directory}/evaluate_pouring_trajectories.py', evaluation_config_json])


if __name__ == '__main__':
    # Script inputs
    input_trajectory_file = [
        os.path.expanduser('~/data/pouring/pouring_processed.hdf5'),
    ]
    output_directory = os.path.expanduser('~/data/pouring/figures/human/')

    # Main
    model_outputs_to_evaluations(
        input_trajectory_file,
        output_directory,
    )