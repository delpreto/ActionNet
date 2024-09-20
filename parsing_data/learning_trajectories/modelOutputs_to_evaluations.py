
from helpers.convert_modelOutputs_to_hdf5s import *
import numpy as np
import os
import json

#################################################
# Specify the input data files.
input_dir = os.path.realpath(os.path.join(actionsense_root_dir, 'results', 'learning_trajectories',
                                         'models', 'state-space',
                                         # '2024-09-06_13-35'
                                         # '2024-09-06_17-25'
                                         # '2024-09-06_19-02'
                                         # '2024-09-07_09-58'
                                         # '2024-09-10_17-10'
                                         '2024-09-13_18-15_forSubmission'
                                           ))
# Specify where to save the outputs.
output_dir = input_dir
modelData_output_filepath = os.path.join(output_dir, 'pouring_modelData.hdf5')
os.makedirs(output_dir, exist_ok=True)

#################################################
# Convert the model data.
# print(init_values.shape) ## shape: batch_size x dim, where dim dimension is set up as: [initial_hand_position,initial_hand_quaternion,glass_position,final_hand_position]
# print(trajectories.shape) ## shape: batch_size x steps x dim, where dim dimension is set up as: [hand_positions,hand_quaternions]
init_values = np.load(os.path.join(input_dir, 'initial_vals_LOSS.npy'))
trajectories = np.load(os.path.join(input_dir, 'trajectories_LOSS.npy'))
convert_modelOutputs_to_hdf5_data(init_values, trajectories, modelData_output_filepath)

#################################################
# Generate desired plots.
evaluation_config = {
  'feature_data_filepaths_byType': {
    'model': os.path.realpath(modelData_output_filepath),
  },
  'output_dir': os.path.realpath(output_dir), # None to not save the plots
  'plot_exports_extension': 'png',
  
  'plot_spout_speedJerk':  True,
  'plot_spout_tilt':  True,
  'plot_spout_height':  True,
  'plot_spout_pouring_projection':  True,
  
  'plot_all_trajectories_singlePlot':  False,
  'plot_all_startingConditions_singlePlot':  False,
  
  'interactively_animate_trajectories_exampleType': None,
  'save_trajectory_animations_eachType': False,
  'save_trajectory_animations_compositeTypes': False,
  
  'plot_body_speedJerk':  False,
  'plot_joint_angles':  False,
  'plot_compare_distribution_body_speedJerk':  False,
  'plot_compare_distribution_spout_speedJerk':  True, # includes Wasserstein distances
  'plot_compare_distribution_joint_angles':  False,
  'plot_compare_distribution_spout_projection':  False,
  'plot_compare_distribution_spout_height':  False,
  'plot_compare_distribution_spout_tilt':  False,
  'plot_distributions_hand_to_pitcher_angles':  False,
  
  'keep_plots_open':  False,
}
evaluation_config_json = json.dumps(evaluation_config)
evaluation_config_json = evaluation_config_json.replace('"', '\\"')
os.system('evaluate_trajectories.py "%s"' % evaluation_config_json)



