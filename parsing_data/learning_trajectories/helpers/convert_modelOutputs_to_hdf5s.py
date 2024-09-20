
import numpy as np
import h5py
import os

def convert_modelOutputs_to_hdf5_data(init_values, trajectories, output_filepath):
  print()
  print('Parsing data')
  model_inputs = {
    'hand_position': init_values[:, 0:3],
    'hand_quaternion': init_values[:, 3:7],
    'glass_position': init_values[:, 7:10],
    'final_hand_position': init_values[:, 10:13],
  }
  model_outputs = {
    'hand_position': trajectories[:, :, 0:3],
    'hand_quaternion': trajectories[:, :, 3:7],
  }
  num_timesteps = model_outputs['hand_position'].shape[1]
  num_trials = model_outputs['hand_position'].shape[0]
  
  #################################################
  # Prepend the initial conditions if needed.
  if num_timesteps == 99:
    print('*** Prepending the initial pose ***')
    new_model_outputs_hand_position = []
    new_model_outputs_hand_quaternion = []
    for trial_index in range(num_trials):
      starting_hand_position = model_inputs['hand_position'][trial_index:trial_index+1, :]
      hand_position = model_outputs['hand_position'][trial_index, :, :]
      hand_position = np.array(starting_hand_position.tolist() + hand_position.tolist())
      new_model_outputs_hand_position.append(hand_position)
  
      starting_hand_quaternion = model_inputs['hand_quaternion'][trial_index:trial_index+1, :]
      hand_quaternion = model_outputs['hand_quaternion'][trial_index, :, :]
      hand_quaternion = np.array(starting_hand_quaternion.tolist() + hand_quaternion.tolist())
      new_model_outputs_hand_quaternion.append(hand_quaternion)
    model_outputs['hand_position'] = np.array(new_model_outputs_hand_position)
    model_outputs['hand_quaternion'] = np.array(new_model_outputs_hand_quaternion)
  
    num_timesteps += 1
  
  #################################################
  # Denormalize
  print('Denormalizing')
  # mins_byFrame = {'hand_location': np.array([-0.53807845, -0.3380863 ,  0.04596044]), 'object_location': np.array([-0.53807845, -0.3380863 ,  0.04596044]), 'hand_location_polar': np.array([ 0.24093589, -3.14143613,  0.75810699]), 'object_location_polar': np.array([ 0.24093589, -3.14143613,  0.75810699])}
  # maxs_byFrame = {'hand_location': np.array([-0.15750682,  0.37952101,  0.4150433 ]), 'object_location': np.array([-0.15750682,  0.37952101,  0.4150433 ]), 'hand_location_polar': np.array([0.63575751, 3.14148258, 1.46840939]), 'object_location_polar': np.array([0.63575751, 3.14148258, 1.46840939])}
  mins_byFrame = {'hand_location': np.array([-0.58123652, -0.3375666 ,  0.14079341]), 'object_location': np.array([-0.58123652, -0.3375666 ,  0.14079341]), 'hand_location_polar': np.array([ 0.30370537, -3.1415501 ,  0.71982327]), 'object_location_polar': np.array([ 0.30370537, -3.1415501 ,  0.71982327])}
  maxs_byFrame = {'hand_location': np.array([-0.15750682,  0.37952101,  0.39201282]), 'object_location': np.array([-0.15750682,  0.37952101,  0.39201282]), 'hand_location_polar': np.array([0.62828742, 3.14153173, 1.31659006]), 'object_location_polar': np.array([0.62828742, 3.14153173, 1.31659006])}

  spatial_mins = mins_byFrame['hand_location'].reshape(1, 1, -1)
  spatial_maxs = maxs_byFrame['hand_location'].reshape(1, 1, -1)
  def denormalize(x, mins, maxs):
    return np.squeeze(x * (maxs - mins) + mins)
  model_inputs['hand_position'] = denormalize(model_inputs['hand_position'], spatial_mins, spatial_maxs)
  model_inputs['glass_position'] = denormalize(model_inputs['glass_position'], spatial_mins, spatial_maxs)
  model_inputs['final_hand_position'] = denormalize(model_inputs['final_hand_position'], spatial_mins, spatial_maxs)
  model_outputs['hand_position'] = denormalize(model_outputs['hand_position'], spatial_mins, spatial_maxs)
  
  #################################################
  # Generate a time vector for each trial.
  print('Generating a time vector')
  time_s_pred = [np.linspace(0, 7.7, num_timesteps)[:,None] for trial_index in range(num_trials)]
  
  #################################################
  # Save the output data.
  print('Saving output data')
  h5file = h5py.File(output_filepath, 'w')
  h5file.create_dataset('hand_position_m', data=model_outputs['hand_position'])
  h5file.create_dataset('hand_quaternion_wijk', data=model_outputs['hand_quaternion'])
  h5file.create_dataset('referenceObject_position_m', data=model_inputs['glass_position'])
  h5file.create_dataset('time_s', data=time_s_pred)
  truth_group = h5file.create_group('truth')
  truth_group.create_dataset('referenceObject_position_m', data=model_inputs['glass_position'])
  truth_group.create_dataset('starting_hand_position_m', data=model_inputs['hand_position'])
  # truth_group.create_dataset('hand_quaternion_wijk', data=)
  h5file.close()
  
  print('Done!')
  print()


if __name__ == '__main__':
  script_dir = os.path.dirname(os.path.realpath(__file__))
  actionsense_root_dir = script_dir
  while os.path.split(actionsense_root_dir)[-1] != 'ActionSense':
    actionsense_root_dir = os.path.realpath(os.path.join(actionsense_root_dir, '..'))
  
  #################################################
  # Specify the output folder.
  data_dir = os.path.realpath(os.path.join(actionsense_root_dir, 'results', 'learning_trajectories',
                                           'models', 'state-space',
                                           # '2024-09-06_13-35'
                                           # '2024-09-06_17-25'
                                           # '2024-09-06_19-02'
                                           # '2024-09-07_09-58'
                                           # '2024-09-10_17-10'
                                           # '2024-09-12_15-08'
                                           # '2024-09-13_08-56'
                                           '2024-09-13_18-15_forSubmission'
                                           ))
  output_dir = data_dir
  output_filepath = os.path.join(output_dir, 'pouring_modelData.hdf5')
  os.makedirs(output_dir, exist_ok=True)
  
  #################################################
  # Read and parse the data.
  # print(init_values.shape) ## shape: batch_size x dim, where dim dimension is set up as: [initial_hand_position,initial_hand_quaternion,glass_position,final_hand_position]
  # print(trajectories.shape) ## shape: batch_size x steps x dim, where dim dimension is set up as: [hand_positions,hand_quaternions]
  init_values = np.load(os.path.join(data_dir, 'initial_vals_LOSS.npy'))
  trajectories = np.load(os.path.join(data_dir, 'trajectories_LOSS.npy'))

  convert_modelOutputs_to_hdf5_data(init_values, trajectories, output_filepath)



