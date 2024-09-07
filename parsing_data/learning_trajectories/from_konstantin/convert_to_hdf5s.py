
import numpy as np
import h5py
import os
script_dir = os.path.dirname(os.path.realpath(__file__))
actionsense_root_dir = script_dir
while os.path.split(actionsense_root_dir)[-1] != 'ActionSense':
  actionsense_root_dir = os.path.realpath(os.path.join(actionsense_root_dir, '..'))

#################################################
# Specify the output folder.
output_dir = os.path.realpath(os.path.join(actionsense_root_dir, 'results', 'learning_trajectories',
                                         'from_konstantin',
                                         # '2024-09-06_13-35'
                                         # '2024-09-06_17-25'
                                         # '2024-09-06_19-02'
                                         '2024-09-07_09-58'
                                           ))
os.makedirs(output_dir, exist_ok=True)

#################################################
# Read and parse the data.
# print(init_values.shape) ## shape: batch_size x dim, where dim dimension is set up as: [initial_hand_position,initial_hand_quaternion,glass_position,final_hand_position]
# print(trajectories.shape) ## shape: batch_size x steps x dim, where dim dimension is set up as: [hand_positions,hand_quaternions]
init_values = np.load(os.path.join(output_dir, 'initial_vals_LOSS.npy'))
trajectories = np.load(os.path.join(output_dir, 'trajectories_LOSS.npy'))

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
# Denormalize
mins_byFrame = {'hand_location': np.array([-0.53807845, -0.3380863 ,  0.04596044]), 'object_location': np.array([-0.53807845, -0.3380863 ,  0.04596044]), 'hand_location_polar': np.array([ 0.24093589, -3.14143613,  0.75810699]), 'object_location_polar': np.array([ 0.24093589, -3.14143613,  0.75810699])}
maxs_byFrame = {'hand_location': np.array([-0.15750682,  0.37952101,  0.4150433 ]), 'object_location': np.array([-0.15750682,  0.37952101,  0.4150433 ]), 'hand_location_polar': np.array([0.63575751, 3.14148258, 1.46840939]), 'object_location_polar': np.array([0.63575751, 3.14148258, 1.46840939])}
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
time_s_pred = [np.linspace(0, 7.7, num_timesteps)[:,None] for trial_index in range(num_trials)]

#################################################
# Save the output data.
h5file = h5py.File(os.path.join(output_dir, 'data_to_evaluate.hdf5'), 'w')
h5file.create_dataset('hand_position_m', data=model_outputs['hand_position'])
h5file.create_dataset('hand_quaternion_wijk', data=model_outputs['hand_quaternion'])
h5file.create_dataset('referenceObject_position_m', data=model_inputs['glass_position'])
h5file.create_dataset('time_s', data=time_s_pred)
truth_group = h5file.create_group('truth')
truth_group.create_dataset('referenceObject_position_m', data=model_inputs['glass_position'])
truth_group.create_dataset('starting_hand_position_m', data=model_inputs['hand_position'])
# truth_group.create_dataset('hand_quaternion_wijk', data=)
h5file.close()
