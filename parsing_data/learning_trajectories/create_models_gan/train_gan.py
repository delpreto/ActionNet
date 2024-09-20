############
#
# Copyright (c) 2024 MIT CSAIL and Joseph DelPreto
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
# IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# See https://action-sense.csail.mit.edu for more usage information.
# Created 2021-2024 for the MIT ActionSense project by Joseph DelPreto [https://josephdelpreto.com].
#
############

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from torch.cuda.amp import GradScaler, autocast # automatic mixed precision (AMP) module
from torch.utils.data import TensorDataset, DataLoader
from torchsummaryX import summary
from torchviz import make_dot
import numpy as np
import h5py
import os
import time
script_dir = os.path.dirname(os.path.realpath(__file__))
actionsense_root_dir = script_dir
while os.path.split(actionsense_root_dir)[-1] != 'ActionSense':
  actionsense_root_dir = os.path.realpath(os.path.join(actionsense_root_dir, '..'))
import matplotlib.pyplot as plt

from learning_trajectories.helpers.printing import *
from learning_trajectories.helpers.plot_animations import plt_wait_for_keyboard_press

###################################################################
# Configuration
###################################################################

# Specify the input training data.
subject_ids = ['S00', 'S10', 'S11']
input_dir = os.path.realpath(os.path.join(actionsense_root_dir, 'results', 'learning_trajectories'))

# Define the dimensions.
input_dims = [3,  # starting hand position
              4,  # starting hand quaternion
              3,  # glass position
              3,  # hand-to-pitcher angles
            ]
input_dim = sum(input_dims)
noise_dim = 32  # Dimensionality of the noise vector
hidden_dim = 128
num_timesteps = 100
trajectory_feature_dims = [
                  3, # hand position
                  4, # hand quaternion
                  ]
trajectory_feature_dim = sum(trajectory_feature_dims)
trajectory_dim = num_timesteps * trajectory_feature_dim
condition_dim = input_dim
duration_dim = 1

# Define the optimizers.
learning_rate_discriminator = 0.00020
learning_rate_generator     = 0.00015
b1, b2 = 0.5, 0.9 # for Adam optimizer

# Define the training loop.
num_epochs = 200000
epoch_ratio_add_starting_hand_position = 0
epoch_ratio_add_starting_hand_quaternion = 0
epoch_ratio_add_reference_object_position = 0
epoch_ratio_add_hand_pitcher_angles = 0
batch_size = 8
n_critic = 5
use_wgangp = False
g_lambda_reconstruction = 3 # Weight for the reconstruction loss
g_lambda_reconstruction_keyPoints = 0  # Weight for the reconstruction loss at key points
g_lambda_reconstruction_duration = 0  # Weight for the duration loss
g_lambda_adversarial = 1 # Weight for the adversarial loss if not using WGAN-GP
g_lambda_wgangp = 1
d_lambda_real = 1
d_lambda_fake = 1
d_lambda_gradients = 2  # Weight for the gradient penalty if use_wgangp
real_label = 0.9 # not using 0/1 may make discriminator less confident and get more meaningful outputs
fake_label = 0.1 # not using 0/1 may make discriminator less confident and get more meaningful outputs
dropout_rate_generator = 0.1
dropout_rate_discriminator = 0.1

# Specify the output directory if desired.
output_dir = os.path.join(input_dir, 'model_training',
                          '%s_epochs%d_batch%d_starts%d_noise%d_hidden%d_wgangp%d' % (
                            get_time_str(format='%Y-%m-%d_%H-%M-%S'),
                            num_epochs, batch_size, epoch_ratio_add_starting_hand_position,
                            noise_dim, hidden_dim, int(use_wgangp)))
if output_dir is not None:
  os.makedirs(output_dir, exist_ok=True)
  os.makedirs(os.path.join(output_dir, 'model_checkpoints'), exist_ok=True)
  config_file = open(os.path.join(output_dir, '_configuration.txt'), 'w')
  config_file.write(get_dict_str(locals()))
  config_file.close()

# Use GPU if available.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

###################################################################
# Helpers
###################################################################

# # Create a class to convert lists of examples to a Pytorch dataset.
# class TrajectoryDataset(Dataset):
#   def __init__(self,
#                hand_position_m_byTrial, hand_quaternion_wijk_byTrial,
#                time_s_byTrial,
#                reference_object_position_m_byTrial,
#                hand_to_pitcher_angles_rad_byTrial):
#     self._hand_position_m_byTrial = hand_position_m_byTrial
#     self._hand_quaternion_wijk_byTrial = hand_quaternion_wijk_byTrial
#     self._time_s_byTrial = time_s_byTrial
#     self._hand_to_pitcher_angles_rad_byTrial = hand_to_pitcher_angles_rad_byTrial
#     self._reference_object_position_m_byTrial = reference_object_position_m_byTrial
#
#     # Normalize.
#     def normalize(x_byTrial):
#       x = np.array(x_byTrial)
#       x = x - np.min(x, axis=tuple(range(0, x.ndim-1)))
#       x = x / np.max(x, axis=tuple(range(0, x.ndim-1)))
#       return x
#     self._hand_position_m_byTrial = normalize(self._hand_position_m_byTrial)
#     self._hand_quaternion_wijk_byTrial = normalize(self._hand_quaternion_wijk_byTrial)
#     self._time_s_byTrial = normalize(self._time_s_byTrial)
#     self._hand_to_pitcher_angles_rad_byTrial = normalize(self._hand_to_pitcher_angles_rad_byTrial)
#     self._reference_object_position_m_byTrial = normalize(self._reference_object_position_m_byTrial)
#
#   def __len__(self):
#     return len(self._time_s_byTrial)
#
#   def __getitem__(self, idx):
#     hand_position_m = self._hand_position_m_byTrial[idx]
#     hand_quaternion_wijk = self._hand_quaternion_wijk_byTrial[idx]
#     time_s = self._time_s_byTrial[idx]
#     hand_to_pitcher_angles_rad = self._hand_to_pitcher_angles_rad_byTrial[idx]
#     reference_object_position_m = self._reference_object_position_m_byTrial[idx]
#     starting_hand_position_m = hand_position_m[0, :]
#     starting_hand_quaternion_wijk = hand_quaternion_wijk[0, :]
#     duration_s = time_s[-1]
#
#     trajectory = np.concatenate([hand_position_m, hand_quaternion_wijk], axis=1)
#
#     # Convert to torch tensors
#     trajectory = torch.tensor(trajectory, dtype=torch.float32)
#     hand_to_pitcher_angles_rad = torch.tensor(hand_to_pitcher_angles_rad, dtype=torch.float32)
#     reference_object_position_m = torch.tensor(reference_object_position_m, dtype=torch.float32)
#     starting_hand_position_m = torch.tensor(starting_hand_position_m, dtype=torch.float32)
#     starting_hand_quaternion_wijk = torch.tensor(starting_hand_quaternion_wijk, dtype=torch.float32)
#     duration_s = torch.tensor(duration_s, dtype=torch.float32)
#
#     return (trajectory, duration_s,
#             starting_hand_position_m, starting_hand_quaternion_wijk,
#             hand_to_pitcher_angles_rad, reference_object_position_m)

# Generator Network
class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()
    self.fc1s_by_input_dim = []
    for x in range(input_dim+1):
      self.fc1s_by_input_dim.append(nn.Linear(x + noise_dim, hidden_dim, device=device))
    self.fc2 = nn.Linear(hidden_dim, hidden_dim, device=device)
    self.fc3 = nn.Linear(hidden_dim, hidden_dim, device=device)
    # Skip connections
    self.fc_skips_by_input_dim = []
    for x in range(input_dim+1):
      self.fc_skips_by_input_dim.append(nn.Linear(x + noise_dim, hidden_dim, device=device))
    # Output layers.
    self.fc_trajectory = nn.Linear(hidden_dim, trajectory_dim, device=device)
    self.fc_duration_s = nn.Linear(hidden_dim, duration_dim, device=device)
    
    if dropout_rate_generator is not None:
      self.dropout = nn.Dropout(dropout_rate_generator)  # Dropout with a 30% rate
    else:
      self.dropout = None
  
  def forward(self, starting_hand_position_m=None,
              starting_hand_quaternion_wijk=None,
              hand_to_pitcher_angles_rad=None,
              reference_object_position_m=None,
              noise_vector=None):
    # Concatenate inputs and noise vector
    to_cat = [starting_hand_position_m, starting_hand_quaternion_wijk,
              hand_to_pitcher_angles_rad, reference_object_position_m,
              noise_vector]
    to_cat = [x for x in to_cat if x is not None]
    x = torch.cat(to_cat, dim=1)
    current_input_dim = x.size(1) - noise_dim
    skip = self.fc_skips_by_input_dim[current_input_dim](x)
    x = F.relu(self.fc1s_by_input_dim[current_input_dim](x))
    if self.dropout is not None:
      x = self.dropout(x)
    x = F.relu(self.fc2(x) + skip)
    if self.dropout is not None:
      x = self.dropout(x)
    x = F.relu(self.fc3(x))
    trajectory = self.fc_trajectory(x)
    duration_s = self.fc_duration_s(x)
    return trajectory.view(-1, num_timesteps, trajectory_feature_dim), duration_s.view(-1)

# Discriminator Network
class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.fc1s_by_condition_dim = []
    for x in range(condition_dim+1):
      self.fc1s_by_condition_dim.append(nn.Linear(trajectory_dim + duration_dim + x, hidden_dim, device=device))
    self.fc2 = nn.Linear(hidden_dim, hidden_dim, device=device)
    self.fc3 = nn.Linear(hidden_dim, 1, device=device)
    if dropout_rate_discriminator is not None:
      self.dropout = nn.Dropout(dropout_rate_discriminator)
    else:
      self.dropout = None
  
  def forward(self, trajectory, duration_s,
              starting_hand_position_m, starting_hand_quaternion_wijk,
              hand_to_pitcher_angles_rad, reference_object_position_m):
    to_cat = [trajectory.view(trajectory.size(0), -1),
              duration_s.view(duration_s.size(0), -1),
              starting_hand_position_m, starting_hand_quaternion_wijk,
              hand_to_pitcher_angles_rad,
              reference_object_position_m]
    to_cat = [x for x in to_cat if x is not None]
    x = torch.cat(to_cat, dim=1)
    current_condition_dim = x.size(1) - trajectory_dim - duration_dim
    # print('trajectory', trajectory.dtype)
    # print('duration_s', duration_s.dtype)
    # print('starting_hand_position_m', starting_hand_position_m.dtype if starting_hand_position_m is not None else None)
    # print('starting_hand_quaternion_wijk', starting_hand_quaternion_wijk.dtype if starting_hand_quaternion_wijk is not None else None)
    # print('hand_to_pitcher_angles_rad', hand_to_pitcher_angles_rad.dtype if hand_to_pitcher_angles_rad is not None else None)
    # print('reference_object_position_m', reference_object_position_m.dtype if reference_object_position_m is not None else None)
    x = F.relu(self.fc1s_by_condition_dim[current_condition_dim](x))
    if self.dropout is not None:
      x = self.dropout(x)
    x = F.relu(self.fc2(x))
    if self.dropout is not None:
      x = self.dropout(x)
    validity = torch.sigmoid(self.fc3(x)) # validity = self.fc3(x)
    return validity

# The gradient penalty enforces the Lipschitz constraint, which is crucial for WGAN-GP.
def compute_gradient_penalty(discriminator, real_samples, fake_samples, durations_s,
                             starting_hand_position_m_byBatchTrial, starting_hand_quaternion_wijk_byBatchTrial,
                             hand_to_pitcher_angles_rad_byBatchTrial, reference_object_position_m_byBatchTrial):
  alpha = torch.randn(real_samples.size(0), 1, 1)
  alpha = alpha.expand(real_samples.size()).to(real_samples.device)
  
  interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
  d_interpolates = discriminator(interpolates, durations_s,
                                 starting_hand_position_m_byBatchTrial, starting_hand_quaternion_wijk_byBatchTrial,
                                 hand_to_pitcher_angles_rad_byBatchTrial, reference_object_position_m_byBatchTrial)
  
  fake = torch.ones(d_interpolates.size()).to(real_samples.device)
  gradients = torch.autograd.grad(
    outputs=d_interpolates,
    inputs=interpolates,
    grad_outputs=fake,
    create_graph=True,
    retain_graph=True,
    only_inputs=True,
  )[0]
  
  gradients = gradients.view(gradients.size(0), -1)
  gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
  return gradient_penalty

# A custom loss function for trajectories.
def reconstruction_loss_keyPoints(gen_trajectories, real_trajectories,
                                  gen_durations_s, real_durations_s,
                                  starting_hand_positions_m, starting_hand_quaternions_wijk,
                                  hand_to_pitcher_angles_rads, reference_object_positions_m,
                                  stationary_period_indexes):
  loss_stationary = ((gen_trajectories[np.arange(gen_trajectories.size(0)), stationary_period_indexes, :]
                      - real_trajectories[np.arange(real_trajectories.size(0)), stationary_period_indexes, :]
                      ) ** 2).mean()
  loss_start = ((gen_trajectories[:, 0, :] - real_trajectories[:, 0, :])**2).mean()
  # loss_end = ((gen_trajectories[:, -1, :] - real_trajectories[:, -1, :])**2).mean()
  
  # loss = nn.functional.mse_loss(gen_trajectories, real_trajectories)
  return loss_stationary + loss_start

# A custom loss function for durations.
def reconstruction_loss_duration(gen_durations_s, real_durations_s):
  return ((real_durations_s - gen_durations_s)**2).mean()

###################################################################
# Start main code.
# Wrapped in __main__ in case we want DataLoader to use multiple workers.
#  But it seemed like doing so actually slows down training a lot.
###################################################################

if __name__ == '__main__':
  
  ###################################################################
  # Load training data
  ###################################################################
  
  # Load the training data and create a dataset.
  print('Loading training data')
  hand_position_m_byTrial = []
  hand_quaternion_wijk_byTrial = []
  time_s_byTrial = []
  hand_to_pitcher_angles_rad_byTrial = []
  reference_object_position_m_byTrial = []
  stationary_period_index_byTrial = []
  subject_ids_byTrial = []
  for subject_id in subject_ids:
    input_filepath = os.path.join(input_dir, 'pouring_trainingData_%s.hdf5' % subject_id)
    print(' Loading file %s' % os.path.basename(input_filepath))
    training_data_file = h5py.File(input_filepath, 'r')
    for trial_index in range(training_data_file['time_s'].shape[0]):
      subject_ids_byTrial.append(subject_id)
      hand_position_m_byTrial.append(np.array(training_data_file['hand_position_m'][trial_index, :, :]))
      hand_quaternion_wijk_byTrial.append(np.array(training_data_file['hand_quaternion_wijk'][trial_index, :, :]))
      time_s_byTrial.append(np.array(training_data_file['time_s'][trial_index, :, :]))
      hand_to_pitcher_angles_rad_byTrial.append(np.squeeze(training_data_file['hand_to_pitcher_angles_rad'][trial_index, :, :]))
      reference_object_position_m_byTrial.append(np.squeeze(training_data_file['referenceObject_position_m'][trial_index, :, :]))
      stationary_period_index_byTrial.append(np.squeeze(training_data_file['stationary_index'][trial_index, :, :]))
    training_data_file.close()
  
  # Convert from lists to numpy arrays.
  hand_position_m_byTrial = np.array(hand_position_m_byTrial)
  hand_quaternion_wijk_byTrial = np.array(hand_quaternion_wijk_byTrial)
  time_s_byTrial = np.array(time_s_byTrial)
  hand_to_pitcher_angles_rad_byTrial = np.array(hand_to_pitcher_angles_rad_byTrial)
  reference_object_position_m_byTrial = np.array(reference_object_position_m_byTrial)
  stationary_period_index_byTrial = np.array(stationary_period_index_byTrial)
  
  # Normalize.
  print('Normalizing training data')
  def normalize(x_byTrial, min_value=None, max_value=None):
    x = np.array(x_byTrial)
    if min_value is None:
      min_value = np.min(x, axis=tuple(range(0, x.ndim-1)))
    if max_value is None:
      max_value = np.max(x, axis=tuple(range(0, x.ndim-1)))
    # Remove the offset.
    x -= min_value
    # Scale to [0, 2]
    x /= (max_value - min_value)/2
    # Shift to [-1, 1]
    x -= 1
    # if min_value is None:
    #   x = x - np.min(x, axis=tuple(range(0, x.ndim-1)))
    # else:
    #   x = x - min_value
    # if max_value is None:
    #   x = x / np.max(x, axis=tuple(range(0, x.ndim-1)))
    # else:
    #   if min_value is None:
    #     x = x / max_value
    #   else:
    #     x = x / (max_value - min_value)
    return x
  # Use the same normalization parameters for all spatial inputs.
  # Independently normalize each coordinate axis.
  min_position_m = np.min(np.stack([np.min(x, axis=tuple(range(0, x.ndim-1))) for x in [
    hand_position_m_byTrial, reference_object_position_m_byTrial
  ]]), axis=0)
  max_position_m = np.max(np.stack([np.max(x, axis=tuple(range(0, x.ndim-1))) for x in [
    hand_position_m_byTrial, reference_object_position_m_byTrial
  ]]), axis=0)
  print('  Using min/max position coordinates (%s, %s) cm' % (100*min_position_m, 100*max_position_m))
  hand_position_m_byTrial = normalize(hand_position_m_byTrial, min_position_m, max_position_m)
  reference_object_position_m_byTrial = normalize(reference_object_position_m_byTrial, min_position_m, max_position_m)
  # Normalize time by the maximum duration.
  min_time_s = np.min(time_s_byTrial)
  max_time_s = np.max(time_s_byTrial)
  print('  Using min/max time values (%s, %s) s' % (min_time_s, max_time_s))
  time_s_byTrial = normalize(time_s_byTrial, min_time_s, max_time_s)
  # Normalize pitcher holding angles by the min/max range.
  min_hand_to_pitcher_angle_rad = np.min(hand_to_pitcher_angles_rad_byTrial, axis=tuple(range(0, hand_to_pitcher_angles_rad_byTrial.ndim-1)))
  max_hand_to_pitcher_angle_rad = np.max(hand_to_pitcher_angles_rad_byTrial, axis=tuple(range(0, hand_to_pitcher_angles_rad_byTrial.ndim-1)))
  print('  Using min/max hand-to-pitcher angles (%s, %s) deg' % (np.degrees(min_hand_to_pitcher_angle_rad), np.degrees(max_hand_to_pitcher_angle_rad)))
  hand_to_pitcher_angles_rad_byTrial = normalize(hand_to_pitcher_angles_rad_byTrial)
  # Quaternions should already be close to normalized.
  print('  Quaternion min/max are unchanged at (%s, %s)' % (np.min(hand_quaternion_wijk_byTrial, axis=tuple(range(0, hand_quaternion_wijk_byTrial.ndim-1))), np.max(hand_quaternion_wijk_byTrial, axis=tuple(range(0, hand_quaternion_wijk_byTrial.ndim-1)))))
  
  # Extract single-timestep metrics.
  starting_hand_position_m_byTrial = [hand_position_m[0, :] for hand_position_m in hand_position_m_byTrial]
  starting_hand_quaternion_wijk_byTrial = [hand_quaternion_wijk[0, :] for hand_quaternion_wijk in hand_quaternion_wijk_byTrial]
  duration_s_byTrial = [time_s[-1] for time_s in time_s_byTrial]
  
  # Concatenate to form trajectory matrices.
  print('Creating a dataset and data loader on the %s' % str(device).upper())
  trajectory_byTrial = []
  for trial_index in range(len(time_s_byTrial)):
    trajectory_byTrial.append(np.concatenate([hand_position_m_byTrial[trial_index],
                                              hand_quaternion_wijk_byTrial[trial_index]], axis=1))
  
  # Convert all data to Torch tensors and move them to the GPU/CPU now instead of during training.
  trajectory_byTrial = torch.stack([torch.tensor(x, dtype=torch.float32) for x in trajectory_byTrial]).to(device)
  starting_hand_position_m_byTrial = torch.stack([torch.tensor(x, dtype=torch.float32) for x in starting_hand_position_m_byTrial]).to(device)
  starting_hand_quaternion_wijk_byTrial = torch.stack([torch.tensor(x, dtype=torch.float32) for x in starting_hand_quaternion_wijk_byTrial]).to(device)
  duration_s_byTrial = torch.stack([torch.tensor(x, dtype=torch.float32) for x in duration_s_byTrial]).to(device)
  hand_to_pitcher_angles_rad_byTrial = torch.stack([torch.tensor(x, dtype=torch.float32) for x in hand_to_pitcher_angles_rad_byTrial]).to(device)
  reference_object_position_m_byTrial = torch.stack([torch.tensor(x, dtype=torch.float32) for x in reference_object_position_m_byTrial]).to(device)
  stationary_period_index_byTrial = torch.stack([torch.tensor(x, dtype=torch.int) for x in stationary_period_index_byTrial]).to(device)
  
  # Create a dataset and data loader.
  dataset = TensorDataset(trajectory_byTrial, duration_s_byTrial,
                          starting_hand_position_m_byTrial, starting_hand_quaternion_wijk_byTrial,
                          hand_to_pitcher_angles_rad_byTrial, reference_object_position_m_byTrial,
                          stationary_period_index_byTrial)
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                          num_workers=0, pin_memory=False)
  
  ###################################################################
  # Train the models
  ###################################################################
  
  print()
  print('Creating the models')
  
  # Instantiate the models
  generator = Generator().to(device)
  discriminator = Discriminator().to(device)
  
  # Visualize the models.
  # Example inputs
  # Forward pass through the generator
  sample_inputs = [
    torch.randn(1, input_dims[0]), torch.randn(1, input_dims[1]),
    torch.randn(1, input_dims[2]), torch.randn(1, input_dims[3])
  ]
  sample_inputs = [x.to(device) for x in sample_inputs]
  sample_noise = torch.randn(1, noise_dim).to(device)
  sample_generated_trajectory, sample_generated_duration = generator(
    *sample_inputs, sample_noise)
  # Visualize the computation graph
  dot = make_dot(sample_generated_trajectory, params=dict(list(generator.named_parameters())))
  dot.format = 'png'
  dot.render('generator')
  # For discriminator
  validity = discriminator(sample_generated_trajectory, sample_generated_duration, *sample_inputs)
  dot = make_dot(validity, params=dict(list(discriminator.named_parameters())))
  dot.format = 'png'
  dot.render('discriminator')
  
  # Print a summary of the network architecture.
  print('Generator Summary:')
  summary(generator, *sample_inputs, sample_noise)
  time.sleep(0.1)
  print()
  print('Discriminator Summary:')
  summary(discriminator, sample_generated_trajectory, sample_generated_duration, *sample_inputs)
  time.sleep(0.1)
  
  # Optimizers.
  # Using AdamW (weight decay) rather than Adam.
  optimizer_G = optim.AdamW(generator.parameters(), lr=learning_rate_generator, betas=(b1, b2))
  optimizer_D = optim.AdamW(discriminator.parameters(), lr=learning_rate_discriminator, betas=(b1, b2))
  
  # Loss functions
  adversarial_loss = nn.BCEWithLogitsLoss() # nn.BCELoss()
  reconstruction_loss = nn.MSELoss() # torch.nn.L1Loss()
  
  # # Initialize a gradient scaler for automatic mixed precision (AMP)
  # scaler = GradScaler()
  
  # Training loop
  discriminator_loss_byEpoch = []
  generator_loss_byEpoch = []
  print()
  print('Training the models using %s' % str(device).upper())
  training_start_time_s = time.time()
  last_print_time_s = time.time()
  last_checkpoint_time_s = time.time()
  num_rows = 1
  num_cols = 1
  subplot_index = 0
  fig = plt.figure()
  figManager = plt.get_current_fig_manager()
  figManager.window.showMaximized()
  plt_wait_for_keyboard_press(0.2)
  fig.add_subplot(num_rows, num_cols, subplot_index+1, projection='3d')
  ax = fig.get_axes()[subplot_index]
  is_first_plot = True
  for epoch in range(num_epochs):
    for i, trial_data in enumerate(dataloader):
      (real_trajectories_byBatchTrial, real_durations_s_byBatchTrial,
       starting_hand_position_m_byBatchTrial, starting_hand_quaternion_wijk_byBatchTrial,
       hand_to_pitcher_angles_rad_byBatchTrial, reference_object_position_m_byBatchTrial,
       stationary_period_index_byBatchTrial) = trial_data
      current_batch_size = real_trajectories_byBatchTrial.size(0)
      
      # Ground truths
      valid = torch.ones(current_batch_size, 1)*real_label
      fake = torch.ones(current_batch_size, 1)*fake_label
      
      # Generate noise vector
      noise_vector = torch.randn(current_batch_size, noise_dim)
      
      # Move data to CPU/GPU as needed.
      # real_trajectories_byBatchTrial = real_trajectories_byBatchTrial.to(device)
      # real_durations_s_byBatchTrial = real_durations_s_byBatchTrial.to(device)
      # starting_hand_position_m_byBatchTrial = starting_hand_position_m_byBatchTrial.to(device)
      # starting_hand_quaternion_wijk_byBatchTrial = starting_hand_quaternion_wijk_byBatchTrial.to(device)
      # hand_to_pitcher_angles_rad_byBatchTrial = hand_to_pitcher_angles_rad_byBatchTrial.to(device)
      # reference_object_position_m_byBatchTrial = reference_object_position_m_byBatchTrial.to(device)
      # stationary_period_index_byBatchTrial = stationary_period_index_byBatchTrial.to(device)
      noise_vector = noise_vector.to(device)
      valid = valid.to(device)
      fake = fake.to(device)
    
      # Gradually add conditioning information.
      if epoch/num_epochs < epoch_ratio_add_starting_hand_position:
        starting_hand_position_m_byBatchTrial = None
      if epoch/num_epochs < epoch_ratio_add_starting_hand_quaternion:
        starting_hand_quaternion_wijk_byBatchTrial = None
      if epoch/num_epochs < epoch_ratio_add_hand_pitcher_angles:
        hand_to_pitcher_angles_rad_byBatchTrial = None
      if epoch/num_epochs < epoch_ratio_add_reference_object_position:
        reference_object_position_m_byBatchTrial = None
        stationary_period_index_byBatchTrial = None
      
      # Generate trajectories.
      # Will use these results in both the discriminator and generator training below.
      # with autocast(): # use automatic mixed precision
      gen_trajectories, gen_durations_s = generator(starting_hand_position_m_byBatchTrial, starting_hand_quaternion_wijk_byBatchTrial,
                                                    hand_to_pitcher_angles_rad_byBatchTrial, reference_object_position_m_byBatchTrial,
                                                    noise_vector)
      
      # ---------------------
      #  Train Discriminator
      # ---------------------
      
      real_pred = discriminator(real_trajectories_byBatchTrial, real_durations_s_byBatchTrial,
                                starting_hand_position_m_byBatchTrial, starting_hand_quaternion_wijk_byBatchTrial,
                                hand_to_pitcher_angles_rad_byBatchTrial, reference_object_position_m_byBatchTrial)
      fake_pred = discriminator(gen_trajectories.detach(), gen_durations_s.detach(),
                                starting_hand_position_m_byBatchTrial, starting_hand_quaternion_wijk_byBatchTrial,
                                hand_to_pitcher_angles_rad_byBatchTrial, reference_object_position_m_byBatchTrial)
      if use_wgangp:
        d_real_loss = -torch.mean(real_pred)
        d_fake_loss =  torch.mean(fake_pred)
        gradient_penalty = compute_gradient_penalty(discriminator, real_trajectories_byBatchTrial,
                                                    gen_trajectories.detach(), gen_durations_s.detach(),
                                                    starting_hand_position_m_byBatchTrial, starting_hand_quaternion_wijk_byBatchTrial,
                                                    hand_to_pitcher_angles_rad_byBatchTrial, reference_object_position_m_byBatchTrial)
        d_loss = d_lambda_real*d_real_loss + d_lambda_fake*d_fake_loss + d_lambda_gradients*gradient_penalty
      else:
        d_real_loss = adversarial_loss(real_pred, valid)
        d_fake_loss = adversarial_loss(fake_pred, fake)
        d_loss = (d_real_loss + d_fake_loss)/2
      optimizer_D.zero_grad()
      d_loss.backward() # scaler.scale(d_loss).backward()
      optimizer_D.step() # scaler.step(optimizer_D) #
      # scaler.update()
      
      # -----------------
      #  Train Generator
      # -----------------
      if i % n_critic == 0:
        # with autocast():
        fake_pred = discriminator(gen_trajectories, gen_durations_s,
                                  starting_hand_position_m_byBatchTrial, starting_hand_quaternion_wijk_byBatchTrial,
                                  hand_to_pitcher_angles_rad_byBatchTrial, reference_object_position_m_byBatchTrial)
        g_reconstruction_loss = reconstruction_loss(gen_trajectories, real_trajectories_byBatchTrial)
        g_reconstruction_loss_keyPoints = reconstruction_loss_keyPoints(
          gen_trajectories, real_trajectories_byBatchTrial,
          gen_durations_s, real_durations_s_byBatchTrial,
          starting_hand_position_m_byBatchTrial, starting_hand_quaternion_wijk_byBatchTrial,
          hand_to_pitcher_angles_rad_byBatchTrial, reference_object_position_m_byBatchTrial,
          stationary_period_index_byBatchTrial)
        g_reconstruction_loss_duration = reconstruction_loss_duration(
          gen_durations_s, real_durations_s_byBatchTrial)
        g_loss = g_lambda_reconstruction*g_reconstruction_loss
        g_loss += g_lambda_reconstruction_keyPoints*g_reconstruction_loss_keyPoints
        g_loss += g_lambda_reconstruction_duration*g_reconstruction_loss_duration
        if use_wgangp:
          g_wgangp_loss = -torch.mean(fake_pred)
          g_loss += g_lambda_wgangp*g_wgangp_loss
        else:
          g_adversarial_loss = adversarial_loss(fake_pred, valid)
          g_loss += g_lambda_adversarial*g_adversarial_loss
        optimizer_G.zero_grad()
        g_loss.backward() # scaler.scale(g_loss).backward()
        optimizer_G.step() # scaler.step(optimizer_G)
        # scaler.update()
    
    # Store the latest losses.
    discriminator_loss_byEpoch.append(d_loss.detach().cpu().numpy())
    generator_loss_byEpoch.append(g_loss.detach().cpu().numpy())
    
    # Print and plot if appropriate.
    if time.time() - last_print_time_s > 5 \
        or epoch == 0 or epoch == num_epochs-1:
      print(' Epoch %d/%d (%0.1f%%) | D loss: %0.3f | G loss: %0.3f | Elapsed: %0.1fs  Remaining: %0.1fs  Epochs/sec: %0.1f' % (
        epoch, num_epochs, 100*(epoch+1)/num_epochs, d_loss.item(), g_loss.item(),
        time.time() - training_start_time_s, (time.time() - training_start_time_s)/(epoch+1)*(num_epochs-(epoch+1)),
        (epoch+1)/(time.time() - training_start_time_s)
      ))
      last_print_time_s = time.time()
      # Plot the latest hand path.
      gen_hand_position_m = gen_trajectories[0, :, 0:3].detach().cpu().numpy()
      gen_hand_quaternion_wijk = gen_trajectories[0, :, 3:7].detach().cpu().numpy()
      gen_duration = gen_durations_s[0].detach().cpu().numpy()
      real_hand_position_m = real_trajectories_byBatchTrial[0, :, 0:3].detach().cpu().numpy()
      real_hand_quaternion_wijk = real_trajectories_byBatchTrial[0, :, 3:7].detach().cpu().numpy()
      real_duration = real_durations_s_byBatchTrial[0, 0].detach().cpu().numpy()
      ax.clear()
      ax.plot3D(gen_hand_position_m[:, 0], gen_hand_position_m[:, 1], gen_hand_position_m[:, 2], alpha=0.8)
      ax.plot3D(real_hand_position_m[:, 0], real_hand_position_m[:, 1], real_hand_position_m[:, 2], alpha=0.8)
      if is_first_plot:
        ax.set_xlabel('X [cm]')
        ax.set_ylabel('Y [cm]')
        ax.set_zlabel('Z [cm]')
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        z_lim = ax.get_zlim()
        ax.set_box_aspect([ub - lb for lb, ub in (x_lim, y_lim, z_lim)])
        is_first_plot = False
      plt.show(block=False)
      plt.draw()
      plt_wait_for_keyboard_press(0.05)
    
    # Save the model if desired.
    if time.time() - last_checkpoint_time_s > 300 or epoch == 0 or epoch == num_epochs-1:
      if output_dir is not None:
        torch.save(discriminator, os.path.join(output_dir, 'model_checkpoints',
                                               'discriminator_epoch%06d.pth' % (epoch+1)))
        torch.save(generator, os.path.join(output_dir, 'model_checkpoints',
                                               'generator_epoch_epoch%06d.pth' % (epoch+1)))
        torch.save(discriminator.state_dict(), os.path.join(output_dir, 'model_checkpoints',
                                               'discriminator_state_epoch%06d.pth' % (epoch+1)))
        torch.save(generator.state_dict(), os.path.join(output_dir, 'model_checkpoints',
                                               'generator_state_epoch%06d.pth' % (epoch+1)))
        last_checkpoint_time_s = time.time()
        if fig is not None:
          plt.savefig(os.path.join(output_dir, 'model_checkpoints',
                                   'sample_trajectory_epoch%06d.jpg' % (epoch+1)), dpi=300)
    
  # Plot the losses.
  plt.figure()
  figManager = plt.get_current_fig_manager()
  figManager.window.showMaximized()
  plt_wait_for_keyboard_press(0.2)
  plt.plot(discriminator_loss_byEpoch, '*-', label='Discriminator')
  plt.plot(generator_loss_byEpoch, '*-', label='Generator')
  plt.grid(True, color='lightgray')
  plt.title('Training Losses')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()
  if output_dir is not None:
    plt.savefig(os.path.join(output_dir, 'training_losses.jpg'), dpi=300)
  
  ###################################################################
  # Using the model
  ###################################################################
  
  # Put the generator in evaluation mode.
  generator.eval()
  
  num_trials = len(dataset)
  num_rows = 3
  num_cols = 3
  num_trials_to_test = num_rows*num_cols
  start_trial_index = 0
  end_trial_index = min(num_trials-1, start_trial_index + num_trials_to_test - 1)
  print('Generating trajectories for all known trials')
  real_durations_s = []
  gen_durations_s = []
  while start_trial_index < num_trials-2:
    subplot_index = 0
    fig = plt.figure()
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt_wait_for_keyboard_press(0.2)
    trial_indexes = np.arange(start_trial_index, end_trial_index+1)
    for trial_index in trial_indexes:
      trial_data = dataset[trial_index]
      (real_trajectory, real_duration,
       starting_hand_position_m, starting_hand_quaternion_wijk,
       hand_to_pitcher_angles_rad, reference_object_position_m,
       stationary_period_index) = trial_data
      noise_vector = torch.randn(1, noise_dim).to(device)
      gen_trajectory, gen_duration = generator(starting_hand_position_m[None, :], starting_hand_quaternion_wijk[None, :],
                                                 hand_to_pitcher_angles_rad[None,:], reference_object_position_m[None,:],
                                               noise_vector)
      gen_hand_position_m = gen_trajectory[0, :, 0:3].detach().cpu().numpy()
      gen_hand_quaternion_wijk = gen_trajectory[0, :, 3:7].detach().cpu().numpy()
      gen_duration = gen_duration[0].detach().cpu().numpy()
      real_hand_position_m = real_trajectory[:, 0:3].detach().cpu().numpy()
      real_hand_quaternion_wijk = real_trajectory[:, 3:7].detach().cpu().numpy()
      real_duration = real_duration.detach().cpu().numpy()
      
      # Store the durations.
      real_durations_s.append((real_duration+1)*((max_time_s - min_time_s)/2) + min_time_s)
      gen_durations_s.append((gen_duration+1)*((max_time_s - min_time_s)/2) + min_time_s)
      
      # Plot the hand path.
      fig.add_subplot(num_rows, num_cols, subplot_index+1, projection='3d')
      ax = fig.get_axes()[subplot_index]
      ax.plot3D(gen_hand_position_m[:, 0], gen_hand_position_m[:, 1], gen_hand_position_m[:, 2], alpha=0.8)
      ax.plot3D(real_hand_position_m[:, 0], real_hand_position_m[:, 1], real_hand_position_m[:, 2], alpha=0.8)
      ax.set_xlabel('X [cm]')
      ax.set_ylabel('Y [cm]')
      ax.set_zlabel('Z [cm]')
      x_lim = ax.get_xlim()
      y_lim = ax.get_ylim()
      z_lim = ax.get_zlim()
      ax.set_box_aspect([ub - lb for lb, ub in (x_lim, y_lim, z_lim)])
      subplot_index += 1
    if output_dir is not None:
      plt.savefig(os.path.join(output_dir, 'sample_trajectories_indexes_%03d-%03d.jpg' % (
        start_trial_index, end_trial_index
      )), dpi=300)
    start_trial_index = end_trial_index+1
    end_trial_index = min(num_trials-1, start_trial_index + num_trials_to_test - 1)
  
  # Plot the durations
  plt.figure()
  figManager = plt.get_current_fig_manager()
  figManager.window.showMaximized()
  plt_wait_for_keyboard_press(0.2)
  plt.plot(real_durations_s, '*-', label='Real')
  plt.plot(gen_durations_s, '*-', label='Generated')
  plt.grid(True, color='lightgray')
  plt.title('Trajectory Durations')
  plt.xlabel('Trial Index')
  plt.ylabel('Duration [s]')
  plt.legend()
  if output_dir is not None:
    plt.savefig(os.path.join(output_dir, 'sample_durations.jpg'), dpi=300)
    
  plt.figure()
  figManager = plt.get_current_fig_manager()
  figManager.window.showMaximized()
  plt_wait_for_keyboard_press(0.2)
  plt.plot(sorted(real_durations_s), '*-', label='Real')
  plt.plot(sorted(gen_durations_s), '*-', label='Generated')
  plt.grid(True, color='lightgray')
  plt.title('Trajectory Durations Sorted')
  plt.xlabel('Index')
  plt.ylabel('Duration [s]')
  plt.legend()
  if output_dir is not None:
    plt.savefig(os.path.join(output_dir, 'sample_durations_sorted.jpg'), dpi=300)
  
  # Show the plots.
  print('Close the plots to exit')
  plt.show()
  
  ###################################################################
  # Clean up
  ###################################################################
  
  print()
  print('Done!')
  print()
  
  
  
  
  
  
  
  
  