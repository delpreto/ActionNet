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
from torch.utils.data import Dataset, DataLoader
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
noise_dim = 64  # Dimensionality of the noise vector
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
epoch_ratio_add_starting_hand_position = 0.1
epoch_ratio_add_starting_hand_quaternion = 0.1
epoch_ratio_add_starting_reference_object_position = 0.1
epoch_ratio_add_hand_pitcher_angles = 0.1
batch_size = 64
n_critic = 5
use_wgangp = True
g_lambda_reconstruction = 25  # Weight for the reconstruction loss
g_lambda_adversarial = 1  # Weight for the adversarial loss
g_lambda_wgangp = 5
d_lambda_real = 1
d_lambda_fake = 1
d_lambda_gradients = 10  # Weight for the gradient penalty if use_wgangp
real_label = 0.9 # not using 0/1 may make discriminator less confident and get more meaningful outputs
fake_label = 0.1 # not using 0/1 may make discriminator less confident and get more meaningful outputs
dropout_rate_generator = 0.1
dropout_rate_discriminator = 0.1

###################################################################
# Load training data
###################################################################

# Create a class to convert lists of examples to a Pytorch dataset.
class TrajectoryDataset(Dataset):
  def __init__(self,
               hand_position_m_byTrial, hand_quaternion_wijk_byTrial,
               time_s_byTrial,
               reference_object_position_m_byTrial,
               hand_to_pitcher_angles_rad_byTrial):
    self._hand_position_m_byTrial = hand_position_m_byTrial
    self._hand_quaternion_wijk_byTrial = hand_quaternion_wijk_byTrial
    self._time_s_byTrial = time_s_byTrial
    self._hand_to_pitcher_angles_rad_byTrial = hand_to_pitcher_angles_rad_byTrial
    self._reference_object_position_m_byTrial = reference_object_position_m_byTrial
    
    # Normalize.
    def normalize(x_byTrial):
      x = np.array(x_byTrial)
      x = x - np.min(x, axis=tuple(range(0, x.ndim-1)))
      x = x / np.max(x, axis=tuple(range(0, x.ndim-1)))
      return x
    self._hand_position_m_byTrial = normalize(self._hand_position_m_byTrial)
    self._hand_quaternion_wijk_byTrial = normalize(self._hand_quaternion_wijk_byTrial)
    self._time_s_byTrial = normalize(self._time_s_byTrial)
    self._hand_to_pitcher_angles_rad_byTrial = normalize(self._hand_to_pitcher_angles_rad_byTrial)
    self._reference_object_position_m_byTrial = normalize(self._reference_object_position_m_byTrial)
  
  def __len__(self):
    return len(self._time_s_byTrial)
  
  def __getitem__(self, idx):
    hand_position_m = self._hand_position_m_byTrial[idx]
    hand_quaternion_wijk = self._hand_quaternion_wijk_byTrial[idx]
    time_s = self._time_s_byTrial[idx]
    hand_to_pitcher_angles_rad = self._hand_to_pitcher_angles_rad_byTrial[idx]
    reference_object_position_m = self._reference_object_position_m_byTrial[idx]
    starting_hand_position_m = hand_position_m[0, :]
    starting_hand_quaternion_wijk = hand_quaternion_wijk[0, :]
    duration_s = time_s[-1]
    
    trajectory = np.concatenate([hand_position_m, hand_quaternion_wijk], axis=1)
    
    # Convert to torch tensors
    trajectory = torch.tensor(trajectory, dtype=torch.float32)
    hand_to_pitcher_angles_rad = torch.tensor(hand_to_pitcher_angles_rad, dtype=torch.float32)
    reference_object_position_m = torch.tensor(reference_object_position_m, dtype=torch.float32)
    starting_hand_position_m = torch.tensor(starting_hand_position_m, dtype=torch.float32)
    starting_hand_quaternion_wijk = torch.tensor(starting_hand_quaternion_wijk, dtype=torch.float32)
    duration_s = torch.tensor(duration_s, dtype=torch.float32)
    
    return (trajectory, duration_s,
            starting_hand_position_m, starting_hand_quaternion_wijk,
            hand_to_pitcher_angles_rad, reference_object_position_m)

# Load the training data and create a dataset.
print('Loading training data')
hand_position_m_byTrial = []
hand_quaternion_wijk_byTrial = []
time_s_byTrial = []
hand_to_pitcher_angles_rad_byTrial = []
reference_object_position_m_byTrial = []
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
  training_data_file.close()

# Create the dataset
dataset = TrajectoryDataset(
  hand_position_m_byTrial, hand_quaternion_wijk_byTrial,
  time_s_byTrial, reference_object_position_m_byTrial, hand_to_pitcher_angles_rad_byTrial)

# Create the DataLoader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

###################################################################
# Create models
###################################################################

# Generator Network
class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()
    self.fc1s_by_input_dim = []
    for x in range(input_dim+1):
      self.fc1s_by_input_dim.append(nn.Linear(x + noise_dim, hidden_dim))
    self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    self.fc3 = nn.Linear(hidden_dim, hidden_dim)
    # Skip connections
    self.fc_skips_by_input_dim = []
    for x in range(input_dim+1):
      self.fc_skips_by_input_dim.append(nn.Linear(x + noise_dim, hidden_dim))
    # Output layers.
    self.fc_trajectory = nn.Linear(hidden_dim, trajectory_dim)
    self.fc_duration_s = nn.Linear(hidden_dim, duration_dim)
    
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
      self.fc1s_by_condition_dim.append(nn.Linear(trajectory_dim + duration_dim + x, hidden_dim))
    self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    self.fc3 = nn.Linear(hidden_dim, 1)
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
    x = F.relu(self.fc1s_by_condition_dim[current_condition_dim](x))
    if self.dropout is not None:
      x = self.dropout(x)
    x = F.relu(self.fc2(x))
    if self.dropout is not None:
      x = self.dropout(x)
    validity = torch.sigmoid(self.fc3(x)) # validity = self.fc3(x)
    return validity

# The gradient penalty enforces the Lipschitz constraint, which is crucial for WGAN-GP.
def compute_gradient_penalty(D, real_samples, fake_samples, durations_s,
                             starting_hand_position_m_byBatchTrial, starting_hand_quaternion_wijk_byBatchTrial,
                             hand_to_pitcher_angles_rad_byBatchTrial, reference_object_position_m_byBatchTrial):
  alpha = torch.randn(real_samples.size(0), 1, 1)
  alpha = alpha.expand(real_samples.size()).to(real_samples.device)
  
  interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
  d_interpolates = D(interpolates, durations_s,
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

###################################################################
# Train the models
###################################################################

print()
print('Creating the models')

# Instantiate the models
generator = Generator()
discriminator = Discriminator()

# Visualize the models.
# Example inputs
# Forward pass through the generator
sample_inputs = [
  torch.randn(1, input_dims[0]), torch.randn(1, input_dims[1]),
  torch.randn(1, input_dims[2]), torch.randn(1, input_dims[3])
]
sample_generated_trajectory, sample_generated_duration = generator(
  *sample_inputs, torch.randn(1, noise_dim))
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
summary(generator, *sample_inputs, torch.randn(1, noise_dim))
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

# Training loop
discriminator_loss_byEpoch = []
generator_loss_byEpoch = []
print()
print('Training the models')
training_start_time_s = time.time()
last_print_time_s = time.time()
for epoch in range(num_epochs):
  for i, trial_data in enumerate(dataloader):
    (real_trajectories_byBatchTrial, real_durations_s_byBatchTrial,
     starting_hand_position_m_byBatchTrial, starting_hand_quaternion_wijk_byBatchTrial,
     hand_to_pitcher_angles_rad_byBatchTrial, reference_object_position_m_byBatchTrial) = trial_data
    
    current_batch_size = real_trajectories_byBatchTrial.size(0)
    
    # Gradually add conditioning information.
    if epoch/num_epochs < epoch_ratio_add_starting_hand_position:
      starting_hand_position_m_byBatchTrial = None
    if epoch/num_epochs < epoch_ratio_add_starting_hand_quaternion:
      starting_hand_quaternion_wijk_byBatchTrial = None
    if epoch/num_epochs < epoch_ratio_add_hand_pitcher_angles:
      hand_to_pitcher_angles_rad_byBatchTrial = None
    if epoch/num_epochs < epoch_ratio_add_starting_reference_object_position:
      reference_object_position_m_byBatchTrial = None
    
    # Ground truths
    valid = torch.ones(current_batch_size, 1)*real_label
    fake = torch.ones(current_batch_size, 1)*fake_label
    
    # Generate noise vector
    noise_vector = torch.randn(current_batch_size, noise_dim)
    
    # ---------------------
    #  Train Discriminator
    # ---------------------
    optimizer_D.zero_grad()
    
    # Real trajectories
    real_pred = discriminator(real_trajectories_byBatchTrial, real_durations_s_byBatchTrial,
                              starting_hand_position_m_byBatchTrial, starting_hand_quaternion_wijk_byBatchTrial,
                              hand_to_pitcher_angles_rad_byBatchTrial, reference_object_position_m_byBatchTrial)
    # Fake trajectories
    gen_trajectories, gen_durations_s = generator(starting_hand_position_m_byBatchTrial, starting_hand_quaternion_wijk_byBatchTrial,
                                                  hand_to_pitcher_angles_rad_byBatchTrial, reference_object_position_m_byBatchTrial,
                                                  noise_vector)
    fake_pred = discriminator(gen_trajectories, gen_durations_s,
                              starting_hand_position_m_byBatchTrial, starting_hand_quaternion_wijk_byBatchTrial,
                              hand_to_pitcher_angles_rad_byBatchTrial, reference_object_position_m_byBatchTrial)
    # Loss
    if use_wgangp:
      d_real_loss = -torch.mean(valid)
      d_fake_loss =  torch.mean(fake)
      gradient_penalty = compute_gradient_penalty(discriminator, real_trajectories_byBatchTrial,
                                                  gen_trajectories, gen_durations_s,
                                                  starting_hand_position_m_byBatchTrial, starting_hand_quaternion_wijk_byBatchTrial,
                                                  hand_to_pitcher_angles_rad_byBatchTrial, reference_object_position_m_byBatchTrial)
      d_loss = d_lambda_real*d_real_loss + d_lambda_fake*d_fake_loss + d_lambda_gradients*gradient_penalty
    else:
      d_real_loss = adversarial_loss(real_pred, valid)
      d_fake_loss = adversarial_loss(fake_pred, fake)
      d_loss = (d_real_loss + d_fake_loss)/2
      
    d_loss.backward()
    optimizer_D.step()
    
    if i % n_critic == 0:
      # -----------------
      #  Train Generator
      # -----------------
      optimizer_G.zero_grad()
      
      gen_trajectories, gen_durations_s = generator(starting_hand_position_m_byBatchTrial, starting_hand_quaternion_wijk_byBatchTrial,
                                                    hand_to_pitcher_angles_rad_byBatchTrial, reference_object_position_m_byBatchTrial,
                                                    noise_vector)
      fake_pred = discriminator(gen_trajectories, gen_durations_s,
                                starting_hand_position_m_byBatchTrial, starting_hand_quaternion_wijk_byBatchTrial,
                                hand_to_pitcher_angles_rad_byBatchTrial, reference_object_position_m_byBatchTrial)
      g_reconstruction_loss = reconstruction_loss(gen_trajectories, real_trajectories_byBatchTrial)
      if use_wgangp:
        g_wgangp_loss = -torch.mean(fake_pred)
        g_loss = g_lambda_wgangp*g_wgangp_loss + g_lambda_reconstruction*g_reconstruction_loss
      else:
        g_adversarial_loss = adversarial_loss(fake_pred, valid)
        g_loss = g_lambda_adversarial*g_adversarial_loss + g_lambda_reconstruction*g_reconstruction_loss
      
      g_loss.backward()
      optimizer_G.step()
  
  discriminator_loss_byEpoch.append(d_loss.detach().numpy())
  generator_loss_byEpoch.append(g_loss.detach().numpy())
  if time.time() - last_print_time_s > 5 \
      or epoch == 0 or epoch == num_epochs-1:
    print(' Epoch %d/%d (%0.1f%%) | D loss: %0.3f | G loss: %0.3f | Elapsed %0.1fs Remaining %0.1fs' % (
      epoch, num_epochs, 100*(epoch+1)/num_epochs, d_loss.item(), g_loss.item(),
      time.time() - training_start_time_s, (time.time() - training_start_time_s)/(epoch+1)*(num_epochs-epoch-1)
    ))
    # print(f" [Epoch {epoch}/{num_epochs}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
    last_print_time_s = time.time()

# Plot the losses.
plt.figure()
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.plot(discriminator_loss_byEpoch, '*-', label='Discriminator')
plt.plot(generator_loss_byEpoch, '*-', label='Generator')
plt.grid(True, color='lightgray')
plt.title('Training Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

###################################################################
# Using the model
###################################################################

# Example: Generating a trajectory.
generator.eval()

trial_index = 0
trial_data = dataset[trial_index]
(real_trajectory, real_duration_s,
 starting_hand_position_m, starting_hand_quaternion_wijk,
 hand_to_pitcher_angles_rad, reference_object_position_m) = trial_data
noise_vector = torch.randn(1, noise_dim)
gen_trajectory, gen_duration_s = generator(starting_hand_position_m[None, :], starting_hand_quaternion_wijk[None, :],
                                           hand_to_pitcher_angles_rad[None,:], reference_object_position_m[None,:],
                                           noise_vector)
gen_hand_position_m = gen_trajectory[0, :, 0:3].detach().numpy()
gen_hand_quaternion_wijk = gen_trajectory[0, :, 3:7].detach().numpy()
gen_duration_s = gen_duration_s[0].detach().numpy()
real_hand_position_m = real_trajectory[:, 0:3].detach().numpy()
real_hand_quaternion_wijk = real_trajectory[:, 3:7].detach().numpy()
real_duration_s = real_duration_s.detach().numpy()

print()
print(real_duration_s, gen_duration_s)

# Plot the hand path.
num_rows = 1
num_cols = 1
subplot_index = 0
fig = plt.figure()
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
plt.show()

###################################################################
# Clean up
###################################################################

print()
print('Done!')
print()









