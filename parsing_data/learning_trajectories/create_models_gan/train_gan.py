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
import numpy as np
import h5py
import os
script_dir = os.path.dirname(os.path.realpath(__file__))
actionsense_root_dir = script_dir
while os.path.split(actionsense_root_dir)[-1] != 'ActionSense':
  actionsense_root_dir = os.path.realpath(os.path.join(actionsense_root_dir, '..'))

###################################################################
# Configuration
###################################################################

# Specify the input training data.
subject_ids = ['S00', 'S10', 'S11']
input_dir = os.path.realpath(os.path.join(actionsense_root_dir, 'results', 'learning_trajectories'))

# Define the dimensions.
input_dim = (3    # starting hand position
             + 4  # starting hand quaternion
             + 3  # starting glass position
             + 3  # hand-to-pitcher angles
             )
noise_dim = 64  # Dimensionality of the noise vector
hidden_dim = 64
num_timesteps = 100
trajectory_feature_dim = (
                  3   # hand position
                  + 4 # hand quaternion
                  + 1 # time
                  )
trajectory_dim = num_timesteps * trajectory_feature_dim
condition_dim = input_dim

# Define the optimizers.
lr = 0.0002
b1, b2 = 0.5, 0.999

# Define the training loop.
num_epochs = 1000
batch_size = 32

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
    
    trajectory = np.concatenate([hand_position_m, hand_quaternion_wijk, time_s], axis=1)
    
    # Convert to torch tensors
    trajectory = torch.tensor(trajectory, dtype=torch.float32)
    hand_to_pitcher_angles_rad = torch.tensor(hand_to_pitcher_angles_rad, dtype=torch.float32)
    reference_object_position_m = torch.tensor(reference_object_position_m, dtype=torch.float32)
    starting_hand_position_m = torch.tensor(starting_hand_position_m, dtype=torch.float32)
    starting_hand_quaternion_wijk = torch.tensor(starting_hand_quaternion_wijk, dtype=torch.float32)
    
    return (trajectory,
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
    self.fc1 = nn.Linear(input_dim + noise_dim, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    self.fc3 = nn.Linear(hidden_dim, trajectory_dim)
  
  def forward(self, starting_hand_position_m,
              starting_hand_quaternion_wijk,
              hand_to_pitcher_angles_rad,
              reference_object_position_m,
              noise_vector):
    # Concatenate start, end positions and noise vector
    x = torch.cat([starting_hand_position_m, starting_hand_quaternion_wijk,
                   hand_to_pitcher_angles_rad, reference_object_position_m,
                   noise_vector], dim=1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    trajectory = self.fc3(x)
    return trajectory.view(-1, num_timesteps, trajectory_feature_dim) # Reshape to sequence of N feature vectors

# Discriminator Network
class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.fc1 = nn.Linear(trajectory_dim + condition_dim, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    self.fc3 = nn.Linear(hidden_dim, 1)
  
  def forward(self, trajectory,
              starting_hand_position_m, starting_hand_quaternion_wijk,
              hand_to_pitcher_angles_rad, reference_object_position_m):
    # Flatten trajectory and concatenate with start, end positions
    x = torch.cat([trajectory.view(trajectory.size(0), -1),
                   starting_hand_position_m, starting_hand_quaternion_wijk,
                   hand_to_pitcher_angles_rad,
                   reference_object_position_m], dim=1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    validity = torch.sigmoid(self.fc3(x))
    return validity

###################################################################
# Train the models
###################################################################

# Instantiate the models
generator = Generator()
discriminator = Discriminator()

optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

# Loss function
adversarial_loss = nn.BCELoss()

# Training loop
for epoch in range(num_epochs):
  for i, trial_data in enumerate(dataloader):
    (real_trajectories_byBatchTrial,
     starting_hand_position_m_byBatchTrial, starting_hand_quaternion_wijk_byBatchTrial,
     hand_to_pitcher_angles_rad_byBatchTrial, reference_object_position_m_byBatchTrial) = trial_data
    
    current_batch_size = real_trajectories_byBatchTrial.size(0)
    
    # Ground truths
    valid = torch.ones(current_batch_size, 1)
    fake = torch.zeros(current_batch_size, 1)
    
    # Generate noise vector
    noise_vector = torch.randn(current_batch_size, noise_dim)
    
    # ---------------------
    #  Train Discriminator
    # ---------------------
    optimizer_D.zero_grad()
    
    # Real trajectories
    real_pred = discriminator(real_trajectories_byBatchTrial,
                              starting_hand_position_m_byBatchTrial, starting_hand_quaternion_wijk_byBatchTrial,
                              hand_to_pitcher_angles_rad_byBatchTrial, reference_object_position_m_byBatchTrial)
    d_real_loss = adversarial_loss(real_pred, valid)
    
    # Fake trajectories
    gen_trajectories = generator(starting_hand_position_m_byBatchTrial, starting_hand_quaternion_wijk_byBatchTrial,
                                 hand_to_pitcher_angles_rad_byBatchTrial, reference_object_position_m_byBatchTrial,
                                 noise_vector)
    fake_pred = discriminator(gen_trajectories,
                              starting_hand_position_m_byBatchTrial, starting_hand_quaternion_wijk_byBatchTrial,
                              hand_to_pitcher_angles_rad_byBatchTrial, reference_object_position_m_byBatchTrial)
    d_fake_loss = adversarial_loss(fake_pred, fake)
    
    # Total discriminator loss
    d_loss = (d_real_loss + d_fake_loss) / 2
    d_loss.backward()
    optimizer_D.step()
    
    # -----------------
    #  Train Generator
    # -----------------
    optimizer_G.zero_grad()
    
    # Generate trajectories
    gen_trajectories = generator(starting_hand_position_m_byBatchTrial, starting_hand_quaternion_wijk_byBatchTrial,
                                 hand_to_pitcher_angles_rad_byBatchTrial, reference_object_position_m_byBatchTrial,
                                 noise_vector)
    fake_pred = discriminator(gen_trajectories,
                              starting_hand_position_m_byBatchTrial, starting_hand_quaternion_wijk_byBatchTrial,
                              hand_to_pitcher_angles_rad_byBatchTrial, reference_object_position_m_byBatchTrial)
    g_loss = adversarial_loss(fake_pred, valid)
    
    g_loss.backward()
    optimizer_G.step()
    
  if epoch % 10 == 0:
    print(f"[Epoch {epoch}/{num_epochs}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")


###################################################################
# Using the model
###################################################################

# Example: Generating a trajectory.
generator.eval()

trial_index = 0
trial_data = dataset[trial_index]
(trajectory,
 starting_hand_position_m, starting_hand_quaternion_wijk,
 hand_to_pitcher_angles_rad, reference_object_position_m) = trial_data
noise_vector = torch.randn(1, noise_dim)
generated_trajectory = generator(starting_hand_position_m[None,:], starting_hand_quaternion_wijk[None,:],
                                 hand_to_pitcher_angles_rad[None,:], reference_object_position_m[None,:],
                                 noise_vector)
hand_position_m = generated_trajectory[0, :, 0:3]
hand_quaternion_wijk = generated_trajectory[0, :, 3:7]
time_s = generated_trajectory[0, :, 7:8]
print(time_s)



###################################################################
# Clean up
###################################################################

print()
print('Done!')
print()






















