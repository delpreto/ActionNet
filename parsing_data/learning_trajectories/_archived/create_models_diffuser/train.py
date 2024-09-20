
print('Starting the script!')

import numpy as np
import torch
import pdb
import os
import time
import json
import copy
from datetime import datetime
script_dir = os.path.dirname(os.path.realpath(__file__))
actionsense_root_dir = script_dir
while os.path.split(actionsense_root_dir)[-1] != 'ActionSense':
  actionsense_root_dir = os.path.realpath(os.path.join(actionsense_root_dir, '..'))

from denoising_diffusion_pytorch.datasets.actionsense_pouring import PouringDataset
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from denoising_diffusion_pytorch.mixer import MixerUnet
# from denoising_diffusion_pytorch.temporal import TemporalMixerUnet
from denoising_diffusion_pytorch.temporal_attention import TemporalUnet

from diffusion.models.mlp import TimeConditionedMLP
from diffusion.models import Config

import matplotlib
import matplotlib.pyplot as plt

from learning_trajectories.helpers.printing import *
from learning_trajectories.helpers.plot_animations import plt_wait_for_keyboard_press

print('Finished imports')
import sys
print()
print()
python_version = sys.version
print('Python version:', python_version)
print()
print()
cuda_is_available = torch.cuda.is_available()
cuda_device_count = torch.cuda.device_count()
cuda_device_names = [torch.cuda.get_device_name(device_index) for device_index in range(torch.cuda.device_count())]
print('CUDA is available?', cuda_is_available)
print('Device count:', cuda_device_count)
for (device_index, device_name) in enumerate(cuda_device_names):
  print('Device %d: ' % device_index, device_name)
print()
print()


num_epochs = 500000
num_diffusion_steps = 2000
epoch_index_to_test = num_epochs-1
horizon_length = 128
train_batch_size = 64
first_layer_size = 64
layer_size_mults = (1, 2, 4, 8)
loss_type = 'l2'

dataset_kwargs = {
   'horizon_length': horizon_length,
   'features_include_hand_position': True,
   'features_include_hand_quaternion': True,
   'features_include_elbow_position': True,
   'features_include_shoulder_position': True,
   'features_include_wrist_joint_angles': False,
   'features_include_elbow_joint_angles': False,
   'features_include_shoulder_joint_angles': False,
}

diffusion_model_dir = os.path.join(actionsense_root_dir,
                   'results', 'learning_trajectories',
                   'outputs', 'pouring_model_horizon%03d_handPosQuat-elbShdPos_%s' % (
                                     horizon_length,
                                     datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')))
os.makedirs(diffusion_model_dir, exist_ok=True)

local_vars_raw = locals().copy()
local_vars = {}
for (var_name, var_value) in local_vars_raw.items():
  try:
    json.dumps(var_value)
    local_vars[var_name] = var_value
  except:
    pass
config_file = open(os.path.join(diffusion_model_dir, '_configuration.txt'), 'w')
config_file.write(json.dumps(local_vars))
config_file.close()
config_file = open(os.path.join(diffusion_model_dir, '_configuration_str.txt'), 'w')
config_file.write(get_dict_str(local_vars))
config_file.close()

headless = True

if headless:
  print('Trying to switch to headless matplotlib')
  try:
    matplotlib.use('Agg')
  except:
    pass
  
# Load the dataset
print('Loading the dataset')
dataset = PouringDataset(**dataset_kwargs)
num_trajectory_dimensions = dataset._num_trajectory_dimensions
num_trajectory_timesteps = dataset._num_timesteps

# renderer = KukaRenderer()

# Create a model.

# model = Unet(
#   width = H,
#   dim = 32,
#   dim_mults = (1, 2, 4, 8),
#   channels = 2,
#   out_dim = 1,
# ).cuda()

# model = MixerUnet(
#   dim = 32,
#   image_size = (H, num_trajectory_dimensions),
#   dim_mults = (1, 2, 4, 8),
#   channels = 2,
#   out_dim = 1,
# ).cuda()

# model = MixerUnet(
#   horizon = H,
#   transition_dim = num_trajectory_dimensions,
#   cond_dim = H,
#   dim = 32,
#   dim_mults = (1, 2, 4, 8),
# ).cuda()

print('Creating the model')
model = TemporalUnet(
  horizon = horizon_length,
  transition_dim = num_trajectory_dimensions,
  cond_dim = horizon_length,
  dim = first_layer_size, # size of the first layer
  dim_mults = layer_size_mults, # subsequent layer sizes will be multiplied by these factors
).cuda()

diffusion = GaussianDiffusion(
  model,
  channels = 1,
  image_size = (horizon_length, num_trajectory_dimensions),
  timesteps = num_diffusion_steps,   # number of diffusion steps
  loss_type = loss_type  # L1 or L2
).cuda()

#### test
print('Testing forward')
x = dataset[0][0].view(1, horizon_length, num_trajectory_dimensions).cuda()
mask = torch.zeros(1, horizon_length).cuda()

loss = diffusion(x, mask)
loss.backward()
print('done')
# pdb.set_trace()
####

trainer = Trainer(
  diffusion,
  dataset,
  renderer=None,
  train_batch_size = train_batch_size,
  train_lr = 2e-5,
  train_num_steps = num_epochs,     # total training steps
  gradient_accumulate_every = 2,  # gradient accumulation steps
  ema_decay = 0.995,        # exponential moving average decay
  fp16 = False,           # turn on mixed precision training with apex
  results_folder = diffusion_model_dir,
  save_every = max(1, num_epochs//20),
  save_last = True,
  sample_every = None,
)

print('Training the model')
trainer.train()

# Sample the model.
print('Loading model from epoch index %d' % epoch_index_to_test)
trainer.load(epoch_index_to_test)
trainer.ema_model.eval()
# trainer.model.eval()

# Conditions is (timestep, state at timestep (unused), desired state at timestep)
for gen_index in range(50):
  msg = 'Generating trajectory %d' % gen_index
  fout = open(os.path.join(diffusion_model_dir, '_log.txt'), 'a')
  fout.write(msg)
  fout.write('\n')
  fout.close()
  print()
  print(msg)
  first_state = trainer.ds[0][0][0, :] # will be size 3
  first_state = first_state[None, :] # make it 2D to be size (1, 3)
  first_state = first_state[None, :] # add the batch dimension to be size (1, 1, 3)
  conditions = [
    (0,  None, first_state), ## first state conditioning
  ]
  trajectory_samples = []
  for start_timestep_index in range(0, num_trajectory_timesteps, horizon_length):
    print('Generating window starting with timestep index %d' % start_timestep_index)
    window_samples = trainer.ema_model.conditional_sample(batch_size=1, conditions=conditions)
    trajectory_samples.append(np.squeeze(window_samples.cpu().numpy()[0,:]))
    last_state = window_samples[0, -1, :] # will be size 3
    last_state = last_state[None, :] # make it 2D to be size (1, 3)
    last_state = last_state[None, :] # add the batch dimension to be size (1, 1, 3)
    conditions = [(0, None, last_state)]
  trajectory_samples = np.concatenate(trajectory_samples, axis=0)
  trajectory_samples = trajectory_samples[0:num_trajectory_timesteps, :]
  
  generated_hand_position_m = trajectory_samples[:, 0:3]
  time_str = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
  gen_filepath = os.path.join(diffusion_model_dir, 'sampled_trajectory_%02d_%s.npy' % (gen_index, time_str))
  np.save(gen_filepath, trajectory_samples)
  
  fig = plt.figure()
  if not headless:
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
  fig.add_subplot(1, 1, 0+1, projection='3d')
  ax = fig.get_axes()[0]
  if not headless:
    plt_wait_for_keyboard_press(0.2)
  ax.plot3D(generated_hand_position_m[:, 0], generated_hand_position_m[:, 1], generated_hand_position_m[:, 2], alpha=0.8)
  ax.set_xlabel('X [cm]')
  ax.set_ylabel('Y [cm]')
  ax.set_zlabel('Z [cm]')
  x_lim = ax.get_xlim()
  y_lim = ax.get_ylim()
  z_lim = ax.get_zlim()
  ax.set_box_aspect([ub - lb for lb, ub in (x_lim, y_lim, z_lim)])
  if not headless:
    plt_wait_for_keyboard_press(0.2)
  fig.savefig(gen_filepath.replace('.npy', '.png'), dpi=300)
  plt.close(fig)
  
# Show the plots.
if not headless:
  print('Close the plots to exit')
  plt.show()

# # Load the model weights
# diffusion.load_state_dict(torch.load(os.path.join(diffusion_model_dir, 'model-)))



# print('Loading model from epoch index %d' % epoch_index_to_test)
# model_checkpoint_filepath = os.path.join(trainer.results_folder, 'model_epoch-%04d.pt' % epoch_index_to_test)
#
# hidden_dims = [128, 128, 128]
# config = Config(
#     model_class=TimeConditionedMLP,
#     time_dim=128,
#     input_dim=num_trajectory_dimensions,
#     hidden_dims=hidden_dims,
#     output_dim=12,
#     savepath="",
# )
# device = torch.device('cuda')
# model = config.make()
# model.to(device)
# model_checkpoint = torch.load(model_checkpoint_filepath)
# model.load_state_dict(model_checkpoint)
#
# # Conditions is (timestep, state at timestep (unused), desired state at timestep)
# conditions = [
#   (0,  None, trainer.ds[0][0][None, :]), ## first state conditioning
# ]
# trajectory_samples = []
# for start_timestep_index in range(0, num_trajectory_timesteps, horizon_length):
#   print('Generating window starting with timestep index %d' % start_timestep_index)
#   window_samples = model.conditional_sample(batch_size=1, conditions=conditions)
#   trajectory_samples.append(np.squeeze(window_samples.cpu().numpy()[0,:]))
#   conditions = [(0, None, window_samples)]
# trajectory_samples = np.concatenate(trajectory_samples, axis=0)
# trajectory_samples = trajectory_samples[0:num_trajectory_timesteps, :]
#
# generated_hand_position_m = trajectory_samples[:, 0:3]
#
# fig = plt.figure()
# figManager = plt.get_current_fig_manager()
# figManager.window.showMaximized()
# fig.add_subplot(1, 1, 0+1, projection='3d')
# ax = fig.get_axes()[0]
# plt_wait_for_keyboard_press(0.2)
# ax.plot3D(generated_hand_position_m[:, 0], generated_hand_position_m[:, 1], generated_hand_position_m[:, 2], alpha=0.8)
# ax.set_xlabel('X [cm]')
# ax.set_ylabel('Y [cm]')
# ax.set_zlabel('Z [cm]')
# x_lim = ax.get_xlim()
# y_lim = ax.get_ylim()
# z_lim = ax.get_zlim()
# ax.set_box_aspect([ub - lb for lb, ub in (x_lim, y_lim, z_lim)])
#
# # Show the plots.
# print('Close the plots to exit')
# plt.show()






