
import collections
import os
script_dir = os.path.dirname(os.path.realpath(__file__))
actionsense_root_dir = script_dir
while os.path.split(actionsense_root_dir)[-1] != 'ActionSense':
  actionsense_root_dir = os.path.realpath(os.path.join(actionsense_root_dir, '..'))

import h5py
import numpy as np
import pdb

import torch
from torch.utils.data import Dataset
from glob import glob


def to_tensor(x, dtype=torch.float, device='cpu'):
  return torch.tensor(x, dtype=dtype, device=device)


class PouringDataset(Dataset):

  conditions = [
    ([], 1), ## none
    # ([0], 1), ## first
    # ([-1], 1), ## last
    # ([0,-1], 1), ## first and last
  ]

  def __init__(self, horizon_length,
               features_include_hand_position=True,
               features_include_hand_quaternion=True,
               features_include_elbow_position=True,
               features_include_shoulder_position=True,
               features_include_wrist_joint_angles=True,
               features_include_elbow_joint_angles=True,
               features_include_shoulder_joint_angles=True,
               ):
    # Load the data.
    subject_ids = ['S00', 'S10', 'S11']
    input_dir = os.path.realpath(os.path.join(actionsense_root_dir, 'results', 'learning_trajectories'))
    print('Loading training data')
    hand_position_m_byTrial = []
    hand_quaternion_wijk_byTrial = []
    wrist_joint_angle_xyz_rad_byTrial = []
    elbow_joint_angle_xyz_rad_byTrial = []
    shoulder_joint_angle_xyz_rad_byTrial = []
    elbow_position_m_byTrial = []
    shoulder_position_m_byTrial = []
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
        wrist_joint_angle_xyz_rad_byTrial.append(np.array(training_data_file['wrist_joint_angle_xyz_rad'][trial_index, :, :]))
        elbow_joint_angle_xyz_rad_byTrial.append(np.array(training_data_file['elbow_joint_angle_xyz_rad'][trial_index, :, :]))
        shoulder_joint_angle_xyz_rad_byTrial.append(np.array(training_data_file['shoulder_joint_angle_xyz_rad'][trial_index, :, :]))
        elbow_position_m_byTrial.append(np.array(training_data_file['elbow_position_m'][trial_index, :, :]))
        shoulder_position_m_byTrial.append(np.array(training_data_file['shoulder_position_m'][trial_index, :, :]))
        time_s_byTrial.append(np.array(training_data_file['time_s'][trial_index, :, :]))
        hand_to_pitcher_angles_rad_byTrial.append(np.squeeze(training_data_file['hand_to_pitcher_angles_rad'][trial_index, :, :]))
        reference_object_position_m_byTrial.append(np.squeeze(training_data_file['referenceObject_position_m'][trial_index, :, :]))
        stationary_period_index_byTrial.append(np.squeeze(training_data_file['stationary_index'][trial_index, :, :]))
      training_data_file.close()
    num_trials = len(subject_ids_byTrial)
    num_trajectory_dimensions = 0 \
                                + features_include_hand_position*hand_position_m_byTrial[0].shape[-1] \
                                + features_include_hand_quaternion*hand_quaternion_wijk_byTrial[0].shape[-1] \
                                + features_include_wrist_joint_angles*wrist_joint_angle_xyz_rad_byTrial[0].shape[-1] \
                                + features_include_elbow_joint_angles*elbow_joint_angle_xyz_rad_byTrial[0].shape[-1] \
                                + features_include_shoulder_joint_angles*shoulder_joint_angle_xyz_rad_byTrial[0].shape[-1] \
                                + features_include_elbow_position*elbow_position_m_byTrial[0].shape[-1] \
                                + features_include_shoulder_position*shoulder_position_m_byTrial[0].shape[-1]
    num_timesteps = hand_position_m_byTrial[0].shape[0]
    
    # Convert from lists to numpy arrays.
    hand_position_m_byTrial = np.array(hand_position_m_byTrial)
    hand_quaternion_wijk_byTrial = np.array(hand_quaternion_wijk_byTrial)
    wrist_joint_angle_xyz_rad_byTrial = np.array(wrist_joint_angle_xyz_rad_byTrial)
    elbow_joint_angle_xyz_rad_byTrial = np.array(elbow_joint_angle_xyz_rad_byTrial)
    shoulder_joint_angle_xyz_rad_byTrial = np.array(shoulder_joint_angle_xyz_rad_byTrial)
    elbow_position_m_byTrial = np.array(elbow_position_m_byTrial)
    shoulder_position_m_byTrial = np.array(shoulder_position_m_byTrial)
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
      return x
    # Use the same normalization parameters for all spatial inputs.
    # Independently normalize each coordinate axis.
    min_position_m = np.min(np.stack([np.min(x, axis=tuple(range(0, x.ndim-1))) for x in [
      hand_position_m_byTrial, reference_object_position_m_byTrial,
      elbow_position_m_byTrial, shoulder_position_m_byTrial
    ]]), axis=0)
    max_position_m = np.max(np.stack([np.max(x, axis=tuple(range(0, x.ndim-1))) for x in [
      hand_position_m_byTrial, reference_object_position_m_byTrial,
      elbow_position_m_byTrial, shoulder_position_m_byTrial
    ]]), axis=0)
    print('  Using min/max position coordinates (%s, %s) cm' % (100*min_position_m, 100*max_position_m))
    hand_position_m_byTrial = normalize(hand_position_m_byTrial, min_position_m, max_position_m)
    elbow_position_m_byTrial = normalize(elbow_position_m_byTrial, min_position_m, max_position_m)
    shoulder_position_m_byTrial = normalize(shoulder_position_m_byTrial, min_position_m, max_position_m)
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
    # Normalize joint angles.
    print('  Using min/max wrist joint angles %s deg' % np.degrees([(np.min(wrist_joint_angle_xyz_rad_byTrial, axis=tuple(range(0, wrist_joint_angle_xyz_rad_byTrial.ndim-1))), np.max(wrist_joint_angle_xyz_rad_byTrial, axis=tuple(range(0, wrist_joint_angle_xyz_rad_byTrial.ndim-1))))]))
    wrist_joint_angle_xyz_rad_byTrial = normalize(wrist_joint_angle_xyz_rad_byTrial)
    print('  Using min/max elbow joint angles %s deg' % np.degrees([(np.min(elbow_joint_angle_xyz_rad_byTrial, axis=tuple(range(0, elbow_joint_angle_xyz_rad_byTrial.ndim-1))), np.max(elbow_joint_angle_xyz_rad_byTrial, axis=tuple(range(0, elbow_joint_angle_xyz_rad_byTrial.ndim-1))))]))
    elbow_joint_angle_xyz_rad_byTrial = normalize(elbow_joint_angle_xyz_rad_byTrial)
    print('  Using min/max shoulder joint angles %s deg' % np.degrees([(np.min(shoulder_joint_angle_xyz_rad_byTrial, axis=tuple(range(0, shoulder_joint_angle_xyz_rad_byTrial.ndim-1))), np.max(shoulder_joint_angle_xyz_rad_byTrial, axis=tuple(range(0, shoulder_joint_angle_xyz_rad_byTrial.ndim-1))))]))
    shoulder_joint_angle_xyz_rad_byTrial = normalize(shoulder_joint_angle_xyz_rad_byTrial)
    
    # # Extract single-timestep metrics.
    # starting_hand_position_m_byTrial = [hand_position_m[0, :] for hand_position_m in hand_position_m_byTrial]
    # starting_hand_quaternion_wijk_byTrial = [hand_quaternion_wijk[0, :] for hand_quaternion_wijk in hand_quaternion_wijk_byTrial]
    # duration_s_byTrial = [time_s[-1] for time_s in time_s_byTrial]
    
    # Create state vectors from trial data.
    q_states = np.zeros((num_trials, num_timesteps, num_trajectory_dimensions))
    path_lengths = np.zeros(num_trials, dtype=int)
    for trial_index in range(num_trials):
      to_concatenate = []
      if features_include_hand_position:
        to_concatenate.append(hand_position_m_byTrial[trial_index])
      if features_include_hand_quaternion:
        to_concatenate.append(hand_quaternion_wijk_byTrial[trial_index])
      if features_include_elbow_position:
        to_concatenate.append(elbow_position_m_byTrial[trial_index])
      if features_include_shoulder_position:
        to_concatenate.append(shoulder_position_m_byTrial[trial_index])
      if features_include_wrist_joint_angles:
        to_concatenate.append(wrist_joint_angle_xyz_rad_byTrial[trial_index])
      if features_include_elbow_joint_angles:
        to_concatenate.append(elbow_joint_angle_xyz_rad_byTrial[trial_index])
      if features_include_shoulder_joint_angles:
        to_concatenate.append(shoulder_joint_angle_xyz_rad_byTrial[trial_index])
      q_state = np.concatenate(to_concatenate, axis=1)
      path_length = q_state.shape[0]
      q_states[trial_index, 0:path_length, :] = q_state
      path_lengths[trial_index] = path_length

    # Create indexes for horizon windows.
    horizon_indexes = []
    for (trial_index, path_length) in enumerate(path_lengths):
      for start in range(path_length - horizon_length + 1):
        end = start + horizon_length
        horizon_indexes.append((trial_index, start, end))
    horizon_indexes = np.array(horizon_indexes)
    
    # Format conditioning indexes
    conditions_indexes, conditions_probabilities = zip(*self.conditions)
    self._conditions_indexes = np.array(conditions_indexes, dtype=object)
    self._conditions_probabilities = np.array(conditions_probabilities) / sum(conditions_probabilities)
    
    # Store state.
    self._num_trials = num_trials
    self._num_trajectory_dimensions = num_trajectory_dimensions
    self._q_states = q_states
    self._path_lengths = path_lengths
    self._num_timesteps = num_timesteps
    self._horizon_indexes = horizon_indexes
    
  def __len__(self):
    return self._num_trials

  def __getitem__(self, window_index):
    # Select the desired window from the desired example.
    path_index, horizon_start_index, horizon_end_index = self._horizon_indexes[window_index]
    q_states = self._q_states[path_index, horizon_start_index:horizon_end_index, :]
    q_states = to_tensor(q_states)
    
    # Create a mask based on conditioned elements if desired.
    mask = torch.zeros_like(q_states[..., -1])
    if self._conditions_indexes.size > 0:
      condition_indexes = np.random.choice(self._conditions_indexes, p=self._conditions_probabilities)
      for condition_index in condition_indexes:
        mask[condition_index] = 1
    
    return (q_states, mask)




















