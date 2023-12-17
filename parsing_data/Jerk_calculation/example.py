import numpy as np
import h5py
import numpy as np
from collections import OrderedDict
import pickle
import os

directory = os.getcwd()


# Edit the below to use 'pouring' or 'scooping'
Human_Raw_data_filepath = directory + '\pouring_paths_humans.hdf5'

training_data_filepath = directory + '\pouring_training_data.hdf5'
referenceObject_position_filepath = directory + '\pouring_training_referenceObject_positions.hdf5'

# Open the file of labeled feature matrices.
training_data_file = h5py.File(training_data_filepath, 'r')
Human_raw_file = h5py.File(Human_Raw_data_filepath,'r')

feature_matrices =  np.squeeze(np.array(training_data_file['feature_matrices']))

labels = training_data_file['labels']
labels = [label.decode('utf-8') for label in labels]
label_map = OrderedDict(list(enumerate(set(labels))))
label_indexes = [list(label_map.values()).index(label) for label in labels]

num_examples = len(labels)
num_timesteps_per_example = feature_matrices[0].shape[0]
num_features = feature_matrices[0].shape[1]
assert(feature_matrices.shape[0] == num_examples)

print()
print('Loaded training data')
print('  Shape of feature_matrices:', feature_matrices.shape)
print('  Number of examples   :', num_examples)
print('  Timesteps per example:', num_timesteps_per_example)
print('  Feature matrix shape :', feature_matrices[0].shape)
print('  Label breakdown:')
for (label_index, label) in label_map.items():
  print('    %02d: %s (label index %d)' % (len([x for x in labels if x == label]), label, label_index))
print()

# Rename variables for future cells to use
N = num_examples  # number of samples
seq_len = num_timesteps_per_example # length (in timesteps) of each sample
input_dim = num_features # number of numerical features of each sample timestep
num_classes = len(set(labels)) # number of possible output classes

# Example of parsing a feature row:
time_index = 0


for row_index in range(2):
  row_time_features = feature_matrices[row_index, time_index, :]
  hand_position_xyz_cm = row_time_features[0:3]
  elbow_position_xyz_cm = row_time_features[3:6]
  shoulder_position_xyz_cm = row_time_features[6:9]
  hand_quaternion_wxyz = row_time_features[9:13]
  lowerarm_quaternion_wxyz = row_time_features[13:17]
  upperarm_quaternion_wxyz = row_time_features[17:21]
  wrist_angles_xzy_rad = row_time_features[21:24]
  elbow_angles_xzy_rad = row_time_features[24:27]
  shoulder_angles_xzy_rad = row_time_features[27:30]

  print('Sample parsed data using example %d timestep %d:' % (row_index, time_index))
  print('  Label                      : %s' % (labels[row_index]))
  print('  Hand position [xyz]        : (%g, %g, %g) cm'  % tuple(hand_position_xyz_cm))
  print('  Elbow position [xyz]       : (%g, %g, %g) cm'  % tuple(elbow_position_xyz_cm))
  print('  Shoulder position [xyz]    : (%g, %g, %g) cm'  % tuple(shoulder_position_xyz_cm))
  print('  Hand quaternion [wxyz]     : (%g, %g, %g %g)'  % tuple(hand_quaternion_wxyz))
  print('  Lower arm quaternion [wxyz]: (%g, %g, %g %g)'  % tuple(lowerarm_quaternion_wxyz))
  print('  Upper arm quaternion [wxyz]: (%g, %g, %g %g)'  % tuple(upperarm_quaternion_wxyz))
  print('  Wrist angles [xzy]         : (%g, %g, %g) rad (Ulnar Deviation/Radial Deviation, Pronation/Supination, Flexion/Extension)' % tuple(wrist_angles_xzy_rad))
  print('  Elbow angles [xzy]         : (%g, %g, %g) rad (Ulnar Deviation/Radial Deviation, Pronation/Supination, Flexion/Extension)' % tuple(elbow_angles_xzy_rad))
  print('  Shoulder angles [xzy]      : (%g, %g, %g) rad (Abduction/Adduction, Internal/External Rotation, Flexion/Extension)' % tuple(shoulder_angles_xzy_rad))

num_examples_human = 0
num_examples_robot = 0

for row_index in range(num_examples):
  if labels[row_index] == 'robot':
    num_examples_human += 1

  elif labels[row_index] == 'human':
    num_examples_robot += 1
  else:
    print("error")


Time_Stationary = np.zeros((num_examples_human,1))
Pose_Stationary = np.zeros((num_examples_human,3))
human_index = 0
Jerk_Experimental_human = np.zeros(num_examples_human)
Jerk_MinimumJerkTheory = np.zeros(num_examples_human)
MinJerk_Experimental_human = np.zeros(num_examples_human)

## Find Stationary point
for row_index in range(num_examples):
  if labels[row_index] == 'human':
    hand_position_xyz_m = feature_matrices[row_index, :, 0:3]
    Time_Step = hand_position_xyz_m.shape[0]
    Variance = np.zeros((Time_Step,1))

    for Time_index in range(Time_Step - 1):
      Variance[Time_index] = np.linalg.norm(hand_position_xyz_m[Time_index+1,:] - hand_position_xyz_m[Time_index,:])
    Time_Stationary[human_index] = np.where(Variance == np.min(Variance[20:80]))[0]
    T_Station = int(Time_Stationary[human_index])
    Pose_Stationary[human_index,:] = hand_position_xyz_m[T_Station,:]

    hand_pos_xyz_m = hand_position_xyz_m
    hand_vel_xyz_m = np.zeros((T_Station-1,3))
    hand_acc_xyz_m = np.zeros((T_Station-2,3))
    hand_jerk_xyz_m = np.zeros((T_Station-3,3))

    MinJerkPos_xyz_m = np.zeros((T_Station,3))
    Minhand_vel_xyz_m = np.zeros((T_Station-1,3))
    Minhand_acc_xyz_m = np.zeros((T_Station-2,3))
    Minhand_jerk_xyz_m = np.zeros((T_Station-3,3))
    td = 1/T_Station
    for time_index in range (T_Station):
      MinJerkPos_xyz_m[time_index,:] = hand_pos_xyz_m[0,:] + (hand_pos_xyz_m[T_Station,:] - hand_pos_xyz_m[0,:])*(10*(time_index*td)**3-15*(time_index*td)**4+6*(time_index*td)**5)

    for time_index in range (T_Station - 1):
      hand_vel_xyz_m[time_index,:] = hand_position_xyz_m[time_index+1,:] - hand_position_xyz_m[time_index,:]
      Minhand_vel_xyz_m[time_index,:] = MinJerkPos_xyz_m[time_index+1,:] - MinJerkPos_xyz_m[time_index,:]

    for time_index in range (T_Station - 2):
      hand_acc_xyz_m[time_index,:] = hand_vel_xyz_m[time_index+1,:] - hand_vel_xyz_m[time_index,:]
      Minhand_acc_xyz_m[time_index,:] = Minhand_vel_xyz_m[time_index+1,:] - Minhand_vel_xyz_m[time_index,:]

    for time_index in range (T_Station - 3):
      hand_jerk_xyz_m[time_index,:] = hand_acc_xyz_m[time_index+1,:] - hand_acc_xyz_m[time_index,:]
      Minhand_jerk_xyz_m[time_index,:] = Minhand_acc_xyz_m[time_index+1,:] - Minhand_acc_xyz_m[time_index,:]
      Jerk_Experimental_human[human_index] += (hand_jerk_xyz_m[time_index,0])**2+(hand_jerk_xyz_m[time_index,1])**2+(hand_jerk_xyz_m[time_index,2])**2
      Jerk_MinimumJerkTheory[human_index] += (Minhand_jerk_xyz_m[time_index,0])**2+(Minhand_jerk_xyz_m[time_index,1])**2+(Minhand_jerk_xyz_m[time_index,2])**2

    human_index +=1
Time_Stationary_robot = np.zeros((num_examples_robot,1))
Pose_Stationary_robot = np.zeros((num_examples_robot,3))
robot_index = 0
Jerk_Experimental_robot = np.zeros(num_examples_robot)
MinJerk_Experimental_robot = np.zeros(num_examples_robot)

## Find Stationary point
for row_index in range(num_examples):
  if labels[row_index] == 'robot':
    hand_position_xyz_m_robot = feature_matrices[row_index, :, 0:3]
    Time_Step = hand_position_xyz_m_robot.shape[0]
    Variance_robot = np.zeros((Time_Step,1))

    for Time_index in range(Time_Step - 1):
      Variance_robot[Time_index] = np.linalg.norm(hand_position_xyz_m_robot[Time_index+1,:] - hand_position_xyz_m_robot[Time_index,:])
    Variance_temp = np.squeeze(Variance_robot)
    Time_Stationary_robot[robot_index] = np.squeeze(np.where(Variance_temp == np.min(Variance_temp[20:80])))[0]
    T_Station = int(Time_Stationary_robot[robot_index])
    Pose_Stationary_robot[robot_index,:] = hand_position_xyz_m_robot[T_Station,:]

    hand_pos_xyz_m_robot = hand_position_xyz_m_robot
    hand_vel_xyz_m_robot = np.zeros((T_Station-1,3))
    hand_acc_xyz_m_robot = np.zeros((T_Station-2,3))
    hand_jerk_xyz_m_robot = np.zeros((T_Station-3,3))

    td = 1/T_Station

    for time_index in range (T_Station - 1):
      hand_vel_xyz_m_robot[time_index,:] = hand_position_xyz_m_robot[time_index+1,:] - hand_position_xyz_m_robot[time_index,:]

    for time_index in range (T_Station - 2):
      hand_acc_xyz_m_robot[time_index,:] = hand_vel_xyz_m_robot[time_index+1,:] - hand_vel_xyz_m_robot[time_index,:]

    for time_index in range (T_Station - 3):
      hand_jerk_xyz_m_robot[time_index,:] = hand_acc_xyz_m_robot[time_index+1,:] - hand_acc_xyz_m_robot[time_index,:]
      Jerk_Experimental_robot[robot_index] += (hand_jerk_xyz_m_robot[time_index,0])**2+(hand_jerk_xyz_m_robot[time_index,1])**2+(hand_jerk_xyz_m_robot[time_index,2])**2

    robot_index +=1

print("Jerk sum of the Robot experimental data is " f"{Jerk_Experimental_robot.mean()}")
print("Jerk sum of the Human model output data is " f"{Jerk_Experimental_human.mean()}")

# Open Model Data
GroundTruth_filepath = directory + '\ground_truth.pkl'
with open(GroundTruth_filepath, 'rb') as f:
    data = pickle.load(f)

Prediction_filepath = directory + '\predictions.pkl'
# Prediction_filepath = '/content/drive/MyDrive/KitchenData/S00/ground_truth.pkl'

with open(Prediction_filepath, 'rb') as f:
    data_prediction = pickle.load(f)


#Jerk of the Model data
Time_Stationary_model = np.zeros((data_prediction.shape[0],1))
Pose_Stationary_model = np.zeros((data_prediction.shape[0],3))
human_index_model = 0
Jerk_Experimental_model = np.zeros(data_prediction.shape[0])
Jerk_MinimumJerkTheory_model = np.zeros(data_prediction.shape[0])
MinJerk_Experimental_model= np.zeros(data_prediction.shape[0])
T_Station_Max = 0

window_size = 4

window = np.ones(int(window_size))/float(window_size)
## Find Stationary point
Starting_point = 5
row_index_model= 0
hand_position_xyz_m_model = data_prediction[row_index_model,:,0:3]
End_point = hand_position_xyz_m_model.shape[0]-1
for row_index_model in range(len(data_prediction)):
    hand_position_xyz_m_model = data_prediction[row_index_model,:,0:3]
    Time_Step_model = hand_position_xyz_m_model.shape[0]
    Variance_model_raw = np.zeros((Time_Step_model,1))

    for Time_index_model in range(Time_Step_model - 1):
      Variance_model_raw[Time_index_model] = np.linalg.norm(hand_position_xyz_m_model[Time_index_model+1,:] - hand_position_xyz_m_model[Time_index_model,:])
    Variance_model = np.convolve(np.squeeze(Variance_model_raw), window, 'valid')

    # print(f"{Variance_model}")
    Time_Stationary_model[human_index_model] = np.where(Variance_model == np.min(Variance_model[Starting_point:End_point]))[0]
    T_Station_model = int(Time_Stationary_model[human_index_model])
    Pose_Stationary_model[human_index_model,:] = hand_position_xyz_m_model[T_Station_model,:]

    hand_pos_xyz_m_model = hand_position_xyz_m_model
    hand_vel_xyz_m_model = np.zeros((T_Station_model-1,3))
    hand_acc_xyz_m_model = np.zeros((T_Station_model-2,3))
    hand_jerk_xyz_m_model = np.zeros((T_Station_model-3,3))
    T_Station_Max = max(T_Station_model,T_Station_Max)
    td_model = 1/T_Station_model

    for time_index in range (T_Station_model - 1):
      hand_vel_xyz_m_model[time_index,:] = hand_position_xyz_m_model[time_index+1,:] - hand_position_xyz_m_model[time_index,:]

    for time_index in range (T_Station_model - 2):
      hand_acc_xyz_m_model[time_index,:] = hand_vel_xyz_m_model[time_index+1,:] - hand_vel_xyz_m_model[time_index,:]

    for time_index in range (T_Station_model - 3):
      hand_jerk_xyz_m_model[time_index,:] = hand_acc_xyz_m_model[time_index+1,:] - hand_acc_xyz_m_model[time_index,:]
      Jerk_Experimental_model[human_index_model] += (hand_jerk_xyz_m_model[time_index,0])**2+(hand_jerk_xyz_m_model[time_index,1])**2+(hand_jerk_xyz_m_model[time_index,2])**2
    human_index_model +=1

human_index_model = 0
Jerk_experimental_model_Time = np.zeros((data_prediction.shape[0],T_Station_Max))
for row_index_model in range(len(data_prediction)):
      hand_position_xyz_m_model = data_prediction[row_index_model,:,0:3]
      hand_pos_xyz_m_model = hand_position_xyz_m_model
      hand_vel_xyz_m_model = np.zeros((T_Station_model-1,3))
      hand_acc_xyz_m_model = np.zeros((T_Station_model-2,3))
      hand_jerk_xyz_m_model = np.zeros((T_Station_model-3,3))

      for time_index in range (T_Station_model - 1):
        hand_vel_xyz_m_model[time_index,:] = hand_position_xyz_m_model[time_index+1,:] - hand_position_xyz_m_model[time_index,:]

      for time_index in range (T_Station_model - 2):
        hand_acc_xyz_m_model[time_index,:] = hand_vel_xyz_m_model[time_index+1,:] - hand_vel_xyz_m_model[time_index,:]

      for time_index in range (T_Station_model - 3):
        hand_jerk_xyz_m_model[time_index,:] = hand_acc_xyz_m_model[time_index+1,:] - hand_acc_xyz_m_model[time_index,:]
        Jerk_experimental_model_Time[human_index_model,time_index] = (hand_jerk_xyz_m_model[time_index,0])**2+(hand_jerk_xyz_m_model[time_index,1])**2+(hand_jerk_xyz_m_model[time_index,2])**2
      human_index_model +=1
# print(f"{Variance_model}")
# print(f"{Time_Stationary_model}")

print("Jerk sum of the human experimental data is " f"{Jerk_Experimental_model.mean()}")
print(f"{Jerk_experimental_model_Time.shape}")

filename = 'Human_experimental_jerk.pkl'
filename_2 = '2Model_jerk.pkl'
filename_4 = 'Model_jerk_Time.pkl'
# print(f"{type(Jerk_Experimental_human)}")

# with open(filename, 'wb') as file:
    # pickle.dump(Jerk_Experimental_human, file)
with open(filename_2, 'wb') as file:
    pickle.dump(Jerk_Experimental_model, file)
with open(filename_4, 'wb') as file:
    pickle.dump(Jerk_experimental_model_Time, file)