import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import pdb

script_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.realpath(os.path.join(script_dir, '..', 'Data'))

# Open Model Data
Filename_model = 'Zahra_predictions.pkl'
FileDirectory_model = os.path.realpath(os.path.join(script_dir, Filename_model))
with open(FileDirectory_model, 'rb') as f:
    data_prediction = pickle.load(f)

# Open ground_truth Data
Filename_ground = 'Zahra_ground_truth.pkl'
FileDirectory_ground = os.path.realpath(os.path.join(script_dir, Filename_ground))
with open(FileDirectory_ground, 'rb') as f:
    data_ground = pickle.load(f)

# Calculate the Jerk of the model
# 1. Find Stationary point (Model)
# 1.1 Initialization (Model)
Time_Stationary_model = np.zeros((data_prediction.shape[0], 1))
Pose_Stationary_model = np.zeros((data_prediction.shape[0], 3))
human_index_model = 0
Jerk_Experimental_model = np.zeros(data_prediction.shape[0])
Jerk_MinimumJerkTheory_model = np.zeros(data_prediction.shape[0])
MinJerk_Experimental_model = np.zeros(data_prediction.shape[0])
T_Station_Max = 0
window_size = 6
window = np.ones(int(window_size)) / float(window_size)
Starting_point = 5
End_point = data_prediction[0, :, 0:3].shape[0] - 15  # size of the time

# 1.2 Calculate the jerk (Model)
for row_index_model in range(len(data_prediction)):
    hand_position_xyz_m_model = data_prediction[row_index_model, :, 0:3]
    Time_Step_model = hand_position_xyz_m_model.shape[0]
    Variance_model_raw = np.zeros((Time_Step_model, 1))

    for Time_index_model in range(Time_Step_model - 1):
        Variance_model_raw[Time_index_model] = np.linalg.norm(
            hand_position_xyz_m_model[Time_index_model + 1, :] - hand_position_xyz_m_model[Time_index_model, :])

    Variance_model = np.convolve(np.squeeze(Variance_model_raw), window, 'valid')  # Moving average with time window 3
    Time_Stationary_model[human_index_model] = \
        np.where(Variance_model == np.min(Variance_model[Starting_point:End_point]))[0]
    T_Station_model = int(Time_Stationary_model[human_index_model])
    Pose_Stationary_model[human_index_model, :] = hand_position_xyz_m_model[T_Station_model, :]

    hand_pos_xyz_m_model = hand_position_xyz_m_model
    hand_vel_xyz_m_model = np.zeros((T_Station_model - 1, 3))
    hand_acc_xyz_m_model = np.zeros((T_Station_model - 2, 3))
    hand_jerk_xyz_m_model = np.zeros((T_Station_model - 3, 3))
    T_Station_Max = max(T_Station_model, T_Station_Max)
    td_model = 1 / T_Station_model

    for time_index in range(T_Station_model - 1):
        hand_vel_xyz_m_model[time_index, :] = hand_position_xyz_m_model[time_index + 1, :] - hand_position_xyz_m_model[
                                                                                             time_index, :]

    for time_index in range(T_Station_model - 2):
        hand_acc_xyz_m_model[time_index, :] = hand_vel_xyz_m_model[time_index + 1, :] - hand_vel_xyz_m_model[time_index,
                                                                                        :]

    for time_index in range(T_Station_model - 3):
        hand_jerk_xyz_m_model[time_index, :] = hand_acc_xyz_m_model[time_index + 1, :] - hand_acc_xyz_m_model[
                                                                                         time_index, :]
        Jerk_Experimental_model[human_index_model] += np.linalg.norm(hand_jerk_xyz_m_model[time_index, :])
    human_index_model += 1

human_index_model = 0
Jerk_experimental_model_Time = np.zeros((data_prediction.shape[0], T_Station_Max))
for row_index_model in range(len(data_prediction)):
    hand_position_xyz_m_model = data_prediction[row_index_model, :, 0:3]
    hand_pos_xyz_m_model = hand_position_xyz_m_model
    hand_vel_xyz_m_model = np.zeros((T_Station_model - 1, 3))
    hand_acc_xyz_m_model = np.zeros((T_Station_model - 2, 3))
    hand_jerk_xyz_m_model = np.zeros((T_Station_model - 3, 3))

    for time_index in range(T_Station_model - 1):
        hand_vel_xyz_m_model[time_index, :] = hand_position_xyz_m_model[time_index + 1, :] - hand_position_xyz_m_model[
                                                                                             time_index, :]

    for time_index in range(T_Station_model - 2):
        hand_acc_xyz_m_model[time_index, :] = hand_vel_xyz_m_model[time_index + 1, :] - hand_vel_xyz_m_model[time_index,
                                                                                        :]

    for time_index in range(T_Station_model - 3):
        hand_jerk_xyz_m_model[time_index, :] = hand_acc_xyz_m_model[time_index + 1, :] - hand_acc_xyz_m_model[
                                                                                         time_index, :]
        Jerk_experimental_model_Time[human_index_model, time_index] = np.linalg.norm(
            hand_jerk_xyz_m_model[time_index, :])
    human_index_model += 1

# Calculate the Jerk of the ground_truth
# 1. Find Stationary point (ground_truth)
# 1.1 Initialization (ground_truth)
Time_Stationary_ground = np.zeros((data_ground.shape[0], 1))
Pose_Stationary_ground = np.zeros((data_ground.shape[0], 3))
human_index_ground = 0
Jerk_Experimental_ground = np.zeros(data_ground.shape[0])
Jerk_MinimumJerkTheory_ground = np.zeros(data_ground.shape[0])
MinJerk_Experimental_ground = np.zeros(data_ground.shape[0])
T_Station_Max = 0

Starting_point = 5
End_point = data_ground[0, :, 0:3].shape[0] - 15  # size of the time

# 1.2 Calculate the jerk (ground)
for row_index_ground in range(len(data_ground)):
    hand_position_xyz_m_ground = data_ground[row_index_ground, :, 0:3]
    Time_Step_ground = hand_position_xyz_m_ground.shape[0]
    Variance_ground_raw = np.zeros((Time_Step_ground, 1))

    for Time_index_ground in range(Time_Step_ground - 1):
        Variance_ground_raw[Time_index_ground] = np.linalg.norm(
            hand_position_xyz_m_ground[Time_index_ground + 1, :] - hand_position_xyz_m_ground[Time_index_ground, :])

    Variance_ground = np.convolve(np.squeeze(Variance_ground_raw), window, 'valid')  # Moving average with time window 3
    Time_Stationary_ground[human_index_ground] = \
        np.where(Variance_ground == np.min(Variance_ground[Starting_point:End_point]))[0]
    T_Station_ground = int(Time_Stationary_ground[human_index_ground])
    Pose_Stationary_ground[human_index_ground, :] = hand_position_xyz_m_ground[T_Station_ground, :]

    hand_pos_xyz_m_ground = hand_position_xyz_m_ground
    hand_vel_xyz_m_ground = np.zeros((T_Station_ground - 1, 3))
    hand_acc_xyz_m_ground = np.zeros((T_Station_ground - 2, 3))
    hand_jerk_xyz_m_ground = np.zeros((T_Station_ground - 3, 3))
    T_Station_Max = max(T_Station_ground, T_Station_Max)
    td_ground = 1 / T_Station_ground

    for time_index in range(T_Station_ground - 1):
        hand_vel_xyz_m_ground[time_index, :] = hand_position_xyz_m_ground[time_index + 1,
                                               :] - hand_position_xyz_m_ground[
                                                    time_index, :]

    for time_index in range(T_Station_ground - 2):
        hand_acc_xyz_m_ground[time_index, :] = hand_vel_xyz_m_ground[time_index + 1, :] - hand_vel_xyz_m_ground[
                                                                                          time_index,
                                                                                          :]

    for time_index in range(T_Station_ground - 3):
        hand_jerk_xyz_m_ground[time_index, :] = hand_acc_xyz_m_ground[time_index + 1, :] - hand_acc_xyz_m_ground[
                                                                                           time_index, :]
        Jerk_Experimental_ground[human_index_ground] += np.linalg.norm(hand_jerk_xyz_m_ground[time_index, :])
    human_index_ground += 1

human_index_ground = 0
Jerk_experimental_ground_Time = np.zeros((data_ground.shape[0], T_Station_Max))
for row_index_ground in range(len(data_ground)):
    hand_position_xyz_m_ground = data_ground[row_index_ground, :, 0:3]
    hand_pos_xyz_m_ground = hand_position_xyz_m_ground
    hand_vel_xyz_m_ground = np.zeros((T_Station_ground - 1, 3))
    hand_acc_xyz_m_ground = np.zeros((T_Station_ground - 2, 3))
    hand_jerk_xyz_m_ground = np.zeros((T_Station_ground - 3, 3))

    for time_index in range(T_Station_ground - 1):
        hand_vel_xyz_m_ground[time_index, :] = hand_position_xyz_m_ground[time_index + 1,
                                               :] - hand_position_xyz_m_ground[time_index, :]

    for time_index in range(T_Station_ground - 2):
        hand_acc_xyz_m_ground[time_index, :] = hand_vel_xyz_m_ground[time_index + 1, :] - hand_vel_xyz_m_ground[
                                                                                          time_index, :]

    for time_index in range(T_Station_ground - 3):
        hand_jerk_xyz_m_ground[time_index, :] = hand_acc_xyz_m_ground[time_index + 1, :] - hand_acc_xyz_m_ground[
                                                                                           time_index, :]
        Jerk_experimental_ground_Time[human_index_ground, time_index] = np.linalg.norm(
            hand_jerk_xyz_m_ground[time_index, :])
    human_index_ground += 1

# Calculate the maximum and average jerk
print('Mean value of human jerk is ' f"{np.mean(Jerk_experimental_ground_Time[Jerk_experimental_ground_Time!=0])}")
print('Mean value of model jerk is 'f"{np.mean(Jerk_experimental_model_Time[Jerk_experimental_model_Time!=0])}")

Max_model = np.zeros(len(data_prediction))
Max_ground = np.zeros(len(data_ground))

for index in range(len(data_ground)):
    Max_ground[index] = np.max(Jerk_experimental_ground_Time[index,:])

for index in range(len(data_prediction)):
    Max_model[index] = np.max(Jerk_experimental_model_Time[index,:])

print('Max value of human jerk is ' f"{np.mean(Max_ground)}")
Jerk_model_nonzero = Jerk_experimental_model_Time[Jerk_experimental_model_Time!=0]
print('Max value of model jerk is 'f"{np.mean(Max_model)}")

# Plot the jerk of the model and the ground truth
ModelData_time = Jerk_experimental_model_Time
GroundData_time = Jerk_experimental_ground_Time

for trial in range(ModelData_time.shape[0]):
    y = ModelData_time[trial, :]
    x = np.arange(y.shape[0])
    plt.plot(x, y)
    plt.title('Jerk_of_Model_data')
    # naming the x and y-axis
    plt.xlabel('Time')
    plt.ylabel('Jerk')

# plt.show()

for trial in range(GroundData_time.shape[0]):
    y = GroundData_time[trial, :]
    x = np.arange(y.shape[0])
    plt.plot(x, y)
    plt.title('Jerk_of_Human_data')
    # naming the x and y-axis
    plt.xlabel('Time')
    plt.ylabel('Jerk')
# plt.show()

y_ground_average = np.zeros(GroundData_time.shape[1])
y_ground_average_low = np.zeros(GroundData_time.shape[1])
y_ground_average_high = np.zeros(GroundData_time.shape[1])
y_ground_variance = np.zeros(GroundData_time.shape[1])

for time in range(GroundData_time.shape[1]):
    y_ground_average[time] = np.mean(GroundData_time[:,time])
    y_ground_variance[time] = np.std(GroundData_time[:,time])
    y_ground_average_low[time] = y_ground_average[time] - 1.96* y_ground_variance[time]
    y_ground_average_high[time] = y_ground_average[time] +1.96*  y_ground_variance[time]

y_model_average = np.zeros(ModelData_time.shape[1])
y_model_variance = np.zeros(ModelData_time.shape[1])
y_model_average_low = np.zeros(ModelData_time.shape[1])
y_model_average_high = np.zeros(ModelData_time.shape[1])
for time in range(ModelData_time.shape[1]):
    y_model_average[time] = np.mean(ModelData_time[:,time])
    y_model_variance[time] = np.std(ModelData_time[:,time])
    y_model_average_low[time] = y_model_average[time] - 1.96* y_model_variance[time]
    y_model_average_high[time] = y_model_average[time] +1.96*  y_model_variance[time]

# Plot shaded graph of the jerk of the model and the ground truth
Package_dir = os.path.realpath(os.path.join(script_dir, '..', 'cable_gripper_plotting'))

sys.path.append('Package_dir')
import plot_utils

fig, ax = plt.subplots()

# Plot the line

ax.plot(x, y_ground_average, label='Line')
error = 1.96* y_model_variance
ax.fill_between(x, y_ground_average_low, y_ground_average_high, alpha=0.4, label='Error')

# Add a legend
ax.legend()

# Show the plot
plt.show()

fig2, ax = plt.subplots()

# Plot the line

ax.plot(x, y_model_average, label='Line')
ax.fill_between(x, y_model_average_low, y_model_average_high, alpha=0.4, label='Error')

# Add a legend
ax.legend()

# Show the plot
plt.show()
