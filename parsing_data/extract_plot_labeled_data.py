
import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
script_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(script_dir, '..', 'data', '2022-06-27_mgh_skills_lab')

# NOTE: HDFView is a helpful program for exploring HDF5 contents.
#   The official download page is at https://www.hdfgroup.org/downloads/hdfview.
#   It can also be downloaded without an account from https://www.softpedia.com/get/Others/Miscellaneous/HDFView.shtml.

# Data will be organized by subject then by the HDF5 hierarchy.
# Some useful examples:
#  data['S00']['myo-ARM']['TYPE']['data']
#  data['S00']['myo-ARM']['TYPE']['time_s']
#  data['S00']['myo-ARM']['TYPE']['time_str']
#  ARM is 'left' or 'right'.
#  TYPE is 'emg', 'acceleration_g', 'angular_velocity_deg_s', or 'orientation_quaternion'.
#  'data' will contain the vector of data from each timestep.
#  'time_s' and 'time_str' will contain the timestamp for each timestep as seconds since epoch or as a string.
data = {}
# Activity data will be organized by subject
#  and will contain 'labels', 'start_times_s', 'end_times_s', 'notes', and 'ratings'.
# For example:
#  activities['S00']['labels']
activities = {}

####################################################
# Load HDF5 files and concatenate them by subject.
####################################################
subject_ids_to_load = ['S00', 'S01']
hdf5_filepaths = glob.glob(os.path.join(data_dir, '**', '*.hdf5'), recursive=True)
for subject_id in subject_ids_to_load:
  data.setdefault(subject_id, {})
  hdf5_filepaths_forSubject = [f for f in hdf5_filepaths if subject_id in f]
  # Concatenate each subject's data across all files for that subject.
  for hdf5_filepath in hdf5_filepaths_forSubject:
    hdf5_file = h5py.File(hdf5_filepath, 'r')
    for (device_name, device_group) in hdf5_file.items(): # for example, 'myo-left'
      data[subject_id].setdefault(device_name, {})
      for (stream_name, stream_group) in device_group.items(): # for example, 'emg'
        data[subject_id][device_name].setdefault(stream_name, {})
        for (dataset_name, dataset) in stream_group.items(): # 'data', 'time_s', or 'time_str'
          dataset = np.squeeze(np.array(dataset))
          # Concatenate the datasets.
          if dataset_name in data[subject_id][device_name][stream_name]:
            data[subject_id][device_name][stream_name][dataset_name] = np.concatenate(
                (data[subject_id][device_name][stream_name][dataset_name], dataset),
                axis=0)
          else:
            data[subject_id][device_name][stream_name][dataset_name] = dataset

####################################################
# Parse label data to get activity start/stop times.
####################################################
device_name = 'experiment-activities'
stream_name = 'activities'
exclude_bad_labels = True # some activities may have been marked as 'Bad' or 'Maybe' by the experimenter; submitted notes with the activity typically give more information
for (subject_id, subject_data) in data.items():
  # Get the timestamped label data.
  # As described in the HDF5 metadata, each row has entries for ['Activity', 'Start/Stop', 'Valid', 'Notes'].
  activity_datas = subject_data[device_name][stream_name]['data']
  activity_times_s = subject_data[device_name][stream_name]['time_s']
  activity_times_s = np.squeeze(np.array(activity_times_s))  # squeeze (optional) converts from a list of single-element lists to a 1D list
  # Convert to strings for convenience.
  activity_datas = [[x.decode('utf-8') for x in datas] for datas in activity_datas]
  # Combine start/stop rows to single activity entries with start/stop times.
  #   Each row is either the start or stop of the label.
  #   The notes and ratings fields are the same for the start/stop rows of the label, so only need to check one.
  activities_labels = []
  activities_start_times_s = []
  activities_end_times_s = []
  activities_ratings = []
  activities_notes = []
  for (row_index, time_s) in enumerate(activity_times_s):
    label    = activity_datas[row_index][0]
    is_start = activity_datas[row_index][1] == 'Start'
    is_stop  = activity_datas[row_index][1] == 'Stop'
    rating   = activity_datas[row_index][2]
    notes    = activity_datas[row_index][3]
    if exclude_bad_labels and rating in ['Bad', 'Maybe']:
      continue
    # Record the start of a new activity.
    if is_start:
      activities_labels.append(label)
      activities_start_times_s.append(time_s)
      activities_ratings.append(rating)
      activities_notes.append(notes)
    # Record the end of the previous activity.
    if is_stop:
      activities_end_times_s.append(time_s)
  
  ####################################################
  # Rename/adjust some activity labels.
  ####################################################
  
  # Separate Da Vinci grasping/releasing from physical grasping/releasing.
  # During data recording, the Da Vinci task wasn't available as a separate activity label
  #  so the notes were used to clarify.
  for activity_index in range(len(activities_labels)):
    activity_label = activities_labels[activity_index]
    activity_start_time_s = activities_start_times_s[activity_index]
    activity_end_time_s = activities_end_times_s[activity_index]
    activity_notes = activities_notes[activity_index]
    if 'grasping/releasing' in activity_label.lower():
      if 'vinci' in activity_notes.lower():
        activities_labels[activity_index] = 'Grasping/Releasing (da Vinci)'
      else:
        activities_labels[activity_index] = 'Grasping/Releasing'
        
  # Ignore the activity of cutting the sutuering string.
  # This occurred once and consisted of cutting the string after sutuering,
  #  but all other instances of cutting were using cardboard.
  edited_list = None
  while edited_list in [None, True]:
    edited_list = False
    for activity_index in range(len(activities_labels)):
      activity_label = activities_labels[activity_index]
      activity_notes = activities_notes[activity_index]
      if 'string' in activity_notes.lower():
        del activities_labels[activity_index]
        del activities_start_times_s[activity_index]
        del activities_end_times_s[activity_index]
        del activities_notes[activity_index]
        del activities_ratings[activity_index]
        # Stop this iteration through the list and start from the beginning to avoid indexing issues.
        edited_list = True
        break

  # Ignore the sutuering activity.
  # Apparently the tools were insufficient, so it was not representative of real sutuering.
  edited_list = None
  while edited_list in [None, True]:
    edited_list = False
    for activity_index in range(len(activities_labels)):
      activity_label = activities_labels[activity_index]
      activity_notes = activities_notes[activity_index]
      if 'sutuering' in activity_label.lower():
        del activities_labels[activity_index]
        del activities_start_times_s[activity_index]
        del activities_end_times_s[activity_index]
        del activities_notes[activity_index]
        del activities_ratings[activity_index]
        # Stop this iteration through the list and start from the beginning to avoid indexing issues.
        edited_list = True
        break

  # Aggregate grasping/releasing activities.
  # During experiments, attempts were made to label each individual grasp/release operation.
  # Now lump them into one activity instead for better comparison with other activities.
  edited_list = None
  while edited_list in [None, True]:
    edited_list = False
    for activity_index in range(len(activities_labels)-1):
      activity_label = activities_labels[activity_index]
      next_activity_label = activities_labels[activity_index+1]
      # Combine these two activities if they are both for grasping/releasing.
      if [activity_label, next_activity_label] == ['Grasping/Releasing']*2:
        activities_end_times_s[activity_index] = activities_end_times_s[activity_index+1]
        del activities_labels[activity_index+1]
        del activities_start_times_s[activity_index+1]
        del activities_end_times_s[activity_index+1]
        del activities_notes[activity_index+1]
        del activities_ratings[activity_index+1]
        # Stop this iteration through the list and start from the beginning to avoid indexing issues.
        edited_list = True
        break
  
  # Store the activities for this subject.
  activities[subject_id] = {
    'labels': activities_labels,
    'start_times_s': activities_start_times_s,
    'end_times_s': activities_end_times_s,
    'notes': activities_notes,
    'ratings': activities_ratings,
  }
  
  # Print a summary of the activities.
  print()
  print('='*50)
  print('Extracted activities for subject %s' % subject_id)
  print('='*50)
  max_label_length = max([len(label) for label in activities_labels])
  for activity_index in range(len(activities_labels)):
    activity_label = activities_labels[activity_index]
    activity_start_time_s = activities_start_times_s[activity_index]
    activity_end_time_s = activities_end_times_s[activity_index]
    activity_notes = activities_notes[activity_index]
    print('  %s [duration %6.2fs] [%s]' % (activity_label.ljust(max_label_length), activity_end_time_s - activity_start_time_s, activity_notes))






####################################################
# Plot each activity!
####################################################
# Helper function for segmenting data based on start/end times.
def segment_data(data, time_s, start_time_s, end_time_s):
  indexes_forLabel = np.where((time_s >= start_time_s) & (time_s <= end_time_s))[0]
  return (data[indexes_forLabel, :], time_s[indexes_forLabel])
  
for subject_id in data:
  subject_data = data[subject_id]
  subject_activities = activities[subject_id]
  unique_activity_labels = list(set(subject_activities['labels']))
  for label in unique_activity_labels:
    # Create a figure and subplots.
    fig, axs = plt.subplots(nrows=9, ncols=2,
                            squeeze=False, # if False, always return 2D array of axes
                            sharex=False, sharey=False,
                            subplot_kw={'frame_on': True},
                            )
    # Find the start/end times associated with instances of this label.
    label_start_times_s = [t for (i, t) in enumerate(subject_activities['start_times_s']) if subject_activities['labels'][i] == label]
    label_end_times_s = [t for (i, t) in enumerate(subject_activities['end_times_s']) if subject_activities['labels'][i] == label]
    # Plot each instance of the activity for this subject.
    plot_start_time_s = 0
    for target_label_instance in range(len(label_start_times_s)):
      label_start_time_s = label_start_times_s[target_label_instance]
      label_end_time_s = label_end_times_s[target_label_instance]
      # Segment and plot EMG data!
      for (arm_index, arm) in enumerate(['left', 'right']):
        emg_data = subject_data['myo-%s' % arm]['emg']['data']
        emg_time_s = subject_data['myo-%s' % arm]['emg']['time_s']
        (emg_data, emg_time_s) = segment_data(emg_data, emg_time_s, label_start_time_s, label_end_time_s)
        emg_time_s = emg_time_s - min(emg_time_s) + plot_start_time_s
        for emg_channel in range(8):
          axs[emg_channel][arm_index].plot(emg_time_s, emg_data[:, emg_channel])
          axs[emg_channel][arm_index].plot([max(emg_time_s)] * 2, [-100, 100], 'k-', linewidth=3)
          axs[emg_channel][arm_index].set_ylim((-100, 100))
        # Segment and plot IMU data!
        accel_data = subject_data['myo-%s' % arm]['acceleration_g']['data']
        accel_time_s = subject_data['myo-%s' % arm]['acceleration_g']['time_s']
        (accel_data, accel_time_s) = segment_data(accel_data, accel_time_s, label_start_time_s, label_end_time_s)
        accel_time_s = accel_time_s - min(accel_time_s) + plot_start_time_s
        axs[-1][arm_index].plot(accel_time_s, accel_data)
        axs[-1][arm_index].plot([max(accel_time_s)] * 2, [-100, 100], 'k-', linewidth=3)
        axs[-1][arm_index].set_ylim((-1.5, 1.5))
      # Update time at which the next instance should start its plot.
      plot_start_time_s = max(emg_time_s) + 1
    plt.suptitle('%s: %s' % (subject_id, label))
    
plt.show()




