import h5py
import numpy as np
from scipy import interpolate # for the resampling example
import json
import os

def extract_stream(filepath, device_name, stream_name):
  # Open the file.
  h5_file = h5py.File(filepath, 'r')

  print(f'Extracting {device_name} data from the HDF5 file')

  # Get the data from the stream, should preserve original dimensions
  data = h5_file[device_name][stream_name]['data']
  data = np.array(data)
  # Get the timestamps for each row as seconds since epoch.
  time_s = h5_file[device_name][stream_name]['time_s']
  time_s = np.squeeze(np.array(time_s)) # squeeze (optional) converts from a list of single-element lists to a 1D list
  # Get the timestamps for each row as human-readable strings.

  print(f'{device_name} Data:')
  print('  Shape', data.shape)
  
  return data, time_s

def extract_label_data(filepath, exclude_bad_labels=True, group='experiment-activities', stream='activities'):
  # Open the file.
  h5_file = h5py.File(filepath, 'r')

  print('Extracting activity labels from the HDF5 file')

  # Get the timestamped label data.
  # As described in the HDF5 metadata, each row has entries for ['Activity', 'Start/Stop', 'Valid', 'Notes'].
  activity_datas = h5_file[group][stream]['data']
  activity_times_s = h5_file[group][stream]['time_s']
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

  return activities_labels, activities_start_times_s, activities_end_times_s

def extract_activity_times(target_activity, activities_labels, activities_start_times_s, activities_end_times_s):
  label_start_times_s = [t for (i, t) in enumerate(activities_start_times_s) if activities_labels[i] == target_activity]
  label_end_times_s = [t for (i, t) in enumerate(activities_end_times_s) if activities_labels[i] == target_activity]
  return label_start_times_s, label_end_times_s

def stream_from_times(data, time_s, label_start_times_s, label_end_times_s):
  segmented_stream = []
  for label_start, label_end in zip(label_start_times_s, label_end_times_s):
    idxs_for_label = np.where((time_s >= label_start) & (time_s <= label_end))[0]
    data_for_label = data[idxs_for_label, :]
    time_s_for_label = time_s[idxs_for_label]
    
    segmented_stream.append([data_for_label, time_s_for_label])
  
  return segmented_stream

def interpolate_stream(interp_data, interp_times, base_times, base_data=None):
  interp_data = np.array(interp_data)
  interp_times = np.squeeze(np.array(interp_times))

  # Resample interp_data to match the base_times timestamps.
  #  Note that the IMU streamed at about 50 Hz while the EMG streamed at about 200 Hz.
  fn_interpolate = interpolate.interp1d(
                                  interp_times, # x values
                                  interp_data,   # y values
                                  axis=0,              # axis of the data along which to interpolate
                                  kind='linear',       # interpolation method, such as 'linear', 'zero', 'nearest', 'quadratic', 'cubic', etc.
                                  fill_value='extrapolate' # how to handle x values outside the original range
                                  )
  time_s_resampled = base_times
  data_resampled = fn_interpolate(time_s_resampled)
  
  print()
  if base_data:
    print('Base Data:')
    print('  Shape', base_data.shape)
    print('  Sampling rate: %0.2f Hz' % ((base_data.shape[0]-1)/(max(base_times) - min(base_times))))
    print()
  print('Original Data:')
  print('  Shape', interp_data.shape)
  print('  Sampling rate: %0.2f Hz' % ((interp_data.shape[0]-1)/(max(interp_times) - min(interp_times))))
  print()
  print('Resampled Data:')
  print('  Shape', data_resampled.shape)
  print('  Sampling rate: %0.2f Hz' % ((data_resampled.shape[0]-1)/(max(time_s_resampled) - min(time_s_resampled))))
  print()
  
  return data_resampled, time_s_resampled


def extract_streams_for_activities(hdf_file, requests_file):
  with open(requests_file, 'r') as f:
      requests = f.read().split('\n')
      
  labels, start_times, end_times = extract_label_data(hdf_file)
  
  extracted_streams = {"time_s":{}}
  interp_master_times = None
  for request in requests:
    try:
      devices, activities = request.split(';')
    except:
      print(f"{request} not formatted correctly.")
      pass
    for stream_path in devices.split(','):
      device_name, stream_name = stream_path.split('/')
              
      data, time_s = extract_stream(hdf_file, device_name, stream_name)
      
      if interp_master_times is None:
        interp_master_times = time_s
      else:
        data, time_s = interpolate_stream(data, time_s, interp_master_times)
             
      for activity in activities.split(','):
        label_start_times, label_end_times = extract_activity_times(activity, labels, start_times, end_times)
        streams = stream_from_times(data, time_s, label_start_times, label_end_times)
        
        for i in range(len(streams)):
          for j in range(len(streams[i])):
            if isinstance(streams[i][j], np.ndarray):
              streams[i][j] = streams[i][j].tolist()
          streams[i] = streams[i][:2]
          
        activity_data, activity_time_s = list(zip(*streams))
          
        if activity not in extracted_streams['time_s']:
          extracted_streams['time_s'][activity] = activity_time_s
        
        if device_name not in extracted_streams:
          extracted_streams[device_name] = {}
        if stream_name not in extracted_streams[device_name]:
          extracted_streams[device_name][stream_name] = {}
        extracted_streams[device_name][stream_name][activity] = activity_data
      
  return extracted_streams

if __name__ == '__main__':
  pass
  data_dir = "C:/Users/2021l/Documents/UROP/data/"
  requests_file = 'streams_to_extract.txt'
  extracted_streams = {}
  for file in os.listdir(data_dir):
    if file[-4:] == "hdf5":
      subj_stream = extract_streams_for_activities(data_dir+file, requests_file)
      extracted_streams[file[-8:-5]] = subj_stream

  with open('file_right.json', 'w') as f:
    f.write(json.dumps(extracted_streams))