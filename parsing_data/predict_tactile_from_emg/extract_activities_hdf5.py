import h5py
import numpy as np
from scipy import interpolate, signal # for the resampling
import yaml
from yaml.loader import SafeLoader
import matplotlib.pyplot as plt

def extract_stream(filepath, device_name, stream_name):
  '''
  Extracts a single stream (saved as device_name/stream_name in the 'filepath' hdf5 file)
  Returns data (with original dimensions) and timestamps of the stream as numpy arrays
  '''
  h5_file = h5py.File(filepath, 'r')
  print(f'Extracting {device_name} data from the HDF5 file')

  # Get the data from the stream, should preserve original dimensions
  data = h5_file[device_name][stream_name]['data']
  data = np.array(data)
  # Get the timestamps for each row as seconds since epoch.
  time_s = h5_file[device_name][stream_name]['time_s']
  time_s = np.squeeze(np.array(time_s)) # squeeze (optional) converts from a list of single-element lists to a 1D list

  print(f'{device_name} Data:')
  print('  Shape', data.shape)
  
  return data, time_s

def extract_label_data(filepath, exclude=['Bad', 'Maybe'], group='experiment-activities', stream='activities'):
  '''
  Extracts activity labels from {group}/{stream} ('experiment-activities/activities' by default) in {filepath} hdf5 file
  Params:
    exclude_bad_labels: list of label types (strings) to exclude, "Bad" and "Maybe" excluded by default
  Returns:
    list of string activity labels, list of activity start times, list of activity end times
    activities_labels[i] corresponds to the label of the activity from activities_start_times_s[i] to activities_end_times_s[i]
  '''
  h5_file = h5py.File(filepath, 'r')
  print('Extracting activity labels from the HDF5 file')

  # Get the timestamped label data.
  # As described in the HDF5 metadata, each row has entries for ['Activity', 'Start/Stop', 'Valid', 'Notes'].
  activity_datas = h5_file[group][stream]['data']
  activity_datas = [[x.decode('utf-8') for x in datas] for datas in activity_datas]  # Convert to strings for convenience.

  activity_times_s = h5_file[group][stream]['time_s']
  activity_times_s = np.squeeze(np.array(activity_times_s))  # squeeze (optional) converts from a list of single-element lists to a 1D list

  # Combine start/stop rows to single activity entries with start/stop times.
  #   Each row is either the start or stop of the label.
  #   The notes and ratings fields are the same for the start/stop rows of the label, so only need to check one.
  #   Note: activity ratings and notes are saved, but not returned
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
    if rating in exclude:
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
  '''
  Returns two lists (start times and end times) for all instances of a specified activity
  Params:
    target_activity: string label to search for instances of
    activities_labels, activities_start_times_s, activities_end_times_s: lists such that the i-th element of activities_labels
                                                                         started at activities_start_times_s[i] and ended at
                                                                         activities_end_times_s[i]
  '''
  label_start_times_s = [t for (i, t) in enumerate(activities_start_times_s) if activities_labels[i] == target_activity]
  label_end_times_s = [t for (i, t) in enumerate(activities_end_times_s) if activities_labels[i] == target_activity]
  return label_start_times_s, label_end_times_s

def stream_from_times(data, time_s, label_start_times_s, label_end_times_s):
  '''
  Returns list of segments of data, where segment start and end are specified by label_start_times_s and label_end_times_s
  
  i-th element of returned list contains two lists (data and timestamps) for activity starting
  at label_start_times_s[i] and ending at label_end_times_s[i]
  '''
  segmented_stream = []
  for label_start, label_end in zip(label_start_times_s, label_end_times_s):
    # extract indices only between label_start and label_end times, inclusive
    idxs_for_label = np.where((time_s >= label_start) & (time_s <= label_end))[0]
    data_for_label = data[idxs_for_label, :]
    time_s_for_label = time_s[idxs_for_label]
    
    segmented_stream.append([data_for_label, time_s_for_label])
  
  return segmented_stream

def interpolate_stream(data_to_interp, times_to_interp, base_times):
  '''
  Linearly resample data_to_interp and times_to_interp to match timestamps in base_times
  '''
  data_to_interp = np.array(data_to_interp)
  times_to_interp = np.squeeze(np.array(times_to_interp))

  # Resample interp_data to match the base_times timestamps.
  fn_interpolate = interpolate.interp1d(
                                  times_to_interp, # x values
                                  data_to_interp,   # y values
                                  axis=0,              # axis of the data along which to interpolate
                                  kind='linear',       # interpolation method, such as 'linear', 'zero', 'nearest', 'quadratic', 'cubic', etc.
                                  fill_value='extrapolate' # how to handle x values outside the original range
                                  )
  time_s_resampled = base_times
  data_resampled = fn_interpolate(time_s_resampled)
  
  ## debugging print statements to show change in sampling rate
  # print()
  # print('Original Data:')
  # print('  Shape', interp_data.shape)
  # print('  Sampling rate: %0.2f Hz' % ((interp_data.shape[0]-1)/(max(interp_times) - min(interp_times))))
  # print()
  # print('Resampled Data:')
  # print('  Shape', data_resampled.shape)
  # print('  Sampling rate: %0.2f Hz' % ((data_resampled.shape[0]-1)/(max(time_s_resampled) - min(time_s_resampled))))
  # print()
  
  return data_resampled, time_s_resampled

def interpolate_stream_by_rate(data_to_interp, times_to_interp, sampling_rate):
  '''
  Linearly resample data_to_interp and times_to_interp to have specified sampling_rate
  Resampled times will start from the same initial time as times_to_interp but with new sampling_rate
  '''
  data_to_interp = np.array(data_to_interp)
  times_to_interp = np.squeeze(np.array(times_to_interp))

  # Set up resampling function
  fn_interpolate = interpolate.interp1d(
                                  times_to_interp, # x values
                                  data_to_interp,   # y values
                                  axis=0,              # axis of the data along which to interpolate
                                  kind='linear',       # interpolation method, such as 'linear', 'zero', 'nearest', 'quadratic', 'cubic', etc.
                                  fill_value='extrapolate' # how to handle x values outside the original range
                                  )
  
  # Create resampled times based on specified sampling_rate
  sampling_period = 1.0/sampling_rate
  time_s_resampled = np.arange(times_to_interp[0],times_to_interp[-1],sampling_period)
  data_resampled = fn_interpolate(time_s_resampled)
  
  ## debugging print statements to show change in sampling rate
  # print('Original Data:')
  # print('  Shape', interp_data.shape)
  # print('  Sampling rate: %0.2f Hz' % ((interp_data.shape[0]-1)/(max(interp_times) - min(interp_times))))
  # print()
  # print('Resampled Data:')
  # print('  Shape', data_resampled.shape)
  # print('  Sampling rate: %0.2f Hz' % ((data_resampled.shape[0]-1)/(max(time_s_resampled) - min(time_s_resampled))))
  # print()
  
  return data_resampled, time_s_resampled

def butter_filter(data, times, cutoff_freq, version='sos'):
  '''
  Returns data after being passed through a Butterworth filter
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
  Params:
    data: list of data to be filtered
    times: list of times corresponding to inputted data (used to calculate current data frequency)
    cutoff_freq: target cutoff frequency to be inputted to Butterworth filter
    version: 'sos' for second-order sections OR 'ba' for numerator/denominator
  '''
  # calculate current frequency
  data_freq = (len(data)-1) / (times[-1] - times[0])
  
  if version == 'sos':
    ## following scipy example
    sos = signal.butter(5, cutoff_freq/(data_freq/2), btype='lowpass', output='sos')
    filtered = signal.sosfilt(sos, data)
  else:
    ## following matlab example
    b,a = signal.butter(5, cutoff_freq/(data_freq/2), btype='lowpass', output='ba')
    filtered = signal.lfilter(b, a, data)
  return filtered

def visualize_intermediate(data, times, label):
  '''
  Simple function to plot given data over given times and save the 
  resulting figure in local visualizations folder
  WARNING: may not work depending on dimensions of data, currently hardcoded
           to plot the 4th index in a stream of data (i.e. for the 8-element emg array)
  '''
  to_plot = data[:, 4]
  plt.plot(times, to_plot)
  plt.title(label)
  plt.savefig(f'visualizations/{label}.png')
  plt.show()
  plt.close()

def extract_streams_for_activities(hdf_file, requests_file): 
  '''
  Uses settings in requests_file to return dictionary of extracted streams from hdf_file
  Returned dictionary maps "time_s" and {device_name} to inner level 1 dictionaries
    Inner level 1 dictionary maps {stream_name} to inner level 2 dictionaries
    Inner level 2 dictionary maps {activity_name} to lists of data, where each internal 
                                                  list corresponds to one instance of the activity 
  '''     
  with open(requests_file) as f:
    requests = yaml.load(f, Loader=SafeLoader)
          
  # Find all valid instances of activities
  labels, start_times, end_times = extract_label_data(hdf_file)
  
  extracted_streams = {"time_s":{}}
  device_requests = requests['devices']
  
  # Loop through all requested streams to find the *smallest* range of times
  # Since we don't want to interpolate over a time frame for which we don't have 
  # recorded data, we will standardize interpolation to start with the latest 
  # stream and end with the earliest stream
  min_time = None
  max_time = None
  for device in device_requests.keys():
    time_s = extract_stream(hdf_file, device, device_requests[device]['stream'])[1]
    if min_time is None or time_s[0] > min_time:
      min_time = time_s[0]
    if max_time is None or time_s[-1] < max_time:
      max_time = time_s[-1]
      
  # If a target sampling frequency is specified in the requests file, use the range
  # of times calculated above to create a list of times sampled at the correct frequency
  interp_master_times = None
  if 'sampling_freq' in requests and requests['sampling_freq'] is not None:
    interp_master_times = interpolate_stream_by_rate([1,1], [min_time,max_time], requests['sampling_freq'])[1]
  
  for device in device_requests.keys():
    data, time_s = extract_stream(hdf_file, device, device_requests[device]['stream'])
    data = data.astype('float')
    # visualize_intermediate(data, time_s, f"Original_{device}")
    
    # Take absolute value of all datapoints if specified in requests file
    if device_requests[device]['absolute_value']:
      data = abs(data)
      # visualize_intermediate(data, time_s, f"Absval_{device}")

    # Use Butterworth filter on data if cutoff frequency is specified in requests file
    if device_requests[device]['cutoff_freq'] is not None:
      orig_shape = data.shape
      data = data.reshape(orig_shape[0], -1).transpose()
      for i, stream in enumerate(data):
        data[i] = butter_filter(stream, time_s, device_requests[device]['cutoff_freq'])
      data = data.transpose().reshape(orig_shape)
      data = data[100:]
      time_s = time_s[100:]
      # visualize_intermediate(data, time_s, f"Filtered_{device}")
      
    # Normalize by subtracting off the nth-percentile,
    # where n is the specified "normalize_offset" in the requests file
    if device_requests[device]['normalize_offset'] is not None:
      offset = np.percentile(data, device_requests[device]['normalize_offset'])
      data = data - offset
      # visualize_intermediate(data, time_s, f"Normalizedoffset_{device}")
      
    # Normalize data to fit in the range ~0-1 by dividing by the nth-percentile,
    # where n is the specified "normalize_scale" in the requests file
    # Clips data to be within -0.2 and 1.2 after normalizing
    if device_requests[device]['normalize_scale'] is not None:
      div_factor = np.percentile(data, device_requests[device]['normalize_scale'])
      data = data / div_factor
      data = np.clip(data, -0.2, 1.2)
      # visualize_intermediate(data, time_s, f"Normalized_{device}")

    # Resamples data, either based on the timestamps of the first requested stream
    # or based on the frequency in the requests file (if specified)
    if interp_master_times is None:
      interp_master_times = time_s
    data, time_s = interpolate_stream(data, time_s, interp_master_times)
            
    # Split data based on requested activities before saving it in the returned dictionary
    for activity in requests['activities'].split('|'): #can't split with commas as some activity names have commas
      label_start_times, label_end_times = extract_activity_times(activity, labels, start_times, end_times)
      streams = stream_from_times(data, time_s, label_start_times, label_end_times)
      
      # converting data from numpy arrays to lists (easier to make a list of lists to store in dictionary)
      for i in range(len(streams)):
        for j in range(len(streams[i])):
          if isinstance(streams[i][j], np.ndarray):
            streams[i][j] = streams[i][j].tolist()
        streams[i] = streams[i][:2]
        
      if len(streams) > 0:
        activity_data, activity_time_s = list(zip(*streams))
          
        # add times for this activity to list of times in returned dictionary 
        if activity not in extracted_streams['time_s']:
          extracted_streams['time_s'][activity] = activity_time_s
        
        # Set up inner level 1 dictionary if device has not been added to returned dictionary yet
        if device not in extracted_streams:
          extracted_streams[device] = {}
          
        # Add data from extracted stream to returned dictionary
        if device_requests[device]['stream'] not in extracted_streams[device]:
          extracted_streams[device][device_requests[device]['stream']] = {}
        extracted_streams[device][device_requests[device]['stream']][activity] = activity_data
      
  return extracted_streams

if __name__ == '__main__':
  pass