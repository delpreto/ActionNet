
from sensor_streamer_handlers.SensorManager import SensorManager
from sensor_streamer_handlers.DataVisualizer import DataVisualizer

import time
import traceback
from utils.time_utils import *
from utils.print_utils import *
import sys
import os
script_dir = os.path.dirname(os.path.realpath(__file__))

# Note that multiprocessing requires the __main__ check.
if __name__ == '__main__':
  # Configure printing to the console.
  print_status = True
  print_debug = False
  
  exclude_bad_labels = True
  
  # Define the log(s) to replay.
  data_dir = os.path.realpath(os.path.join(script_dir, '..', '..', 'data'))
  experiments_dir = os.path.join(data_dir, 'experiments')
  log_dirs_byVideo = {
    'S00': [
      os.path.join(experiments_dir, '2022-06-07_experiment_S00', '2022-06-07_17-18-17_actionNet-wearables_S00'),
      os.path.join(experiments_dir, '2022-06-07_experiment_S00', '2022-06-07_18-10-55_actionNet-wearables_S00'),
    ],
    'S01': [
      os.path.join(experiments_dir, '2022-06-13_experiment_S01_recordingStopped', '2022-06-13_18-13-12_actionNet-wearables_S01'),
    ],
    'S02': [
      os.path.join(experiments_dir, '2022-06-13_experiment_S02', '2022-06-13_21-39-50_actionNet-wearables_S02'),
      os.path.join(experiments_dir, '2022-06-13_experiment_S02', '2022-06-13_21-47-57_actionNet-wearables_S02'),
      os.path.join(experiments_dir, '2022-06-13_experiment_S02', '2022-06-13_22-34-45_actionNet-wearables_S02'),
      os.path.join(experiments_dir, '2022-06-13_experiment_S02', '2022-06-13_23-16-47_actionNet-wearables_S02'),
      os.path.join(experiments_dir, '2022-06-13_experiment_S02', '2022-06-13_23-22-21_actionNet-wearables_S02'),
    ],
    'S03': [
      # os.path.join(experiments_dir, '2022-06-14_experiment_S03', '2022-06-14_13-01-32_actionNet-wearables_S03'),
      os.path.join(experiments_dir, '2022-06-14_experiment_S03', '2022-06-14_13-11-44_actionNet-wearables_S03'),
      os.path.join(experiments_dir, '2022-06-14_experiment_S03', '2022-06-14_13-52-21_actionNet-wearables_S03'),
    ],
    'S04': [
      os.path.join(experiments_dir, '2022-06-14_experiment_S04', '2022-06-14_16-38-18_actionNet-wearables_S04'),
    ],
    'S05': [
      os.path.join(experiments_dir, '2022-06-14_experiment_S05', '2022-06-14_20-36-27_actionNet-wearables_S05'),
      os.path.join(experiments_dir, '2022-06-14_experiment_S05', '2022-06-14_20-45-43_actionNet-wearables_S05'),
    ]
  }
  
  # Loop through each specified log directory to process.
  for (video_key, log_dirs) in log_dirs_byVideo.items():
    print('\n'*2)
    print('='*75)
    print('Creating composite visualization labels for video %s\n' % video_key)
    chapter_start_times_s = []
    chapter_end_times_s = []
    chapter_labels = []
    video_start_time_s = 0
    for (i, log_dir) in enumerate(log_dirs):
      print('\n\n' + '='*75)
      print('  Processing log directory %s\n' % log_dir)
      print()
      
      log_player_options = {
        'log_dir': log_dir,
        'load_datasets_into_memory': False,
      }
      
      # Configure where and how to save sensor data.
      datalogging_options = None
      # Configure visualizations to be shown as a simulation of real-time streaming.
      visualization_options = None
      composite_video_filepath = os.path.join(log_dir, 'composite_visualization_postProcessed')
      # Create a sensor manager.
      sensor_manager = SensorManager(sensor_streamer_specs=None,
                                     log_player_options=log_player_options,
                                     data_logger_options=datalogging_options,
                                     data_visualizer_options=visualization_options,
                                     print_status=print_status, print_debug=print_debug)
      # Load streams from the saved logs for each streamer.
      sensor_manager.connect()
      
      # Get the start and end times that the visualizer used when making the video.
      frame_size = (1500, 2500) # height, width # (1800, 3000)
      col_width = int(frame_size[1]/3)
      row_height = int(frame_size[0]/3)
      visualizer = DataVisualizer(sensor_streamers=sensor_manager.get_streamers(),
                                  update_period_s = 0.1,
                                  use_composite_video=False,
                                  print_status=False, print_debug=False)
      (start_time_s, end_time_s) = visualizer.get_loggedData_start_end_times_s(
          start_offset_s=None, end_offset_s=None,
          duration_s=None)
      
      # Get label times.
      experimentControl_streamer = sensor_manager.get_streamers(class_name='ExperimentControlStreamer')[0]
      activity_data_dict = experimentControl_streamer.get_data('experiment-activities', 'activities')
      if activity_data_dict is None:
        print()
        print('Advancing video time but skipping labels since no activities were recorded')
        video_start_time_s += end_time_s - start_time_s
        continue
      activity_times_s = activity_data_dict['time_s']
      activity_datas = activity_data_dict['data']
      activity_datas = [[x.decode('utf-8') for x in datas] for datas in activity_datas]
      # activity_labels = [data[0].decode('utf-8') for data in activity_datas]
      # activity_startStop = [data[1].decode('utf-8') for data in activity_datas]
      # activity_rating = [data[2].decode('utf-8') for data in activity_datas]
      # activity_notes = [data[3].decode('utf-8') for data in activity_datas]
  
      activities_labels = []
      activities_start_times_s = []
      activities_end_times_s = []
      activities_ratings = []
      activities_notes = []
      for (i, time_s) in enumerate(activity_times_s):
        label    = activity_datas[i][0]
        is_start = activity_datas[i][1] == 'Start'
        is_stop  = activity_datas[i][1] == 'Stop'
        rating   = activity_datas[i][2]
        notes    = activity_datas[i][3]
        if exclude_bad_labels and rating in ['Bad', 'Maybe']:
          continue
        if is_start:
          activities_labels.append(label)
          activities_start_times_s.append(time_s)
          activities_ratings.append(rating)
          activities_notes.append(notes)
        if is_stop:
          activities_end_times_s.append(time_s)

      # print_var(activity_labels)
      # print_var(activity_start_times_s)
      # print_var(activity_end_times_s)
      # print_var(activity_ratings)
      # print_var(activity_notes)
      
      # Group unique activities.
      activity_labels_unique = list(set(activities_labels))
      
      # Convert activity times to chapter times.
      for activity_label in activity_labels_unique:
        activity_start_times_s = [x for (i, x) in enumerate(activities_start_times_s) if activities_labels[i] == activity_label]
        activity_end_times_s = [x for (i, x) in enumerate(activities_end_times_s) if activities_labels[i] == activity_label]
        activity_start_time_s = min(activity_start_times_s)
        activity_end_time_s = max(activity_end_times_s)
        chapter_start_times_s.append(activity_start_time_s - start_time_s + video_start_time_s)
        chapter_end_times_s.append(activity_end_time_s - start_time_s + video_start_time_s)
        chapter_labels.append(activity_label)
        
      video_start_time_s += end_time_s - start_time_s
    
    # Sort.
    chapter_labels = [x for _,x in sorted(zip(chapter_start_times_s, chapter_labels))]
    chapter_end_times_s = [x for _,x in sorted(zip(chapter_start_times_s, chapter_end_times_s))]
    chapter_start_times_s = sorted(chapter_start_times_s)
    
    # Print the result
    print()
    print('-'*50)
    print('Video chapters for %s\n' % video_key)
    for i in range(len(chapter_labels)):
      activity_start_time_s = chapter_start_times_s[i]
      label = chapter_labels[i]
      print('%02d:%02d:%02d %s' % (int(activity_start_time_s/3600),
                                   int((activity_start_time_s % 3600)/60),
                                   round(activity_start_time_s % 60),
                                   label))
  
  
