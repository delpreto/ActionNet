############
#
# Copyright (c) 2022 MIT CSAIL and Joseph DelPreto
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
# See https://action-net.csail.mit.edu for more usage information.
# Created 2021-2022 for the MIT ActionNet project by Joseph DelPreto [https://josephdelpreto.com].
#
############

# This script will print out activity label start/end times
#  in a format that can be copied into YouTube descriptions to create chapters.

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
    ],
    'S06': [
      os.path.join(experiments_dir, '2022-07-12_experiment_S06', '2022-07-12_14-30-38_actionNet-wearables_S06'),
      os.path.join(experiments_dir, '2022-07-12_experiment_S06', '2022-07-12_15-07-50_actionNet-wearables_S06'),
    ],
    'S07': [
      os.path.join(experiments_dir, '2022-07-13_experiment_S07', '2022-07-13_11-01-18_actionNet-wearables_S07'),
    ],
    'S08': [
      os.path.join(experiments_dir, '2022-07-13_experiment_S08', '2022-07-13_14-15-03_actionNet-wearables_S08'),
    ],
    'S09': [
      os.path.join(experiments_dir, '2022-07-14_experiment_S09', '2022-07-14_09-47-52_actionNet-wearables_S09'),
      os.path.join(experiments_dir, '2022-07-14_experiment_S09', '2022-07-14_09-58-40_actionNet-wearables_S09'),
      os.path.join(experiments_dir, '2022-07-14_experiment_S09', '2022-07-14_11-13-55_actionNet-wearables_S09'),
    ],
    # 'Badminton': [
    #   os.path.join(experiments_dir, '2022-07-14_badminton_test_wearing', '2022-07-14_17-43-05_testing in holodeck'),
    # ],
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
      # Create a sensor manager.
      sensor_manager = SensorManager(sensor_streamer_specs=None,
                                     log_player_options=log_player_options,
                                     data_logger_options=datalogging_options,
                                     data_visualizer_options=visualization_options,
                                     print_status=print_status, print_debug=print_debug)
      # Load streams from the saved logs for each streamer.
      sensor_manager.connect()
      
      # Get the start and end times that the visualizer used when making the video.
      visualizer = DataVisualizer(sensor_streamers=sensor_manager.get_streamers(),
                                  update_period_s = 0.1,
                                  use_composite_video=False,
                                  print_status=False, print_debug=False)
      (start_time_s, end_time_s) = visualizer.get_loggedData_start_end_times_s(
          start_offset_s=None, end_offset_s=None,
          duration_s=None)
      
      # Get label information.
      experimentControl_streamer = sensor_manager.get_streamers(class_name='ExperimentControlStreamer')[0]
      activity_data_dict = experimentControl_streamer.get_data('experiment-activities', 'activities')
      if activity_data_dict is None:
        print()
        print('Advancing video time but skipping labels since no activities were recorded')
        video_start_time_s += (end_time_s - start_time_s)
        continue
      activity_times_s = activity_data_dict['time_s']
      activity_datas = activity_data_dict['data']
      activity_datas = [[x.decode('utf-8') for x in datas] for datas in activity_datas]
      
      # Get start/end times for each label instance.
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
      
      # Get a list of unique activity labels.
      activity_labels_unique = list(set(activities_labels))
      
      # Convert activity times to chapter times.
      # For each label, will use the start of the first instance and the end of the last instance.
      for activity_label in activity_labels_unique:
        activity_start_times_s = [x for (i, x) in enumerate(activities_start_times_s) if activities_labels[i] == activity_label]
        activity_end_times_s = [x for (i, x) in enumerate(activities_end_times_s) if activities_labels[i] == activity_label]
        activity_start_time_s = min(activity_start_times_s)
        activity_end_time_s = max(activity_end_times_s)
        chapter_start_times_s.append(activity_start_time_s - start_time_s + video_start_time_s)
        chapter_end_times_s.append(activity_end_time_s - start_time_s + video_start_time_s)
        chapter_labels.append(activity_label)
        
      video_start_time_s += end_time_s - start_time_s
    
    # Sort chapters by their start times.
    chapter_labels = [x for _,x in sorted(zip(chapter_start_times_s, chapter_labels))]
    chapter_end_times_s = [x for _,x in sorted(zip(chapter_start_times_s, chapter_end_times_s))]
    chapter_start_times_s = sorted(chapter_start_times_s)
    
    # Print the result in a format that can be copied into YouTube descriptions.
    print()
    print('-'*50)
    print('Video chapters for %s\n' % video_key)
    print('00:00:00 Calibration/Setup')
    for i in range(len(chapter_labels)):
      activity_start_time_s = chapter_start_times_s[i]
      label = chapter_labels[i]
      print('%02d:%02d:%02d %s' % (int(activity_start_time_s/3600),
                                   int((activity_start_time_s % 3600)/60),
                                   round(activity_start_time_s % 60),
                                   label))
  
  
