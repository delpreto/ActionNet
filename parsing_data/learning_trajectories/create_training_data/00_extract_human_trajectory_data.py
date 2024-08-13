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

import h5py
import numpy as np
from scipy import interpolate
from scipy.spatial.transform import Rotation
from collections import OrderedDict
import os
script_dir = os.path.dirname(os.path.realpath(__file__))
actionsense_root_dir = script_dir
while os.path.split(actionsense_root_dir)[-1] != 'ActionSense':
  actionsense_root_dir = os.path.realpath(os.path.join(actionsense_root_dir, '..'))
import cv2

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d

from learning_trajectories.helpers.configuration import *
from learning_trajectories.helpers.parse_process_demonstration_data import *
from learning_trajectories.helpers.timeseries_processing import *
from learning_trajectories.helpers.numpy_scipy_utils import *
from learning_trajectories.helpers.plot_animations import *
from learning_trajectories.helpers.printing import *

###################################################################
# Configuration
###################################################################

# Specify the subjects to consider.
subject_id_toProcess = 'S00' # S00, S10, S11, ted_S00
subject_ids_filter = None # None to consider all subjects

data_dir = os.path.realpath(os.path.join(actionsense_root_dir, 'data'))
results_dir = os.path.realpath(os.path.join(actionsense_root_dir, 'results'))

# Specify the folder of experiments to parse.
if subject_id_toProcess == 'S00':
  experiments_dir = os.path.join(data_dir, 'experiments', '2023-09-10_experiment_%s' % subject_id_toProcess)
elif subject_id_toProcess == 'S10':
  experiments_dir = os.path.join(data_dir, 'experiments', '2023-08-18_experiment_%s' % subject_id_toProcess)
elif subject_id_toProcess == 'S11':
  experiments_dir = os.path.join(data_dir, 'experiments', '2023-09-10_experiment_%s' % subject_id_toProcess)
elif subject_id_toProcess == 'ted_S00':
  experiments_dir = os.path.join(data_dir, 'experiments', '2024-03-04_experiment_S00_selectedRun')

# Specify the output folder.
output_dir = os.path.realpath(os.path.join(results_dir, 'learning_trajectories'))
os.makedirs(output_dir, exist_ok=True)

# Specify outputs.
animate_trajectory_plots = False # show an animated plot of the skeleton for each trial
plot_all_trajectories = False # make a subplot for each subject, which shows all paths from that subject
save_plot_all_trajectories = True # make a subplot for each subject, which shows all paths from that subject
save_eye_videos = True # save the eye-tracking video for each trial
save_animation_videos = True
save_composite_videos = True # save the eye-tracking video and animated plot for each trial
save_results_data = True

# Choose the activity to process.
activity_mode = 'pouring' # 'pouring', 'scooping'

if activity_mode == 'pouring':
  target_activity_label = 'Pour water from a pitcher into a glass'
  target_activity_keyword = 'pouring'
  stationary_position_use_variance = True # Use a rolling buffer and look at hand variance
elif activity_mode == 'scooping':
  target_activity_label = 'Scoop from a pan to a plate'
  target_activity_keyword = 'scooping'
  stationary_position_use_variance = False # Use a hard-coded fraction into the trial.
else:
  raise AssertionError('Unknown activity mode "%s"' % activity_mode)


###################################################################
# Main processing
###################################################################

def main_processing():
  # Find folders of log data, and record filepaths for the HDF5s and first-person videos.
  hdf5_filepaths = OrderedDict() # map subject IDs to list of filepaths
  eyeVideo_filepaths = OrderedDict() # map subject IDs to list of filepaths
  for subdir, dirs, filenames in os.walk(experiments_dir):
    hdf5_filepath = [filename for filename in filenames if '.hdf5' in filename]
    eyeVideo_filepath = [filename for filename in filenames if 'eye-tracking-video-worldGaze_frame.avi' in filename]
    log_filepath = [filename for filename in filenames if 'log_history.txt' in filename]
    try:
      subject_id = int(subdir.split('_')[-1][1:])
    except:
      subject_id = None
    is_a_root_log_folder = len(hdf5_filepath) == 1 and len(eyeVideo_filepath) == 1 \
                           and len(log_filepath) == 1 and subject_id is not None
    if is_a_root_log_folder and (subject_ids_filter is None or subject_id in subject_ids_filter):
      hdf5_filepaths.setdefault(subject_id, [])
      eyeVideo_filepaths.setdefault(subject_id, [])
      hdf5_filepaths[subject_id].append(os.path.join(subdir, hdf5_filepath[0]))
      eyeVideo_filepaths[subject_id].append(os.path.join(subdir, eyeVideo_filepath[0]))
  
  # Loop through experiment files to extract trajectories for each action instance.
  print()
  num_subjects = len(hdf5_filepaths)
  handles_allPaths = None
  allPaths_subplot_index = 0
  time_s_byTrial_bySubject = OrderedDict()
  bodyPath_data_byTrial_bySubject = OrderedDict()
  stationary_time_s_byTrial_bySubject = OrderedDict()
  stationary_pose_byTrial_bySubject = OrderedDict()
  referenceObject_position_m_byTrial_bySubject = OrderedDict()
  for subject_id, subject_hdf5_filepaths in hdf5_filepaths.items():
    print('Processing subject %02d' % subject_id)
    time_s_byTrial_bySubject.setdefault(subject_id, [])
    bodyPath_data_byTrial_bySubject.setdefault(subject_id, [])
    stationary_time_s_byTrial_bySubject.setdefault(subject_id, [])
    stationary_pose_byTrial_bySubject.setdefault(subject_id, [])
    referenceObject_position_m_byTrial_bySubject.setdefault(subject_id, [])
    targetActivity_trial_index_start = 0
    for (filepath_index, hdf5_filepath) in enumerate(subject_hdf5_filepaths):
      print(' ', hdf5_filepath)
      eyeVideo_filepath = eyeVideo_filepaths[subject_id][filepath_index]
      # Open the HDF5 file.
      h5_file = h5py.File(hdf5_filepath, 'r')
      
      # Determine the start/end times of each hand path.
      print('  Getting activity start/end times')
      (activities_labels, activities_start_times_s, activities_end_times_s) = \
        get_targetActivity_startEnd_times_s(h5_file, target_activity_label,
                                            exclude_bad_labels=True)
      if activities_labels is None:
        continue
      num_trials_inFile = len(activities_labels)
      # if use_manual_startEnd_times:
      #   (activities_labels, activities_start_times_s, activities_end_times_s) = \
      #     get_manual_pouring_startEnd_times_s(subject_id)
      #   activities_labels = activities_labels[targetActivity_trial_index_start:(targetActivity_trial_index_start + num_trials_inFile)]
      #   activities_start_times_s = activities_start_times_s[targetActivity_trial_index_start:(targetActivity_trial_index_start + num_trials_inFile)]
      #   activities_end_times_s = activities_end_times_s[targetActivity_trial_index_start:(targetActivity_trial_index_start + num_trials_inFile)]
      #   # print([activities_start_times_s[i] - activities_start_times_s[i] for i in range(5)])
      #   # print([activities_end_times_s[i] - activities_end_times_s[i] for i in range(5)])
      #   # print([get_time_str(t) for t in activities_start_times_s])
      #   # print([get_time_str(t) for t in activities_start_times_s])
  
      # Get the hand paths.
      print('    Getting body path data')
      (time_s_byTrial, bodyPath_data_byTrial) = get_bodyPath_data_byTrial(h5_file, activities_start_times_s, activities_end_times_s)
      print('    Transforming body path data')
      (time_s_byTrial, bodyPath_data_byTrial) = transform_bodyPath_data_personFrame(time_s_byTrial, bodyPath_data_byTrial)
      print('    Resampling body path data')
      (time_s_byTrial, bodyPath_data_byTrial) = resample_bodyPath_data(time_s_byTrial, bodyPath_data_byTrial)
      # Infer the hand position while being relatively stationary
      print('    Inferring stationary poses')
      (stationary_time_s_byTrial, stationary_pose_byTrial) = infer_stationary_poses(
        time_s_byTrial, bodyPath_data_byTrial, use_variance=stationary_position_use_variance,
        hand_segment_key='RightHand')
      # Infer the reference object position
      print('    Inferring reference object positions')
      referenceObject_start_time_s_byTrial = []
      referenceObject_end_time_s_byTrial = []
      for (trial_index, stationary_pose) in enumerate(stationary_pose_byTrial):
        referenceObject_start_time_s_byTrial.append(time_s_byTrial[trial_index][stationary_pose['start_time_index']])
        referenceObject_end_time_s_byTrial.append(time_s_byTrial[trial_index][stationary_pose['end_time_index']])
      referenceObject_position_m_byTrial = infer_referenceObject_position_m_byTrial(
        bodyPath_data_byTrial, time_s_byTrial,
        referenceObject_start_time_s_byTrial, referenceObject_end_time_s_byTrial,
        referenceObject_bodySegment_name)
  
      # Store the results
      time_s_byTrial_bySubject[subject_id].extend(time_s_byTrial)
      bodyPath_data_byTrial_bySubject[subject_id].extend(bodyPath_data_byTrial)
      stationary_time_s_byTrial_bySubject[subject_id].extend(stationary_time_s_byTrial)
      stationary_pose_byTrial_bySubject[subject_id].extend(stationary_pose_byTrial)
      referenceObject_position_m_byTrial_bySubject[subject_id].extend(referenceObject_position_m_byTrial)
  
      # Plot the paths.
      if animate_trajectory_plots:
        print('    Animating the trajectories')
        for trial_index in range(len(bodyPath_data_byTrial)):
          fig_animatePath = animate_trajectory(
            bodyPath_data=bodyPath_data_byTrial[trial_index], time_s=time_s_byTrial[trial_index],
            referenceObject_position_m=referenceObject_position_m_byTrial[trial_index],
            subject_id=subject_id, num_total_subjects=1, subplot_index=0,
            trial_index=trial_index, trial_start_index_offset_forTitle=targetActivity_trial_index_start,
            include_skeleton=True, delay_s_between_timesteps='realtime',
            wait_for_user_after_timesteps=True, hide_figure_window=False,
            timestep_interval=5)
          plt.close(fig_animatePath)
      
      if plot_all_trajectories or save_plot_all_trajectories:
        print('    Plotting the trajectories')
        for trial_index in range(len(bodyPath_data_byTrial)):
          handles_allPaths = plot_timestep(
            time_s_byTrial[trial_index], time_index=0,
            bodyPath_data=bodyPath_data_byTrial[trial_index],
            referenceObject_position_m=referenceObject_position_m_byTrial[trial_index],
            subject_id=subject_id, num_total_subjects=num_subjects, subplot_index=allPaths_subplot_index,
            trial_index=trial_index, trial_start_index_offset_forTitle=targetActivity_trial_index_start,
            previous_handles=handles_allPaths, clear_previous_timestep=False,
            redraw_trajectory_each_timestep=True,
            include_skeleton=False, include_pitcher=False, include_hand=False,
            include_spout_projection=False, include_referenceObject=False,
            wait_for_user_after_plotting=False, hide_figure_window=False)
        
      if save_eye_videos:
        save_trial_eyeVideos(h5_file, eyeVideo_filepath, time_s_byTrial, subject_id,
                             trial_start_index_offset=targetActivity_trial_index_start,
                             trial_indexes_filter=None)
      if save_composite_videos:
        save_activity_composite_videos(h5_file, eyeVideo_filepath,
                                       time_s_byTrial, bodyPath_data_byTrial,
                                       referenceObject_position_m_byTrial,
                                       subject_id, trial_indexes_filter=None,
                                       trial_start_index_offset=targetActivity_trial_index_start)
      if save_animation_videos:
        save_activity_animation_videos(time_s_byTrial, bodyPath_data_byTrial,
                                       referenceObject_position_m_byTrial,
                                       subject_id, trial_indexes_filter=None,
                                       trial_start_index_offset=targetActivity_trial_index_start)
      # Close the HDF5 file.
      h5_file.close()
  
      # Increment the trial index offset counter for this subject.
      targetActivity_trial_index_start += num_trials_inFile
      
    # Advance subplot index if putting each subject in a new subplot
    if plot_all_trajectories or save_plot_all_trajectories:
      allPaths_subplot_index += 1
  
  # Save the all-trajectories plot if desired.
  if save_plot_all_trajectories:
    print('    Saving the plot of all trajectories')
    plot_dir = os.path.join(output_dir, '%s_paths_human')
    os.makedirs(plot_dir)
    handles_allPaths[0].savefig(
      os.path.join(plot_dir, '%s_paths_human_allPaths.jpg' % (target_activity_keyword)),
      dpi=300)
    
  # Export the results if desired
  if save_results_data:
    print('    Exporting data to HDF5 file')
    export_path_data(time_s_byTrial_bySubject, bodyPath_data_byTrial_bySubject,
                     stationary_time_s_byTrial_bySubject, stationary_pose_byTrial_bySubject,
                     referenceObject_position_m_byTrial_bySubject)
    
  # Show the final plot
  if animate_trajectory_plots or plot_all_trajectories:
    print('    Close the plots to continue')
    plt.show(block=True)
    
  print('Done!')
  print()
  
###################################################################
# Helpers to save videos
###################################################################

# Save the first-person videos during each hand path.
def save_trial_eyeVideos(h5_file, eyeVideo_filepath, times_s, subject_id, trial_indexes_filter=None, trial_start_index_offset=0):
  device_name = 'eye-tracking-video-worldGaze'
  stream_name = 'frame_timestamp'
  frames_time_s = h5_file[device_name][stream_name]['data']
  
  for trial_index, time_s in enumerate(times_s):
    trial_index_withOffset = trial_index+trial_start_index_offset
    print('    Saving eye video for Subject S%02d trial %02d' % (subject_id, trial_index_withOffset))
    if trial_indexes_filter is not None and trial_index not in trial_indexes_filter:
      continue
    start_time_s = min(time_s)
    end_time_s = max(time_s)
    frame_indexes = np.where((frames_time_s >= start_time_s) & (frames_time_s <= end_time_s))[0]
    start_frame_index = min(frame_indexes)
    num_frames = len(frame_indexes)
    video_reader = cv2.VideoCapture(eyeVideo_filepath)
    eyeVideo_output_dir = os.path.join(output_dir, '%s_eyeVideos' % target_activity_keyword)
    os.makedirs(eyeVideo_output_dir, exist_ok=True)
    video_writer = cv2.VideoWriter(os.path.join(eyeVideo_output_dir, '%s_eyeVideo_S%02d_%02d.mp4' % (target_activity_keyword, subject_id, trial_index_withOffset)),
                                   cv2.VideoWriter_fourcc(*'MP4V'),  # for AVI: cv2.VideoWriter_fourcc(*'MJPG'),
                                   video_reader.get(cv2.CAP_PROP_FPS),
                                   (int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                                   )
    video_reader.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index-1)
    for i in range(num_frames):
      res, frame = video_reader.read()
      video_writer.write(frame)
    video_reader.release()
    video_writer.release()

# Save the first-person videos along with the body animation.
def save_activity_composite_videos(h5_file, eyeVideo_filepath,
                                   time_s_byTrial, bodyPath_data_byTrial,
                                   referenceObject_position_m_byTrial,
                                   subject_id, trial_indexes_filter=None,
                                   trial_start_index_offset=0):
  device_name = 'eye-tracking-video-worldGaze'
  stream_name = 'frame_timestamp'
  frames_time_s = h5_file[device_name][stream_name]['data']
  
  for (trial_index, time_s) in enumerate(time_s_byTrial):
    trial_index_withOffset = trial_index + trial_start_index_offset
    print('    Saving composite video for Subject S%02d trial %02d' % (subject_id, trial_index_withOffset))
    if trial_indexes_filter is not None and trial_index not in trial_indexes_filter:
      continue
    start_time_s = time_s[0]
    end_time_s = time_s[-1]
    frame_indexes = np.where((frames_time_s >= start_time_s) & (frames_time_s <= end_time_s))[0]
    start_frame_index = min(frame_indexes)
    num_frames = len(frame_indexes)
    video_reader = cv2.VideoCapture(eyeVideo_filepath)
    video_reader.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index-1)
    video_writer = None
    previous_handles = None
    for frame_index in range(num_frames):
      if frame_index % (np.ceil(num_frames/5)) == 0 and frame_index > 0:
        print('      Completed %d/%d frames (%0.1f%%)' % (frame_index, num_frames, 100*frame_index/num_frames))
      # Get the frame of the eye video
      _, eye_frame = video_reader.read()
      # Plot the skeleton at the closest time to the eye-video frame timestamp
      target_time_s = frames_time_s[start_frame_index + frame_index]
      plot_time_index = time_s.searchsorted(target_time_s)[0]
      previous_handles = plot_timestep(
                          time_s, plot_time_index,
                          referenceObject_position_m_byTrial[trial_index],
                          bodyPath_data=bodyPath_data_byTrial[trial_index],
                          subject_id=subject_id,
                          trial_index=trial_index, trial_start_index_offset_forTitle=trial_start_index_offset,
                          previous_handles=previous_handles, include_skeleton=True,
                          wait_for_user_after_plotting=False, hide_figure_window=True)
      fig = previous_handles[0]
      plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
      plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
      plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)
      # Concatenate the eye-video frame and the plot
      eye_frame = resize_image(eye_frame, target_height=plot_img.shape[0])
      composite_frame = cv2.hconcat([plot_img, eye_frame])
      # cv2.imshow('my frame', composite_frame)
      # cv2.waitKey(10)
      
      # Write to the output video
      if video_writer is None:
        composite_output_dir = os.path.join(output_dir, '%s_composites' % target_activity_keyword)
        os.makedirs(composite_output_dir, exist_ok=True)
        video_writer = cv2.VideoWriter(os.path.join(composite_output_dir, '%s_composite_S%02d_%02d.mp4' % (target_activity_keyword, subject_id, trial_index_withOffset)),
                                       cv2.VideoWriter_fourcc(*'MP4V'), # for AVI: cv2.VideoWriter_fourcc(*'MJPG'),
                                       video_reader.get(cv2.CAP_PROP_FPS),
                                       (composite_frame.shape[1], composite_frame.shape[0])
                                       )
      video_writer.write(composite_frame)
    video_reader.release()

# Save the body animation.
def save_activity_animation_videos(time_s_byTrial, bodyPath_data_byTrial,
                                   referenceObject_position_m_byTrial,
                                   subject_id, trial_indexes_filter=None,
                                   trial_start_index_offset=0):
  
  for (trial_index, time_s) in enumerate(time_s_byTrial):
    trial_index_withOffset = trial_index + trial_start_index_offset
    print('Saving animation video for Subject S%02d trial %02d' % (subject_id, trial_index_withOffset))
    if trial_indexes_filter is not None and trial_index not in trial_indexes_filter:
      continue
    start_time_s = time_s[0]
    end_time_s = time_s[-1]
    num_frames = len(time_s)
    fps = (num_frames-1)/(end_time_s - start_time_s)
    video_writer = None
    previous_handles = None
    for frame_index in range(num_frames):
      if frame_index % (np.ceil(num_frames/5)) == 0 and frame_index > 0:
        print('      Completed %d/%d frames (%0.1f%%)' % (frame_index, num_frames, 100*frame_index/num_frames))
      # Plot the scene
      previous_handles = plot_timestep(
                          time_s, frame_index,
                          referenceObject_position_m_byTrial[trial_index],
                          bodyPath_data=bodyPath_data_byTrial[trial_index],
                          subject_id=subject_id,
                          trial_index=trial_index, trial_start_index_offset_forTitle=trial_start_index_offset,
                          previous_handles=previous_handles, include_skeleton=True,
                          wait_for_user_after_plotting=False, hide_figure_window=True)
      fig = previous_handles[0]
      plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
      plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
      plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)
      # cv2.imshow('my frame', plot_img)
      # cv2.waitKey(10)
      
      # Write to the output video
      if video_writer is None:
        animation_output_dir = os.path.join(output_dir, '%s_animations' % target_activity_keyword)
        os.makedirs(animation_output_dir, exist_ok=True)
        video_writer = cv2.VideoWriter(os.path.join(animation_output_dir, '%s_animation_S%02d_%02d.mp4' % (target_activity_keyword, subject_id, trial_index_withOffset)),
                                       cv2.VideoWriter_fourcc(*'MP4V'), # for AVI: cv2.VideoWriter_fourcc(*'MJPG'),
                                       fps,
                                       (plot_img.shape[1], plot_img.shape[0])
                                       )
      video_writer.write(plot_img)

###################################################################
# Helpers to save processed data
###################################################################

def export_path_data(time_s_byTrial_bySubject, bodyPath_data_byTrial_bySubject,
                     stationary_time_s_byTrial_bySubject, stationary_pose_byTrial_bySubject,
                     referenceObject_position_m_byTrial_bySubject):
  # Open the output HDF5 file
  hdf5_output_filepath = os.path.join(output_dir, '%s_paths_humans_%s.hdf5' % (target_activity_keyword, subject_id_toProcess))
  if os.path.exists(hdf5_output_filepath):
    print()
    print('Output file exists at [%s]' % hdf5_output_filepath)
    print('  Overwrite the file? [y/N] ', end='')
    overwrite_file = input()
    if overwrite_file.lower().strip() != 'y':
      print('  Aborting')
      return
  hdf5_file = h5py.File(hdf5_output_filepath, 'w')
  
  body_segment_names = None
  body_joint_names = None
  
  for subject_id in bodyPath_data_byTrial_bySubject:
    num_trials = len(bodyPath_data_byTrial_bySubject[subject_id])
    subject_group = hdf5_file.create_group('subject_%02d' % subject_id)
    for trial_index in range(num_trials):
      trial_group = subject_group.create_group('trial_%02d' % trial_index)
      # Add timestamps
      time_s = time_s_byTrial_bySubject[subject_id][trial_index]
      time_s = time_s - time_s[0]
      trial_group.create_dataset('time_s', data=time_s)
      # time_str = [get_time_str(t, '%Y-%m-%d %H:%M:%S.%f') for t in time_s]
      # trial_group.create_dataset('time_str', data=time_str, dtype='S26')
      
      # Add body segment position data
      data_segmentDict = bodyPath_data_byTrial_bySubject[subject_id][trial_index]['position_m']
      data = list(data_segmentDict.values())
      data = np.stack(data, axis=0)
      data = np.moveaxis(data, 0, 1) # convert from [segment][time][xyz] to [time][segment][xyz]
      trial_group.create_dataset('body_segment_position_m', data=data)
      
      # Add body segment orientation data
      data_segmentDict = bodyPath_data_byTrial_bySubject[subject_id][trial_index]['quaternion_wijk']
      data = list(data_segmentDict.values())
      data = np.stack(data, axis=0)
      data = np.moveaxis(data, 0, 1) # convert from [segment][time][wxyz] to [time][segment][wxyz]
      trial_group.create_dataset('body_segment_quaternion_wijk', data=data)
      
      # Add body joint angle data
      data_jointDict = bodyPath_data_byTrial_bySubject[subject_id][trial_index]['joint_angle_eulerZXY_xyz_rad']
      data = list(data_jointDict.values())
      data = np.stack(data, axis=0)
      data = np.moveaxis(data, 0, 1) # convert from [joint][time][xyz] to [time][joint][xzy]
      trial_group.create_dataset('joint_angle_eulerZXY_xyz_rad', data=data)
      data_jointDict = bodyPath_data_byTrial_bySubject[subject_id][trial_index]['joint_angle_eulerXZY_xyz_rad']
      data = list(data_jointDict.values())
      data = np.stack(data, axis=0)
      data = np.moveaxis(data, 0, 1) # convert from [joint][time][xyz] to [time][joint][xzy]
      trial_group.create_dataset('joint_angle_eulerXZY_xyz_rad', data=data)
      
      # Add estimated stationary position
      stationary_group = trial_group.create_group('stationary')
      data_segmentDict = stationary_pose_byTrial_bySubject[subject_id][trial_index]['position_m']
      data = list(data_segmentDict.values())
      data = np.stack(data, axis=0)
      stationary_group.create_dataset('body_segment_position_m', data=data)
      
      # Add estimated stationary quaternion
      data_segmentDict = stationary_pose_byTrial_bySubject[subject_id][trial_index]['quaternion_wijk']
      data = list(data_segmentDict.values())
      data = np.stack(data, axis=0)
      stationary_group.create_dataset('body_segment_quaternion_wijk', data=data)
      # Add stationary time
      stationary_group.create_dataset('time_s',
                                      data=stationary_time_s_byTrial_bySubject[subject_id][trial_index] - min(time_s))
      
      # Add estimated reference object position
      referenceObject_group = trial_group.create_group('reference_object')
      data = referenceObject_position_m_byTrial_bySubject[subject_id][trial_index]
      data = np.stack(data, axis=0)
      referenceObject_group.create_dataset('position_m', data=data)
      
      # Store body segment and joint names.
      # This should be the same for every subject/trial.
      body_segment_names_forTrial = list(bodyPath_data_byTrial_bySubject[subject_id][0]['position_m'].keys())
      body_joint_names_forTrial = list(bodyPath_data_byTrial_bySubject[subject_id][0]['joint_angle_eulerZXY_xyz_rad'].keys())
      if body_segment_names is None:
        body_segment_names = body_segment_names_forTrial
        body_joint_names = body_joint_names_forTrial
      else:
        assert body_segment_names_forTrial == body_segment_names
        assert body_joint_names_forTrial == body_joint_names
  
  # Add segment names
  hdf5_file.create_dataset('body_segment_names', data=body_segment_names)
  # Add joint names
  hdf5_file.create_dataset('body_joint_names', data=body_joint_names)

  # Close the output file
  hdf5_file.close()

###################################################################
# Various helpers
###################################################################

# Resize an image.
def resize_image(img, target_width=None, target_height=None):
  img_width = img.shape[1]
  img_height = img.shape[0]
  if target_width is not None and target_height is not None:
    # Check if the width or height will be the controlling dimension.
    scale_factor_fromWidth = target_width/img_width
    scale_factor_fromHeight = target_height/img_height
    if img_height*scale_factor_fromWidth > target_height:
      scale_factor = scale_factor_fromHeight
    else:
      scale_factor = scale_factor_fromWidth
  elif target_width is not None:
    scale_factor = target_width/img_width
  elif target_height is not None:
    scale_factor = target_height/img_height
  else:
    raise AssertionError('No target dimension provided when resizing the image')
  # Resize the image.
  return cv2.resize(img, (0,0), None, scale_factor, scale_factor)


###################################################################
# Run
###################################################################

if __name__ == '__main__':
  main_processing()