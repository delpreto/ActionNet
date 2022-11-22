import rosbag
import rospy
import os
import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sensor_msgs import point_cloud2
from rosbags import image
from mpl_toolkits import mplot3d
from datetime import datetime

def data_generation(bagfile, depth, start_offset=0, duration=None):
  '''
  Extract depth data (xyz points w/ rgb values, timestamps) from a ROS bagfile
  Parameters:
      bagfile: absolute path to the bagfile
      depth (bool): True for parsing depth-depth 3d images, False for parsing depth-raw 2d images
      start_offset: time (in sec) after start of bagfile to start saving depth data from
                    0sec by default (start from the beginning of the file)
      duration: duration (in sec) of bagfile stream to save depth data from
                None by default (parse the whole file)
  Returns:
      xyz: for depth only, 3d, float-32 numpy array w/ dims (frames, points in frame, 3 - xyz points)
      rgb: for depth only, 3d, uint8 numpy array w/ dims (frames, points in frame, 3 - rgb values)
      frames: for raw only, 4d, uint8 numpy array w/ dims (frames, (dims of image), 3 - rgb values)
      time_stamps: 1d, float-64 numpy array with epoch timestamps
      time_stamp_strs: 1d, numpy byte array with timestamps in form "YYYY-mm-dd HH:MM:SS.ffffff"
  '''
  bag = rosbag.Bag(bagfile)
  
  # convert start_offset and duration to ROS Time objects
  start_time = rospy.Time.from_sec(bag.get_start_time())
  start_offset_s = start_time + rospy.Duration.from_sec(start_offset)
  if duration is None:
    duration_s = rospy.Duration.from_sec(bag.get_end_time() - bag.get_start_time())
  else:
    duration_s = rospy.Duration.from_sec(duration)
    
  frames = []
  longest_frame = 0  
  time_stamps = np.array([])
  time_stamp_strs = np.array([])
  
  # use correct topics title for each type of bag file (depth vs. raw)
  topics = "/kitchen_depth/depth/color/points" if depth else "/kitchen_depth/color/image_raw"

  for topic, msg, t in bag.read_messages(topics=topics, start_time=start_offset_s, end_time=start_offset_s+duration_s):
    # save timestamp information
    time_stamp = msg.header.stamp.secs + (msg.header.stamp.nsecs/1000000000.0)
    time_stamp_str = datetime.utcfromtimestamp(time_stamp)
    time_stamps = np.append(time_stamps, time_stamp)
    time_stamp_strs = np.append(time_stamp_strs, bytes(time_stamp_str.strftime("%Y-%m-%d %H:%M:%S.%f"), 'utf-8'))
    
    if depth:
      counter = 0
      points = np.array([[0, 0, 0]])
      colors = np.array([])
      for point in point_cloud2.read_points(msg, skip_nans=True):
        counter += 1
        # save point data
        colors = np.append(colors, point[3])
        points = np.append(points,[[*point[:3]]], axis = 0)
      print(counter)
      
      # extract rgb data
      packed = colors.astype('>f').tobytes()
      unpacked = np.frombuffer(packed, dtype='>l')
      colors = unpacked.astype('int32')   
      unpacked_colors = ((colors & 0x00FF0000)>> 16, (colors & 0x0000FF00)>> 8, (colors & 0x000000FF))
      rgbs = np.dstack(unpacked_colors)[0]
      points = points[1:]
      
      # look for frame with the most points, for standardizing array dimensions later 
      if points.shape[0] > longest_frame:
        longest_frame = points.shape[0]
      
      frames.append(np.concatenate((points,rgbs), axis=1))
    else:
      # save image array (converted to bgr8 for opencv)
      img = image.message_to_cvimage(msg, 'bgr8')
      frames.append(img)
        
  # add rows of 0s as padding to standardize array dimensions for depth arrays
  if depth:
    for i, frame in enumerate(frames):
      frames[i] = np.pad(frame, [(0,longest_frame-frame.shape[0]),(0,0)], mode='constant', constant_values=0)
  
    # save as smaller datatypes to save space
    frames = np.stack(frames, axis=0)
    xyz = frames[:,:,:3].astype('float32')
    rgb = frames[:,:,3:].astype('uint8')
    return xyz,rgb,time_stamps,time_stamp_strs
  else:
    frames = np.stack(frames, axis=0)       
    return frames,time_stamps,time_stamp_strs

def save_to_hdf5(file, dataset_names, data, compression_level=9):
  '''
  Save numpy arrays to an hdf5 file
  Automatically compresses with gzip level 9 (https://docs.h5py.org/en/stable/high/dataset.html#filter-pipeline)
  Parameters:
      file: absolute path to the bagfile to write to
      dataset_names: list of paths to data within the hdf5 file
                     i.e. "/camera1/stream1" will save a dataset called "stream1" in the camera1 group
      data: list of numpy arrays to save as datasets, must correspond to stream names in dataset_names
  Returns:
      None
  '''
  f = h5py.File(file, 'a')
  for name, arr in zip(dataset_names, data):
    f.create_dataset(name, data=arr, compression='gzip', compression_opts=compression_level)
  f.close()
  
def load_from_hdf5(file, dataset_names):
  f = h5py.File(file, 'r')
  datasets = []
  for name in dataset_names:
    datasets.append(f[name])
  return datasets
  
def plot_3d_frame(frames, index, timestamps=None, xlim=(-1.5, 1.5), ylim=(-0.5,0.0), zlim=(0,1.5)):
  frame = frames[index]
  fig = plt.figure()
  fig.set_size_inches(12, 9)
  ax = plt.axes(projection='3d')
  ax.scatter3D(frame[:, 0], frame[:, 1], frame[:, 2], c=frame[:, 3:]/255.0)
  
  ax.set_xlim(*xlim)
  ax.set_ylim(*ylim)
  ax.set_zlim(*zlim)
  ax.view_init(235, 270)
  
  title = f"Frame {index}/{frames.shape[0]}"
  if timestamps is not None:
    title = f"Timestamp: {timestamps[index].decode('utf-8')} ({title})"
  ax.set_title(title)
  plt.show()
  
def get_snippet(hdf5_file, depth, start_offset, duration=None):
  if depth == 'depth':
    timestamps, timestrings, xyz, rgb = load_from_hdf5(hdf5_file, ['depth-data/time_s', 'depth-data/time_str', 'depth-data/xyz', 'depth-data/rgb'])
    data = np.concatenate((xyz, rgb), axis=2)
  else:
    timestamps, timestrings, data = load_from_hdf5(hdf5_file, ['depth-raw/time_s', 'depth-raw/time_str', 'depth-raw/rgb'])
    
  # convert start_offset and duration to ROS Time objects
  # start_time = timestamps[0]
  start_time = timestamps[0] + start_offset
  end_time = timestamps[-1] if duration is None else start_time + duration 
  print(start_time, end_time)
  timestamps = list(timestamps)
  
  start_ind = 0
  end_ind = -1
  for i, time in enumerate(timestamps):
    if time > start_time: 
      start_ind = i
      break
    
  for i, time in enumerate(timestamps[::-1], 1):
    if time < end_time:
      end_ind = -i+1
      break

  return data[start_ind:end_ind], timestrings[start_ind:end_ind]

def depth_video_from_frames(frames, video_dir, video_name, timestamps, save_intermediates=False, xlim=(-1.5, 1.5), ylim=(-0.5,0.0), zlim=(0,1.5)):
  if not os.path.exists(video_dir):
    os.mkdir(video_dir)
  if not os.path.exists(video_dir+'/intermediate_frames'):
    os.mkdir(video_dir+'/intermediate_frames')
      
  for i, frame in enumerate(frames):
    fig = plt.figure()
    fig.set_size_inches(12, 9)
    ax = plt.axes(projection='3d')

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.view_init(235, 270)

    ax.scatter3D(frame[:, 0], frame[:, 1], frame[:, 2], c=frame[:, 3:]/255.0)
    ax.set_title(f"Timestamp: {timestamps[i].decode('utf-8')} (Frame {i}/{frames.shape[0]})")
    
    filename = f'{video_dir}/intermediate_frames/depth_{i}.png'
    plt.savefig(filename, dpi=100)
    
    if i == 0:
      frame = cv2.imread(filename)
      height = frame.shape[0]
      width = frame.shape[1]
      if video_name[-3:] == 'mp4':
        video = cv2.VideoWriter(f'{video_dir}/{video_name}', cv2.VideoWriter_fourcc(*'mp4v'), 1, (width, height))
      else:
        video = cv2.VideoWriter(f'{video_dir}/{video_name}', 0, 1, (width, height))
    video.write(cv2.imread(filename))
    
    if not save_intermediates:
      os.remove(filename)
    
    plt.close()
    plt.clf()
    plt.cla()
    
  if not save_intermediates: os.rmdir(video_dir+'/intermediate_frames')
  video.release()
  cv2.destroyAllWindows()
  
def raw_video_from_frames(frames, video_dir, video_name, timestamps=None, save_intermediates=False):
  if not os.path.exists(video_dir):
    os.mkdir(video_dir)
  if not os.path.exists(video_dir+'/intermediate_frames'):
    os.mkdir(video_dir+'/intermediate_frames')
      
  for i, frame in enumerate(frames):
    if timestamps is not None:
      frame = cv2.putText(frame,
                          f"Timestamp: {timestamps[i].decode('utf-8')} (Frame {i}/{frames.shape[0]})",
                          (20,frames.shape[1]-20),
                          cv2.FONT_HERSHEY_PLAIN, 1,
                          (0,0,0))
    
    if i == 0:
      height = frame.shape[0]
      width = frame.shape[1]
      if video_name[-3:] == 'mp4':
        video = cv2.VideoWriter(f'{video_dir}/{video_name}', cv2.VideoWriter_fourcc(*'mp4v'), 1, (width, height))
      else:
        video = cv2.VideoWriter(f'{video_dir}/{video_name}', 0, 1, (width, height))
    video.write(frame)
    
    if save_intermediates:
      cv2.imwrite(f'{video_dir}/intermediate_frames/raw_{i}.png', frame)
    
  if not save_intermediates: os.rmdir(video_dir+'/intermediate_frames')
  video.release()
    
if __name__ == '__main__':
  pass
  data_dir = "C:/Users/2021l/Documents/UROP/test_depth_conversion/"
  filename3d = "kitchen_depth-depth_2022-06-07-17-31-35.bag"
  # filename2d = "kitchen_depth-raw_2022-06-07-17-31-35.bag"
  # frames = data_generation(data_dir+filename3d, True, 100, 2)
  # save_to_hdf5(data_dir+filename3d[:-4]+'_cameras.hdf5', ['depth-data/xyz', 'depth-data/rgb', 'depth-data/time_s', 'depth-data/time_str'], frames)
  # frames = data_generation(data_dir+filename2d, False, 100, 2)
  # save_to_hdf5(data_dir+filename3d[:-4]+'_cameras.hdf5', ['depth-raw/rgb', 'depth-raw/time_s', 'depth-raw/time_str'], frames)
  # xyz, rgb, timestamps = load_from_hdf5(data_dir+filename3d[:-4]+'_cameras.hdf5', ['depth-data/xyz', 'depth-data/rgb', 'depth-data/time_str'])
  # frames = np.concatenate((xyz, rgb), axis=2)
  # depth_video_from_frames(frames, data_dir+'camera', "test_vectorized.mp4", timestamps, save_intermediates=True)
  # frames, timestamps = load_from_hdf5(data_dir+filename3d[:-4]+'_cameras.hdf5', ['depth-raw/rgb', 'depth-raw/time_str'])
  # raw_video_from_frames(frames, data_dir+'camera', "test_bgr8.mp4", timestamps, save_intermediates=True)
  # data, times = get_snippet(data_dir+filename3d[:-4]+'_cameras.hdf5', True, 1, 0.5)
  # plot_3d_frame(frames, 5, timestamps)