import rosbag
import rospy
import os
import numpy as np
from sensor_msgs import point_cloud2
import struct, ctypes
import h5py
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from datetime import datetime
import cv2

def data_generation(bagfile, start_offset=0, duration=None):
  '''
  Extract depth data (xyz points w/ rgb values, timestamps) from a ROS bagfile
  Parameters:
      bagfile: absolute path to the bagfile
      start_offset: time (in sec) after start of bagfile to start saving depth data from
                    0sec by default (start from the beginning of the file)
      duration: duration (in sec) of bagfile stream to save depth data from
                None by default (parse the whole file)
  Returns:
      xyz: 3d, float-32 numpy array w/ dimensions (# of frames, # of points in frame, 3 - xyz points)
      rgb: 3d, uint8 numpy array w/ dimensions (# of frames, # of points in frame, 3 - rgb values)
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

  for topic, msg, t in bag.read_messages(topics="/kitchen_depth/depth/color/points", start_time=start_offset_s, end_time=start_offset_s+duration_s):
    # save timestamp information
    time_stamp = msg.header.stamp.secs + (msg.header.stamp.nsecs/1000000000.0)
    time_stamp_str = datetime.utcfromtimestamp(time_stamp)
    time_stamps = np.append(time_stamps, time_stamp)
    time_stamp_strs = np.append(time_stamp_strs, bytes(time_stamp_str.strftime("%Y-%m-%d %H:%M:%S.%f"), 'utf-8'))
    
    counter = 0
    points = np.array([[0, 0, 0, 0, 0, 0]])
    for point in point_cloud2.read_points(msg, skip_nans=True):
      counter += 1
      # extract rgb values
      if counter % 5 == 0:
        s = struct.pack('>f' ,point[3])
        i = struct.unpack('>l',s)[0]
        pack = ctypes.c_uint32(i).value
        r = (pack & 0x00FF0000)>> 16
        g = (pack & 0x0000FF00)>> 8
        b = (pack & 0x000000FF)
                            
        points = np.append(points,[[*point[:3], r, g, b]], axis = 0)
    points = points[1:]
    print(counter)
    # look for frame with the most points, for standardizing array dimensions later 
    if points.shape[0] > longest_frame:
      longest_frame = points.shape[0]
      
    frames.append(points)
        
  # add rows of 0s as padding to standardize array dimensions  
  for i, frame in enumerate(frames):
    frames[i] = np.pad(frame, [(0,longest_frame-frame.shape[0]),(0,0)], mode='constant', constant_values=0)
  
  # save as smaller datatypes to save space
  frames = np.stack(frames, axis=0)
  xyz = frames[:,:,:3].astype('float32')
  rgb = frames[:,:,3:].astype('uint8')
  return xyz,rgb,time_stamps,time_stamp_strs

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
  f = h5py.File(file, 'w')
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
  if timestamps:
    title = f"Timestamp: {timestamps[index].decode('utf-8')} ({title})"
  ax.set_title(title)
  plt.show()
  
def video_from_frames(frames, timestamps, video_dir, video_name, save_intermediates=False, xlim=(-1.5, 1.5), ylim=(-0.5,0.0), zlim=(0,1.5)):
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
    
    filename = f'{video_dir}/intermediate_frames/{i}.png'
    plt.savefig(filename, dpi=100)
    
    if i == 0:
      frame = cv2.imread(filename)
      height, width, layers = frame.shape
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
  cv2.destroyAllWindows()
  video.release()

if __name__ == '__main__':
  pass
  # data_dir = "C:/Users/2021l/Documents/UROP/test_depth_conversion/"
  # filename = "kitchen_depth-depth_2022-06-07-17-31-35.bag"
  # # frames = data_generation(data_dir+filename)
  # # save_to_hdf5(data_dir+'kitchen_depth-depth_2022-06-07-17-31-35_cameras-condensed.hdf5', ['depth-data/xyz', 'depth-data/rgb', 'depth-data/time_s', 'depth-data/time_str'], frames)
  # xyz, rgb, timestamps = load_from_hdf5(data_dir+'kitchen_depth-depth_2022-06-07-17-31-35_camera_test_test.hdf5', ['depth-data/xyz', 'depth-data/rgb', 'depth-data/time_str'])
  # frames = np.concatenate((xyz, rgb), axis=2)
  # video_from_frames(frames, timestamps, data_dir+'depth-camera', "test_compressed.mp4", save_intermediates=True)
  # # # plot_3d_frame(frames, 5, timestamps)