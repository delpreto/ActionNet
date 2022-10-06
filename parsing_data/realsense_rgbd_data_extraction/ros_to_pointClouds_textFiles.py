#!/usr/bin/env pythonfrom sensor_msgs.msg import PointCloud2

from ros_numpy.point_cloud2 import pointcloud2_to_array
import rosbag
import os
import numpy as np

def data_generation(data_dir, filename):
  bagfile = data_dir + filename
  bag = rosbag.Bag(bagfile)
  output_dir = data_dir + filename[filename.index('-')+1:-4]
  os.mkdir(output_dir)

  index = 0
  for topic, msg, t in bag.read_messages(topics="/kitchen_depth/depth/color/points"):
    time_stamp_sec = msg.header.stamp.secs
    time_stamp_nsec = msg.header.stamp.nsecs
    # pointcloud_data = msg.data
    pointcloud_array_data = pointcloud2_to_array(msg)
    pointcloud_file = open(output_dir+"/frame%06i.txt" % index, 'w')
    pointcloud_file.write("time_stamp_sec: " + str(time_stamp_sec) + "\n")
    pointcloud_file.write("time_stamp_nsec: " +
                          str(time_stamp_nsec) + "\n")
    pointcloud_file.write("height: " + str(msg.height) + "\n")
    pointcloud_file.write("width: " + str(msg.width) + "\n")
    pointcloud_file.write("point step: " + str(msg.point_step) + "\n")
    pointcloud_file.write("row step: " + str(msg.row_step) + "\n")
    # pointcloud_file.write(str(pointcloud_array_data))
    pointcloud_file.close()
    with open(output_dir+"/frame%06i.txt" % index, "ab") as pointcloud_file:
        pointcloud_file.write(b"data:\n")
        np.savetxt(pointcloud_file, pointcloud_array_data)
    index += 1

if __name__ == '__main__':
  data_dir = "/run/user/1000/gvfs/smb-share:server=actionnet-data.local,share=data/experiments/2022-06-07-S00/camera/"
  filename = "kitchen_depth-depth_2022-06-07-17-31-35.bag"
  data_generation(data_dir, filename)




