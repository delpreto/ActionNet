from ros_to_images import *

# Script to loop through all bagfiles listed in "available_ros_files.txt",
# parse the depth camera data, and save it to an hdf5 file

data_dir = "C:/Users/2021l/Documents/UROP/test_depth_conversion/" #TODO modify based on lab computer

with open("parsing_data/realsense_rgbd_data_extraction/available_ros_files.txt", 'r') as f:
    files = f.read().split('\n')
    
for filename in files:
    if filename[-4:] == '.bag':
        frames = data_generation(data_dir+filename)
        save_to_hdf5(data_dir+f'{filename[:-4]}_cameras.hdf5', ['depth-data/xyz', 'depth-data/rgb', 'depth-data/time_s', 'depth-data/time_str'], frames)
