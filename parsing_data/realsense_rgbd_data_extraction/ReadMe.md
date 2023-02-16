
# Realsense RGBD Depth Camera Data Extraction

See [https://action-net.csail.mit.edu](https://action-net.csail.mit.edu) for more information about the ActionSense project and dataset.

## About the Files

### .txt Files
The text files in this directory are passed in as parameters to the python functions.

| Filename      | Description |
| ------------- | ----------- |
| *available_ros_files.txt* | - lists the filenames of .bag ROS files that contain depth camera data<br>- each line should be a different filename ending in .bag<br> - listed .bag files will be parsed by parse_ros_depth_data.py in order to save the relevant data in a more accesible hdf5 file|
| *convert_to_video.txt*   | - each line is a comma-delimited list of parameters for generating videos from the hdf5 depth camera file <br> - the format is ```name_of_hdf5_file.hdf5, "depth" or "raw" for the video type, start offset in seconds, duration in seconds``` <br> - Ex) ```kitchen_depth-depth_2022-06-07-17-31-35_cameras.hdf5,depth,0,2``` for a 2 second-long 3D depth video starting from the beginning of the given file


### .py Files
```
generate_depth_video.py
parse_ros_depth_data.py
ros_to_images.py
```


### previous_iterations
```
ros_to_images_rgb.m
ros_to_pointClouds_textFiles.py
ros_to_pointClouds_xyz-rgb.m
```
The files in this directory are previous iterations (either in Python or MATLAB) of the methods that have now been fully translated into Python. Please do not use the files in previous_iterations for transforming/using any depth camera data.