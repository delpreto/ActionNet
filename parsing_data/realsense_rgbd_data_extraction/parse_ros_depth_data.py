from ros_to_images import *
import sys

# Script to loop through all bagfiles listed in "available_ros_files.txt",
# parse the depth camera data, and save it to an hdf5 file

try:
    data_dir = sys.argv[1] #"C:/Users/DRL Public/Desktop/ActionNet/external_data/2022-06-07_experiment_S00/"
except:
    print("Please run this script with the absolute path to a directory with .bag files")
    
with open("available_ros_files.txt", 'r') as f:
    files = f.read().split('\n')
    
for filename in files:
    if filename[-4:] == '.bag':
        if 'raw' in filename:
            frames = data_generation(data_dir+filename, False, 0, 1)
            if frames is not None:
                save_to_hdf5(data_dir+f'{filename[-23:-4]}_cameras.hdf5', ['depth-raw/rgb', 'depth-raw/time_s', 'depth-raw/time_str'], frames)
        else:
            pass
            parsing_chunks, chunk_time, longest_frame = make_parsing_chunks(data_dir+filename, True, 0, 2)
            for start_time in parsing_chunks:
                frames = data_generation(data_dir+filename, True, longest_frame, start_time, chunk_time)
                if frames is not None:
                    save_to_hdf5(data_dir+f'{filename[-23:-4]}_cameras.hdf5', ['depth-data/xyz', 'depth-data/rgb', 'depth-data/time_s', 'depth-data/time_str'], frames)