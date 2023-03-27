from ros_to_images import *
import time
# Script to loop through all bagfiles listed in "available_ros_files.txt",
# parse the depth camera data, and save it to an hdf5 file

data_dir = "C:/Users/2021l/Documents/UROP/test_depth_conversion/" #"C:/Users/DRL Public/Desktop/ActionNet/external_data/2022-06-07_experiment_S00/" #TODO modify based on lab computer

with open("available_ros_files.txt", 'r') as f:
    files = f.read().split('\n')
    
for filename in files:
    if filename[-4:] == '.bag':
        if 'raw' in filename:
            pass
            frames = data_generation(data_dir+filename, False)
            save_to_hdf5(data_dir+f'{filename[-23:-4]}_cameras.hdf5', ['depth-raw/rgb', 'depth-raw/time_s', 'depth-raw/time_str'], frames)
        else:
            pass
            timer = time.perf_counter()
            # parsing_chunks, chunk_time, longest_frame = make_parsing_chunks(data_dir+filename)
            parsing_chunks = [400, 450]
            chunk_time = 50
            longest_frame = 275367
            print("Finished prep", time.perf_counter()-timer)
            print(longest_frame, len(parsing_chunks))
            for start_time in parsing_chunks:
                print(start_time)
                frames = data_generation(data_dir+filename, True, longest_frame, start_time, chunk_time)
                print("Parsed ", start_time, time.perf_counter()-timer)
                if frames is not None:
                    save_to_hdf5(data_dir+f'{filename[-23:-4]}_sixth_try.hdf5', ['depth-data/xyz', 'depth-data/rgb', 'depth-data/time_s', 'depth-data/time_str'], frames)
                    print("Saved ", start_time, time.perf_counter()-timer)

            print("Finished!", time.perf_counter()-timer)