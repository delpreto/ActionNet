from ros_to_images import *
import sys
# Script to loop through all hdf5 files listed in 'convert_to_video.txt'
# with the proper format (filename, depth/raw, start offset in seconds, duration in seconds)
# saves the video in specified directory

try:
    data_dir = sys.argv[1]
    video_dir = data_dir+'videos/'
except:
    print("Please run this script with the absolute path to a directory that contains .hdf5 files ending with a backslash (/)")
    sys.exit()
    
with open("convert_to_video.txt", 'r') as f:
    jobs = f.read().split('\n')
    
for job in jobs:
    job = job.split(',')
    filename = job[0][:-5]
    try:
        assert len(job) == 4
        job[0] = data_dir+job[0]
        job[2] = float(job[2])
        job[3] = float(job[3])
    except:
        print(job, " is not in the proper format (hdf5 filename, depth/raw, start offset, duration)")
        continue
    
    data, timestrings = get_snippet(*job)
    video_name = f'{filename}_{job[1]}_{job[2]}s_to_{job[3]+job[2]}s.mp4'
    if job[1] == 'depth':
        depth_video_from_frames(data, video_dir, video_name, timestrings)
    else:
        raw_video_from_frames(data, video_dir, video_name, timestrings)
