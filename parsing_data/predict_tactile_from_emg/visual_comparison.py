from parse_hdf5_data import *
from extract_activities_hdf5 import *
import ujson
import matplotlib.pyplot as plt
import numpy as np

data_dir = "C:/Users/2021l/Documents/UROP/data/"
visualization_file = "2022-06-07_18-11-37_streamLog_actionNet-wearables_S00.hdf5"

# # get all activities labels
# activities_labels, activities_start_times_s, activities_end_times_s = extract_label_data(data_dir+visualization_file)
# for activity in set(activities_labels):
#     print(f'{activity}|', end='')
    
# # save all data in a json
# requests_file = 'visual_comparison_extract.txt'
# extracted_streams = {}
# subj_stream = extract_streams_for_activities(data_dir+visualization_file, requests_file)
# extracted_streams[visualization_file[-8:-5]] = subj_stream

# with open('extracted_streams_2acts.json', 'w') as f:
#     f.write(ujson.dumps(extracted_streams))

# Plot all channels
with open('extracted_streams_1.json') as json_file:
    data = ujson.load(json_file)['S00']
    activities = data['time_s'].keys()
    
for act in activities:
    # visualizing tactile data
    times = sum(data['time_s'][act], [])
    tact_lh = sum(data['tactile-glove-left']['tactile_data'][act], [])
    tact_rh = sum(data['tactile-glove-right']['tactile_data'][act], [])
            
    avg_tact_lh = [find_fingers(frame) for frame in tact_lh]
    avg_tact_lh = list(zip(*avg_tact_lh))
    avg_tact_rh = [find_fingers(frame) for frame in tact_rh]
    avg_tact_rh = list(zip(*avg_tact_rh))
        
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    # visualizing emg data
    emg_lh = np.array(sum(data['myo-left']['emg'][act], []))
    emg_rh = np.array(sum(data['myo-right']['emg'][act], []))

    avg_emg_lh = np.average(emg_lh, 1)
    avg_emg_rh = np.average(emg_rh, 1)
    
    ax3 = ax1.twinx()
    ax4 = ax2.twinx()
    ax3.plot(times, avg_emg_lh)
    ax4.plot(times, avg_emg_rh)
    
    for i in range(len(avg_tact_lh)):
        filtered1 = butter_filter(avg_tact_lh[i], times, 2)
        filtered2 = butter_filter(avg_tact_rh[i], times, 2)

        ax1.plot(times[100:], filtered1[100:])
        ax2.plot(times[100:], filtered2[100:])
        

    plt.show()
    
#       tactiles = data['S00']['tactile-glove-right']['tactile_data']['Slice bread']
#   times = data['S00']['time_s']['Slice bread']
#   tactiles = tactiles[0] # + tactiles[1] + tactiles[2]
#   times = times[0]
#   avg_tactiles = [find_fingers(frame) for frame in tactiles]
  
#   avg_tactiles_by_finger = list(zip(*avg_tactiles))
  
#   plt.plot(times, avg_tactiles_by_finger[1])
#   filtered1 = butter_filter(avg_tactiles_by_finger[1], times) #scipy example
#   filtered2 = butter_filter(avg_tactiles_by_finger[1], times, version='matlab') #matlab example  
#   plt.plot(times, filtered1)
#   plt.plot(times, filtered2)
#   plt.xlim(left=1654641154.59)
#   plt.ylim((560, 580))
#   plt.show()
# for stream in data.keys():
#     print(stream)
