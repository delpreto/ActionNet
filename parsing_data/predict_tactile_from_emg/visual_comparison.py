from parse_hdf5_data import *
from extract_activities_hdf5 import *
import ujson
import matplotlib.pyplot as plt
import numpy as np
import os

data_dir = "C:/Users/2021l/Documents/UROP/data/"
visualization_file = "2022-06-07_18-11-37_streamLog_actionNet-wearables_S00.hdf5"
save_dir = data_dir+"comparison_pictures/"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
# # get all activities labels
# activities_labels, activities_start_times_s, activities_end_times_s = extract_label_data(data_dir+visualization_file)
# for activity in set(activities_labels):
#     print(f'{activity}|', end='')
    
#### save all data in a json (only needs to be run once to create the json)
# data_dir = "C:/Users/2021l/Documents/UROP/data/"
# requests_file = 'request_yamls/all_streams.yaml'
# extracted_streams = {}
# for file in os.listdir(data_dir):
#     if file[-4:] == "hdf5":
#         subj_stream = extract_streams_for_activities(data_dir+file, requests_file)
#         extracted_streams[file[-8:-5]] = subj_stream

# with open('all_streams.json', 'w') as f:
#     f.write(ujson.dumps(extracted_streams))

# Plot all channels
with open('short.json') as json_file:
    data = ujson.load(json_file)['S00']
    activities = data['time_s'].keys()
    
for act in activities:
    combine_streams = zip(data['time_s'][act], data['tactile-glove-left']['tactile_data'][act], data['tactile-glove-right']['tactile_data'][act], data['myo-left']['emg'][act], data['myo-right']['emg'][act])
    activity_count = 1
    for times, tact_lh, tact_rh, emg_lh, emg_rh in combine_streams:
        # times = sum(data['time_s'][act], [])
        # tact_lh = sum(data['tactile-glove-left']['tactile_data'][act], [])
        # tact_rh = sum(data['tactile-glove-right']['tactile_data'][act], [])
                
        avg_tact_lh = [find_fingers(frame) for frame in tact_lh]
        avg_tact_lh = list(zip(*avg_tact_lh))
        avg_tact_rh = [find_fingers(frame) for frame in tact_rh]
        avg_tact_rh = list(zip(*avg_tact_rh))
            
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(12, 6)
                    
        # visualizing emg data
        # emg_lh = np.array(sum(data['myo-left']['emg'][act], []))
        # emg_rh = np.array(sum(data['myo-right']['emg'][act], []))

        avg_emg_lh = np.average(emg_lh, 1)
        avg_emg_rh = np.average(emg_rh, 1)
        
        ax3 = ax1.twinx()
        ax4 = ax2.twinx()
        ax1.plot(times[50:], avg_emg_lh[50:], 'k')
        ax2.plot(times[50:], avg_emg_rh[50:], 'k')
        
        for i in range(len(avg_tact_lh)):
            filtered1 = butter_filter(avg_tact_lh[i], times, 2)
            filtered2 = butter_filter(avg_tact_rh[i], times, 2)

            ax3.plot(times[50:], filtered1[50:])
            ax4.plot(times[50:], filtered2[50:])

        # plt.show()
        act = act.replace(':', '')
        act = act.replace('/', ' or ')
        plt.legend(["Thumb", "Index", "Middle", "Ring", "Pinky"])
        plt.title(f'{act}_{activity_count}')
        plt.savefig(f'{save_dir}{act}_{activity_count}.png')
        activity_count += 1
        plt.close()