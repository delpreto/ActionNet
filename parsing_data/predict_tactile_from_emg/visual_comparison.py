from parse_hdf5_data import *
from extract_activities_hdf5 import *
import ujson
import matplotlib.pyplot as plt
import numpy as np
import os

# script to create plots overlaying tactile signal and emg signal for each activity
# gets data for the plots from a json file

data_dir = "FILL IN CORRECT DIRECTORY LATER"
json_filename = "all_activities.json"
save_dir = data_dir+"comparison_pictures/"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
    
# Plot all activities from data of first subject saved in json
with open(json_filename) as json_file:
    data = ujson.load(json_file)
    data = data[data.keys()[0]] # getting data from first subject only
    activities = data['time_s'].keys()
    
for act in activities:
    combine_streams = zip(data['time_s'][act], data['tactile-glove-left']['tactile_data'][act], data['tactile-glove-right']['tactile_data'][act], data['myo-left']['emg'][act], data['myo-right']['emg'][act])
    activity_count = 1
    
    for times, tact_lh, tact_rh, emg_lh, emg_rh in combine_streams:    
        # plot new line for each finger (explanation of how "finger" areas were approximated can be found in parse_hdf5_data.py)
        avg_tact_lh = [find_fingers(frame) for frame in tact_lh]
        avg_tact_lh = list(zip(*avg_tact_lh))
        avg_tact_rh = [find_fingers(frame) for frame in tact_rh]
        avg_tact_rh = list(zip(*avg_tact_rh))
            
        # create 2 plots, one for each hand
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(12, 6)
                    
        # visualizing emg data
        avg_emg_lh = np.average(emg_lh, 1)
        avg_emg_rh = np.average(emg_rh, 1)
        
        # plot emg and tactile data on two axes on the same plot to deal with difference in scaling
        ax3 = ax1.twinx()
        ax4 = ax2.twinx()
        ax1.plot(times[50:], avg_emg_lh[50:], 'k')
        ax2.plot(times[50:], avg_emg_rh[50:], 'k')
        
        for i in range(len(avg_tact_lh)):
            ax3.plot(times[50:], avg_tact_lh[i][50:])
            ax4.plot(times[50:], avg_tact_rh[i][50:])

        # plt.show()
        
        # Save file, named with activity name and # to show which instance of the activity it is
        act = act.replace(':', '')
        act = act.replace('/', ' or ')
        plt.legend(["Thumb", "Index", "Middle", "Ring", "Pinky"])
        plt.title(f'{act}_{activity_count}')
        plt.savefig(f'{save_dir}{act}_{activity_count}.png')
        activity_count += 1
        plt.close()