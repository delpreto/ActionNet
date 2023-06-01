from parse_hdf5_data import *
from extract_activities_hdf5 import *
import ujson
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras import layers
import pydot
import graphviz

def save_training_json(yaml_file, data_dir, json_dump):
    '''
    Uses the functions from extract_activities_hdf5.py to extract requested data ({yaml_file}
    holds the requests) from all hdf5 files in {data_dir} and save it in the file called {json_dump}
    
    If {json_dump} already exists, the data in it will be overwritten
    Does not return the extracted streams, only saved it to a json file
    '''
    # format of extracted streams is {Subject #: {extracted stream from that file}}
    extracted_streams = {}
    
    # loops through all files in {data_dir} and saves streams from all hdf5 files in the directory
    for file in os.listdir(data_dir):
        if file[-4:] == "hdf5":
            subj_stream = extract_streams_for_activities(data_dir+file, yaml_file)
            extracted_streams[file[-8:-5]] = subj_stream

    with open(json_dump, 'w') as f:
        f.write(ujson.dumps(extracted_streams))
        
def window_data(data, stream_relations, training_window):
    '''
    Combines all activities from all x and y streams (specified in stream_relations) into a single set of
    training x, training y, test x, and test y datasets
    
    20% of the activities from each stream will be randomly set aside for the test datasets
    Data from each instance of each activity will be windowed into {training_window}-length segments before
    being added to a list for each dataset. Thus, each outputted dataset will be some "dp x window_size" list
    where dp is the number of windows that came from the aggregated activities/streams.
    Params:
        data: all data to be windowed, must contain all streams listed in stream_relations
        stream_relations: list of tuples, each tuple contains (name of x stream, name of y stream)
        training_window (number): number of seconds in each window
    '''
    subjects = list(data.keys())
    activities = list(data[subjects[0]]['time_s'].keys())

    # randomly set aside 20% (hard-coded) of the activities to use as part of the testing set
    training_size = (int)(0.8 * len(activities))
    testing_size = len(activities) - training_size
    testing_gen = np.random.default_rng()
    testing_activities = set(testing_gen.choice(activities, size=testing_size, replace=False))
    
    # calculate number of frames per window based on window length (in seconds) and sampling frequency of the data
    freq = (len(data[subjects[0]]['time_s'][activities[0]][0])-1) / (data[subjects[0]]['time_s'][activities[0]][0][-1] - data[subjects[0]]['time_s'][activities[0]][0][0])
    frames_per_window = (int)(freq * training_window) # freq in cycles/sec * seconds in training window

    print("Set aside for testing: ", testing_activities)
    print("Frames per window: ", frames_per_window)
    
    train_x = np.array([])
    train_y = np.array([])
    test_x = np.array([])
    test_y = np.array([])
    
    for subject in subjects:
        subject_data = data[subject]
        
        for x_stream, y_stream in stream_relations:
            # get device and stream names to access correct lists from subject_data
            x_device, x_stream = x_stream.split('/')
            y_device, y_stream = y_stream.split('/')
            
            x_data = subject_data[x_device][x_stream]
            y_data = subject_data[y_device][y_stream]
            
            for activity in activities:
                activity_x_data = x_data[activity]
                activity_y_data = y_data[activity]
                
                # create windows of each instance of an activity separately
                for run in range(len(activity_x_data)):
                    x_shape = np.array(activity_x_data[run]).shape
                    y_shape = np.array(activity_y_data[run]).shape

                    total_windows = (int)(x_shape[0] / frames_per_window)
                    
                    # will truncate the arrays if there is any overflow (i.e. # frames is not evenly divisible by frames_per_window)
                    activity_x_data[run] = np.resize(activity_x_data[run], (total_windows, frames_per_window, *x_shape[1:]))
                    activity_y_data[run] = np.resize(activity_y_data[run], (total_windows, frames_per_window, *y_shape[1:]))
                
                # aggregate all instances of each activity
                activity_x_data = np.concatenate(activity_x_data)
                activity_y_data = np.concatenate(activity_y_data)
                
                # set correct array sizes if this is the first set of activities that 
                # are being added to the originally empty numpy datasets
                if (train_x.size == 0 and test_x.size == 0):
                    train_x = train_x.reshape((0, *activity_x_data.shape[1:]))
                    train_y = train_y.reshape((0, *activity_y_data.shape[1:]))
                    test_x = test_x.reshape((0, *activity_x_data.shape[1:]))
                    test_y = test_y.reshape((0, *activity_y_data.shape[1:]))
                                    
                # add activities accordingly to either test or train datasets
                if (activity in testing_activities):
                    test_x = np.concatenate((test_x, activity_x_data))
                    test_y = np.concatenate((test_y, activity_y_data))        
                else:
                    train_x = np.concatenate((train_x, activity_x_data))
                    train_y = np.concatenate((train_y, activity_y_data))    

    return train_x, train_y, test_x, test_y
    
def setup_model(x_train):
    '''
    Params:
        x_train: uses shape of training data to specify shape of input layer
    '''
    inputs = keras.Input(shape=(x_train.shape[1:])) # each input is a length 100 vector -> 2 seconds of 50 Hz
    lstm = layers.LSTM(64) # TODO see if we can change what is being outputted (time series or just single output) -> probably single for now
    fully_conn = layers.Dense(100, activation='relu')
    reshape = layers.Reshape((100, 1, 1,))
    cnn = layers.Conv2DTranspose(1, 1)
    outputs = cnn(reshape(fully_conn(lstm(inputs))))

    model = keras.Model(inputs=inputs, outputs=outputs, name="prediction_model")
    model.compile(optimizer=keras.optimizers.SGD(), loss=keras.losses.MeanSquaredError(), metrics=[keras.metrics.MeanSquaredError()])
    model.summary()
    
    # save diagram of the model as model_diagram.png in local directory
    keras.utils.plot_model(model, "model_diagram.png", show_shapes=True)
    return model
      
                
if __name__ == '__main__':
    pass