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
    ### save all data in a json (only needs to be run once to create the json)
    extracted_streams = {}
    for file in os.listdir(data_dir):
        if file[-4:] == "hdf5":
            subj_stream = extract_streams_for_activities(data_dir+file, yaml_file)
            extracted_streams[file[-8:-5]] = subj_stream

    with open(json_dump, 'w') as f:
        f.write(ujson.dumps(extracted_streams))
        
def window_data(data, stream_relations, training_window):
    print("start windowing")
    subjects = list(data.keys())
    activities = list(data[subjects[0]]['time_s'].keys())

    # randomly set aside 20% of the activities to use as part of the testing set
    training_size = (int)(0.8 * len(activities))
    testing_size = len(activities) - training_size
    testing_gen = np.random.default_rng()
    testing_activities = set(testing_gen.choice(activities, size=testing_size, replace=False))
    
    freq = (len(data[subjects[0]]['time_s']['Clean a plate with a towel'][0])-1) / (data[subjects[0]]['time_s']['Clean a plate with a towel'][0][-1] - data[subjects[0]]['time_s']['Clean a plate with a towel'][0][0])
    frames_per_window = (int)(freq * training_window) # freq in cycles/sec * seconds in training window

    print(testing_activities)
    print(frames_per_window)
    train_x = np.array([])#.reshape(0, frames_per_window)
    train_y = np.array([])#.reshape(0, frames_per_window)
    test_x = np.array([])#.reshape(0, frames_per_window)
    test_y = np.array([])#.reshape(0, frames_per_window)
    
    for subject in subjects:
        subject_data = data[subject]
        
        for x_stream, y_stream in stream_relations:
            x_device, x_stream = x_stream.split('/')
            y_device, y_stream = y_stream.split('/')
            
            x_data = subject_data[x_device][x_stream]
            y_data = subject_data[y_device][y_stream]
            
            add = 0
            for activity in activities:
                activity_x_data = x_data[activity]
                activity_y_data = y_data[activity]
                
                for run in range(len(activity_x_data)):
                    x_shape = np.array(activity_x_data[run]).shape
                    y_shape = np.array(activity_y_data[run]).shape
                    print("original shape", x_shape, y_shape)
                    total_windows = (int)(x_shape[0] / frames_per_window)
                    print("   ", activity, run, total_windows)
                    add += total_windows
                    # will truncate the arrays if there is any overflow (i.e. # frames is not evenly divisible by frames_per_window)
                    activity_x_data[run] = np.resize(activity_x_data[run], (total_windows, frames_per_window, *x_shape[1:]))
                    activity_y_data[run] = np.resize(activity_y_data[run], (total_windows, frames_per_window, *y_shape[1:]))
                
                activity_x_data = np.concatenate(activity_x_data)
                activity_y_data = np.concatenate(activity_y_data)
                
                if (train_x.size == 0 and test_x.size == 0):
                    train_x = train_x.reshape((0, *activity_x_data.shape[1:]))
                    train_y = train_y.reshape((0, *activity_y_data.shape[1:]))
                    test_x = test_x.reshape((0, *activity_x_data.shape[1:]))
                    test_y = test_y.reshape((0, *activity_y_data.shape[1:]))
                                    
                if (activity in testing_activities):
                    test_x = np.concatenate((test_x, activity_x_data))
                    test_y = np.concatenate((test_y, activity_y_data))        
                else:
                    train_x = np.concatenate((train_x, activity_x_data))
                    train_y = np.concatenate((train_y, activity_y_data))    
                print("total windows = ", add)    
    return train_x, train_y, test_x, test_y
    
def setup_model():
    inputs = keras.Input(shape=(1,100,)) # each input is a length 100 vector -> 2 seconds of 50 Hz
    lstm = layers.LSTM(64) # TODO see if we can change what is being outputted (time series or just single output) -> probably single for now
    fully_conn = layers.Dense(64, activation='relu')
    reshape = layers.Reshape((64, 1, 1,))
    cnn = layers.Conv2DTranspose(1, 1) # don't understand right now
    outputs = cnn(reshape(fully_conn(lstm(inputs))))

    model = keras.Model(inputs=inputs, outputs=outputs, name="prediction_model")

    model.compile(optimizer=keras.optimizers.SGD(), loss=keras.losses.MeanSquaredError(), metrics=[])
    model.summary()
    # Visualization has import errors right now, trying to fix
    keras.utils.plot_model(model, "model_diagram.png", show_shapes=True)
    return model
      
                
if __name__ == '__main__':
    data_dir = "C:/Users/2021l/Documents/UROP/data/"
    requests_file = 'request_yamls/training.yaml'
    saved_json = 'short.json'
    
    setup_model()
