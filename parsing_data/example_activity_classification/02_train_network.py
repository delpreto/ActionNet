############
#
# Copyright (c) 2022 MIT CSAIL and Joseph DelPreto
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
# IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# See https://action-net.csail.mit.edu for more usage information.
# Created 2021-2022 for the MIT ActionNet project by Joseph DelPreto [https://josephdelpreto.com].
#
############

# Input: an HDF5 file with processed data from all desired experiments.
# Outputs: a folder for each holdout iteration with:
#            a trained network for each epoch (as HDF5 files)
#            a plotted confusion matrix for the final network for each train/validation/test set
#            learning curves of the train/validation/test set accuracies
#            an HDF5 file for each network (each holdout iteration) with:
#              per-epoch accuracies and correct counts
#              accuracies and confusions for the final network on the training/validation/test sets

import h5py
from collections import OrderedDict
import os, sys
import shutil
import time
import pickle
import pyperclip

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import backend as keras_backend

from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

from helpers import *
from utils.print_utils import *
from utils.dict_utils import *
from utils.time_utils import *

script_dir = os.path.dirname(os.path.realpath(__file__))
print()

#####################################
# Configuration
#####################################

data_processed_root_dir = os.path.realpath(os.path.join(script_dir, '..', '..', '..', 'data_processed'))
training_networks_root_dir = os.path.realpath(os.path.join(script_dir, '..', '..', '..', 'training_networks'))

sensor_subsets = [
  'allStreams',
  'noEMG',
  'noEye',
  'noTactile',
  'noBody',
  'onlyEMG',
  'onlyEye',
  'onlyTactile',
  'onlyBody',
  ]
use_random_label_indexes = False # True to experimentally find the chance level

save_outputs = True

# Training settings.
num_epochs = 200
batch_size = 32
# Define the number of features per sensor in the processed data files that will be used.
#  Will use these to create mini networks for each sensor path.
num_features_perSensor = {
  'eye': 2, # x/y gaze position
  'emg': 8*2, # 8 channels for each arm
  'tactile': (4*4)*2, # aggregated 4x4 matrix for each hand
  'body': 22*3 # x/y/z for 22 joints
}

holdout_subject_ids = 'each' # 'each' to iteratively hold out each experiment, None, or a list of IDs
percentExamples_test_validation_sets = 20 # ignored if doing holdouts # a percent 0-100, or 'subject' to use the average number of examples from a subject

# See below for the network architecture specification.

##################################################

accuracy_table_toCopy = '' # will print an accuracy table that can be copied into Excel
for sensor_subset in sensor_subsets:
  # Select input data.
  data_processed_filename = 'data_processed_allStreams_10s_10hz_5subj_ex20-20_allActs.hdf5'
  data_processed_filepath = os.path.join(data_processed_root_dir, data_processed_filename)
  
  # Define the output folder based on the desired sensor subset.
  output_dir = os.path.join(training_networks_root_dir,
                            'training_networks_lstmPaths5-main50_batch32_epochs200_LOSO_%s_%s' % (sensor_subset, os.path.splitext(os.path.basename(data_processed_filepath))[0])
                            )
  save_outputs = save_outputs and (output_dir is not None)
  
  # Summarize input/output settings.
  print()
  print('Input file with processed data:')
  print(data_processed_filepath)
  print()
  if save_outputs:
    print('Output folder: %s' % output_dir)
  else:
    print('NOTE: not saving outputs')
    time.sleep(3)
  print()
  
  # Check output existence.
  if save_outputs:
    if os.path.exists(output_dir):
      user_input = input('The output folder already exists! Clear it? [y/N] ')
      print()
      if user_input.lower() in ['y', 'yes']:
        shutil.rmtree(output_dir)
      else:
        print('Aborting!')
        sys.exit()
    os.makedirs(output_dir, exist_ok=True)
  
  #####################################
  # Load processed data
  #####################################
  
  fin = h5py.File(data_processed_filepath, 'r')
  data_processing_metadata = OrderedDict()
  data_processing_metadata.update(fin.attrs)
  feature_matrices = np.array(fin['example_matrices'])
  feature_label_indexes = np.array(fin['example_label_indexes'])
  feature_subject_ids = np.array([x.decode('utf-8') for x in fin['example_subject_ids']])
  labels_all = np.array(eval(data_processing_metadata['activities_to_classify']))
  # Generate random labels if desired.
  # Will ensure that each class has the same number of examples.
  if use_random_label_indexes:
    print('\n'*5)
    print('GENERATING RANDOM LABELS')
    print('\n'*5)
    min_index = min(feature_label_indexes)
    max_index = max(feature_label_indexes)
    feature_label_indexes_random = []
    for i in range(min_index, 1+max_index):
      num_instances = np.sum(feature_label_indexes == i)
      feature_label_indexes_random.extend([i]*num_instances)
    feature_label_indexes_random = np.array(feature_label_indexes_random)
    feature_label_indexes_random = np.random.permutation(feature_label_indexes_random)
    feature_label_indexes[0:] = feature_label_indexes_random
    print_var(feature_label_indexes)
    print(list(feature_label_indexes))
    for i in range(min(feature_label_indexes), 1+max(feature_label_indexes)):
      print(i, np.sum(feature_label_indexes == i))
    print('\n'*5)
    time.sleep(5)
  
  num_experiments = len(np.unique(feature_subject_ids))
  num_labels = len(labels_all)
  label_indexes_all = list(range(num_labels))
  segment_length = np.squeeze(feature_matrices[0,:,:]).shape[0]
  num_features = np.squeeze(feature_matrices[0,:,:]).shape[1]
  num_examples_per_subject = [sum(feature_subject_ids == subject_id) for subject_id in np.unique(feature_subject_ids)]
  print()
  print('Number of labels:', num_labels)
  print('Segment length:', segment_length)
  print('Number of features:', num_features)
  print('Number of examples per subject:', num_examples_per_subject)
  print()

  #####################################
  # Network architecture
  #####################################

  # Define the layers that will constitute the network.
  network_layers_perSensor = OrderedDict()
  feature_index_start = 0
  num_features_forSensor = num_features_perSensor['eye']
  network_layers_perSensor['eye'] = [
    {'type': 'lambda', 'lambda_fn' : lambda x, _feature_index_start=feature_index_start, _num_features_forSensor=num_features_forSensor: x[:,:, _feature_index_start:(_feature_index_start+_num_features_forSensor)],
                       'output_shape': (segment_length, num_features_forSensor)},
    {'type': 'lstm', 'size': 5, 'return_sequences': True}
  ]
  feature_index_start += num_features_forSensor
  num_features_forSensor = num_features_perSensor['emg']
  network_layers_perSensor['emg'] = [
    {'type': 'lambda', 'lambda_fn' : lambda x, _feature_index_start=feature_index_start, _num_features_forSensor=num_features_forSensor: x[:,:, _feature_index_start:(_feature_index_start+_num_features_forSensor)],
                       'output_shape': (segment_length, num_features_forSensor)},
    {'type': 'lstm', 'size': 5, 'return_sequences': True}
  ]
  feature_index_start += num_features_forSensor
  num_features_forSensor = num_features_perSensor['tactile']
  network_layers_perSensor['tactile'] = [
    {'type': 'lambda', 'lambda_fn' : lambda x, _feature_index_start=feature_index_start, _num_features_forSensor=num_features_forSensor: x[:,:, _feature_index_start:(_feature_index_start+_num_features_forSensor)],
                       'output_shape': (segment_length, num_features_forSensor)},
    {'type': 'lstm', 'size': 5, 'return_sequences': True}
  ]
  feature_index_start += num_features_forSensor
  num_features_forSensor = num_features_perSensor['body']
  network_layers_perSensor['body'] = [
    {'type': 'lambda', 'lambda_fn' : lambda x, _feature_index_start=feature_index_start, _num_features_forSensor=num_features_forSensor: x[:,:, _feature_index_start:(_feature_index_start+_num_features_forSensor)],
                       'output_shape': (segment_length, num_features_forSensor)},
    {'type': 'lstm', 'size': 5, 'return_sequences': True}
  ]
  # Only use the desired sensor types.
  if sensor_subset == 'noEMG':
    del network_layers_perSensor['emg']
  if sensor_subset == 'noEye':
    del network_layers_perSensor['eye']
  if sensor_subset == 'noTactile':
    del network_layers_perSensor['tactile']
  if sensor_subset == 'noBody':
    del network_layers_perSensor['body']
  if sensor_subset == 'onlyEMG':
    network_layers_perSensor = {'emg': network_layers_perSensor['emg']}
  if sensor_subset == 'onlyEye':
    network_layers_perSensor = {'eye': network_layers_perSensor['eye']}
  if sensor_subset == 'onlyTactile':
    network_layers_perSensor = {'tactile': network_layers_perSensor['tactile']}
  if sensor_subset == 'onlyBody':
    network_layers_perSensor = {'body': network_layers_perSensor['body']}

  # Create the overall network that includes the sub-networks.
  network_layers = [
    # Pathways for each sensor
    list(network_layers_perSensor.values()),
    # Merged pathway
    {'type': 'lstm', 'size' : 50, 'return_sequences': False},
    {'type': 'dropout', 'ratio': 0.2},
    {'type': 'dense', 'size': num_labels, 'activation':'softmax'},
  ]

  ################################################
  # Train and evaluate for each holdout experiment
  ################################################
  
  if holdout_subject_ids == 'each':
    holdout_subject_ids = sorted(np.unique(feature_subject_ids))
  elif not isinstance(holdout_subject_ids, (list, tuple)):
    holdout_subject_ids = [holdout_subject_ids]
  
  holdout_accuracies_50epochs = []
  for holdout_subject_id in holdout_subject_ids:
    while os.path.exists(os.path.join(training_networks_root_dir, '_pause_script_02_train_networks.txt')):
      time.sleep(30)
    
    print('==================================================')
    print('==================================================')
    print('          HOLDOUT SUBJECT %s' % holdout_subject_id)
    print('==================================================')
    print('==================================================')
    
    # Prepare outputs
    if save_outputs:
      output_dir_forHoldout = os.path.join(output_dir, 'holdout_subject_%s' % holdout_subject_id
                                                        if holdout_subject_id is not None else 'no_holdout_subject')
      os.makedirs(output_dir_forHoldout, exist_ok=True)
      
      fout = h5py.File(os.path.join(output_dir_forHoldout, 'results.hdf5'), 'w')
    else:
      output_dir_forHoldout = None
      fout = None
    
    ##########################################
    # Create training/validation/testing sets
    ##########################################
    print('Dividing data into training/validation/test sets')
    
    if holdout_subject_id is not None:
      # Get the vectors/labels for the subject left out.
      holdout_indexes = np.where(feature_subject_ids == holdout_subject_id)
      toSplit_indexes = np.where(feature_subject_ids != holdout_subject_id)
      X_test = np.squeeze(feature_matrices[holdout_indexes, :, :])
      y_test = np.squeeze(feature_label_indexes[holdout_indexes])
      
      # Divide the remaining examples into training/validation sets.
      data_split_count_val = len(X_test) # make validation and testing sets the same size
      feature_matrices_toSplit = np.squeeze(feature_matrices[toSplit_indexes, :, :])
      feature_label_indexes_toSplit = np.squeeze(feature_label_indexes[toSplit_indexes])
      X_train, X_val, y_train, y_val = train_test_split(feature_matrices_toSplit,
                                                        feature_label_indexes_toSplit,
                                                        test_size=data_split_count_val,  # absolute number or a percent
                                                        shuffle=True, stratify=feature_label_indexes_toSplit,
                                                        random_state=12)
    else:
      # Divide the examples to create a testing set.
      if percentExamples_test_validation_sets == 'subject':
        data_split_count_test = int(np.round(np.mean(num_examples_per_subject)))
      else:
        data_split_count_test = int(round(percentExamples_test_validation_sets/100*len(feature_matrices)))
      feature_matrices_toSplit = np.squeeze(feature_matrices)
      feature_label_indexes_toSplit = np.squeeze(feature_label_indexes)
      X_remaining, X_test, y_remaining, y_test = train_test_split(feature_matrices_toSplit,
                                                                  feature_label_indexes_toSplit,
                                                                  test_size=data_split_count_test,  # absolute number or a percent
                                                                  shuffle=True, stratify=feature_label_indexes_toSplit,
                                                                  random_state=12)
      # Divide the remaining examples into training/validation sets.
      data_split_count_val = len(X_test) # make validation and testing sets the same size
      feature_matrices_toSplit = X_remaining
      feature_label_indexes_toSplit = y_remaining
      X_train, X_val, y_train, y_val = train_test_split(feature_matrices_toSplit,
                                                        feature_label_indexes_toSplit,
                                                        test_size=data_split_count_val,  # absolute number or a percent
                                                        shuffle=True, stratify=feature_label_indexes_toSplit,
                                                        random_state=12)
    
    print('  Total examples: %5d' % (len(feature_matrices)))
    print('  Subject left out: %s' % holdout_subject_id)
    print('  Training examples  : %5d (%5.1f%%)' % (len(X_train), 100*len(X_train)/len(feature_matrices)))
    print('  Validation examples: %5d (%5.1f%%)' % (len(X_val), 100*len(X_val)/len(feature_matrices)))
    print('  Testing examples   : %5d (%5.1f%%)' % (len(X_test), 100*len(X_test)/len(feature_matrices)))
    print()
    print('Breakdown by label:')
    print(' Train: %5d   [%s]' % (len(y_train), ' '.join(['%5d' % sum([y == lbli for y in y_train]) for lbli in label_indexes_all])))
    print(' Test : %5d   [%s]' % (len(y_test), ' '.join(['%5d' % sum([y == lbli for y in y_test]) for lbli in label_indexes_all])))
    print(' Val  : %5d   [%s]' % (len(y_val), ' '.join(['%5d' % sum([y == lbli for y in y_val]) for lbli in label_indexes_all])))
    print()
    
    # # Save the data to the output file (takes up a lot of space though).
    # if fout is not None:
    #   fout.create_group('data')
    #   fout['data'].create_dataset('X_train', data=X_train)
    #   fout['data'].create_dataset('X_val', data=X_val)
    #   fout['data'].create_dataset('X_test', data=X_test)
    #   fout['data'].create_dataset('y_train', data=X_train)
    #   fout['data'].create_dataset('y_val', data=X_val)
    #   fout['data'].create_dataset('y_test', data=X_test)
    
    # Convert to datasets
    # Note that it will also convert labels to one-hot encodings
    
    def convert_to_dataset(x, y, shuffle=True, batch_size=32):
      ds = tf.data.Dataset.from_tensor_slices((x, tf.one_hot(y, num_labels)))
      if shuffle:
        ds = ds.shuffle(buffer_size=len(y))
      ds = ds.batch(batch_size)
      ds = ds.prefetch(batch_size)
      return ds
    
    time.sleep(0.5)
    print('Converting to datasets')
    train_ds = convert_to_dataset(X_train, y_train, shuffle=True, batch_size=batch_size)
    val_ds = convert_to_dataset(X_val, y_val, shuffle=True, batch_size=batch_size)
    test_ds = convert_to_dataset(X_test, y_test, shuffle=True, batch_size=batch_size)
    print('Done converting to datasets')
    
    # if output_dir_forHoldout is not None:
    #   tf.data.experimental.save(train_ds, os.path.join(output_dir_forHoldout, 'dataset_train'), compression=None, shard_func=None)
    #   tf.data.experimental.save(val_ds, os.path.join(output_dir_forHoldout, 'dataset_val'), compression=None, shard_func=None)
    #   tf.data.experimental.save(test_ds, os.path.join(output_dir_forHoldout, 'dataset_test'), compression=None, shard_func=None)
    
    ##########################################
    # Create a model!
    ##########################################
    
    print('Creating a model')
    
    # Helper to add a layer to a sequence.
    def add_layer(current_layers, layer_info):
      if layer_info['type'] == 'reshape_for_lstm':
        layer = tf.keras.layers.Reshape([segment_length, num_features])
      elif layer_info['type'] == 'lstm':
        layer = tf.keras.layers.LSTM(eval(layer_info['size']) if isinstance(layer_info['size'], str) else layer_info['size'],
                                     return_sequences=layer_info['return_sequences'])
      elif layer_info['type'] == 'dense':
        layer = tf.keras.layers.Dense(eval(layer_info['size']) if isinstance(layer_info['size'], str) else layer_info['size'],
                                      activation=layer_info['activation'] or 'relu')
      elif layer_info['type'] == 'dropout':
        layer = tf.keras.layers.Dropout(layer_info['ratio'])
      elif layer_info['type'] == 'lambda':
        layer = tf.keras.layers.Lambda(layer_info['lambda_fn'], layer_info['output_shape'])
      return layer(current_layers)
      
    # Add an input layer.
    input_layer = tf.keras.layers.Input(shape=np.squeeze(feature_matrices[0,:,:]).shape)
    # Add the specified layers.
    output_layer = input_layer
    for layer_info in network_layers:
      # print('layer_info')
      # print(layer_info)
      # Create parallel pathways
      if isinstance(layer_info, (list, tuple)):
        # print('Starting pathways!')
        pathways = []
        for pathway_layers in layer_info:
          # print('pathway_layers')
          # print(pathway_layers)
          pathway = None
          for pathway_layer_info in pathway_layers:
            # print('pathway_layer_info')
            # print(pathway_layer_info)
            # print('pathway')
            # print(pathway)
            if pathway is None:
              pathway = add_layer(output_layer, pathway_layer_info)
            else:
              pathway = add_layer(pathway, pathway_layer_info)
          pathways.append(pathway)
        output_layer = tf.keras.layers.Concatenate()(pathways)
      # Create individual layers.
      else:
        output_layer = add_layer(output_layer, layer_info)
    
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    
    model.compile(optimizer='adam', #tf.keras.optimizers.SGD(learning_rate = 0.01, momentum = 0.0, nesterov = False),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), # from_logits is False since already outputting probabilities from the network (via softmax)
                  metrics=['accuracy'],
                  )
    
    # Visualize the model (will save model.png in the current folder).
    if output_dir is not None:
      # rankdir='LR' is used to make the graph horizontal.
      # Requires `pip install pydot` and graphviz (https://graphviz.gitlab.io/download/)
      tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")
      # Move the resulting PNG to the current processing directory.
      if save_outputs:
        shutil.move(os.path.join(script_dir, 'model.png'), os.path.join(output_dir, 'model.png'))
    
    print(model.summary())
    print_var(network_layers, 'network_layers')
    
    
    ##########################################
    # Train the model!
    ##########################################
    
    X_sets = OrderedDict([('Training', X_train), ('Validation', X_val), ('Testing', X_test)])
    Y_sets = OrderedDict([('Training', y_train), ('Validation', y_val), ('Testing', y_test)])
    set_names = list(X_sets.keys())
    X_sets_np = OrderedDict([(set_name, np.array(X_set)) for (set_name, X_set) in X_sets.items()])
    
    # Add a callback to save the model after every epoch.
    callbacks_list = []
    if output_dir_forHoldout is not None:
      filepath = os.path.join(output_dir_forHoldout, "checkpoint_model_epoch-{epoch:03d}_trainAcc-{accuracy:.4f}_valAcc-{val_accuracy:.4f}.hdf5")
      checkpoint_callback = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=0,
                                            save_best_only=False, save_weights_only=False,
                                            mode='auto', save_freq='epoch')
      callbacks_list.append(checkpoint_callback)
    
    # Add a callback to evaluate the model on train/validation/test sets after every epoch.
    #   Note that the training accuracy may be different than printed by TensorFlow due to the dropout layer.
    epoch_accuracies = OrderedDict([(set_name, []) for set_name in set_names])
    epoch_correct_counts = OrderedDict([(set_name, []) for set_name in set_names])
    class CustomCallback(tf.keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs=None):
        for set_name in set_names:
          y_predicted = model.predict(X_sets_np[set_name])
          y_truth = Y_sets[set_name]
          
          y_predicted_indexes = []
          correct_count = 0
          for (i, y_probabilities) in enumerate(y_predicted):
            label_index_predicted = np.argmax(y_probabilities)
            y_predicted_indexes.append(label_index_predicted)
            label_index_truth = y_truth[i]
            if label_index_predicted == label_index_truth:
              correct_count = correct_count+1
          correct_percent = 100*float(correct_count)/float(len(y_predicted))
          
          epoch_accuracies[set_name].append(correct_percent)
          epoch_correct_counts[set_name].append(correct_count)
          print(' Epoch callback: %s set accuracy: %d/%d = %0.2f%%' % (set_name, correct_count, len(y_predicted), correct_percent))
    callbacks_list.append(CustomCallback())
    
    # Try to make sure the GPU will be used.
    # See, for example, https://stackoverflow.com/questions/45662253/can-i-run-keras-model-on-gpu
    print('See the following available GPUs:', keras_backend._get_available_gpus())
    config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 8} )
    sess = tf.compat.v1.Session(config=config)
    keras_backend.set_session(sess)
    
    # Train it!
    start_time_s = time.time()
    with tf.device("GPU:0"): # Try to make sure the GPU will be used.
      training_history = model.fit(train_ds, epochs=num_epochs, validation_data=val_ds, callbacks=callbacks_list)
    training_duration_s = time.time() - start_time_s
    
    # Print results that can be copied and passed to eval() later if needed.
    print()
    print()
    print('Post-epoch accuracies:')
    print(epoch_accuracies)
    print(epoch_correct_counts)
    print()
    print('Training history:')
    print(training_history.history)
    print()
    print('Training history params:')
    print(training_history.params)
    print()
    print()
    
    # Save the epoch accuracies (note that the training history variables will be saved later).
    if fout is not None:
      fout.create_group('epoch_results')
      fout['epoch_results'].create_group('accuracies')
      fout['epoch_results'].create_group('correct_counts')
      for set_name in set_names:
        fout['epoch_results']['accuracies'].create_dataset(set_name, data=epoch_accuracies[set_name])
        fout['epoch_results']['correct_counts'].create_dataset(set_name, data=epoch_correct_counts[set_name])
    
    ##########################################
    # Test the model
    ##########################################
    
    # Example of predicting a single sample:
    # x_np = np.array(X_train[0])
    # x_np = x_np.reshape(1, sequence_length, num_features) # implicitly define the batch size (predict expects batches)
    # prediction = model.predict(x_np)[0]
    # print(prediction)
    # print(sum(prediction))
    
    # Apply the model to each set (including test, which has not been used yet)
    #   Note that the training accuracy may be different than printed by TensorFlow due to the dropout layer.
    if fout is not None:
      fout.create_group('accuracy')
      fout.create_group('confusion')
      for set_name in set_names:
        fout['accuracy'].create_group(set_name)
        fout['confusion'].create_group(set_name)
    for set_name in set_names:
      y_predicted = model.predict(X_sets_np[set_name])
      y_truth = Y_sets[set_name]
      
      y_predicted_indexes = []
      correct_count = 0
      for (i, y_probabilities) in enumerate(y_predicted):
        label_index_predicted = np.argmax(y_probabilities)
        y_predicted_indexes.append(label_index_predicted)
        label_index_truth = y_truth[i]
        if label_index_predicted == label_index_truth:
          correct_count = correct_count+1
      correct_percent = 100*float(correct_count)/float(len(y_predicted))
      print('%s set accuracy: %d/%d = %0.2f%%' % (set_name, correct_count, len(y_predicted), correct_percent))
      
      # Compute a confusion matrix
      confusion = metrics.confusion_matrix(y_truth, y_predicted_indexes)
      confusion = np.array(confusion)
      confusion_percent = confusion/confusion.sum(axis=1)[:,None]
      
      # Plot the confusion matrix.
      plt.clf()
      ax = sns.heatmap(confusion_percent, cmap="YlGnBu")
      plt.title('Confusion for %s set with experiment %s left out: %0.1f%%' % (set_name, holdout_subject_id, 100 * float(correct_count) / float(len(y_predicted))))
      plt.show(block=False)
      # Save the confusion matrix.
      if output_dir_forHoldout is not None:
        filepath_noExt = os.path.join(output_dir_forHoldout, 'holdout-%s_confusion-%s' % (holdout_subject_id, set_name))
        plt.savefig('%s.pdf' % filepath_noExt)
        pickle.dump(plt.gcf(), open('%s.pickle' % filepath_noExt,'wb'))
      
      # Store results.
      if fout is not None:
        fout['accuracy'][set_name].create_dataset('correct_count', data=correct_count)
        fout['accuracy'][set_name].create_dataset('correct_percent', data=correct_percent)
        fout['confusion'][set_name].create_dataset('confusion', data=confusion)
        fout['confusion'][set_name].create_dataset('confusion_percent', data=confusion_percent)
      
      # Keep a list of the holdout accuracies.
      if set_name == 'Testing':
        holdout_accuracies_50epochs.append(epoch_accuracies['Testing'])
      
    # Plot the learning curves
    plt.clf()
    for (set_name, epoch_set_accuracies) in epoch_accuracies.items():
      plt.plot(epoch_set_accuracies, '-*')
    plt.legend(set_names)
    plt.grid(True, color='lightgray')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy [%]')
    plt.title('Learning curves with experiment %s left out' % holdout_subject_id)
    plt.show(block=False)
    if output_dir_forHoldout is not None:
      filepath_noExt = os.path.join(output_dir_forHoldout, 'holdout-%s_learning-curves' % (holdout_subject_id))
      plt.savefig('%s.pdf' % filepath_noExt)
      pickle.dump(plt.gcf(), open('%s.pickle' % filepath_noExt,'wb'))
    
    print()
    print('Label order for above plots: ')
    for i in range(num_labels):
      print('%3d: %s' % (i, labels_all[i]))
    
    # Save additional metadata
    if fout is not None:
      metadata = OrderedDict()
      metadata['model_summary'] = model.summary()
      metadata['network_layers'] = network_layers
      metadata['holdout_experiment_id'] = holdout_subject_id
      metadata['data_processed_filepath'] = data_processed_filepath
      metadata['num_epochs'] = num_epochs
      metadata['batch_size'] = batch_size
      metadata['training_history'] = training_history.history
      metadata['training_params'] = training_history.params
      metadata['training_duration_s'] = training_duration_s
      metadata['set_names'] = set_names
      
      metadata = convert_dict_values_to_str(metadata, preserve_nested_dicts=False)
      fout.attrs.update(metadata)
      fout.close()
  
  print()
  print()
  print('*****')
  print('Input data: ', os.path.basename(data_processed_filepath))
  holdout_accuracies_50epochs = np.array(holdout_accuracies_50epochs)
  holdout_accuracies = np.median(holdout_accuracies_50epochs[:,-50:], axis=1)
  print('Holdout accuracies (median over last 50 epochs): ', [round(x,1) for x in holdout_accuracies])
  print('Average median holdout accuracy: ', np.mean(holdout_accuracies))
  print('Min median holdout accuracy: ', np.amin(holdout_accuracies))
  print('Max median holdout accuracy: ', np.amax(holdout_accuracies))
  s = sensor_subset + '\t'
  s += os.path.basename(data_processed_filepath) + '\t'
  s += os.path.basename(output_dir) + '\t'
  s += '\t'.join(['%0.10f' % x for x in list(holdout_accuracies)])
  print(s)
  accuracy_table_toCopy += '\n' + s
  print('.....')
  print()
  print()
  
  print('Done!!')
  if save_outputs:
    print('Outputs are saved to: %s' % output_dir)
  else:
    print('Outputs were not saved')
  print()
  print()
  print('Accuracy table so far:')
  print(accuracy_table_toCopy)
  try:
    pyperclip.copy(accuracy_table_toCopy)
  except:
    pass
  print()
  print()
  
  
  # Copy the PyCharm console output to the output folder.
  if save_outputs:
    output_filepath = os.path.join(output_dir,
                                   'pycharm_console_streaming_trainNetwork_%s.txt' % get_time_str())
    output_filepath = append_to_get_unique_filepath(output_filepath, append_format='_run%02d')
    
    console_output_filepath = os.path.join(data_processed_root_dir, 'pycharm_console_streaming_trainNetwork.txt')
    
    try:
      time.sleep(2) # helps make sure console output was flushed to the streaming file
      shutil.copy(console_output_filepath, output_filepath)
    except:
      print('Could not copy the pycharm console output text file')
      print('Check that the console is set to save logs to %s' % console_output_filepath)
      print()
  
  print()
  

