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

# Input: processed data, trained networks, and classification results using leave-one-subject-out cross validation
# Outputs: a confusion matrix that aggregate across holdout sets, for each sensor subset

import h5py
from collections import OrderedDict
import os, sys
import shutil
import time
import pickle
import pyperclip

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
  'RAND_allStreams',
]
sensor_subset_titles = dict([
  ('allStreams', 'All Streams'),
  ('noEMG', 'No EMG'),
  ('noEye', 'No Gaze'),
  ('noTactile', 'No Tactile'),
  ('noBody', 'No Joints'),
  ('onlyEMG', 'Only EMG'),
  ('onlyEye', 'Only Gaze'),
  ('onlyTactile', 'Only Tactile'),
  ('onlyBody', 'Only Joints'),
  ('RAND_allStreams', 'Random Labels'),
  ])
activity_shortnames = dict([
  ('None', 'None'),
  ('Get/replace items from refrigerator/cabinets/drawers', 'Fetch: Various'),
  ('Peel a cucumber', 'Peel: Cucumber'),
  ('Clear cutting board', 'Clear Cutting Board'),
  ('Slice a cucumber', 'Slice: Cucumber'),
  ('Peel a potato', 'Peel: Potato'),
  ('Slice a potato', 'Slice: Potato'),
  ('Slice bread', 'Slice: Bread'),
  ('Spread almond butter on a bread slice', 'Spread: Almond Butter'),
  ('Spread jelly on a bread slice', 'Spread: Jelly'),
  ('Open/close a jar of almond butter', 'Open/Close Jar'),
  ('Pour water from a pitcher into a glass', 'Pour Water'),
  ('Clean a plate with a sponge', 'Clean: Plate, Sponge'),
  ('Clean a plate with a towel', 'Clean: Plate, Towel'),
  ('Clean a pan with a sponge', 'Clean: Pan, Sponge'),
  ('Clean a pan with a towel', 'Clean: Pan, Towel'),
  ('Get items from cabinets: 3 each large/small plates, bowls, mugs, glasses, sets of utensils',
      'Fetch: Tableware'),
  ('Set table: 3 each large/small plates, bowls, mugs, glasses, sets of utensils', 'Set Table'),
  ('Stack on table: 3 each large/small plates, bowls', 'Stack Tableware'),
  ('Load dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils', 'Dishwasher: Load'),
  ('Unload dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils', 'Dishwasher: Unload'),
])
label_order = [
  10, # Open/Close Jar
  11, # Pour Water
  2,  # Peel: Cucumber
  5,  # Peel: Potato
  4,  # Slice: Cucumber
  6,  # Slice: Potato
  7,  # Slice: Bread
  8,  # Spread: Almond Butter
  9,  # Spread: Jelly
  12, # Clean: Plate, Sponge
  13, # Clean: Plate, Towel
  14, # Clean: Pan, Sponge
  15, # Clean: Pan, Towel
  3,  # Clear Cutting Board
  20, # Dishwasher: Load
  19, # Dishwasher: Unload
  18, # Stack Tableware
  17, # Set Table
  1,  # Fetch: Various
  16, # Fetch: Tableware
  0,  # None
]

save_outputs = True

#####################################
# Aggregate confusions!
#####################################

for sensor_subset in sensor_subsets:
  data_processed_filename = 'data_processed_allStreams_10s_10hz_5subj_ex20-20_allActs.hdf5'
  data_processed_filepath = os.path.join(data_processed_root_dir, data_processed_filename)
  
  training_dir = os.path.join(training_networks_root_dir,
                            'training_networks_lstmPaths5-main50_batch32_epochs200_LOSO_%s_%s'
                              % (sensor_subset, os.path.splitext(os.path.basename(data_processed_filepath))[0])
                              )
  
  # Summarize input/output settings.
  print('Training folder: %s' % training_dir)
  
  #####################################
  # Load data info and results
  #####################################

  fin = h5py.File(data_processed_filepath, 'r')
  data_processing_metadata = OrderedDict()
  data_processing_metadata.update(fin.attrs)
  feature_matrices = np.array(fin['example_matrices'])
  feature_label_indexes = np.array(fin['example_label_indexes'])
  feature_subject_ids = np.array([x.decode('utf-8') for x in fin['example_subject_ids']])
  labels_all = np.array(eval(data_processing_metadata['activities_to_classify']))
  num_experiments = len(np.unique(feature_subject_ids))
  num_labels = len(labels_all)
  fin.close()
  
  # Create a combined confusion matrix.
  holdout_dirs = next(os.walk(training_dir))[1]
  confusion = np.zeros(shape=(num_labels, num_labels))
  for holdout_dir in holdout_dirs:
    results_fin = h5py.File(os.path.join(training_dir, holdout_dir, 'results.hdf5'), 'r')
    holdout_confusion = results_fin['confusion']['Testing']['confusion']
    confusion += holdout_confusion
    results_fin.close()
  num_examples_perRow = np.sum(confusion, axis=1)
  num_examples_perRow = np.atleast_2d(num_examples_perRow).T
  confusion_percent = 100 * (confusion / num_examples_perRow)
  assert np.all(np.sum(confusion_percent, axis=1) == 100)
  
  # Apply renaming and reordering of labels.
  labels_all = np.array([activity_shortnames[label] for label in labels_all])
  confusion_percent = (confusion_percent[label_order,:])[:,label_order]
  labels_all = labels_all[label_order]
  
  # Plot the combined confusion matrix.
  def plot_save_confusion(data, xticklabels, yticklabels, title, filepath_noExt,
                          vmin=None, vmax=None, annot_data_condition_fn=None,
                          ax_postprocess_fn=None):
    plt.figure(figsize=np.array([6.4, 4.8])*1.2)
    plt.clf()
    annot_labels = np.empty_like(data, dtype='<U10')
    if callable(annot_data_condition_fn):
      for r in range(annot_labels.shape[0]):
        for c in range(annot_labels.shape[1]):
          if annot_data_condition_fn(data[r][c]):
            annot_labels[r][c] = '%d' % round(data[r][c])
          else:
            annot_labels[r][c] = ''
    ax = sns.heatmap(data, cmap="YlGnBu",
                     xticklabels=xticklabels, yticklabels=yticklabels,
                     annot=annot_labels, fmt='', annot_kws={"fontsize":5},
                     linewidths=0.1, linecolor='lightgray',
                     vmin=vmin, vmax=vmax)
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
    ax.figure.subplots_adjust(left=0.25, right=0.99, bottom=0.35, top=0.95)
    if callable(ax_postprocess_fn):
      ax_postprocess_fn(ax)
    plt.title(title)
    plt.xlabel('Prediction')
    plt.ylabel('True Gesture')
    # plt.tight_layout()
    if save_outputs:
      plt.show(block=False)
      plt.savefig('%s.pdf' % filepath_noExt)
      plt.savefig('%s.png' % filepath_noExt, dpi=300)
      pickle.dump(plt.gcf(), open('%s.pickle' % filepath_noExt,'wb'))

  plot_save_confusion(data=confusion_percent,
                      xticklabels=labels_all,
                      yticklabels=labels_all,
                      title='Confusion Matrix Across All Holdouts: %s' % sensor_subset_titles[sensor_subset],
                      filepath_noExt=os.path.join(training_dir, 'confusion_percent_%s' % sensor_subset),
                      vmin=0, vmax=100.1,
                      annot_data_condition_fn=lambda data_element: data_element > 0,
                      ax_postprocess_fn=None)
      
  

