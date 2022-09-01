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

import numpy as np
import os

# Get the length of a list/tuple or np array
def length(x):
  if isinstance(x, (list, tuple)):
    return len(x)
  return max(x.shape)


# Find an element in an array
def find(x, target=True, num=None, return_list=False):
  if num == 'first':
    num = 1
  elif num == 'last':
    num = -1
  x = np.array(x)
  target = np.array(target).astype(x.dtype)
  matches = np.argwhere(x == target)
  if x.ndim == 1:
    matches = matches.flatten()
  if return_list:
    try:
      matches = [list(x) for x in matches]
    except:
      matches = list(matches)
  if num is not None:
    if num == 1:
      matches = matches[0]
    elif num == -1:
      matches = matches[-1]
    elif num > 0:
      matches = matches[0:num]
    elif num < 0:
      matches = matches[-num:]
  return matches


# Get indexes of rising edges.
# Each index will be the first high value of the step.
def get_rising_edges(vals, target=None, threshold=0.5):
  if target is None:
    vals = vals > threshold
  else:
    vals = vals == target
  rising_indexes = np.flatnonzero((~vals[0:-1]) & (vals[1:]))+1
  return rising_indexes


# Get indexes of falling edges.
# Each index will be the first low value after the step.
def get_falling_edges(vals, target=None, threshold=0.5):
  if target is None:
    vals = vals > threshold
  else:
    vals = vals == target
  falling_indexes = np.flatnonzero((vals[0:-1]) & (~vals[1:]))+1
  return falling_indexes

# x = np.array([1,0,0,0,1,1,1,1,1,0,0,0,0,1,1,1])
# print(get_rising_edges(x, target=1))
# print(get_falling_edges(x, target=1))
# print(get_label_windows(x, target=1))

def append_to_get_unique_filepath(filepath, append_format='_%02d'):
  if not os.path.exists(filepath):
    return filepath
  filepath_original = filepath
  extension = os.path.splitext(filepath)[1]
  num_to_append = 1
  while os.path.exists(filepath):
    filepath = filepath_original.replace(extension, '%s%s' % (append_format % num_to_append, extension))
    num_to_append = num_to_append+1
  return filepath

def heatmap(data, title=None):
  import pyqtgraph as pg

  gr_wid = pg.GraphicsLayoutWidget(show=True)
  gr_wid.setWindowTitle(title or 'Data!')
  gr_wid.resize(700,500)
  gr_wid.show()
  
  i2 = pg.ImageItem(image=data.T)
  p2 = gr_wid.addPlot(1,0, 1,1, title="interactive")
  p2.addItem( i2, title='' )
  p2.invertY(True)
  p2.setAspectLocked(True)
  b = p2.addColorBar(i2, colorMap='inferno')
  i2.setImage(data.T)
  b.setLevels(i2.getLevels())
  pg.exec()
  
  
