import numpy as np
import json
from extract_activities_hdf5 import *
import matplotlib.pyplot as plt 

def average_grid(grid, averaging_coords, single_dim=False):
  '''
  Given a grid or list of values, return a list of averaged values at the specified coordinates
  Params:
    grid: 1d or 2d list/numpy array of values to get averages of
    averaging_coords: list of coordinate groups. Each coordinate group should be a list of tuple coordinates (i.e. [(0,0), (0,-1)] for the two top corners)
    single_dim: True if "grid" is only a 1d list, False if "grid" is a true 2d grid, False by default
  Returns:
    averaged_points: list of value averages corresponding to the coordinate groups in averaging_coords
  '''
  if not isinstance(grid, np.ndarray):
    grid = np.array(grid)
    
  averaged_points = []
  
  for coords in averaging_coords:
    zipped_coords = list(zip(*coords))
    if single_dim:
      to_avg = grid[zipped_coords]
    else:
      to_avg = grid[zipped_coords[0],zipped_coords[1]]
    averaged_points.append(np.average(to_avg))
    
  return averaged_points

def average_grid_dims(grid, new_dims):
  '''
  Given a grid with dimensions old_dims, return a grid with dimensions new_dims such that combined sections were averaged
  i.e. Given a 4x4 grid with new_dims 2x2, each quadrant of the 4x4 grid gets averaged
  e.g. For 4x4 -> 2x2, the average of the old values at (0,0), (0,1), (1,0), and (1,1) -> the new value at (0,0)
  Params:
    grid: 2d list/numpy array of values to get averages of
    new_dims: dimensions of averaged grid that will be output, should be factors of the corresponding dimensions of "grid"
  Returns:
    averaged_points: grid of averages corresponding to sections that were combined in order to decrease the size of the grid
  '''
  if not isinstance(grid, np.ndarray):
    grid = np.array(grid)

  curr_dims = grid.shape    
  scale_dims = np.divide(curr_dims, new_dims).astype(int)
  averaged_points = []
  
  for r in range(scale_dims[0], curr_dims[0]+1, scale_dims[0]):
    for c in range(scale_dims[1], curr_dims[1]+1, scale_dims[1]):
      to_avg = grid[r-scale_dims[0]:r,c-scale_dims[1]:c]
      averaged_points.append(np.average(to_avg))

  averaged_points = np.array(averaged_points).reshape(new_dims)    
  return averaged_points

def find_fingers(grid):
  '''
  Function to find averages of the tactile sensor coordinates that (probably) correspond to the 5 fingers of the hand
  Corresponding coordinates are hard-coded based on visual analysis of a tactile sensor image
  '''
  thumb =  [(23,6), (23,7), (23,8), (23,9), (24,6), (24,7), (24,8), (24,9), (25,6), (25,7), (25,8), (25,9), (26,6), (26,7), (26,8), (26,9), (27,6), (27,7), (27,8), (27,9), (28,6), (28,7), (28,8), (28,9), (29,6), (29,7), (29,8), (29,9), (30,6), (30,7), (30,8), (30,9)]
  index =  [(19,20), (19,21), (19,22), (19,23), (19,24), (19,25), (19,26), (19,27), (19,28), (19,29), (19,30), (20,20), (20,21), (20,22), (20,23), (20,24), (20,25), (20,26), (20,27), (20,28), (20,29), (20,30), (21,20), (21,21), (21,22), (21,23), (21,24), (21,25), (21,26), (21,27), (21,28), (21,29), (21,30)]
  middle = [(11,20), (11,21), (11,22), (11,23), (11,24), (11,25), (11,26), (11,27), (11,28), (11,29), (11,30), (12,20), (12,21), (12,22), (12,23), (12,24), (12,25), (12,26), (12,27), (12,28), (12,29), (12,30), (13,20), (13,21), (13,22), (13,23), (13,24), (13,25), (13,26), (13,27), (13,28), (13,29), (13,30)]
  ring =   [(6,20), (6,21), (6,22), (6,23), (6,24), (6,25), (6,26), (6,27), (6,28), (6,29), (6,30), (7,20), (7,21), (7,22), (7,23), (7,24), (7,25), (7,26), (7,27), (7,28), (7,29), (7,30), (8,20), (8,21), (8,22), (8,23), (8,24), (8,25), (8,26), (8,27), (8,28), (8,29), (8,30)]
  pinky =  [(0,20), (0,21), (0,22), (0,23), (0,24), (0,25), (0,26), (0,27), (0,28), (0,29), (0,30), (1,20), (1,21), (1,22), (1,23), (1,24), (1,25), (1,26), (1,27), (1,28), (1,29), (1,30), (2,20), (2,21), (2,22), (2,23), (2,24), (2,25), (2,26), (2,27), (2,28), (2,29), (2,30)]
  hand_coords = [thumb, index, middle, ring, pinky]
  return average_grid(grid, hand_coords)

if __name__ == '__main__':
  #### Code to load tactile data from testing_tactile.json and plot the data (filtered and unfiltered) for the index finger
  with open('testing_tactile.json') as json_file:
      data = json.load(json_file)
  tactiles = data['S00']['myo-right']['emg']['Slice a cucumber']
  times = data['S00']['time_s']['Slice a cucumber']
  tactiles = tactiles[0] # + tactiles[1] + tactiles[2]
  times = times[0]
  avg_tactiles = np.absolute(np.average(np.array(tactiles), 1))
      
  plt.plot(times, avg_tactiles)
  plt.show()
    
  # #### Currently testing with data from tactile-glove-right/tactile_data;Slice bread
  # #### Code to load tactile data from testing_tactile.json and plot the data for the index finger
  # with open('testing_tactile.json') as json_file:
  #     data = json.load(json_file)
  # tactiles = data['S00']['tactile-glove-right']['tactile_data']['Slice a cucumber']
  # times = data['S00']['time_s']['Slice a cucumber']
  # tactiles = tactiles[0] # + tactiles[1] + tactiles[2]
  # times = times[0]
  # avg_tactiles = [find_fingers(frame) for frame in tactiles]
  
  # avg_tactiles_by_finger = list(zip(*avg_tactiles))
  
  # plt.plot(times, avg_tactiles_by_finger[1])
  # plt.show()
  
  # #### To visualize ^ all fingers
  # for y in avg_tactiles_by_finger:
  #   plt.plot(times, y)
  #   plt.show()

  ##### Testing averaging functions
  # z = np.array([[[1,2,3],[4,5,6],[7,8,9]],[[-2,-3,-4],[-5,-6,-7],[-8,-9,-10]],[[3,4,5],[6,7,8],[9,10,11]]])
  # y = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
  # x = np.array([1,2,3,4,5,6])
  # big_array = np.arange(1,257).reshape((16,16))
  # print(big_array)
  # print(average_grid(y, [[(0,0),(0,1),(1,0),(1,1)],[(0,2),(0,3),(1,2),(1,3)],[(2,0),(3,0),(2,1),(3,1)],[(2,2),(2,3),(3,2),(3,3)]]))
  # print(average_grid(x,[[(1,),(3,),(5,)],[(0,),(2,),(4,)]], True))
  # print(average_grid_dims(big_array,(1,1)))