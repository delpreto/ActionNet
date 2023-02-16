
# ActionNet

See https://action-net.csail.mit.edu for more information about the ActionNet project and dataset.

## Experiment Framework

The folder `recording_data` contains the framework for streaming, recording, visualizing, and post-processing data from the suite of sensors used for the ActionNet experiments. It also includes a GUI for interactively labeling activities during experiments.

See the ReadMe file in that folder for more information and for installation/usage instructions.

## Using the Data

The folder `parsing_data` contains example scripts for parsing data offered by the ActionNet dataset.


## Environment Setup
The various methods in this repository use the following packages for interfacing with ROS files, saving to HDF5, plotting and data processing, etc.

The following packages can be installed with a typical

```
pip install package
```

- [h5py](https://docs.h5py.org/en/stable/build.html)
- [opencv-python](https://pypi.org/project/opencv-python/)
- [rosbags](https://pypi.org/project/rosbags/)
- catkin-pkg (before ros_numpy)
- [PyYAML](https://pypi.org/project/PyYAML/)
- [tensorflow](https://www.tensorflow.org/install/pip) (includes keras)
- ujson
- pydot (for visualizing keras model)
- pydotplus (for visualizing keras model)
- graphviz (for visualizing keras model)

Please be sure to install the proper numpy version
- pip install numpy==1.23.4

After installing catkin-pkg, install ros_numpy via
```
git clone https://github.com/eric-wieser/ros_numpy.git
cd ros_numpy/
python setup.py install
```

After all of these installations, be sure to restart your computer!!!

- Pillow
- scipy









