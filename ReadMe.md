
# ActionNet

See https://action-net.csail.mit.edu for more information about the ActionNet project and dataset.

## Experiment Framework

The folder `recording_data` contains the framework for streaming, recording, visualizing, and post-processing data from the suite of sensors used for the ActionNet experiments. It also includes a GUI for interactively labeling activities during experiments.

See the ReadMe file in that folder for more information and for installation/usage instructions.

## Using the Data

The folder `parsing_data` contains example scripts for parsing data offered by the ActionNet dataset.


## Environment Setup
Please create a conda environment for installing all of the necessary packages for interfacing with ROS files, saving to HDF5, plotting and data processing, etc.

```
conda create --name actionnet python=3.9
conda activate actionnet
conda install pip
```

Use cd to enter the ActionNet folder. From here, you can use the following line to install all dependencies, or you can install them individually with a typical `pip install package_name`.

```
pip install -r requirements.txt
```

Here are all of the required dependencies:
- numpy==1.23.4 (please be sure to install this correct version)
- [h5py](https://docs.h5py.org/en/stable/build.html)
- h5py_cache
- [opencv-python](https://pypi.org/project/opencv-python/)
- [rosbags](https://pypi.org/project/rosbags/)
- [rosbags-image](https://pypi.org/project/rosbags-image/)
- catkin-pkg (before ros_numpy)
- [PyYAML](https://pypi.org/project/PyYAML/)
- [tensorflow](https://www.tensorflow.org/install/pip) (includes keras)
- tensorboard
- ujson
- pydot (for visualizing keras model)
- pydotplus (for visualizing keras model)
- graphviz (for visualizing keras model)

After finishing these installations, especially catkin-pkg, install ros_numpy via
```
git clone https://github.com/eric-wieser/ros_numpy.git
cd ros_numpy/
python setup.py install
```

Finally, please be sure to restart your computer!!!








