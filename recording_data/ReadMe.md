# Installation

A brief summary of the required software and the currently tested versions is below.  The following subsections contain more information about each one.

- Python 3.9
- HDFView 3.1.3 [optional but useful]

### Python
- For reference, the versions currently used are listed below:
  - numpy==1.23.4
  - h5py==3.1.0
  - opencv-python==4.5.5.62
  - PyAudio-0.2.11-cp39-cp39-win_amd64.whl
  - Wave==0.0.2
  - pyqtgraph==0.12.4
  - PyQt6==6.3.0
  - PyOpenGL==3.1.6
  - matplotlib==3.5.1
  - msgpack==1.0.3
  - pyzmq==22.3.0
  - cffi==1.15.0
  - pandas==1.3.5
  - openpyxl==3.0.9

### HDFView

While not required, this lightweight tool is great for exploring HDF5 data.  The official download page is at https://www.hdfgroup.org/downloads/hdfview, and it can also be downloaded without an account from https://www.softpedia.com/get/Others/Miscellaneous/HDFView.shtml.

# Running

# Code and Usage Overview

## Streaming Data
The code is based around the abstract **SensorStreamer** class, which provides methods for streaming data.  Each streamer can have one or more *devices*, and each device can have one or more data *streams*.  Data streams can have arbitrary sampling rates, data types, and dimensions.  Each subclass specifies the expected streams and their data types.

Each stream has a few channels that are created automatically: the data itself, a timestamp as seconds since epoch, and the timestamp formatted as a string.  Subclasses can also add extra channels if desired.  Timestamps are always created by using `time.time()` when data arrives.
