# Installation

A brief summary of the required software and the currently tested versions is below.  The following subsections contain more information about each one.

- Python 3.9
- Xsens MVN 2022.0.0
- Manus Dashboard 1.0.0
- Manus Core 1.7.0 
- Pupil Capture 3.5.1
- Myo Connect 1.0.1
- HDFView 3.1.3 [optional but useful]

### Python
- Install Python 3.9 such as the version available from https://www.python.org/downloads/release/python-399.
- Add the `code` folder of this repository to the Python path.  On Windows, this can be done as follows:
  - Search for Environment Variables from the Start menu, and select `Edit the system environment variables`
  - Click `Environment Variables`
  - If `PYTHONPATH` is not listed, add it via `New...`
  - Add the full path to the `code` folder to the variable.
  - You may need to relaunch the terminal or IDE that you were using to run the Python program.
- Install the following packages:
  - _For general processing:_
  - `pip install numpy`
  - _For reading/writing data via HDF5 files:_ 
  - `pip install h5py` 
  - _For handling videos/images:_
  - `pip install opencv-python`
  - _For handling audio:_
  - `pip install pyaudio`
  - `pip install wave`
  - _For data visualization:_
    - Visualizers can be configured to either use matplotlib or pyqtgraph.  Using pyqtgraph is faster and thus recommended:
    - `pip install pyqtgraph` 
    - `pip install PyQt6`
    - `pip install PyQt5` might also be needed
    - `pip install PyOpenGL`
    - To use matplotlib instead:
    - `pip install matplotlib`
    - For both pipelines, OpenCV is also used for some processing:
    - `pip install opencv-python`
  - _For the Pupil Labs eye tracker:_
  - `pip install msgpack`
  - `pip install pyzmq`
  - _For the Myo:_
  - `pip install cffi`
  - _For post-processing data exported from the Xsens software:_
  - `pip install pandas`
  - `pip install openpyxl`
  - `pip install beautifulsoup4`
  - _For the tactile sensors:_
  - `pip install pyserial`
  - _For the Dymo scale used for tactile sensor calibration:_
  - `pip install pyusb`
- For reference, the versions currently used are listed below:
  - numpy==1.19.5
  - h5py==3.1.0
  - opencv-python==4.5.5.62
  - PyAudio-0.2.11-cp39-cp39-win_amd64.whl
  - Wave==0.0.2
  - pyqtgraph==0.12.4
  - PyQt6==6.3.0
  - PyOpenGL==3.1.6
  - matplotlib==3.5.1
  - msgpack==1.0.3
  - myo-python==1.0.5
  - pyzmq==22.3.0
  - cffi==1.15.0
  - pandas==1.3.5
  - openpyxl==3.0.9
  - beautifulsoup4==4.10.0
  - pyserial==3.5
  - pyusb==1.2.1
### Myo

#### Myo Connect
The official download site has been discontinued, but the last releases are generously available at https://github.com/NiklasRosenstein/myo-python/releases.

Install `Myo Connect` and follow its setup instructions to pair with your Myo device.

#### Myo SDK and myo-python code

The Myo SDK should be added to the system path.  Add the folder `code > myo_library > myo-sdk-win-0.9.0 > bin` to the `Path` environment variable.  See notes under the Python section above regarding how to do this on Windows.  If needed, the SDK can be re-downloaded using the GitHub page mentioned above.

Set up the myo-python code by running `setup.py install` from the folder `code > myo_library > myo-python-master`.  For reference, this code is copied from https://github.com/NiklasRosenstein/myo-python.

After completing the above steps, restart the terminal or the IDE that you are using.

### Xsens MVN

Install `MVN Analyze` from https://www.xsens.com/software-downloads.  Version 2021.2 has been tested so far.

Note that the USB dongle with our license key on it must be plugged in when the software is launched (and remain plugged in while the software is running).

### Manus Core and Manus Dashboard

Manus Dashboard 1.0.0 with Manus Core 1.9.0, or Manus Dashboard 0.9.7 with Manus Core 1.7.0 for our older license key, are available at https://drive.google.com/drive/folders/1m5VDg5bOwAs4BWEzibTFwe97DDgRa0U5?usp=sharing.  Simply copy the folder to a convenient location such as `C:\Program Files` and then run `ManusVRPrime > ManusCore > ManusCore.exe`.  The dashboard can then be launched using the taskbar tray icon (or manually be running `ManusVRPrime > ManusCore > Manus_Dashboard > Manus_Dashboard.exe`).

Newer versions have not been tested, but if you want to try them and you have a license that is compatible with the newer framework, the official download page is at https://resources.manus-meta.com/downloads.

### Pupil Capture

The Pupil Core software to interface with the eye tracker can be downloaded from https://docs.pupil-labs.com/core.  Version 3.5.1 has been tested so far.

### Drivers for the Dymo scale

The Dymo M25 scale is used to calibrate the tactile sensors on the gloves; the person can press their hand onto the scale to stream ground truth force readings along with the tactile sensor readings.

To interface with the scale via `pyusb`, the drivers and a device filter need to be installed:

- Install `libusb`, for example from https://sourceforge.net/projects/libusb-win32/files/latest/download.
- Extract the downloaded zip file, and then run `inf-wizard.exe`.
- Connect the Dymo scale via USB and turn it on.
- Select the scale from the device list.  The vendor should be 0922, and the product ID should be 8003 (or similar if a different scale such as the M10 is being used).
- Save the INF file in a folder where the driver files can be unpacked/installed.
- Click to install the drivers now.

Note that the above may encounter an error due to Windows disallowing unsigned drivers.  The below steps seemed to work to temporarily disable this security feature on Windows 11 (based on the first option in [this article](https://www.maketecheasier.com/install-unsigned-drivers-windows10/)):

- Hold down Shift while restarting to get advanced options.
- Select `Troubleshoot > Advanced options > Start-up settings > Restart`.
- Select the option that disables driver signature enforcement (currently option 7).
- Once the drivers are installed, restart normally to restore the default security.
- Note: I had also disabled Secure Boot in the BIOS settings, and enabled Test Mode according to the previously linked article, but these were likely unnecessary (and they did not allow the driver installation on their own).

### HDFView

While not required, this lightweight tool is great for exploring HDF5 data.  The official download page is at https://www.hdfgroup.org/downloads/hdfview, and it can also be downloaded without an account from https://www.softpedia.com/get/Others/Miscellaneous/HDFView.shtml.

# Running

See `stream_and_save_data_multiProcess.py` for an example of streaming and saving data.  Adjust `sensor_streamer_specs` to reflect the available sensors, adjust `datalogging_options` as desired to select when/where/how data is saved, and adjust `duration_s` to the desired maximum runtime.

See `stream_and_save_data_singleProcess.py` for an example of streaming and saving data without using SensorManager.  This will run in a single process and thus will not leverage multiple cores.

# Code and Usage Overview

## Streaming Data
The code is based around the abstract **SensorStreamer** class, which provides methods for streaming data.  Each streamer can have one or more *devices*, and each device can have one or more data *streams*.  Data streams can have arbitrary sampling rates, data types, and dimensions.  Each subclass specifies the expected streams and their data types.

For example, the MyoStreamer class may have a left-arm device and a right-arm device connected.  Each one has streams for EMG, acceleration, angular velocity, gesture predictions, etc.  EMG data uses 200Hz, IMU uses 50Hz, and prediction data is asynchronous.

Each stream has a few channels that are created automatically: the data itself, a timestamp as seconds since epoch, and the timestamp formatted as a string.  Subclasses can also add extra channels if desired.  Timestamps are always created by using `time.time()` when data arrives.  Some devices such as the Xsens also have their own timestamps; these are treated as simply another data stream, and can be used in post-processing if desired.

## Implemented Streamers

### NotesStreamer
The NotesStreamer allows the user to enter notes at any time to describe experiment updates.  Each note will be timestamped in the same way as any other data, allowing notes to be syncronized with sensor data.

### MyoStreamer
The MyoStreamer can connect to one or more Myo devices.  Each device will stream EMG, acceleration, angular velocity, quaternion orientation estimates, predicted gestures, battery levels, and RSSI levels.

Myo Connect must be running and connected to all devices before starting the Python scripts.  The devices should also be already synced via the wave-out gesture.

### XsensStreamer
The XsensStreamer streams data from the Xsens body tracking suit as well as two Manus gloves if they are available.

The Xsens MVN software should be running, calibrated, and configured for network streaming before starting the Python scripts.  Network streaming can be configured in `Options > Network Streamer` with the following options:
- IP address `127.0.0.1`
- Port `9763`
- Protocol `UDP`
- Stream rate as desired
- Currently supported Datagram selections include:
  - `Position + Orientation (Quaternion)`: segment positions and orientations, which will include finger data if enabled
  - `Position + Orientation (Euler)`: segment positions and orientations, which will include finger data if enabled
  - `Time Code`: the device timestamp of each frame
  - `Send Finger Tracking Data` (if Manus gloves are connected - see below for more details)
  - `Center of Mass`: position, velocity, and acceleration of the center of mass
  - `Joint Angles`: angle of each joint, but only for the main body (finger joint angles do not seem to be supported by Xsens)

Note that it seems like the selected stream rate is not generally achieved in practice.  During some testing with a simple loop that only read raw data from the stream when only the `Time Code` was being streamed, the message rate was approximately half the selected rate up to a selection of 60Hz.  After that, the true rate remained constant at about 30-35Hz.

A few optional Xsens configuration settings in `Options > Preferences` that might be useful are noted below:
- Check `Enable simple calibration routines` to allow calibration without movement.  This is not recommended for 'real' experiments, but can make debugging/iterating faster.
- Uncheck `Maximum number of frames kept in memory` if long experiments are anticipated and memory is not a large concern.

#### Manus Gloves

To include finger tracking data from the Manus gloves:
- Connect the Manus USB dongle and the USB license key dongle to the computer
- Turn on the gloves
- Run Manus Core
- Run Manus Dashboard (if desired) and check that the live display moves as the gloves move
- In Xsens MVN, go to the `Fingers` tab of the live configuration window (the initial screen if starting a new record, or `File > Edit Live Configuration`)
- Select `Glove Type > Manus VR Gloves`
- Check that the fingers of the model move with the gloves
- In `Options > Network Streamer`, check `Send Finger Tracking Data`.  

*Note:* Xsens apparently does *not* stream joint angle data for fingers though, so only segment positions/orientations will be captured.


### EyeStreamer
The EyeStreamer streams gaze and video data from the Pupil Labs eye tracker.  Pupil Capture should be running, calibrated, and streaming before starting the Python scripts.  The following outlines the setup procedure:
- Start Pupil Capture
- Calibrate according to https://docs.pupil-labs.com/core/#_4-calibration
- In Pupil Capture, select `Network API` on the right.  The code currently expects:
  - Port `50020`
  - Frames in `BGR` format

### TouchStreamer
The TouchStreamer streams pressure data from one or more of the custom tactile sensors.  It expects the connected Arduino to be running the code in `TouchStreamer_arduino > TouchStreamer_arduino.ino`.  The configuration options in that file should match those in the TouchStreamer `__init__` method.  Currently, the Arduino continuously prints data as a new line for each timestep.  A legacy option can be enabled in which the Python script explicitly requests each batch of data.

Configuration options such as what sensors are available, their COM ports, baud rates, etc. are currently found in the TouchStreamer `__init__` method.

## Saving Data
The **DataLogger** class provides functionality to save data that is streamed from SensorStreamer objects.  It can write data to HDF5 and/or CSV files.  Video data will be excluded from these files, and instead written to video files.  Data can be saved incrementally throughout the recording process, and/or at the end of an experiment.  Data can optionally be cleared from memory after it is incrementally saved, thus reducing RAM usage.

Data from all streamers given to a DataLogger object will be organized into a single hierarchical HDF5 file that also contains metadata.  If you would prefer data from different streamers be saved in separate HDF5 files, multiple DataLogger objects can simply be created.  Note that when using CSVs, a separate file will always be created for each data stream.

N-dimensional data will be written to HDF5 files directly, and will be unwrapped into individual columns for CSV files.

## Sensor Manager
The **SensorManager** class is a helpful wrapper for coordinating multiple streamers and data loggers.  It connects and launches all streamers, and creates and configures all desired data loggers.  It does so using multiprocessing, so that multiple cores can be leveraged; see below for more details on this.

## Multiprocessing and threading

Note that in Python, using `Threads` is useful for allowing operations to happen concurrently and for using `Locks` to coordinate access, but it does not leverage multiple cores due to Python's Global Interpreter Lock.  Using `Multiprocessing` creates separate system processes and can thus leverage multiple cores, but data cannot be shared between such processes as easily as threads.

The current framework uses both threads and processes.

SensorStreamer and DataLogger launch their run() methods in new threads, and use locks to coordinate data access.  You can thus use these classes directly, and have non-blocking operations but only use a single core.

SensorManager spawns multiple processes so multiple cores can be used.
- It starts each sensor streamer in its own process, unless the streamer specifies that it must be in the main process (currently only NotesStreamer, since user input does not work from child processes).
- Data logging happens on the main process.
- To allow data to be passed from the child processes to the main process for logging, each streamer class is registered with Python's `BaseManager` to create a `Proxy` class that pickles data for communication.

So hopefully, incrementally writing to files will not unduly impact streaming performance.  This will facilitate saving data in the event of a crash, and reducing RAM usage during experiments.

