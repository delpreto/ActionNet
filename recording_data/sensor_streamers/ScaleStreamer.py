
from sensor_streamers.SensorStreamer import SensorStreamer
from visualizers.LinePlotVisualizer import LinePlotVisualizer

import usb.core
import usb.util

import numpy as np
import time
from collections import OrderedDict
import traceback

from utils.print_utils import *

################################################
################################################
# A class to interface with the Dymo M25 scale.
# Interface code based on http://steventsnyder.com/reading-a-dymo-usb-scale-using-python/
################################################
################################################
class ScaleStreamer(SensorStreamer):
  
  ########################
  ###### INITIALIZE ######
  ########################
  
  def __init__(self, streams_info=None,
               log_player_options=None, visualization_options=None,
               print_status=True, print_debug=False, log_history_filepath=None):
    SensorStreamer.__init__(self, streams_info,
                            log_player_options=log_player_options,
                            visualization_options=visualization_options,
                            print_status=print_status, print_debug=print_debug,
                            log_history_filepath=log_history_filepath)
    
    self._log_source_tag = 'scale'
    
    # Configuration.
    self._device_vendor_id = 0x0922
    self._device_product_id = 0x8003 # might be slightly different if using, for example, an M10 scale instead of the M25
    
    # Initialize state.
    self._device = None
    self._device_endpoint = None
    
    # Add streams for the weight data and associated information.
    self._device_name = 'tactile-calibration-scale'
    self.add_stream(device_name=self._device_name,
                    stream_name='weight_g',
                    data_type='float32',
                    sample_size=[1],
                    sampling_rate_hz=5,
                    extra_data_info={},
                    data_notes=OrderedDict([
                      ('Description', 'Weight readings from the Dymo M25 scale '
                                      'that is used to calibrate the tactile sensors '
                                      'on the gloves.  See the calibration streams '
                                      'for the relevant time windows, and the tactile '
                                      'data streams for the corresponding sensor data.'),
                    ]))
    self.add_stream(device_name=self._device_name,
                    stream_name='raw_data',
                    data_type='int',
                    sample_size=[6],
                    sampling_rate_hz=5,
                    extra_data_info={},
                    data_notes=OrderedDict([
                      ('Description', 'Raw data from the scale. '
                                      'data[2] indicates the scale used kg mode (2) or lbs/ounces mode (11); '
                                      'The weight magnitude can be calculated as (scale_factor * (data[4] + (256*data[5]))) '
                                      'where scale_factor is 10**(data[3]-256), then if data[1] is 5 it should be negative.'),
                    ]))
    self.add_stream(device_name=self._device_name,
                    stream_name='accuracy_plusMinus_g',
                    data_type='float32',
                    sample_size=[1],
                    sampling_rate_hz=5,
                    extra_data_info={},
                    data_notes=OrderedDict([
                      ('Description', 'The expected accuracy of the reading. '
                                      'According to the user manual for the Dymo M25, readings '
                                      'are accurate to +/- 0.2oz if the reading is under '
                                      'half the capacity (11kg/2) and 0.4oz otherwise.'),
                    ]))
    
    # Units.
    self._g_per_oz = 28.3495
    self._oz_per_lb = 16
    self._g_per_lb = self._g_per_oz * self._oz_per_lb
    
  def _connect(self, timeout_s=10, suppress_failure_prints=False):
    # Disconnect the device.
    del(self._device)
    del(self._device_endpoint)
    
    try:
      # Find the USB device.
      self._device = usb.core.find(idVendor=self._device_vendor_id, idProduct=self._device_product_id)
  
      # Use the first/default configuration.
      self._device.set_configuration()
      # Use the first endpoint.
      self._device_endpoint = self._device[0][(0,0)][0]
    except:
      self._device = None
      self._device_endpoint = None
    
    # Try to read data to ensure a successful connection.
    (time_s, weight_g, accuracy_plusMinus_g, raw_data) = self._read_weight_g()
    if time_s is not None:
      self._log_status('Successfully connected to the Dymo scale')
      return True
    else:
      if not suppress_failure_prints:
        self._log_warn('WARNING: Could not connect to a Dymo scale')
      return False
  
  #######################################
  ###### INTERFACE WITH THE SENSOR ######
  #######################################

  # Helper methods for units.
  def _oz_to_g(self, weight_oz):
    return weight_oz * self._g_per_oz
  def _g_to_oz(self, weight_g):
    return weight_g / self._g_per_oz
  def _g_to_lb(self, weight_g):
    return weight_g / self._g_per_lb
  def _lb_to_lbOz(self, weight_lb):
    return (int(weight_lb), (weight_lb-int(weight_lb))*self._oz_per_lb)
  def _g_to_lbOz(self, weight_g):
    return self._lb_to_lbOz(self._g_to_lb(weight_g))
  
  # Read a weight measurement from the scale.
  # Returns a tuple of (time_s, weight_g, accuracy_plusMinus_g, raw_data)
  def _read_weight_g(self):
    try:
      data = self._device.read(self._device_endpoint.bEndpointAddress,
                               self._device_endpoint.wMaxPacketSize)
      time_s = time.time()
    except:
      return (None, None, None, None)
    
    # Parse the data.
    # The values in the 6-element array are as follows:
    #   0) 3
    #   1) Seems to be 4 when positive, 2 when 0, and 5 when negative
    #   2) 2=kg mode, 11=lbs/oz mode (but seems to always be 11 if reading 0)
    #   3) scaling factor for ounces; for example 255 is signed value -1
    #      indicating raw value is in tenths (scale factor 10^-1).  254 would be 10^-2.
    #   4 and 5) Calculate the weight as scale_factor * (data[4] + (256*data[5]))
    mode_g = 2
    mode_oz = 11
    if data[2] == mode_g:
      scale_factor = 1
      weight_g = (scale_factor * (data[4] + (256*data[5])))
      if data[1] == 5:
        weight_g = -1*weight_g
    elif data[2] == mode_oz:
      scale_factor = 10**(data[3]-256)
      weight_oz = (scale_factor * (data[4] + (256*data[5])))
      if data[1] == 5:
        weight_oz = -1*weight_oz
      weight_g = self._oz_to_g(weight_oz)
    else:
      weight_g = None
    
    # Determine the accuracy according to the user manual.
    if weight_g < 11000/2: # under half the capacity, which is 11kg for the M25 scale
      accuracy_plusMinus_g = self._oz_to_g(0.2)
    else:
      accuracy_plusMinus_g = self._oz_to_g(0.4)
    
    # Return the results
    return (time_s, weight_g, accuracy_plusMinus_g, data)

  
  ###########################
  ###### VISUALIZATION ######
  ###########################
  
  # Specify how the streams should be visualized.
  def get_default_visualization_options(self, visualization_options=None):
    # Start by not visualizing any streams.
    processed_options = {}
    for (device_name, device_info) in self._streams_info.items():
      processed_options.setdefault(device_name, {})
      for (stream_name, stream_info) in device_info.items():
        processed_options[device_name].setdefault(stream_name, {'class': None})
        
    # Use a line plot to visualize the weight.
    processed_options[self._device_name]['weight_g'] = {'class': LinePlotVisualizer}
    
    # Override with any provided options.
    if isinstance(visualization_options, dict):
      for (device_name, device_info) in self._streams_info.items():
        if device_name in visualization_options:
          device_options = visualization_options[device_name]
          # Apply the provided options for this device to all of its streams.
          for (stream_name, stream_info) in device_info.items():
            for (k, v) in device_options.items():
              processed_options[device_name][stream_name][k] = v
    
    return processed_options
  
  
  #####################
  ###### RUNNING ######
  #####################
  
  # Loop until self._running is False
  def _run(self):
    try:
      while self._running:
        # Read data from the scale
        (time_s, weight_g, accuracy_plusMinus_g, raw_data) = self._read_weight_g()
        # Try to re-connect if needed (for example if the scale turned off).
        if time_s is None:
          self._log_warn('WARNING: Did not receive data from the Dymo scale. Attempting to reconnect.')
          time.sleep(0.5)
          while (not self._connect(suppress_failure_prints=True)) and self._running:
            time.sleep(5)
            self._log_debug('Attempting to reconnect to the Dymo scale')
          if self._running:
            self._log_status('Successfully reconnected to the Dymo scale!')
          continue
        # Append the data
        self.append_data(self._device_name, 'weight_g', time_s, weight_g)
        self.append_data(self._device_name, 'accuracy_plusMinus_g', time_s, accuracy_plusMinus_g)
        self.append_data(self._device_name, 'raw_data', time_s, list(raw_data))
      # Close the device connection.
      del(self._device)
      del(self._device_endpoint)
    except KeyboardInterrupt: # The program was likely terminated
      pass
    except:
      self._log_error('\n\n***ERROR RUNNING ScaleStreamer:\n%s\n' % traceback.format_exc())
    finally:
      pass
  
  # Clean up and quit
  def quit(self):
    self._log_debug('ScaleStreamer quitting')
    SensorStreamer.quit(self)


#####################
###### TESTING ######
#####################
if __name__ == '__main__':
  # Configuration.
  duration_s = 30
  
  # Connect to the device(s).
  scale_streamer = ScaleStreamer(print_status=True, print_debug=False)
  scale_streamer.connect()
  
  # Run for the specified duration and periodically print the sample rate.
  print('\nRunning for %gs!' % duration_s)
  scale_streamer.run()
  start_time_s = time.time()
  try:
    while time.time() - start_time_s < duration_s:
      time.sleep(2)
      device_name = scale_streamer.get_device_names()[0]
      stream_name = scale_streamer.get_stream_names(device_name=device_name)[0]
      num_timesteps = scale_streamer.get_num_timesteps(device_name, stream_name)
      print(' Duration: %6.2fs | Timesteps: %4d | Fs: %6.2f' % (time.time() - start_time_s, num_timesteps, ((num_timesteps)/(time.time() - start_time_s))))
  except:
    pass
  scale_streamer.stop()
  print('\nDone!\n')
  
  print_var(scale_streamer._data)
  
  
  
  











