
import pyaudio
import sys
import time
import numpy as np
import array

input_device_keywords = 'realtek'

bytes_per_sample = 2
num_channels = 1
sampling_rate_hz = 48000
chunk_size = 1024 # number of samples to acquire before running the callback function
array_decoding_codes = { # based on https://docs.python.org/3/library/array.html
  1: 'B',
  2: 'h',
  4: 'l',
  8: 'q',
}
array_decoding_code = array_decoding_codes[bytes_per_sample]



# Initialize PyAudio
p = pyaudio.PyAudio()
format = p.get_format_from_width(bytes_per_sample)
print_var(format, 'format for width %d' % bytes_per_sample)

# Discover input and output devices
info = p.get_host_api_info_by_index(0)
num_devices = info.get('deviceCount')
devices_info = [p.get_device_info_by_host_api_device_index(0, i) for i in range(num_devices)]
input_devices_info = {}
output_devices_info = {}
for (device_index, device_info) in enumerate(devices_info):
  if device_info.get('maxInputChannels') > 0:
    input_devices_info[device_index] = device_info
  elif device_info.get('maxOutputChannels') > 0:
    output_devices_info[device_index] = device_info
# Print available devices
print('Input devices:')
for (device_index, device_info) in input_devices_info.items():
  print(' ID %2d: %s' % (device_index, device_info.get('name')))
print('Output devices:')
for (device_index, device_info) in output_devices_info.items():
  print(' ID %2d: %s' % (device_index, device_info.get('name')))

# Select the desired input/output devices.
input_device_index = None
for (device_index, device_info) in input_devices_info.items():
  if input_device_keyword.lower() in device_info.get('name').lower():
    input_device_index = device_index
    break
output_device_index = None
for (device_index, device_info) in output_devices_info.items():
  if output_device_keyword.lower() in device_info.get('name').lower():
    output_device_index = device_index
    break
print('Using input device ID %s and output device ID %s' % (input_device_index, output_device_index))

last_frame = None
last_time_info = None
num_frames = 0
first_frame_time_s = None
last_frame_time_s = None
offsets_fromCallback = []
offsets_streamTimes_fromCallback = []
samples = []
timestamps_firstSamples_s = []
timestamps_lastSamples_s = []
def callback(in_data, frame_count, time_info, status):
  global last_frame, last_time_info, count, first_frame_time_s, last_frame_time_s, num_frames, offsets_fromCallback, stream, samples, timestamps_s
  last_frame_time_s = time.time()
  if first_frame_time_s is None:
    first_frame_time_s = last_frame_time_s
  num_frames = num_frames + 1
  last_frame = in_data
  last_time_info = time_info
  offsets_fromCallback.append(time.time() - time_info['current_time'])
  offsets_streamTimes_fromCallback.append(stream.get_time() - time_info['current_time'])
  # print(len(in_data), in_data[0:10])
  samples.append(list(np.array(array.array(array_decoding_code, in_data))))
  timestamp_s = time.time()
  timestamps_lastSamples_s.append(timestamp_s)
  timestamps_firstSamples_s.append(timestamp_s - (len(in_data)/bytes_per_sample-1)/sampling_rate_hz)
  return (in_data, pyaudio.paContinue)

stream = p.open(format=format,
                input=True,
                output=True,
                input_device_index=input_device_index,
                output_device_index=output_device_index,
                channels=num_channels,
                rate=sampling_rate_hz,
                frames_per_buffer=chunk_size,
                stream_callback=callback)

offsets_preStream = []
t0 = time.time()
for i in range(25):
  offsets_preStream.append(time.time() - stream.get_time())
  time.sleep(0.05)
print('offset test duration pre-stream:', time.time() - t0)
offsets_preStream = np.array(offsets_preStream)

stream.start_stream()

start_time_s = time.time()
while stream.is_active() and time.time() - start_time_s < 2:
  time.sleep(0.1)

# Clean up.
stream.stop_stream()
# stream.close()
# p.terminate()

# Test
print()
print()
print(len(last_frame), 'length of last frame')
print_var(last_frame, 'last_frame')

print()
stream_duration_s = last_frame_time_s - first_frame_time_s
frame_rate_hz = (num_frames-1)/stream_duration_s
sample_rate_hz = frame_rate_hz * chunk_size
print('Num frames:', num_frames)
print('Streaming duration: %0.2f' % (stream_duration_s))
print('Frame rate: %0.2f Hz' % (frame_rate_hz))
print('Sample rate: %0.2f Hz' % (sample_rate_hz))

print()
last_frame_decoded = np.array(array.array(array_decoding_code, last_frame)) # last_frame_decoded.frombytes(b''.join([last_frame]))
print_var(last_frame_decoded, 'last frame decoded')
if sys.byteorder == 'big':
  print('swap!')
  last_frame_decoded = last_frame_decoded.byteswap()
  print_var(last_frame_decoded, 'last frame decoded')

print()
print_var(last_time_info, 'last_time_info')
print_var(stream.get_time(), 'stream time')
print_var(stream.get_input_latency(), 'stream input latency')
print_var(stream.get_output_latency(), 'stream ouput latency')

offsets_fromCallback = np.array(offsets_fromCallback)

offsets_postStream = []
t0 = time.time()
for i in range(25):
  offsets_postStream.append(time.time() - stream.get_time())
  time.sleep(0.05)
print('offset test duration post-stream:', time.time() - t0)
offsets_postStream = np.array(offsets_postStream)

offsets_preStream = offsets_preStream[1:]
offsets_postStream = offsets_postStream[1:]
offsets_fromCallback = offsets_fromCallback[1:]

print('offsets pre-stream :', np.mean(offsets_preStream), np.std(offsets_preStream), np.mean(offsets_preStream) - min(offsets_preStream), max(offsets_preStream) - np.mean(offsets_preStream))
print('offsets in-stream  :', np.mean(offsets_fromCallback), np.std(offsets_fromCallback), np.mean(offsets_fromCallback) - min(offsets_fromCallback), max(offsets_fromCallback) - np.mean(offsets_fromCallback))
print('offsets post-stream:', np.mean(offsets_postStream), np.std(offsets_postStream), np.mean(offsets_postStream)-min(offsets_postStream), max(offsets_postStream)-np.mean(offsets_postStream))
print('stream time offsets in-stream:', np.mean(offsets_streamTimes_fromCallback), np.std(offsets_streamTimes_fromCallback), np.mean(offsets_streamTimes_fromCallback)-min(offsets_streamTimes_fromCallback), max(offsets_streamTimes_fromCallback)-np.mean(offsets_streamTimes_fromCallback))

import matplotlib.pyplot as plt
h = plt.plot(1000 * offsets_preStream, label='offsets during pre-stream loop')
plt.plot(plt.xlim(), np.ones([2]) * np.mean(1000 * offsets_preStream), '--', color=h[0].get_color())
h = plt.plot(1000 * offsets_postStream, label='offsets during post-stream loop')
plt.plot(plt.xlim(), np.ones([2]) * np.mean(1000 * offsets_postStream), '--', color=h[0].get_color())
h = plt.plot(1000*offsets_fromCallback, label='offsets during streaming callback')
plt.plot(plt.xlim(), np.ones([2])*np.mean(1000*offsets_fromCallback), '--', color=h[0].get_color())
plt.legend()
plt.ylabel('offset [ms]')
plt.xlabel('index')
plt.grid(True, color='lightgray')
plt.show()

stream.close()
p.terminate()

print_var(len(samples))
print_var(len(timestamps_lastSamples_s))
dt = np.diff(timestamps_lastSamples_s)
print('frame last-sample timestamp diffs :', np.mean(dt), np.std(dt), np.min(dt), np.max(dt))
dt = np.diff(timestamps_firstSamples_s)
print('frame first-sample timestamp diffs:', np.mean(dt), np.std(dt), np.min(dt), np.max(dt))
dt = [timestamps_lastSamples_s[i] - timestamps_firstSamples_s[i] for i in range(len(timestamps_lastSamples_s))]
print('intraframe first-to-last-sample timestamp diffs:', np.mean(dt), np.std(dt), np.min(dt), np.max(dt))
dt = [timestamps_firstSamples_s[i] - timestamps_lastSamples_s[i-1] for i in range(1, len(timestamps_lastSamples_s))]
print('interframe last-to-first-sample timestamp diffs:', np.mean(dt), np.std(dt), np.min(dt), np.max(dt))
h = plt.plot(1000*np.array(dt))
plt.plot(plt.xlim(), np.ones([2])*np.mean(1000*dt), '--', color=h[0].get_color())
plt.grid(True, color='lightgray')
plt.title('Time from last sample to next first sample [ms]')
plt.show()

#####################
# Timing conclusions
#####################
# Note that stream.get_time() - time_info['current_time'] is very small (~5e-05 seconds),
#   so both are effectively the same and indicate the time of the last sample in the frame.
# Note that all entries of time_info are 0 if the stream is only an input (no output).
# Note that time_info['input_buffer_adc_time'] was always 0 when using the laptop
#   microphone or the Movo USB conference microphone.
# Note that measuring the system-to-stream time offsets before/after streaming yields
#   a mean about 2ms greater than measuring it during the callback.  Measuring during
#   the callback yields more erratic offsets with peak-to-peak of ~3.5ms, while measuring
#   before/after streaming has a peak-to-peak of ~0.25ms.  The before or after loops yields
#   effectively the same distribution of offsets (at least with a 60s stream).
# So if we want to use the stream time, we should add a dedicated loop of about ~1-2s
#   at the beginning to estimate the offset and then use that offset in the callback
#   to adjust stream.get_time().
# But since the in-stream and before/after-stream offsets are within ~2ms of each other,
#   the callback is being called at ~11Hz, and the sampling rate is 48kHz,
#   it seems the time_info time is just trying to estimate the system time of the callback
#   (not necessarily of the last frame/sample or something else interesting). So it likely
#   wouldn't provide more information than simply getting the system time in the callback.


