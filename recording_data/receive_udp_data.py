
import socket
import json
import pickle
import time

###################
# Configuration
###################
socket_ports = {
  'emg_envelope': 65432,
  'emg_stiffness': 65431,
}

###################
# Initialization
###################
emg_sockets = {}
emg_connections = {}
for (stream_name, socket_port) in socket_ports.items():
  print('Creating a socket for stream [%s]' % stream_name)
  emg_sockets[stream_name] = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  emg_sockets[stream_name].bind(('0.0.0.0', socket_port))

###################
# Receive data
###################
start_time_s = time.time()
while time.time() - start_time_s < 300:
  for (stream_name, socket) in emg_sockets.items():
    data_packet, _ = socket.recvfrom(4096)
    if len(data_packet) == 0:
      break
    try:
      data_packet = data_packet.decode()
      data_packet = json.loads(data_packet)
      (time_s, data) = data_packet
      # Do something with your data!
      print('Received data for stream [%s]: time %0.2fs | data %s' % (stream_name, time_s, data))
    except:
      pass


