
import h5py
import numpy as np

# Specify the input HDF5 file.
hdf5_filepath = '2022-06-07_17-18-46_streamLog_actionNet-wearables_S00.hdf5'

# Load the HDF5 file.
h5_file = h5py.File(hdf5_filepath, 'r')

# Get the desired matrix of data.
data_group = h5_file['xsens-sensors']['free_acceleration_cm_ss']
acceleration_cm_ss = np.squeeze(data_group['data'])

# Get metadata about the data ordering.
data_group_metadata = dict(data_group.attrs.items())
data_headings = eval(data_group_metadata['Data headings']) # the value is a list printed as a string, so use eval() to convert it to an actual list
matrix_ordering = data_group_metadata['Matrix ordering'] # just a human-readable description for reference
print()
print('Data headings:')
for (data_heading_index, data_heading) in enumerate(data_headings):
  print('  %2d: %s' % (data_heading_index, data_heading))
print()
print('Matrix ordering description:', matrix_ordering)

# Unwrap the data matrix to fit the headings.
# The standard "reshape" method should create the desired order described in this matrix_ordering.
acceleration_cm_ss = acceleration_cm_ss.reshape(acceleration_cm_ss.shape[0], -1)

# Print the extracted sizes for reference.
print()
print('Number of data headings: ', len(data_headings))
print('Size of the unwrapped data matrix: ', acceleration_cm_ss.shape)

# Close the HDF5 file.
h5_file.close()





