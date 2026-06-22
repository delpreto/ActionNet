import h5py
import numpy as np


hdf5_filepaths_to_merge = [
  r'P:\MIT\Lab\Wearativity\data\tests\2026-06-14_tennis_S02\2026-06-14_12-29-37_tennis_S02\2026-06-14_12-30-23_streamLog_tennis_S02.hdf5',
  r'P:\MIT\Lab\Wearativity\data\tests\2026-06-14_tennis_S02\2026-06-14_12-18-51_tennis_S02\2026-06-14_12-21-43_streamLog_tennis_S02.hdf5',
  r'P:\MIT\Lab\Wearativity\data\tests\2026-06-14_tennis_S02\2026-06-14_12-47-09_tennis_S02_myo\2026-06-14_12-47-14_streamLog_tennis_S02.hdf5',
  r'P:\MIT\Lab\Wearativity\data\tests\2026-06-14_tennis_S02\2026-06-14_12-23-16_tennis_S02_myo\2026-06-14_12-23-21_streamLog_tennis_S02.hdf5',
]
hdf5_output_filepath = r'P:\MIT\Lab\Wearativity\data\tests\2026-06-14_tennis_S02\2026-06-14_tennis_S02_xsens_myo_data_02.hdf5'
# start_time_s = 1781454291.321439  # 12:24:51.321439
# end_time_s = 1781454363.321371    # 12:26:03.321371
start_time_s = 1781454616.521129 # 12:30:16.521129
end_time_s = 1781454981.420781   # 12:36:21.420781


def _copy_attrs(source_obj, dest_obj):
  for (attr_key, attr_value) in source_obj.attrs.items():
    if attr_key not in dest_obj.attrs:
      dest_obj.attrs[attr_key] = attr_value


def _create_dataset_with_metadata(dest_group, dataset_name, source_dataset, data_to_copy):
  create_kwargs = {
    'dtype': source_dataset.dtype,
  }
  if source_dataset.ndim > 0:
    create_kwargs['maxshape'] = (None, *source_dataset.shape[1:])
    create_kwargs['chunks'] = source_dataset.chunks if source_dataset.chunks is not None else True
  if source_dataset.compression is not None:
    create_kwargs['compression'] = source_dataset.compression
  if source_dataset.compression_opts is not None:
    create_kwargs['compression_opts'] = source_dataset.compression_opts
  if source_dataset.shuffle:
    create_kwargs['shuffle'] = source_dataset.shuffle
  if source_dataset.fletcher32:
    create_kwargs['fletcher32'] = source_dataset.fletcher32

  dest_dataset = dest_group.create_dataset(dataset_name, data=data_to_copy, **create_kwargs)
  _copy_attrs(source_dataset, dest_dataset)
  return dest_dataset


def _append_or_create_dataset(dest_group, dataset_name, source_dataset, data_to_copy, allow_append):
  if dataset_name not in dest_group:
    _create_dataset_with_metadata(dest_group, dataset_name, source_dataset, data_to_copy)
    return

  dest_dataset = dest_group[dataset_name]
  if not allow_append:
    return

  if dest_dataset.ndim == 0:
    if np.array_equal(dest_dataset[...], data_to_copy):
      return
    raise ValueError('Cannot append scalar dataset at %s' % dest_dataset.name)

  if dest_dataset.shape[1:] != data_to_copy.shape[1:]:
    raise ValueError('Dataset shape mismatch at %s: %s vs %s'
                     % (dest_dataset.name, str(dest_dataset.shape[1:]), str(data_to_copy.shape[1:])))

  original_length = dest_dataset.shape[0]
  dest_dataset.resize(original_length + data_to_copy.shape[0], axis=0)
  dest_dataset[original_length:] = data_to_copy


def _get_time_slice(stream_group):
  if 'time_s' not in stream_group:
    return None

  time_values_s = np.asarray(stream_group['time_s'][...]).reshape(-1)
  start_indexes = np.flatnonzero(time_values_s >= start_time_s)
  end_indexes = np.flatnonzero(time_values_s <= end_time_s)
  if len(start_indexes) == 0 or len(end_indexes) == 0:
    return None

  start_index = int(start_indexes[0])
  end_index = int(end_indexes[-1])
  if end_index < start_index:
    return None
  return slice(start_index, end_index + 1)


def _copy_group_contents(source_group, dest_group):
  _copy_attrs(source_group, dest_group)

  for (child_name, child_obj) in source_group.items():
    if isinstance(child_obj, h5py.Group):
      child_group = dest_group.require_group(child_name)
      _copy_group_contents(child_obj, child_group)

  dataset_names = [name for name in source_group.keys()
                   if isinstance(source_group[name], h5py.Dataset)]
  if len(dataset_names) == 0:
    return

  time_slice = _get_time_slice(source_group)
  allow_append = time_slice is not None
  if 'time_s' in dataset_names and time_slice is None:
    return

  for dataset_name in dataset_names:
    source_dataset = source_group[dataset_name]
    if allow_append and source_dataset.ndim > 0:
      data_to_copy = source_dataset[time_slice, ...]
    else:
      data_to_copy = source_dataset[...]
    print('Append or create', dest_group, dataset_name, data_to_copy.shape)
    _append_or_create_dataset(dest_group, dataset_name, source_dataset, data_to_copy, allow_append)


with h5py.File(hdf5_output_filepath, 'w') as h5_fout:
  for hdf5_filepath in hdf5_filepaths_to_merge:
    print('Processing %s' % hdf5_filepath)
    with h5py.File(hdf5_filepath, 'r') as h5_fin:
      _copy_attrs(h5_fin, h5_fout)
      _copy_group_contents(h5_fin, h5_fout)
