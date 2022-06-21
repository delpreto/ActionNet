
import subprocess
import os
script_dir = os.path.dirname(os.path.realpath(__file__))


data_dir = os.path.realpath(os.path.join(script_dir, '..', '..', 'data'))
experiments_dir = os.path.join(data_dir, 'experiments')

log_dirs = [
  # os.path.join(experiments_dir, '2022-06-07_experiment_S00', '2022-06-07_17-18-17_actionNet-wearables_S00'),
  # os.path.join(experiments_dir, '2022-06-07_experiment_S00', '2022-06-07_18-10-55_actionNet-wearables_S00'),
  # os.path.join(experiments_dir, '2022-06-13_experiment_S01_recordingStopped', '2022-06-13_18-13-12_actionNet-wearables_S01'),
  # os.path.join(experiments_dir, '2022-06-13_experiment_S02', '2022-06-13_21-39-50_actionNet-wearables_S02'),
  # os.path.join(experiments_dir, '2022-06-13_experiment_S02', '2022-06-13_21-47-57_actionNet-wearables_S02'),
  # os.path.join(experiments_dir, '2022-06-13_experiment_S02', '2022-06-13_22-34-45_actionNet-wearables_S02'),
  # os.path.join(experiments_dir, '2022-06-13_experiment_S02', '2022-06-13_23-16-47_actionNet-wearables_S02'),
  # os.path.join(experiments_dir, '2022-06-13_experiment_S02', '2022-06-13_23-22-21_actionNet-wearables_S02'),
  # os.path.join(experiments_dir, '2022-06-14_experiment_S03', '2022-06-14_13-01-32_actionNet-wearables_S03'),
  # os.path.join(experiments_dir, '2022-06-14_experiment_S03', '2022-06-14_13-11-44_actionNet-wearables_S03'),
  # os.path.join(experiments_dir, '2022-06-14_experiment_S03', '2022-06-14_13-52-21_actionNet-wearables_S03'),
  # os.path.join(experiments_dir, '2022-06-14_experiment_S04', '2022-06-14_16-38-18_actionNet-wearables_S04'),
  os.path.join(experiments_dir, '2022-06-14_experiment_S05', '2022-06-14_20-36-27_actionNet-wearables_S05'),
  os.path.join(experiments_dir, '2022-06-14_experiment_S05', '2022-06-14_20-45-43_actionNet-wearables_S05'),
  ]

for log_dir in log_dirs:
  print('\n'*5)
  print('|'*75)
  print('Calling script with log directory', log_dir)
  print('\n')
  stream = os.system('py offline_visualizations.py %s' % log_dir)
  # stream = os.popen('py offline_visualizations.py %s' % log_dir)
  # output = stream.read()
  # print(output)


