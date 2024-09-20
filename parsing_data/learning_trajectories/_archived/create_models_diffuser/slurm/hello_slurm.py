

import time

time.sleep(30)
print('HELLO SLURM')
filepath = '/afs/csail.mit.edu/u/d/delpreto/ActionSense/code/parsing_data/learning_trajectories/create_models_diffuser/test.txt'
file = open(filepath, 'w')
file.write('HELLO')
file.close()