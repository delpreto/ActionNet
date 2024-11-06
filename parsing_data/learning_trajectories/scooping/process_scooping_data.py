"""
This script processes the 'parsed' 'training' scooping data to create a new HDF5 with additional fields
Namely, the reference object positions are estimated based on initial conditions and included in the trajectory
Also, rotation matrix entries are added for each trajectory
The HDF5 is also reformatted to separate trajectories
"""
import os
import h5py
import numpy as np
import scipy.spatial.transform as tf

from constants import *

if __name__ == '__main__':
    # Script inputs
    data_file = os.path.expanduser(f'~/data/scooping/scooping_trainingData_S11.hdf5')
    output_file = os.path.expanduser(f'~/data/scooping/scooping_processed_S11.hdf5')
    
    # Read raw HDF5 and pull relevant trajectory fields
    with h5py.File(data_file, 'r') as f:
        time = np.array(f['time_s'])
        hand_position_m = np.array(f['hand_position_m'])
        hand_quaternion_wijk = np.array(f['hand_quaternion_wijk'])
        reference_object_position_m = np.array(f['referenceObject_position_m'])
        n = hand_position_m.shape[0] # num trials

    # Output HDF5
    with h5py.File(output_file, 'w') as f:

        # Process each trajectory
        for i in range(n):
            if i % 10 == 0:
                print(f'Processing trajectory {i}')
            
            t = time[i]
            pos_world_to_hand_W = hand_position_m[i]
            quat_world_to_hand = np.vstack([ # wxyz -> xyzw
                hand_quaternion_wijk[i][:,1],
                hand_quaternion_wijk[i][:,2],
                hand_quaternion_wijk[i][:,3],
                hand_quaternion_wijk[i][:,0],
            ]).T
            pos_world_to_plate_W = reference_object_position_m[i]
            rot_hand_to_world = np.array([tf.Rotation.from_quat(quat).as_matrix() for quat in quat_world_to_hand])

            # Transforms between world frame and spoon end (scoop)
            rot_spoon_to_world = rot_hand_to_world[0] @ rot_spoon_to_hand
            pos_world_to_spoon_W = rot_spoon_to_world @ pos_hand_to_spoon_S + pos_world_to_hand_W[0]
            
            # Combine positions to get scoop position at timestep 0
            pos_world_to_pan_W = np.array([pos_world_to_spoon_W[0], pos_world_to_spoon_W[1], pan_z_offset])

            # Add trajectory to HDF5
            traj_group = f.create_group(f'trajectory_{i+1:03d}')
            traj_group.attrs['description'] = f'Trajectory {i+1} data and reference positions'

            # Store datasets
            data_group = traj_group.create_group('data')
            data_group.attrs['description'] = f'Hand pose trajectory'
            data_group.create_dataset('time', data=t)
            data_group.create_dataset('position', data=pos_world_to_hand_W)
            data_group.create_dataset('quaternion', data=quat_world_to_hand)
            data_group.create_dataset('rotation', data=rot_hand_to_world)

            # Store references
            ref_group = traj_group.create_group('reference')
            ref_group.create_dataset('pan_position', data=pos_world_to_pan_W)
            ref_group.create_dataset('plate_position', data=pos_world_to_plate_W)