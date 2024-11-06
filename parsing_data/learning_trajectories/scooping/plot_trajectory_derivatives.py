"""
Generates aggregate derivative plots for trajectories given a pre-processed HDF5 file for the scooping task
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.transform as tf
import h5py


def quat_to_delta_omega(q1, q2):
    """quat in ijkw format"""
    return 2 * np.array([q1[3]*q2[0] - q1[0]*q2[3] - q1[1]*q2[2] + q1[2]*q2[1], 
                         q1[3]*q2[1] + q1[0]*q2[2] - q1[1]*q2[3] - q1[2]*q2[0],
                         q1[3]*q2[2] - q1[0]*q2[1] + q1[1]*q2[0] - q1[2]*q2[3]])


def differentiate_quat(time, quat):
    """Given time series of quaternions, returns 6-element angular velocity vectors"""
    q1 = quat[1:].T
    q2 = quat[:-1].T
    delta_omega = quat_to_delta_omega(q1, q2).T
    do_dt = delta_omega / np.diff(time[:,np.newaxis], axis=0)
    do_dt_buffered = np.concatenate([np.zeros((1, 3)), do_dt], axis=0) # Prepend 0 value for lost timestep

    return do_dt_buffered


def differentiate(time, data, order=1, axis=0):
    """Returns the nth order derivative of data time-series along specified axis"""
    assert isinstance(order, int) and order >= 0
    
    if order == 0:
        return data
    else:
        ddata_dt = np.gradient(data, axis=axis) / np.gradient(time[:,np.newaxis], axis=axis)
        return differentiate(time, ddata_dt, order=order-1, axis=axis)


# - Main - #

def plot_trajectory_derivatives(input_trajectory_file,
                                output_figure_dir,
                                ):
    # Read trajectory file
    with h5py.File(input_trajectory_file, 'r') as f:

        # Figure init
        fig, ax = plt.subplots(3,2,figsize=(8,8))

        # Process all trajectories
        for trajectory_key in f.keys():
            traj = f[trajectory_key]
            data = traj['data']
            time = np.array(data['time'])
            pos = np.array(data['pos_world_to_hand_W'])
            quat = np.array(data['quat_world_to_hand_ijkw'])
            dataset_name = traj.attrs['name']

            # Linear
            d1pos = differentiate(time, pos, order=1, axis=0)
            d2pos = differentiate(time, pos, order=2, axis=0)
            d3pos = differentiate(time, pos, order=3, axis=0)
            ax[0, 0].plot(time, np.linalg.norm(d1pos, axis=1))
            ax[1, 0].plot(time, np.linalg.norm(d2pos, axis=1))
            ax[2, 0].plot(time, np.linalg.norm(d3pos, axis=1))
            
            # Angular
            d1rot = differentiate_quat(time, quat) # First derivative calculated differently
            d2rot = differentiate(time, d1rot, order=1, axis=0)
            d3rot = differentiate(time, d1rot, order=2, axis=0)
            ax[0, 1].plot(time, np.linalg.norm(d1rot, axis=1))
            ax[1, 1].plot(time, np.linalg.norm(d2rot, axis=1))
            ax[2, 1].plot(time, np.linalg.norm(d3rot, axis=1))

    # Save figures
    fig.suptitle('1st, 2nd, 3rd order derivatives')
    ax[0, 0].set_title('Linear')
    ax[0, 1].set_title('Angular')
    ax[0, 0].set_ylabel('Velocity (m/s)')
    ax[1, 0].set_ylabel('Accel (m/s/s)')
    ax[2, 0].set_ylabel('Jerk (m/s/s/s)')
    ax[2, 0].set_xlabel('Time (s)')
    ax[2, 1].set_xlabel('Time (s)')
    fig.savefig(output_figure_dir + f'{dataset_name}_traj_derivatives.png')
    plt.close()


if __name__ == '__main__':
    # Script inputs
    input_trajectory_file = os.path.expanduser('~/data/scooping/scooping_processed_S00.hdf5')
    output_figure_dir = os.path.expanduser('~/data/scooping/')

    # Run script
    plot_trajectory_derivatives(input_trajectory_file,
                                output_figure_dir)