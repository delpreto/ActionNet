"""
Generates aggregate plots for trajectories given a pre-processed HDF5 file
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.stats as stats

import utils
from constants import *


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

def plot_scooping_aggregates(
    input_trajectory_file,
    output_figure_dir,
):
    # Figure init
    lin_fig, lin_ax = plt.subplots(3,figsize=(16,12))
    ang_fig, ang_ax = plt.subplots(3,figsize=(16,12))
    z_fig, z_ax = plt.subplots(1,figsize=(16,9))
    tilt_fig, tilt_ax = plt.subplots(1,figsize=(16,9))
    loc_fig, loc_ax = plt.subplots(1,2,figsize=(16,9))
    dist_fig, dist_ax = plt.subplots(1,figsize=(16,9))

    # Store all wasserstein distances, if 'truth' trajectory field is provided
    wass_dist = []

    # Read trajectory file
    with h5py.File(input_trajectory_file, 'r') as f:

        # Process all trajectories
        for trajectory_key in f.keys():
            traj = f[trajectory_key]
            dataset_name = traj.attrs['name']
            
            data = traj['data']
            time = np.array(data['time'])
            pos_world_to_hand_W = np.array(data['pos_world_to_hand_W'])
            rot_world_to_hand = np.array(data['rot_world_to_hand'])
            quat_world_to_hand_ijkw = np.array([utils.rot_matrix_to_quat(R) for R in rot_world_to_hand])
            
            ref = traj['reference']
            pos_world_to_pan_W = np.array(ref['pos_world_to_pan_W'])
            pos_world_to_plate_W = np.array(ref['pos_world_to_plate_W'])

            # Linear
            d1pos = differentiate(time, pos_world_to_hand_W, order=1, axis=0)
            d2pos = differentiate(time, pos_world_to_hand_W, order=2, axis=0)
            d3pos = differentiate(time, pos_world_to_hand_W, order=3, axis=0)
            lin_ax[0].plot(time, np.linalg.norm(d1pos, axis=1), color='darkblue', alpha=0.5)
            lin_ax[1].plot(time, np.linalg.norm(d2pos, axis=1), color='darkblue', alpha=0.5)
            lin_ax[2].plot(time, np.linalg.norm(d3pos, axis=1), color='darkblue', alpha=0.5)
            
            # Angular
            d1rot = differentiate_quat(time, quat_world_to_hand_ijkw) # First derivative calculated differently
            d2rot = differentiate(time, d1rot, order=1, axis=0)
            d3rot = differentiate(time, d1rot, order=2, axis=0)
            ang_ax[0].plot(time, np.linalg.norm(d1rot, axis=1), color='darkblue', alpha=0.5)
            ang_ax[1].plot(time, np.linalg.norm(d2rot, axis=1), color='darkblue', alpha=0.5)
            ang_ax[2].plot(time, np.linalg.norm(d3rot, axis=1), color='darkblue', alpha=0.5)

            # Height
            rot_world_to_spoon = rot_world_to_hand @ ROT_HAND_TO_SPOON
            pos_hand_to_spoon_W = rot_world_to_spoon @ POS_HAND_TO_SPOON_S
            pos_world_to_spoon_W = pos_world_to_hand_W + pos_hand_to_spoon_W
            z_ax.plot(time, pos_world_to_spoon_W[:,2], color='darkblue', alpha=0.5)

            # Tilt
            z_world_to_spoon = rot_world_to_spoon[:,2]
            tilt_angle = np.arccos(np.dot(z_world_to_spoon, np.array([0, 0, 1])))
            tilt_ax.plot(time, tilt_angle, color='darkblue', alpha=0.5)

            # Pickup/dropoff location
            pos_world_to_pickup_W = pos_world_to_spoon_W[np.argmin(np.linalg.norm(pos_world_to_spoon_W[:,:2] - pos_world_to_pan_W[:2], axis=1))]
            d_pos_world_to_pickup_W = pos_world_to_pickup_W - pos_world_to_pan_W
            pos_world_to_dropoff_W = pos_world_to_spoon_W[np.argmin(np.linalg.norm(pos_world_to_spoon_W[:,:2] - pos_world_to_plate_W[:2], axis=1))]
            d_pos_world_to_dropoff_W = pos_world_to_dropoff_W - pos_world_to_plate_W
            loc_ax[0].scatter(d_pos_world_to_pickup_W[0], d_pos_world_to_pickup_W[1])
            loc_ax[1].scatter(d_pos_world_to_dropoff_W[0], d_pos_world_to_dropoff_W[1])

            # Wasserstein distances
            if 'truth' in traj.keys():
                truth = traj['truth']
                pos_world_to_hand_W_truth = truth['pos_world_to_hand_W']
                wass_dist_x = stats.wasserstein_distance(pos_world_to_hand_W[:,0], pos_world_to_hand_W_truth[:,0])
                wass_dist_y = stats.wasserstein_distance(pos_world_to_hand_W[:,1], pos_world_to_hand_W_truth[:,1])
                wass_dist_z = stats.wasserstein_distance(pos_world_to_hand_W[:,2], pos_world_to_hand_W_truth[:,2])
                avg_wass_dist = (wass_dist_x + wass_dist_y + wass_dist_z) / 3
                wass_dist.append(avg_wass_dist)

    # Complete figures & save

    # Linear derivatives
    lin_fig.suptitle('Hand trajectory: Linear 1st, 2nd, 3rd order derivatives')
    lin_ax[0].set_ylabel('Velocity (m/s)')
    lin_ax[1].set_ylabel('Accel (m/s/s)')
    lin_ax[2].set_ylabel('Jerk (m/s/s/s)')
    lin_ax[2].set_xlabel('Time (s)')
    lin_fig.savefig(output_figure_dir + f'linear_derivatives.png')

    # Angular derivatives
    ang_fig.suptitle('Hand trajectory: Angular 1st, 2nd, 3rd order derivatives')
    ang_ax[0].set_ylabel('Velocity (m/s)')
    ang_ax[1].set_ylabel('Accel (m/s/s)')
    ang_ax[2].set_ylabel('Jerk (m/s/s/s)')
    ang_ax[2].set_xlabel('Time (s)')
    ang_fig.savefig(output_figure_dir + f'angular_derivatives.png')

    # Height
    z_fig.suptitle('Spoon height')
    z_ax.set_ylabel('Height (m)')
    z_ax.set_xlabel('Time (s)')
    z_fig.savefig(output_figure_dir + f'spoon_height.png')

    # Tilt
    tilt_fig.suptitle('Spoon tilt angle')
    tilt_ax.set_ylabel('Tilt (rad)')
    tilt_ax.set_xlabel('Time (s)')
    tilt_fig.savefig(output_figure_dir + f'spoon_tilt.png')

    # Pickup / dropoff
    loc_fig.suptitle('Pickup and dropoff locations')
    theta = np.linspace(0, 2*np.pi, 50)
    x = PAN_RADIUS * np.cos(theta)
    y = PAN_RADIUS * np.sin(theta)
    loc_ax[0].plot(x, y)
    loc_ax[0].set_ylabel('Y (m)')
    loc_ax[0].set_xlabel('X (m)')
    loc_ax[0].set_title('Pickup location')
    loc_ax[0].set_aspect('equal')
    x = PLATE_RADIUS * np.cos(theta)
    y = PLATE_RADIUS * np.sin(theta)
    loc_ax[1].plot(x, y)
    loc_ax[1].set_ylabel('Y (m)')
    loc_ax[1].set_xlabel('X (m)')
    loc_ax[1].set_title('Dropoff location')
    loc_ax[1].set_aspect('equal')
    loc_fig.savefig(output_figure_dir + f'pickup_dropoff.png')

    # Wasserstein
    dist_fig.suptitle('Trajectory Wasserstein distances (avg over x,y,z)')
    dist_ax.hist(np.array(wass_dist), alpha=0.5, edgecolor='black')
    dist_ax.set_xlabel('Wasserstein distance (m)')
    dist_ax.set_ylabel('Occurences')
    dist_fig.savefig(output_figure_dir + f'wasserstein_distance.png')

    plt.close()


if __name__ == '__main__':
    # Script inputs
    input_trajectory_file = os.path.expanduser('~/data/scooping/inference_LinOSS_scooping_5678.hdf5')
    output_figure_dir = os.path.expanduser('~/data/scooping/figures/inference_LinOSS_scooping_5678/')

    # Run script
    plot_scooping_aggregates(
        input_trajectory_file,
        output_figure_dir,
    )