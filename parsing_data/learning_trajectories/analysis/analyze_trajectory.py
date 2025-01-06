"""
Generates aggregate plots and metrics for learned vs. true trajectories given a pre-processed HDF5 file
Namely:
    - Aggregate trajectories
    - Linear and Angular velocity, acceleration, jerk
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py
import scipy.stats as stats

import utils
from constants import *


# - Helpers - #


def quat_to_delta_omega(q1, q2):
    """"""
    # Check for sign flips
    dot = np.dot(q1, q2)
    if dot < 0:
        q2 = -q2  # Flip the second quaternion to avoid angular velocity spikes

    R1 = tf.Rotation.from_quat(q1).as_matrix()
    R2 = tf.Rotation.from_quat(q2).as_matrix()
    R = R2 @ R1.T

    theta = np.arccos(np.trace(R)/2 - 1/2)
    W = theta * (R - R.T) / np.sin(theta) / 2
    v = np.array([W[2,1], W[0,2], W[1,0]])
    
    return v


def differentiate_quat(time, quat):
    """Given time series of quaternions, returns 6-element angular velocity vectors"""
    delta_omega = np.array([quat_to_delta_omega(quat[i], quat[i+1]) for i in range(quat.shape[0] - 1)])
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
    input_trajectory_files,
    output_figure_directory,
):
    # Figure init
    traj_fig = plt.figure()
    traj_ax = traj_fig.add_subplot(111, projection='3d')
    truth_traj_fig = plt.figure()
    truth_traj_ax = truth_traj_fig.add_subplot(111, projection='3d')
    lin_fig, lin_ax = plt.subplots(3,figsize=(9,9))
    ang_fig, ang_ax = plt.subplots(3,figsize=(9,9))
    energy_fig, energy_ax = plt.subplots(2,figsize=(9,9))
    truth_energy_fig, truth_energy_ax = plt.subplots(2,figsize=(9,9))
    freq_fig, freq_ax = plt.subplots(3,figsize=(9,9))
    truth_freq_fig, truth_freq_ax = plt.subplots(3,figsize=(9,9))

    # Store MSEs
    mse = []

    # Store all wasserstein distances, if 'truth' trajectory field is provided
    lin_vel_wass_dist = []
    lin_accel_wass_dist = []
    lin_jerk_wass_dist = []
    ang_vel_wass_dist = []
    ang_accel_wass_dist = []
    ang_jerk_wass_dist = []

    for input_trajectory_file in input_trajectory_files:

        # Read trajectory file
        with h5py.File(input_trajectory_file, 'r') as f:

            # Process all trajectories
            for trajectory_key in f.keys():
                traj = f[trajectory_key]
                
                data = traj['data']
                time = np.array(data['time'])
                pos_world_to_hand_W = np.array(data['pos_world_to_hand_W'])
                rot_world_to_hand = np.array(data['rot_world_to_hand'])
                quat_world_to_hand_ijkw = np.array([utils.rot_matrix_to_quat(R) for R in rot_world_to_hand])

                # Trajectories
                traj_ax.plot(*pos_world_to_hand_W.T, color='darkblue', alpha=0.5)

                # Calculate derivatives -- linear
                d1pos = differentiate(time, pos_world_to_hand_W, order=1, axis=0)
                d2pos = differentiate(time, pos_world_to_hand_W, order=2, axis=0)
                d3pos = differentiate(time, pos_world_to_hand_W, order=3, axis=0)
                lin_ax[0].plot(time, np.linalg.norm(d1pos, axis=1), color='darkblue', alpha=0.5)
                lin_ax[1].plot(time, np.linalg.norm(d2pos, axis=1), color='darkblue', alpha=0.5)
                lin_ax[2].plot(time, np.linalg.norm(d3pos, axis=1), color='darkblue', alpha=0.5)
                
                # Calculate derivatives -- angular
                d1rot = differentiate_quat(time, quat_world_to_hand_ijkw) # First derivative calculated differently
                d2rot = differentiate(time, d1rot, order=1, axis=0)
                d3rot = differentiate(time, d1rot, order=2, axis=0)
                ang_ax[0].plot(time, np.linalg.norm(d1rot, axis=1), color='darkblue', alpha=0.5)
                ang_ax[1].plot(time, np.linalg.norm(d2rot, axis=1), color='darkblue', alpha=0.5)
                ang_ax[2].plot(time, np.linalg.norm(d3rot, axis=1), color='darkblue', alpha=0.5)

                # Energy
                lin_spec_kin_energy = 1/2 * np.linalg.norm(d1pos, axis=1) ** 2
                spec_pot_energy = 9.81 * pos_world_to_hand_W[:,2]
                ang_spec_kin_energy = 1/2 * np.linalg.norm(d1rot, axis=1) ** 2
                energy_ax[0].plot(lin_spec_kin_energy + spec_pot_energy, color='darkgreen', alpha=0.5)
                energy_ax[1].plot(ang_spec_kin_energy, color='darkgreen', alpha=0.5)

                # Frequency
                m = len(d1pos)
                n = int(2 ** np.ceil(np.log2(m))) # Zero-padding to the next power of 2
                d1pos_norm_padded = np.pad(np.linalg.norm(d1pos, axis=1), (0, n - m), mode='constant')
                d2pos_norm_padded = np.pad(np.linalg.norm(d2pos, axis=1), (0, n - m), mode='constant')
                d3pos_norm_padded = np.pad(np.linalg.norm(d3pos, axis=1), (0, n - m), mode='constant')
                freq = np.fft.fftshift(np.fft.fftfreq(n))[n//2:] # Only keep positive frequencies
                d1pos_freq = np.fft.fftshift(np.fft.fft(d1pos_norm_padded))[n//2:]
                d2pos_freq = np.fft.fftshift(np.fft.fft(d2pos_norm_padded))[n//2:]
                d3pos_freq = np.fft.fftshift(np.fft.fft(d3pos_norm_padded))[n//2:]
                freq_ax[0].plot(freq, np.abs(d1pos_freq), color='darkred', alpha=0.5)
                freq_ax[1].plot(freq, np.abs(d2pos_freq), color='darkred', alpha=0.5)
                freq_ax[2].plot(freq, np.abs(d3pos_freq), color='darkred', alpha=0.5)

                if 'truth' in traj.keys():
                    truth = traj['truth']
                    pos_world_to_hand_W_truth = np.array(truth['pos_world_to_hand_W'])
                    rot_world_to_hand_truth = np.array(truth['rot_world_to_hand'])
                    quat_world_to_hand_ijkw_truth = np.array([utils.rot_matrix_to_quat(R) for R in rot_world_to_hand_truth])

                    # Truth trajectories
                    truth_traj_ax.plot(*pos_world_to_hand_W_truth.T, color='darkblue', alpha=0.5)

                    # MSE
                    mse.append(
                        np.mean(np.linalg.norm(pos_world_to_hand_W - pos_world_to_hand_W_truth, axis=1))
                    )
 
                    # Wasserstein distances
                    # Calculate truth derivatives -- linear
                    d1pos_truth = differentiate(time, pos_world_to_hand_W_truth, order=1, axis=0)
                    d2pos_truth = differentiate(time, pos_world_to_hand_W_truth, order=2, axis=0)
                    d3pos_truth = differentiate(time, pos_world_to_hand_W_truth, order=3, axis=0)

                    # Calculate truth derivatives -- angular
                    d1rot_truth = differentiate_quat(time, quat_world_to_hand_ijkw_truth) # First derivative calculated differently
                    d2rot_truth = differentiate(time, d1rot_truth, order=1, axis=0)
                    d3rot_truth = differentiate(time, d1rot_truth, order=2, axis=0)

                    # Linear wasserstein distances
                    lin_vel_wass_dist.append(
                        stats.wasserstein_distance(np.linalg.norm(d1pos, axis=1), np.linalg.norm(d1pos_truth, axis=1))
                    )
                    lin_accel_wass_dist.append(
                        stats.wasserstein_distance(np.linalg.norm(d2pos, axis=1), np.linalg.norm(d2pos_truth, axis=1))
                    )
                    lin_jerk_wass_dist.append(
                        stats.wasserstein_distance(np.linalg.norm(d3pos, axis=1), np.linalg.norm(d3pos_truth, axis=1))
                    )

                    # Angular wasserstein distances
                    ang_vel_wass_dist.append(
                        stats.wasserstein_distance(np.linalg.norm(d1rot, axis=1), np.linalg.norm(d1rot_truth, axis=1))
                    )
                    ang_accel_wass_dist.append(
                        stats.wasserstein_distance(np.linalg.norm(d2rot, axis=1), np.linalg.norm(d2rot_truth, axis=1))
                    )
                    ang_jerk_wass_dist.append(
                        stats.wasserstein_distance(np.linalg.norm(d3rot, axis=1), np.linalg.norm(d3rot_truth, axis=1))
                    )

                    # Truth Energy
                    lin_spec_kin_energy_truth = 1/2 * np.linalg.norm(d1pos_truth, axis=1) ** 2
                    ang_spec_kin_energy_truth = 1/2 * np.linalg.norm(d1rot_truth, axis=1) ** 2
                    spec_pot_energy_truth = 9.81 * pos_world_to_hand_W_truth[:,2]
                    truth_energy_ax[0].plot(lin_spec_kin_energy_truth + spec_pot_energy_truth, color='darkgreen', alpha=0.5)
                    truth_energy_ax[1].plot(ang_spec_kin_energy_truth, color='darkgreen', alpha=0.5)

                    # Truth frequency
                    m = len(d1pos_truth)
                    n = int(2 ** np.ceil(np.log2(m))) # Zero-padding to the next power of 2
                    d1pos_truth_norm_padded = np.pad(np.linalg.norm(d1pos_truth, axis=1), (0, n - m), mode='constant')
                    d2pos_truth_norm_padded = np.pad(np.linalg.norm(d2pos_truth, axis=1), (0, n - m), mode='constant')
                    d3pos_truth_norm_padded = np.pad(np.linalg.norm(d3pos_truth, axis=1), (0, n - m), mode='constant')
                    freq = np.fft.fftshift(np.fft.fftfreq(n))[n//2:] # Only keep positive frequencies
                    d1pos_truth_freq = np.fft.fftshift(np.fft.fft(d1pos_truth_norm_padded))[n//2:]
                    d2pos_truth_freq = np.fft.fftshift(np.fft.fft(d2pos_truth_norm_padded))[n//2:]
                    d3pos_truth_freq = np.fft.fftshift(np.fft.fft(d3pos_truth_norm_padded))[n//2:]
                    truth_freq_ax[0].plot(freq, np.abs(d1pos_truth_freq), color='darkred', alpha=0.5)
                    truth_freq_ax[1].plot(freq, np.abs(d2pos_truth_freq), color='darkred', alpha=0.5)
                    truth_freq_ax[2].plot(freq, np.abs(d3pos_truth_freq), color='darkred', alpha=0.5)

    # Complete figures & save
    os.makedirs(output_figure_directory, exist_ok=True)

    # Trajectories
    traj_fig.suptitle('Hand trajectories')
    traj_ax.set_xlabel('X (m)')
    traj_ax.set_ylabel('Y (m)')
    traj_ax.set_zlabel('Z (m)')
    traj_ax.set_aspect('equal')
    plt.tight_layout()
    traj_fig.savefig(output_figure_directory + f'trajectories.png')

    # Linear derivatives
    lin_fig.suptitle('Hand trajectory: Linear 1st, 2nd, 3rd order derivatives')
    lin_ax[0].set_ylabel('$Velocity (m/s)$')
    lin_ax[1].set_ylabel('$Accel (m/s^2)$')
    lin_ax[2].set_ylabel('$Jerk (m/s^3)$')
    lin_ax[2].set_xlabel('Time (s)')
    plt.tight_layout()
    lin_fig.savefig(output_figure_directory + f'linear_derivatives.png')

    # Angular derivatives
    ang_fig.suptitle('Hand trajectory: Angular 1st, 2nd, 3rd order derivatives')
    ang_ax[0].set_ylabel('$Velocity (m/s)$')
    ang_ax[1].set_ylabel('$Accel (m/s^2)$')
    ang_ax[2].set_ylabel('$Jerk (m/s^3)$')
    ang_ax[2].set_xlabel('Time (s)')
    plt.tight_layout()
    ang_fig.savefig(output_figure_directory + f'angular_derivatives.png')

    # Energy
    energy_fig.suptitle('Specific energy')
    energy_ax[0].set_xlabel(r'$\frac{1}{2} v^2 + gz$')
    energy_ax[0].set_ylabel(r'$m^2/s^2$')
    energy_ax[1].set_xlabel(r'$\frac{1}{2} \omega^2$')
    energy_ax[1].set_ylabel(r'$rad^2/s^2$')
    plt.tight_layout()
    energy_fig.savefig(output_figure_directory + f'energy.png')

    # Frequency
    freq_fig.suptitle('Frequency spectrum of linear derivatives')
    freq_ax[2].set_xlabel('Frequency (Hz)')
    freq_ax[0].text(0.95, 0.95, r'$Velocity (m/s)$', transform=freq_ax[0].transAxes, fontsize=12, 
                        verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    freq_ax[1].text(0.95, 0.95, r'$Acceleration (m/s^2)$', transform=freq_ax[1].transAxes, fontsize=12, 
                        verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    freq_ax[2].text(0.95, 0.95, r'$Jerk (m/s^3)$', transform=freq_ax[2].transAxes, fontsize=12, 
                        verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    freq_ax[0].set_ylabel('Amplitude')
    freq_ax[1].set_ylabel('Amplitude')
    freq_ax[2].set_ylabel('Amplitude')
    plt.tight_layout()
    freq_fig.savefig(output_figure_directory + f'frequency.png')

    # Trajectory comparisons
    if len(lin_vel_wass_dist) > 0:

        # Truth trajectories
        truth_traj_fig.suptitle('Truth hand trajectories')
        truth_traj_ax.set_xlabel('X (m)')
        truth_traj_ax.set_ylabel('Y (m)')
        truth_traj_ax.set_zlabel('Z (m)')
        truth_traj_ax.set_aspect('equal')
        plt.tight_layout()
        truth_traj_fig.savefig(output_figure_directory + f'truth_trajectories.png')

        # Truth Energy
        truth_energy_fig.suptitle('Specific Energy')
        truth_energy_ax[0].set_xlabel(r'$\frac{1}{2} v^2 + gz$')
        truth_energy_ax[0].set_ylabel(r'$m^2/s^2$')
        truth_energy_ax[1].set_xlabel(r'$\frac{1}{2} \omega^2$')
        truth_energy_ax[1].set_ylabel(r'$rad^2/s^2$')
        plt.tight_layout()
        truth_energy_fig.savefig(output_figure_directory + f'truth_energy.png')

        # Frequency
        truth_freq_fig.suptitle('Frequency spectrum of linear derivatives')
        truth_freq_ax[2].set_xlabel('Frequency (Hz)')
        truth_freq_ax[0].text(0.95, 0.95, r'$Velocity (m/s)$', transform=truth_freq_ax[0].transAxes, fontsize=12, 
                        verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        truth_freq_ax[1].text(0.95, 0.95, r'$Acceleration (m/s^2)$', transform=truth_freq_ax[1].transAxes, fontsize=12, 
                        verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        truth_freq_ax[2].text(0.95, 0.95, r'$Jerk (m/s^3)$', transform=truth_freq_ax[2].transAxes, fontsize=12, 
                        verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        truth_freq_ax[0].set_ylabel('Amplitude')
        truth_freq_ax[1].set_ylabel('Amplitude')
        truth_freq_ax[2].set_ylabel('Amplitude')
        plt.tight_layout()
        truth_freq_fig.savefig(output_figure_directory + f'truth_frequency.png')

        # MSE
        mse_fig, mse_ax = plt.subplots(1,figsize=(9,5))
        mse_fig.suptitle(f'Model vs. Truth Trajectory MSE')
        mse_ax.hist(np.array(mse), alpha=0.5, edgecolor='black')
        mse_ax.text(0.95, 0.95, f'Mean: {np.mean(mse):03f} (m)', transform=mse_ax.transAxes, fontsize=12, 
                        verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        mse_ax.set_ylabel('Occurences')
        mse_ax.set_xlabel('Position MSE (m)')
        plt.tight_layout()
        mse_fig.savefig(output_figure_directory + f'mse.png')

        # Linear Wasserstein
        lin_wass_fig, lin_wass_ax = plt.subplots(3,figsize=(9,9))
        lin_wass_fig.suptitle('Linear Wasserstein distances by derivative')
        lin_wass_ax[0].hist(np.array(lin_vel_wass_dist), alpha=0.5, edgecolor='black')
        lin_wass_ax[0].text(0.95, 0.95, f'Mean: {np.mean(lin_vel_wass_dist):03f} (m/s)', transform=lin_wass_ax[0].transAxes, fontsize=12, 
                        verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        lin_wass_ax[1].hist(np.array(lin_accel_wass_dist), alpha=0.5, edgecolor='black')
        lin_wass_ax[1].text(0.95, 0.95, f'Mean: {np.mean(lin_accel_wass_dist):03f} (m/s/s)', transform=lin_wass_ax[1].transAxes, fontsize=12, 
                        verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        lin_wass_ax[2].hist(np.array(lin_jerk_wass_dist), alpha=0.5, edgecolor='black')
        lin_wass_ax[2].text(0.95, 0.95, f'Mean: {np.mean(lin_jerk_wass_dist):03f} (m/s/s/s)', transform=lin_wass_ax[2].transAxes, fontsize=12, 
                        verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        [lin_wass_ax[i].set_ylabel('Occurences') for i in range(len(lin_wass_ax))]
        lin_wass_ax[0].set_xlabel('Velocity Wasserstein distance (m/s)')
        lin_wass_ax[1].set_xlabel('Acceleration Wasserstein distance (m/s/s)')
        lin_wass_ax[2].set_xlabel('Jerk Wasserstein distance (m/s/s/s)')
        plt.tight_layout()
        lin_wass_fig.savefig(output_figure_directory + f'linear_wasserstein_distance.png')

        # Angular Wasserstein
        ang_wass_fig, ang_wass_ax = plt.subplots(3,figsize=(9,9))
        ang_wass_fig.suptitle('Angular Wasserstein distances by derivative')
        ang_wass_ax[0].hist(np.array(ang_vel_wass_dist), alpha=0.5, edgecolor='black')
        ang_wass_ax[0].text(0.95, 0.95, f'Mean: {np.mean(ang_vel_wass_dist):03f} (rad/s)', transform=ang_wass_ax[0].transAxes, fontsize=12, 
                        verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ang_wass_ax[1].hist(np.array(ang_accel_wass_dist), alpha=0.5, edgecolor='black')
        ang_wass_ax[1].text(0.95, 0.95, f'Mean: {np.mean(ang_accel_wass_dist):03f} (rad/s/s)', transform=ang_wass_ax[1].transAxes, fontsize=12, 
                        verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ang_wass_ax[2].hist(np.array(ang_jerk_wass_dist), alpha=0.5, edgecolor='black')
        ang_wass_ax[2].text(0.95, 0.95, f'Mean: {np.mean(ang_jerk_wass_dist):03f} (rad/s/s/s)', transform=ang_wass_ax[2].transAxes, fontsize=12, 
                        verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        [ang_wass_ax[i].set_ylabel('Occurences') for i in range(len(ang_wass_ax))]
        ang_wass_ax[0].set_xlabel('Velocity Wasserstein distance (rad/s)')
        ang_wass_ax[1].set_xlabel('Acceleration Wasserstein distance (rad/s/s)')
        ang_wass_ax[2].set_xlabel('Jerk Wasserstein distance (rad/s/s/s)')
        plt.tight_layout()
        ang_wass_fig.savefig(output_figure_directory + f'angular_wasserstein_distance.png')

    plt.close()


if __name__ == '__main__':
    # Script inputs
    input_trajectory_files = [
        os.path.expanduser('~/data/scooping/scooping_processed.hdf5'),
    ]
    output_figure_directory = os.path.expanduser('~/data/scooping/figures/human/')

    # Run script
    plot_scooping_aggregates(
        input_trajectory_files,
        output_figure_directory,
    )