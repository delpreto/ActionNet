import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import scipy.spatial.transform as tf
import h5py


def load_npy_from_dir(dir_path):
    """Given a path to a directory, loads all .npy files found"""
    files = glob.glob(os.path.join(dir_path, '*.npy'))
    arrays = [np.load(file) for file in files]

    return arrays

    
def load_traj_hdf5(filename):
  with h5py.File(filename, "r") as f:
    data = f["trajectory"][:]
    columns = f["trajectory"].attrs["columns"]

    return data, columns


def flat_R_to_quat(R):
    """Converts 9-element flattened rotation matrix(s) to quaternion"""
    if R.ndim == 2:
        R = R[np.newaxis, :]

    fun = lambda r: tf.Rotation.from_matrix(r.reshape((3,3))).as_quat()
    quat = np.apply_along_axis(fun, axis=R.ndim-1, arr=R)
    normalized_quat = quat / np.linalg.norm(quat, axis=R.ndim-1)[:,:,np.newaxis]

    return normalized_quat.squeeze()


def quat_to_delta_omega(q1, q2):
    """quat = (x,y,z,w)"""
    return 2 * np.array([q1[3]*q2[0] - q1[0]*q2[3] - q1[1]*q2[2] + q1[2]*q2[1], 
                         q1[3]*q2[1] + q1[0]*q2[2] - q1[1]*q2[3] - q1[2]*q2[0],
                         q1[3]*q2[2] - q1[0]*q2[1] + q1[1]*q2[0] - q1[2]*q2[3]])


def differentiate_quat(time, quat):
    """Given time series of quaternions, returns 6-element angular velocity vectors"""
    if quat.ndim == 2:
        quat = quat[np.newaxis, :]

    q1 = quat[:,1:,:].T
    q2 = quat[:,:-1,:].T
    delta_omega = quat_to_delta_omega(q1, q2).T
    do_dt = delta_omega / np.diff(time, axis=1)
    do_dt_buffered = np.concatenate([np.zeros((do_dt.shape[0], 1, do_dt.shape[2])), do_dt], axis=1) # Prepend 0 value for lost timestep

    return do_dt_buffered


def differentiate(time, data, order=1, axis=0):
    """Returns the nth order derivative of data time-series along specified axis"""
    assert isinstance(order, int) and order >= 0
    
    if order == 0:
        return data
    else:
        ddata_dt = np.gradient(data, axis=axis) / np.gradient(time, axis=axis)
        return differentiate(time, ddata_dt, order=order-1, axis=axis)


def plot_vector_series(time, data):
    """Plots vector timeseries for 2D or 3D array. 
    Assumes (i,j,k) where i are unique trajectories, j are timesteps, and k are vector components"""
    if data.ndim == 2:
        data = data[np.newaxis, :, :]

    i,j,k = data.shape
    
    fig, ax = plt.subplots(k,1,figsize=(8,8))
    for ki in range(k):
        ax[ki].plot(time.T, data[:,:,ki].T)
    
    return fig, ax


if __name__ == '__main__':
    # input_file = os.path.expanduser('~/drl/ActionNet/robot_control/offline_baxter_planning/data/input_trajectory/scooping_human/processed_traj_s00_s11.h5')
    input_file = os.path.expanduser('~/drl/ActionNet/robot_control/offline_baxter_planning/data/input_trajectory/scooping_learned/inference_LinOSS_scooping_5678.h5')
    output_dir = input_file.split('.')[0]

    # Load trajectories
    trajs, _ = load_traj_hdf5(input_file)
    time = trajs[:,:,0:1]

    use_quat = True # Is the trajectory in quaternions or rotation matrices?
    if use_quat:
        data = trajs[:,:,1:]
    else:
        pos = trajs[:,:,1:4]
        quat = flat_R_to_quat(trajs[:,:,4:])
        data = np.concatenate([pos, quat], axis=2)

    # Plot position
    fig, ax = plot_vector_series(time.squeeze(), data)
    fig.suptitle(f'Trajectory')
    fig.savefig(output_dir + '_traj.png')

    # First derivative
    vel = differentiate(time, data[:,:,:3], order=1, axis=1)
    ang_vel = differentiate_quat(time, data[:,:,3:])
    ddata_dt = np.concatenate([vel, ang_vel], axis=2)
    
    # Plot Derivatives (Vel, Accel, Jerk)
    for order in range(1,4):
        o = order - 1 # Start with 1st derivative
        traj_derivative = differentiate(time, ddata_dt, order=o, axis=1)
        fig, ax = plot_vector_series(time.squeeze(), traj_derivative)
        fig.suptitle(f'Order {order} Trajectory Derivative')
        fig.savefig(output_dir + f'_order_{order}.png')