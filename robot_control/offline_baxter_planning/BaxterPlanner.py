import numpy as np
import quaternion
import cvxpy as qp
from typing import List, Union
from collections import OrderedDict
from scipy.interpolate import interp1d

from src.baxter_pykdl.src.baxter_pykdl import baxter_kinematics


# - Helpers - #

def interpolate_waypoints(time_in, time_out, pose):
    """
    Linear (spherical linear) interpolation to densify positions (quaternions)
    
    Args:
        time_in (np.ndarray): (n,) input array of time (s)
        time_out (np.ndarray): (m,) output array of time (s)
        pose (np.ndarray): (n,7) array of poses (x,y,z,i,j,k,w)
        
    Returns:
        (np.ndarray): (m,7) array of poses (x,y,z,i,j,k,w)
    """
    # Interpolate positions
    pos = pose[:,:3]
    interp_function = interp1d(time_in, pos, axis=0, kind='linear')
    pos_interp = interp_function(time_out)

    # Interpolate quaternions
    quat = np.concatenate([pose[:, 6:7], pose[:,3:6]], axis=1) # quaternion package assumes (w,x,y,z)
    quat_interp = np.zeros((len(time_out), 4))
    for i in range(len(time_out)):
        j = np.sum(time_in <= time_out[i])
        ratio = (time_out[i] - time_in[j - 1])/(time_in[j] - time_in[j - 1])
        q1 = quaternion.from_float_array(quat[j - 1])
        q2 = quaternion.from_float_array(quat[j])
        q_slerp = quaternion.slerp_evaluate(q1, q2, ratio)
        quat_interp[i] = np.array([q_slerp.x, q_slerp.y, q_slerp.z, q_slerp.w])
    
    # Concatenate into pose
    pose_interp = np.concatenate([pos_interp, quat_interp], axis=1)

    return pose_interp


def quat_to_delta_omega(q1, q2):
    """Calculates the angular delta between two quaternions
    quat in ijkw format
    """
    # Check for sign flips
    dot = np.dot(q1, q2)
    if dot < 0:
        q2 = -q2  # Flip the second quaternion to avoid angular velocity spikes

    return 2 * np.array([q1[3]*q2[0] - q1[0]*q2[3] - q1[1]*q2[2] + q1[2]*q2[1], 
                         q1[3]*q2[1] + q1[0]*q2[2] - q1[1]*q2[3] - q1[2]*q2[0],
                         q1[3]*q2[2] - q1[0]*q2[1] + q1[1]*q2[0] - q1[2]*q2[3]])


def differentiate_quat(time, quat):
    """Given time series of quaternions, returns 6-element angular velocity vectors"""
    delta_omega = np.array([quat_to_delta_omega(quat[i+1], quat[i]) for i in range(quat.shape[0] - 1)])
    do_dt = delta_omega / np.diff(time[:,np.newaxis], axis=0)

    return do_dt


def differentiate_waypoints(time, pose):
    """
    Approximates linear and angular velocities across a sequence of poses
    
    Args:
        time (np.ndarray): (n,) array of time (s)
        pose (np.ndarray): (n,7) array of poses (x,y,z,i,j,k,w)
    
    Returns:
        (np.ndarray): (n-1,3) array of linear velocities (vx,vy,vz)
        (np.ndarray): (n-1,3) array of angular velocities (wx,wy,wz)
    """
    # Calculate linear velocities
    pos = pose[:,:3]
    vel = np.diff(pos, axis=0) / np.diff(time[:,np.newaxis], axis=0)
    
    # Calculate angular velocities
    quat = pose[:,3:]
    ang = differentiate_quat(time, quat)
    
    return vel, ang


def wrap_joint_angle(joint_names, joint_angle):
    """Wraps a vector of joint angles into an ordered dictionary with joint name labels"""
    assert len(joint_names) == len(joint_angle)
    return OrderedDict(zip(joint_names, joint_angle))


def unwrap_joint_angle(joint_angle):
    """Unwraps an ordered dictionary of joint angles into an array"""
    return np.array(list(joint_angle.values()))


def joint_angle_in_safe_limits(joint_angle, limits_low, limits_high):
    """Returns True if a vector of joint angles lies within the safety limits"""
    return np.all((limits_low < joint_angle) & (joint_angle < limits_high))


def joint_velocity_in_safe_limits(joint_velocity, limits_low, limits_high):
    """Returns True if a vector of joint velocities lies within the safety limits"""
    return np.all((limits_low < joint_velocity) & (joint_velocity < limits_high))


# - BaxterPlanner - #

class BaxterPlanner:
    """
    Handles planning through joint angle space
    Jacobian based technique for circumventing Baxter IK discontinuity
    """
    # Limits TODO
    joint_angle_limits_low = np.array([-np.inf] * 7)
    joint_angle_limits_high = np.array([np.inf] * 7)
    joint_velocity_limits_low = np.array([-np.inf] * 7)
    joint_velocity_limits_high = np.array([np.inf] * 7)


    def __init__(
        self, 
        urdf: str, 
        limb: str, 
        joint_names: List[str], 
        nominal_joint_angle: dict,
    ):
        """
        Args:
        urdf (str): path to baxter urdf .xml 
        limb (str): 'left' or 'right'
        joint_names (list[str]): list of joint names, prefixed by limb
        nominal_joint_angle (dict): labeled dictionary of 'centered' joint angles (rad)
        """
        assert limb == 'left' or limb == 'right'
        self._kin = baxter_kinematics(urdf, limb)
        self._joint_names = joint_names

        if not isinstance(nominal_joint_angle, dict):
            raise ValueError(f'An ordered dictionary of nominal joint angles should be passed, not {type(nominal_joint_angle)}')
        self._nominal_joint_angle = nominal_joint_angle
        self._nominal_pose = self._kin.forward_position_kinematics(nominal_joint_angle)
    

    def plan(
        self, 
        time: np.ndarray, 
        pose: np.ndarray, 
        output_frequency: float = None, 
        start_joint_angle: Union[dict, np.ndarray, List] = None,
    ):
        """
        Given a sequence of poses, plans a dense trajectory in joint angles
        
        Args:
        time (np.ndarray): (n,) timestamps from start of trajectory
        pose (np.ndarray): (n,7) pose vector of gripper through time
        output_frequency (optional, float): output trajectory frequency, hz. If None, does not interpolate
        start_joint_angle (optional, list, np.ndarray, OrderedDict): starting joint angles (rad)
        
        Returns:
        (np.ndarray): (m,) timestamps from start of trajectory. re-interpolated from input
        (list[OrderedDict]): joint angles through time, wrapped in dictionaries
        (np.ndarray): (m,7) pose vector of gripper through time, re-interpolated and passed through forward kinematics
        """
        if not isinstance(time, np.ndarray):
            time = np.array(time)

        if not isinstance(pose, np.ndarray):
            pose = np.array(pose)

        if isinstance(start_joint_angle, dict):
            start_joint_angle = unwrap_joint_angle(start_joint_angle)
        elif isinstance(start_joint_angle, list):
            start_joint_angle = np.array(start_joint_angle)
        
        # Solve for starting joint angle with inverse kinematics if necessary
        if start_joint_angle is None:
            pos = pose[0, :3]
            quat = pose[0, 3:] / np.linalg.norm(pose[0, 3:])
            quat = [quat[3], quat[0], quat[1], quat[2]] # ijkw -> wijk for IK usage
            q_nominal = unwrap_joint_angle(self._nominal_joint_angle)
            start_joint_angle = self._kin.inverse_kinematics(pos, quat, q_nominal)
            
        # Inverse kinematics failure
        if start_joint_angle is None:
            print('Inverse kinematics failure for starting joint angle.')
            return None, None, None
            # print('Inverse kinematics failure for starting joint angle. Solving a proxy trajectory from nominal angle to initial pose.')
            # proxy_time = np.array([0,7]) # Arbitrarily choose 7 seconds
            # proxy_pose = np.array([self._nominal_pose, pose[0]])
            # _, angles, _ = self.plan(proxy_time, proxy_pose, output_frequency, start_joint_angle=self._nominal_joint_angle)
            # start_joint_angle = unwrap_joint_angle(angles[-1])
            # print('Initial joint angle found: ', start_joint_angle)

        # Interpolate
        if output_frequency is None:
            time_interp = time
            pose_interp = pose
        else:
            time_interp = np.arange(time[0], time[-1], 1 / output_frequency)
            pose_interp = interpolate_waypoints(time, time_interp, pose)
        
        # Differentiate
        vel, ang_vel = differentiate_waypoints(time_interp, pose_interp)
        dt = np.diff(time_interp)
        
        # Iteratively plan velocities and build trajectory of joint angles
        qs = [start_joint_angle]
        vs = []
        for i in range(len(dt)):
            # Get current angle
            q = qs[-1].copy()

            # Retrieve jacobian
            J = self._kin.jacobian(wrap_joint_angle(self._joint_names, q))
            
            # Solve for local plan
            v = self._solve(q, J, vel[i], ang_vel[i], kp=0, kd=0)
            
            # Solving error: trash trajectory
            if v is None:
                print('QP solver failure: returning none')
                return None, None, None
            
            # Integrate velocity
            q += v * dt[i]
            
            # Bookkeep
            qs.append(q)
            vs.append(v)
        
        # Check joint angle/velocity safety constraints
        if not joint_angle_in_safe_limits(q, self.joint_angle_limits_low, self.joint_angle_limits_high) \
            or not joint_velocity_in_safe_limits(v, self.joint_velocity_limits_low, self.joint_velocity_limits_high):
            raise Exception("Warning: planned trajectory violates safe joint angle or velocity limits")
        
        # Wrap angles
        angles = [wrap_joint_angle(self._joint_names, q) for q in qs]
        
        # Evaluate resulting gripper poses for validation
        poses = [self._kin.forward_position_kinematics(a) for a in angles]
        
        return time_interp, angles, poses
  

    def _solve(self, joint_angle, jacobian, velocity, angular_velocity, kp = 1., kd = 1.):
        """
        Solves for joint velocities that will accomplish our near term goal
        Includes PD terms
        
        Args:
        joint_angle (np.ndarray): (7,) vector of joint angles (rad)
        jacobian (np.ndarray): (7,6) array for baxter's kinematics
        velocity (np.ndarray): (3,) array of linear velocities (vx,vy,vz)
        angular_velocity (np.ndarray): (3,) array of angular velocities (wx,wy,wz)
        kp (optional, float): gain for proportional control to joint neutralizing position
        kd (optional, float): gain for derivative control
        
        Returns:
        (np.ndarray): (7,) array of joint velocities (rad/s)
        """
        # Prep weight matrices
        n = len(joint_angle)
        vel = np.concatenate([velocity, angular_velocity])
        P = np.eye(n) - np.linalg.pinv(jacobian).dot(jacobian)
        q_nominal = unwrap_joint_angle(self._nominal_joint_angle)
        dq = q_nominal - joint_angle

        # Form QP
        v = qp.Variable(n)
        problem = qp.Problem(
        qp.Minimize(qp.quad_form(v, jacobian.T @ jacobian) - 2 * vel.T @ jacobian @ v + vel.T @ vel + qp.quad_form(v, P.T @ P) \
                    - 2 * kp * v.T @ P.T @ P @ dq + kp**2 * dq.T @ P.T @ P @ dq + kd**2 * qp.quad_form(v, P.T @ P)),
        [] # Constraints can go here -- TODO
        )
        problem.solve()

        return v.value