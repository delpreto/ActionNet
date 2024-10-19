import numpy as np
import quaternion
import cvxpy as qp
from collections import OrderedDict
from scipy.interpolate import interp1d
from src.baxter_pykdl.src.baxter_pykdl import baxter_kinematics

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

  def __init__(self, urdf, limb, joint_names, nominal_joint_angle):
    """
    Args:
      urdf (str): path to baxter urdf .xml 
      limb (str): 'left' or 'right'
      joint_names (list[str]): list of joint names, prefixed by limb
      nominal_joint_angle (list, np.ndarray, OrderedDict): vector of 'centered' joint angles (rad)
    """
    assert limb == 'left' or limb == 'right'
    self._kin = baxter_kinematics(urdf, limb)
    self._joint_names = joint_names
    
    if isinstance(nominal_joint_angle, dict):
      self._nominal_joint_angle = unwrap_joint_angle(nominal_joint_angle)
    else:
      self._nominal_joint_angle = np.array(nominal_joint_angle)
  
  def plan(self, t, pose, freq, q0=None):
    """
    Given a sequence of poses, plans a dense trajectory in joint angles
    
    Args:
      t (list, np.ndarray): timestamps from start of trajectory
      pose (list[list], np.ndarray): (7,) pose vector of gripper through time
      freq (float): output trajectory frequency, hz
      q0 (optional, list, np.ndarray, OrderedDict): starting joint angles (rad)
      
    Returns:
      (list): timestamps from start of trajectory. re-interpolated from input
      (list[OrderedDict]): joint angles through time, wrapped in dictionaries
      (list[list]): (7,) pose vector of gripper through time, re-interpolated and passed through forward kinematics
    """
    if not isinstance(t, np.ndarray):
      t = np.array(t)
      assert np.all(t >= 0)
    if not isinstance(pose, np.ndarray):
      pose = np.array(pose)
    if isinstance(q0, dict):
      q0 = unwrap_joint_angle(q0)
    elif q0:
      q0 = np.array(q0)
    else:
      q0 = self._kin.inverse_kinematics(pose[0,:3], pose[0,3:], self._nominal_joint_angle)

    # Interpolate
    t_int, pose_int = interpolate_waypoints(t, pose, freq)
    
    # Differentiate
    dt, vel, ang_vel = differentiate_waypoints(t_int, pose_int)
    
    # Iteratively plan velocities and build trajectory of joint angles
    qs = [q0]
    vs = []
    for i in range(len(dt)):
      # Get current angle
      q = qs[-1].copy()

      # Retrieve jacobian
      J = self._kin.jacobian(wrap_joint_angle(self._joint_names, q))
      
      # Solve for local plan
      v = self._solve(q, J, vel[i], ang_vel[i])
      
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
    
    return t_int, angles, poses
  

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
    dq = self._nominal_joint_angle - joint_angle

    # Form QP
    v = qp.Variable(n)
    problem = qp.Problem(
      qp.Minimize(qp.quad_form(v, jacobian.T @ jacobian) - 2 * vel.T @ jacobian @ v + vel.T @ vel \
                  + qp.quad_form(v, P.T @ P) - 2 * kp * v.T @ P.T @ P @ dq + kp**2 * dq.T @ P.T @ P @ dq \
                  + kd**2 * qp.quad_form(v, P.T @ P)),
      [] # Constraints can go here -- TODO
    )
    problem.solve()

    return v.value


def interpolate_waypoints(t, pose, freq):
  """
  Linear (spherical linear) interpolation to densify positions (quaternions)
  
  Args:
    t (np.ndarray): (n,) array of time (s)
    pose (np.ndarray): (n,7) array of poses (x,y,z,q0,q1,q2,q3)
    freq (float): re-interpolation frequency (Hz)
    
  Returns:
    (np.ndarray): (m,) array of time (s)
    (np.ndarray): (m,7) array of poses (x,y,z,q0,q1,q2,q3)
  """
  # New timestamps
  t_int = np.arange(t[0], t[-1], 1/float(freq))
  
  # Interpolate positions
  pos = pose[:,:3]
  interp_fun = interp1d(t, pos, axis=0, kind='linear')
  pos_int = interp_fun(t_int)

  # Pre-process quaternions to avoid angular velocity explosion
    # HACK the quaternion slerp'ing code doesn't handle negative q3 quaternions well
  pose[:,3:][pose[:, -1] <= 0] *= -1

  # Interpolate quaternions
  quat = np.concatenate([pose[:, 6:7], pose[:,3:6]], axis=1) # quaternion package assumes (w,x,y,z)
  quat_int = np.zeros((len(t_int), 4))
  for i in range(len(t_int)):
    j = np.sum(t <= t_int[i])
    ratio = (t_int[i] - t[j - 1])/(t[j] - t[j - 1])
    q1 = quaternion.from_float_array(quat[j - 1])
    q2 = quaternion.from_float_array(quat[j])
    q_slerp = quaternion.slerp_evaluate(q1, q2, ratio)
    quat_int[i] = np.array([q_slerp.x, q_slerp.y, q_slerp.z, q_slerp.w])
  
  # Concatenate into pose
  pose_int = np.concatenate([pos_int, quat_int], axis=1)

  return t_int, pose_int
  

def differentiate_waypoints(t, pose):
  """
  Approximates linear and angular velocities across a sequence of poses
  
  Args:
    t (np.ndarray): (n,) array of time (s)
    pose (np.ndarray): (n,7) array of poses (x,y,z,q0,q1,q2,q3)
  
  Returns:
    (np.ndarray): (n-1,) array of time deltas (s)
    (np.ndarray): (n-1,3) array of linear velocities (vx,vy,vz)
    (np.ndarray): (n-1,3) array of angular velocities (wx,wy,wz)
  """
  dt = t[1:] - t[:-1]
  
  # Calculate linear velocities
  pos = pose[:,:3]
  vel = (pos[1:] - pos[:-1]) / dt.reshape((-1,1))
  
  # Calculate angular velocities
  quat = np.concatenate([pose[:, 6:7], pose[:,3:6]], axis=1) # quaternion package assumes (w,x,y,z)
  ang = np.zeros((len(dt), 3))
  for i in range(len(quat) - 1):
    # Calculate dq
    q2 = quaternion.from_float_array(quat[i+1])
    q1 = quaternion.from_float_array(quat[i])
    q_diff = q2 * q1.conjugate()
    q_diff = np.array([q_diff.x, q_diff.y, q_diff.z, q_diff.w])
    
    # Convert to angular velocity
    theta = 2 * np.arccos(np.clip(q_diff[3], -1, 1))
    if theta < 1e-6: # Negligible
      w = np.zeros(3)
    else:
      axis = q_diff[:3] / np.sin(theta / 2)
      w = axis * theta / dt[i]
    
    # Concatenate
    ang[i] = w
  
  return dt, vel, ang


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