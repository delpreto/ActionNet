
# Heavily based on the following code:
# https://github.com/RethinkRobotics/baxter_examples/blob/master/scripts/joint_trajectory_client.py

from __future__ import print_function

import rospy
import baxter_interface
from time import sleep

import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint
from copy import copy


class BaxterTrajectory(object):
  def __init__(self, limb_name, goal_time_tolerance_s=0.1):
    self._goal_time_tolerance_s = goal_time_tolerance_s
    self._limb_name = limb_name
    self._limb = baxter_interface.Limb(limb_name)
    self._trajectory_duration_s = 0
    self._joint_angles_rad = []
    self.clear()
    try: # node may already be initialized by someone using this class
      rospy.init_node('baxterTrajectory')
    except Exception as e:
      errMessage = str(e)
      if 'init_node' in errMessage.lower() and 'already been called' in errMessage.lower():
        pass
      else:
        raise
    ns = 'robot/limb/' + self._limb_name + '/'
    self._client = actionlib.SimpleActionClient(
      ns + "follow_joint_trajectory",
      FollowJointTrajectoryAction,
    )
    server_up = self._client.wait_for_server(timeout=rospy.Duration(10.0))
    if not server_up:
      msg = "Timed out waiting for Joint Trajectory Action Server to connect." \
            "Start the action server via joint_trajectory_action_server.py before running example."
      print(msg)
      rospy.logerr(msg)
      raise AssertionError(msg)

  def add_point(self, joint_angles_rad, time_from_start_s):
    point = JointTrajectoryPoint()
    point.positions = copy(joint_angles_rad)
    point.time_from_start = rospy.Duration(time_from_start_s)
    self._goal.trajectory.points.append(point)
    self._trajectory_duration_s = max(self._trajectory_duration_s, time_from_start_s)
    self._joint_angles_rad.append(joint_angles_rad)

  def start(self):
    self._goal.trajectory.header.stamp = rospy.Time.now()
    self._client.send_goal(self._goal)

  def stop(self):
    self._client.cancel_goal()
    
  def run(self, wait_for_completion=True):
    self.start()
    if wait_for_completion:
      self.wait()

  def wait(self, timeout_s=None):
    if timeout_s is None:
      timeout_s = 1.5*self._trajectory_duration_s
    self._client.wait_for_result(timeout=rospy.Duration(timeout_s))

  def result(self):
    return self._client.get_result()
  
  def clear(self):
    self._goal = FollowJointTrajectoryGoal()
    self.set_goal_time_tolerance_s(self._goal_time_tolerance_s)
    self._goal.trajectory.joint_names = [self._limb_name + '_' + joint for joint in \
      ['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2']]
    # Add the current position at time 0,
    #  since otherwise it seems to often do nothing when the trajectory is run.
    joint_angles_rad = self._limb.joint_angles()
    joint_angles_rad = [joint_angles_rad[joint_name] for joint_name in self._goal.trajectory.joint_names]
    self.add_point(joint_angles_rad, 0)
    self._joint_angles_rad = []
    
  def set_goal_time_tolerance_s(self, goal_time_tolerance_s):
    self._goal_time_tolerance_s = goal_time_tolerance_s
    self._goal.goal_time_tolerance = rospy.Time(self._goal_time_tolerance_s)
    
  def get_joint_angles_rad(self, step_index):
    return self._joint_angles_rad[step_index]

if __name__ == '__main__':
  limb_name = 'right'
  trajectory = BaxterTrajectory(limb_name=limb_name)
  a1 = [0.80, 0.21, 0.00, 0.73, 0.00, -0.94, 0]
  a2 = [-0.80, 0.21, 0.00, 0.73, 0.00, -0.94, 0]
  a3 = [0.80, 0.21, 0.00, 0.73, 0.00, -0.94, 0]
  trajectory.add_point(a1, 5)
  trajectory.add_point(a2, 10)
  trajectory.add_point(a3, 15)
  trajectory.run(True)
