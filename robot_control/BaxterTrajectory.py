
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
    self._trajectory_duration_s = 0
    self.clear()
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
    
  def set_goal_time_tolerance_s(self, goal_time_tolerance_s):
    self._goal_time_tolerance_s = goal_time_tolerance_s
    self._goal.goal_time_tolerance = rospy.Time(self._goal_time_tolerance_s)
    