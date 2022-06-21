import math
import numpy as np

def wrapTo180(angle_deg):
  # reduce the angle
  angle_deg = angle_deg % 360
  
  # force it to be the positive remainder, so that 0 <= angle < 360
  angle_deg = (angle_deg + 360) % 360
  
  # force into the minimum absolute value residue class, so that -180 < angle <= 180
  if angle_deg > 180:
    angle_deg = angle_deg - 360
  return angle_deg


def euler_from_quaternion(w, x, y, z):
  t0 = +2.0 * (w * x + y * z)
  t1 = +1.0 - 2.0 * (x * x + y * y)
  roll_x_rad = math.atan2(t0, t1)
  
  t2 = +2.0 * (w * y - z * x)
  t2 = +1.0 if t2 > +1.0 else t2
  t2 = -1.0 if t2 < -1.0 else t2
  pitch_y_rad = math.asin(t2)
  
  t3 = +2.0 * (w * z + x * y)
  t4 = +1.0 - 2.0 * (y * y + z * z)
  yaw_z_rad = math.atan2(t3, t4)
  
  eulers_rad = np.array([roll_x_rad, pitch_y_rad, yaw_z_rad])
  return eulers_rad*180/np.pi


