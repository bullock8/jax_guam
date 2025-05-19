import jax.numpy as jnp
import numpy as np

from jax_guam.guam_types import RefInputs
from jax_guam.utils.batch_spline import get_spline
# from jax_guam.functional.guam_new import FuncGUAM, GuamState
import math
from scipy.interpolate import CubicSpline
from math import radians, cos, sin, sqrt

import jax.numpy as jnp

def QrotZ(chi):
  '''
  Return quaternion(s) representing rotation(s) of _a_ about the Z axis.

  Usage: q = QrotZ(a)

  Input:
    a = vector of rotation angle (rads), either row or column

  Output:
    q = array of quaternion corresponding to the rotations.
  '''
  s = jnp.cos(chi / 2)
  v = jnp.sin(chi / 2)

  q = jnp.array([s, 0 * v, 0 * v, v])

  return q



def Qtrans(q1, v1):
  '''
  Transform given vector(s) by given (rotation) quaternion(s).

  Usage: v = Qtrans(q1,v1);

  Description:

    Transform a given vector(s) v1 by given (rotation) quaternion(s) q1,
    returning the resulting vector(s) v.

  Input:     q1 = quaternion(s) describing the desired rotation(s),
                  either (N1x4) or (4xN1).
             v1 = original (non-rotated) vector(s), either (N2x3) or
                  (3xN2).
           Either N1 must equal N2, or either N1 or N2 must be 1.

  Outputs:   v = the transformed (rotated) vector(s), (Nx3) or (3xN),
                 where N is max(N1,N2).
  '''
  b1 = v1[0]
  c1 = v1[1]
  d1 = v1[2]

  a2 = q1[0]
  b2 = q1[1]
  c2 = q1[2]
  d2 = q1[3]

  # compute (V1*Q1)

  a3 = - b1*b2 - c1*c2 - d1*d2
  b3 =   b1*a2 + c1*d2 - d1*c2
  c3 =   c1*a2 + d1*b2 - b1*d2
  d3 =   d1*a2 + b1*c2 - c1*b2

  b2 = -b2
  c2 = -c2
  d2 = -d2

  # compute Q1~ * (V1*Q1)

  v = jnp.array([a2*b3 + b2*a3 + c2*d3 - d2*c3, a2*c3 + c2*a3 + d2*b3 - b2*d3, a2*d3 + d2*a3 + b2*c3 - c2*b3 ])

  return v

def Qinv(q):
  '''
  Find the inverse of a quaternion

  Input:
    q: a quaternion

  Output:
    q_inv:  the inverse quaternion of q
  '''
  q0, q1, q2, q3 = q

  q_inv = jnp.array([q0, -q1, -q2, -q3]) / (np.linalg.norm(q)**2)

  return q_inv

def Q2Heading(q):
  '''
  Calculate the vehicle heading based on its quaternion orientation

  Input:
    q: a quaternion representing the vehicle's heading

  Output:
    heading: the heading (rad) of the vehicle, in the range [-pi, pi]
  '''
  Qw, Qx, Qy, Qz = q
  heading = jnp.arctan2(2.0 * (Qz * Qw + Qx * Qy) , - 1.0 + 2.0 * (Qw * Qw + Qx * Qx))

  return heading

def quaternion_to_euler(q):
    """
    Convert a quaternion to Euler angles (roll, pitch, yaw)
    
    Parameters:
    - q: Quaternion as jnp.array of shape (4,) [w, x, y, z]
    
    Returns:
    - roll: Rotation around x-axis (in radians)
    - pitch: Rotation around y-axis (in radians)
    - yaw (heading): Rotation around z-axis (in radians)
    """
    w, x, y, z = q

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = jnp.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    # Clamp sinp to [-1, 1] to avoid NaNs due to numerical errors
    sinp = jnp.clip(sinp, -1.0, 1.0)
    pitch = jnp.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = jnp.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles (roll, pitch, yaw) to a quaternion.

    Parameters:
    - roll: Rotation around x-axis (in radians)
    - pitch: Rotation around y-axis (in radians)
    - yaw (heading): Rotation around z-axis (in radians)

    Returns:
    - Quaternion as jnp.array([w, x, y, z])
    """
    hr = roll * 0.5
    hp = pitch * 0.5
    hy = yaw * 0.5

    cr = jnp.cos(hr)
    sr = jnp.sin(hr)
    cp = jnp.cos(hp)
    sp = jnp.sin(hp)
    cy = jnp.cos(hy)
    sy = jnp.sin(hy)

    # Quaternion formula for yaw-pitch-roll (Z-Y-X intrinsic rotation)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return jnp.array([w, x, y, z])
'''
heading = 210 * jnp.pi/180
vel_inertial = 100 * jnp.array([jnp.cos(heading)+0.01, jnp.sin(heading), 0])

print(f"inertial: {vel_inertial}")

q = QrotZ(heading)

print(f"heading: {(180 / jnp.pi) * Q2Heading(q)} degrees")
vel_heading = Qtrans(q, vel_inertial)

print(f"heading: {vel_heading}")

print(f"inv q: {Qinv(q)}")

heading2inertial = Qinv(q)
print(f"inertial v: {Qtrans(heading2inertial, vel_heading)}")
'''

def circular_turn_calc(position, velocity, turn_rate=3, time_step=0.1):
    # Convert turn rate to radians per second
    turn_rate_rad = np.deg2rad(turn_rate)  # 3 degrees/sec → radians/sec

    # Compute initial speed (magnitude of velocity)
    speed = np.linalg.norm(velocity)

    # Compute turning radius (R = v / ω)
    omega = np.abs(turn_rate_rad)  # Angular velocity in rad/sec
    
    if turn_rate != 0:
        # We have a non-COC turn command
        R = speed / omega  # Turning radius

        # Compute center of rotation based on turn direction
        x, y = position
        vx, vy = velocity
        if turn_rate < 0:
            perp_direction = np.array([-vy, vx])  # Counterclockwise normal
        else:# turn_rate < 0:
            perp_direction = np.array([vy, -vx])  # Clockwise normal
        #else:
            #raise ValueError("turn_direction must be 'left' or 'right'.")

        perp_direction = perp_direction / np.linalg.norm(perp_direction)  # Normalize
        center = np.array(position) + perp_direction * R  # Compute center

        x_c, y_c = center

        circ_angle = np.arctan2(y - y_c, x - x_c)

        #dt = 0.1
        delta_theta = -turn_rate * (np.pi/180) * time_step

        x_des = x_c + R * np.cos(circ_angle + delta_theta)
        y_des = y_c + R * np.sin(circ_angle + delta_theta)

        vx_des = 100 * np.sin(circ_angle + delta_theta) * np.sign(turn_rate)
        vy_des = -100 * np.cos(circ_angle + delta_theta) * np.sign(turn_rate)
    else:
        # we have a CoC command
        x, y = position
        vx, vy = velocity
        
        # Propagate desired states forward by 1 time step (normalize velocity to 100 ft/s)
        x_des = x + vx * (100 / jnp.linalg.norm(velocity)) * time_step
        y_des = y + vy * (100 / jnp.linalg.norm(velocity)) * time_step
        
        # Keep current velocity (normalized to 100 ft/s)
        vx_des = vx * (100 / jnp.linalg.norm(velocity))
        vy_des = vy * (100 / jnp.linalg.norm(velocity))
        
        center = jnp.array([0,0]) # Dummy value for the center
        

    return x_des, y_des, vx_des, vy_des, center
    #return x_des, y_des
    
def get_frame_des(dt, pos_ned, vel_body, quat, turn_rate_deg, down_des=-10, vz_des=0):

  # Extract NED position
  north = pos_ned[0]
  east = pos_ned[1]
  down = pos_ned[2]
  # Re-frame coordinates in a right-hand xy frame
  pos_xy = np.array([east, north])

  # Calculate the quaternion to represent rotation from heading to inertial frame
  # i.e. the inverse of the quaternion from inertial to heading
  inv_quat = Qinv(quat)
  vel_inertial = Qtrans(inv_quat, vel_body)

  v_north, v_east, v_down = vel_inertial # Extract ned inertial velocity

  # Change to a right-hand xy coordinate frame
  vel_xy = np.array([v_east, v_north])
  #dt = 0.1

  des_heading = Q2Heading(quat) + turn_rate_deg * (np.pi/180) * dt

  x_des, y_des, vx_des, vy_des, center = circular_turn_calc(position = pos_xy, velocity = vel_xy, turn_rate = -1.5, time_step=dt)

  north_des = y_des
  east_des = x_des
  down_des = down_des # -10

  des_pos = np.array([north_des, east_des, down_des])

  v_north_des = vy_des
  v_east_des = vx_des
  v_down_des = vz_des #0

  des_vel_inertial = np.array([v_north_des, v_east_des, v_down_des])

  des_vel_body = Qtrans(quat, des_vel_inertial)

  return des_pos, des_vel_body, des_heading


def acas_reference_inputs(dt, state, cmd, des_down=-10, des_vz=0):
    '''
    Calculate the GUAM reference inputs for a given ACAS advisory
    
    Input:
        dt: sample period of the simulation (s)
        state: GuamState object containing the current state of the aircraft
        cmd: the ACAS command to be executed
        des_down: desired down position (ft) (default: -10)
        des_vz: desired vertical speed (ft/s) (default: 0)
    Output:
        RefInputs object containing the reference inputs for the aircraft
    '''
    turn_advisories = [0, -1.5, 1.5, -3, 3] 
    init_pos = state.aircraft[ 6:9].flatten()
    init_vel = state.aircraft[ 0:3].flatten()
    
    #surfengstates = state.surf_eng.ctrl_surf_state
    #print(f"surf eng: {surfengstates}")
    
    #print(f"pos: {init_pos}")
    #print(f"vel: {init_vel}")
    #init_speed = sqrt(sum(init_vel**2))
    #init_heading = math.atan2(init_vel[0], init_vel[1])
    quat_i2b = state.aircraft[ 9:13].flatten()
    
    turn_rate = radians(turn_advisories[cmd])
    turn_rate_deg = turn_advisories[cmd]
    
    #print(f"init_pos:{init_pos}")
    #print(f"init_vel: {init_vel}")
    #print(f"quat: {quat_i2b}")
    
    new_pos, new_vel_body, new_heading = get_frame_des(dt = dt, pos_ned = init_pos, vel_body = init_vel, quat = quat_i2b, turn_rate_deg = turn_rate_deg, down_des = des_down, vz_des = des_vz)
    
    return RefInputs(Pos_des=new_pos, Vel_bIc_des=new_vel_body, Chi_des=new_heading, Chi_dot_des=jnp.array(turn_rate))