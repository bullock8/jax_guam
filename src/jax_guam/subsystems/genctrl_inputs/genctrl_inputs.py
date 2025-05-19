import jax.numpy as jnp
import numpy as np

from jax_guam.guam_types import RefInputs
from jax_guam.utils.batch_spline import get_spline
# from jax_guam.functional.guam_new import FuncGUAM, GuamState
import math
from scipy.interpolate import CubicSpline
from math import radians, cos, sin, sqrt


def lift_cruise_reference_inputs(time):
    # return lift_cruise_reference_inputs_1(time)
    return lift_cruise_reference_inputs_2(time)


def lift_cruise_reference_inputs_1(time):
    """Uses linear interpolation between specified desired velocities (vel_bIc_des) and positions (pos_des)."""
    timeseries = [0, 20, 40]
    vel_bIc_des = jnp.array([[0, 0, -8], [0, 0, 0], [15, 0, 0]])  # SimPar.RefInputs.SimInput.Vel_bIc_des
    pos_des = jnp.array([[0, 0, 0], [0, 0, -80], [150, 0, -100]])  # SimPar.RefInputs.SimInput.pos_des
    chi_des = jnp.array([0, 0, 0])  # SimPar.RefInputs.SimInput.chi_des
    chi_dot_des = jnp.array([0, 0, 0])  # SimPar.RefInputs.SimInput.chi_dot_des

    # Linear interpolate between the points.
    if time in timeseries:
        index = timeseries.index(time)
        return RefInputs(
            Vel_bIc_des=vel_bIc_des[index],
            Pos_des=pos_des[index],
            Chi_des=chi_des[index],
            Chi_dot_des=chi_dot_des[index],
        )
    elif time > timeseries[0] and time < timeseries[1]:
        first = (vel_bIc_des[1] - vel_bIc_des[0]) / (timeseries[1] - timeseries[0]) * time + vel_bIc_des[0]
        second = (pos_des[1] - pos_des[0]) / (timeseries[1] - timeseries[0]) * time + pos_des[0]
        return RefInputs(Vel_bIc_des=first, Pos_des=second, Chi_des=chi_des[0], Chi_dot_des=chi_dot_des[0])
    elif time > timeseries[1] and time < timeseries[2]:
        first = (vel_bIc_des[2] - vel_bIc_des[1]) / (timeseries[2] - timeseries[1]) * (
            time - timeseries[1]
        ) + vel_bIc_des[1]
        second = (pos_des[2] - pos_des[1]) / (timeseries[2] - timeseries[1]) * (time - timeseries[1]) + pos_des[1]
        return RefInputs(Vel_bIc_des=first, Pos_des=second, Chi_des=chi_des[0], Chi_dot_des=chi_dot_des[0])
    else:
        return RefInputs(Vel_bIc_des=np.zeros(3), Pos_des=pos_des[2], Chi_des=chi_des[2], Chi_dot_des=chi_dot_des[2])


def lift_cruise_reference_inputs_2(time: float):
    """" Vehicle hovers, then moves in a square pattern. It defines time points (T_t), desired velocities (T_vel_bIc), 
    and positions (T_pos_bii) for the pattern. Spline interpolation is used to smoothly interpolate between these points, and 
    the interpolated values are returned as 'RefInputs' object."""
    # First, hover. Then, go in a square.
    # (5, )
    T_t = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
    # (5, 3)
    T_vel_bIc = np.array([[0, 0, -8], [20, 0, 0], [0, 20, 0], [-20, 0, 0], [0, 0, 0], [0, 0, 0]])
    # (5, 3)
    T_pos_bii = np.array([[0, 0, 0], [0, 0, -80], [200, 0, -80], [200, 200, -80], [0, 200, -80], [0, 200, -80]])

    T_scale = T_t[-1]
    spl_vel_bIc = get_spline(T_t / T_scale, T_vel_bIc, k=1, s=0)
    spl_pos_bii = get_spline(T_t / T_scale, T_pos_bii, k=1, s=0)

    vel_bIc = spl_vel_bIc(time / T_scale)
    pos_bii = spl_pos_bii(time / T_scale)

    assert vel_bIc.shape == (3,) and pos_bii.shape == (3,)
    # print((time, pos_bii))
    return RefInputs(vel_bIc, pos_bii, Chi_des=np.array(0.0), Chi_dot_des=np.array(0.0))

def lift_cruise_reference_inputs_coc(time, time_bound, state):
    """" clear of collision command """
    init_pos = state.aircraft[6:9]
    init_vel = state.aircraft[0:3]
    final_pos = init_pos + init_vel * time_bound
    # First, hover. Then, go in a square.
    # (2, )
    T_t = np.array([0.0, time_bound])
    # (2, 3)
    T_vel_bIc = np.array([init_vel, init_vel])
    # (2, 3)
    T_pos_bii = np.array([init_pos, final_pos])

    T_scale = T_t[-1]
    spl_vel_bIc = get_spline(T_t / T_scale, T_vel_bIc, k=1, s=0)
    spl_pos_bii = get_spline(T_t / T_scale, T_pos_bii, k=1, s=0)

    vel_bIc = spl_vel_bIc(time / T_scale)
    pos_bii = spl_pos_bii(time / T_scale)

    assert vel_bIc.shape == (3,) and pos_bii.shape == (3,)
    print((time, pos_bii, vel_bIc))
    return RefInputs(vel_bIc, pos_bii, Chi_des=np.array(0.0), Chi_dot_des=np.array(0.0))
# def lift_cruise_reference_inputs_coc(dt, state):
#     """ command with clear of collision """
#     curr_pos = state.aircraft[6:9]
#     curr_vel = state.aircraft[0:3]
#     new_pos = curr_pos + curr_vel * dt
#     new_vel = curr_vel
#     return RefInputs(new_vel, new_pos, Chi_des=np.array(0.0), Chi_dot_des=np.array(0.0))
    
    
    
# def lift_cruise_reference_inputs_cmd(time, time_bound, init_state, curr_state, dt, cmd):
    
#     """ command with a turn angle rate """
#     turn_advisories = [0.0, 0.0, 1.5, -3, 3]
#     init_vx, init_vy, init_vz = init_state.aircraft[0], init_state.aircraft[1], init_state.aircraft[2]
#     init_x, init_y, init_z = init_state.aircraft[6], init_state.aircraft[7], init_state.aircraft[8]
#     curr_vx, curr_vy, curr_vz = curr_state.aircraft[0], curr_state.aircraft[1], curr_state.aircraft[2]
#     curr_x, curr_y, curr_z = curr_state.aircraft[6], curr_state.aircraft[7], curr_state.aircraft[8]
#     qw, qx, qy, qz = curr_state.aircraft[9], curr_state.aircraft[10], curr_state.aircraft[11], curr_state.aircraft[12]
#     init_qw, init_qx, init_qy, init_qz = init_state.aircraft[9], init_state.aircraft[10], init_state.aircraft[11], init_state.aircraft[12]
    
   
#     curr_speed = np.sqrt(curr_vx**2 + curr_vy**2 + curr_vz**2)
#     turning_rate = turn_advisories[cmd] *np.pi/180
    
#     vx_dot = -curr_vy * turning_rate
#     vy_dot = curr_vx * turning_rate
#     vz_dot = 0.0
    
#     new_vx = init_vx + vx_dot * dt
#     new_vy = init_vy + vy_dot * dt
#     new_vz = init_vz + vz_dot * dt
    

#     new_x = init_x + new_vx * time
#     new_y = init_y + new_vy * time
#     new_z = init_z + new_vz * time
#     new_vel = np.array([new_vx, new_vy, new_vz])
#     new_pos = np.array([new_x, new_y, new_z])
#     print((time, vy_dot, curr_vy))
#     return RefInputs(new_vel, new_pos, Chi_des=np.array(0.0), Chi_dot_des=np.array(0.0))
def get_spline(t, y, k=1, s=0):
    return CubicSpline(t, y, bc_type='natural', extrapolate=True)   

def lift_cruise_reference_inputs_turn_right(time, time_bound, state, cmd):
    turn_advisories = [0, -1.5, 1.5, -3, 3] 
    init_pos = np.array(np.squeeze(state.aircraft[0, 6:9]))
    #print(init_pos.shape)
    init_vel = np.squeeze(state.aircraft[0, 0:3])
    #print(f"init_vel: {init_vel}")
    init_speed = sqrt(sum(init_vel**2))
    #print(f"init speed: {init_speed}")
    init_heading = math.atan2(init_vel[0], init_vel[1])
    turn_rate = radians(turn_advisories[cmd])  # 3 degrees per second in radians, neg values represent a left turn 
    print(f"init_heading: {init_heading}")
    
    #print(f"initPos type: {type(init_pos)}")
    #print(f"initVel type: {type(init_vel)}")
    #print(f"inithead type: {type(init_heading)}")

    #print(f"speed: {init_speed}")
    
    # Generate time array
    T_t = np.linspace(0, 10, num=1000)#int(10*time_bound) + 1)

    # Generate velocity and position arrays
    T_vel_bIc = []
    T_pos_bii = []
    for t in T_t:
        new_heading = init_heading + turn_rate * t
        # if time < 0.01:
        #     print(time, cmd, turn_rate, new_heading, t)
        
        init_speed = 100
        new_vx = init_speed * sin(new_heading)
        new_vy = init_speed * cos(new_heading)
        
        
        T_vel_bIc.append([new_vx, new_vy, 0]) # init_vel[2]])

        if t == 0:
            T_pos_bii.append(init_pos)
        else:
            new_pos = T_pos_bii[-1] + np.array([new_vx, new_vy, 0]) * (T_t[1] - T_t[0]) # init_vel[2]]) * (T_t[1] - T_t[0])#
            new_pos[2] = -10 # Try to force altitude of 10 m (12/2)
            T_pos_bii.append(new_pos)

    T_vel_bIc = np.array(T_vel_bIc)
    T_pos_bii = np.array(T_pos_bii)

    # Create splines
    T_scale = T_t[-1]
    spl_vel_bIc = get_spline(T_t / T_scale, T_vel_bIc, k=1, s=0)
    spl_pos_bii = get_spline(T_t / T_scale, T_pos_bii, k=1, s=0)

    # Evaluate splines at the given time
    vel_bIc = spl_vel_bIc(1 / T_scale)#np.array([new_vx, new_vy, init_vel[2]]) * 100 /(np.sqrt(new_vx**2 + new_vy**2 + init_vel[2]**2)) #spl_vel_bIc(time / T_scale)
    pos_bii = spl_pos_bii(1 / T_scale) #new_pos #spl_pos_bii(time / T_scale)

    assert vel_bIc.shape == (3,) and pos_bii.shape == (3,)
    # print((time, pos_bii, vel_bIc))
    return RefInputs(vel_bIc, pos_bii, Chi_des=np.array(0.0), Chi_dot_des=np.array(0.0))

def lift_cruise_reference_inputs_turn_right2(time, time_bound, state, cmd):
    turn_advisories = [0, -1.5, 1.5, -3, 3] 
    init_pos = np.array(np.squeeze(state.aircraft[0, 6:9]))
    #print(init_pos.shape)
    init_vel = np.squeeze(state.aircraft[0, 0:3])
    #print(f"init_vel: {init_vel}")
    init_speed = sqrt(sum(init_vel**2))
    #print(f"init speed: {init_speed}")
    init_heading = math.atan2(init_vel[0], init_vel[1])
    turn_rate = radians(turn_advisories[cmd])  # 3 degrees per second in radians, neg values represent a left turn 
    
    
    #print(f"initPos type: {type(init_pos)}")
    #print(f"initVel type: {type(init_vel)}")
    #print(f"inithead type: {type(init_heading)}")

    #print(f"speed: {init_speed}")
    
    # Generate time array
    T_t = np.linspace(0, 0.3, num=3)#int(10*time_bound) + 1)
    dt = 0.1

    # Generate velocity and position arrays
    T_vel_bIc = []
    T_pos_bii = []
    
    new_heading = init_heading + 4*turn_rate * dt
    
    print(f"advisory: {turn_advisories[cmd]}")
        # if time < 0.01:
        #     print(time, cmd, turn_rate, new_heading, t)
        
    init_speed = 100
    new_vx = init_speed * sin(new_heading)
    new_vy = init_speed * cos(new_heading)
    
    new_pos = init_pos + np.array([new_vx, new_vy, 0]) * (dt) # init_vel[2]]) * (T_t[1] - T_t[0])#
    new_pos[2] = -10 # Try to force altitude of 10 m (12/2)
    T_pos_bii.append(new_pos)
    
    # Update vel
    vel_heading = init_heading + turn_rate * 4*2 * dt
    init_speed = 100
    newer_vx = init_speed * sin(vel_heading)
    newer_vy = init_speed * cos(vel_heading)
    
    T_vel_bIc = np.array(T_vel_bIc)
    T_pos_bii = np.array(T_pos_bii)

    # Create splines
    #T_scale = T_t[-1]
    #spl_vel_bIc = get_spline(T_t / T_scale, T_vel_bIc, k=1, s=0)
    #spl_pos_bii = get_spline(T_t / T_scale, T_pos_bii, k=1, s=0)

    # Evaluate splines at the given time
    vel_bIc = np.array([newer_vx, newer_vy, 0]) * 100 /(np.sqrt(newer_vx**2 + newer_vy**2 ))#spl_vel_bIc(.1 / T_scale)#np.array([new_vx, new_vy, init_vel[2]]) * 100 /(np.sqrt(new_vx**2 + new_vy**2 + init_vel[2]**2)) #spl_vel_bIc(time / T_scale)
    pos_bii = new_pos #spl_pos_bii(.1 / T_scale) #new_pos #spl_pos_bii(time / T_scale)

    assert vel_bIc.shape == (3,) and pos_bii.shape == (3,)
    # print((time, pos_bii, vel_bIc))
    return RefInputs(Vel_bIc_des=vel_bIc, Pos_des=pos_bii, Chi_des=np.array(0.0), Chi_dot_des=np.array(0.0)), new_heading

def lift_cruise_reference_inputs_turn_right3(time, time_bound, state, cmd):
    turn_advisories = [0, -1.5, 1.5, -3, 3] 
    init_pos = np.array(np.squeeze(state.aircraft[0, 6:9]))
    #print(init_pos.shape)
    init_vel = np.squeeze(state.aircraft[0, 0:3])
    #print(f"init_vel: {init_vel}")
    init_speed = sqrt(sum(init_vel**2))
    #print(f"init speed: {init_speed}")
    init_heading = math.atan2(init_vel[0], init_vel[1])
    turn_rate = radians(turn_advisories[cmd])  # 3 degrees per second in radians, neg values represent a left turn 
    
    
    #print(f"initPos type: {type(init_pos)}")
    #print(f"initVel type: {type(init_vel)}")
    #print(f"inithead type: {type(init_heading)}")

    #print(f"speed: {init_speed}")
    
    # Generate time array
    T_t = np.linspace(0, 0.3, num=3)#int(10*time_bound) + 1)
    dt = 0.1

    # Generate velocity and position arrays
    T_vel_bIc = []
    T_pos_bii = []
    
    new_heading = math.atan2(math.sqrt(100**2 - 50**2), 50) + time * turn_rate#init_heading + turn_rate * dt
    
    print(f"advisory: {turn_advisories[cmd]}")
        # if time < 0.01:
        #     print(time, cmd, turn_rate, new_heading, t)
        
    init_speed = 100
    new_vx = init_speed * sin(new_heading)
    new_vy = init_speed * cos(new_heading)
    
    new_pos = init_pos + np.array([new_vx, new_vy, 0]) * (dt) # init_vel[2]]) * (T_t[1] - T_t[0])#
    new_pos[2] = -10 # Try to force altitude of 10 m (12/2)
    T_pos_bii.append(new_pos)
    
    # Update vel
    vel_heading = math.atan2(math.sqrt(100**2 - 50**2), 50) + time * turn_rate
    init_speed = 100
    newer_vx = init_speed * sin(vel_heading)
    newer_vy = init_speed * cos(vel_heading)
    
    T_vel_bIc = np.array(T_vel_bIc)
    T_pos_bii = np.array(T_pos_bii)

    # Create splines
    #T_scale = T_t[-1]
    #spl_vel_bIc = get_spline(T_t / T_scale, T_vel_bIc, k=1, s=0)
    #spl_pos_bii = get_spline(T_t / T_scale, T_pos_bii, k=1, s=0)

    # Evaluate splines at the given time
    vel_bIc = np.array([newer_vx, newer_vy, 0]) * 100 /(np.sqrt(newer_vx**2 + newer_vy**2 ))#spl_vel_bIc(.1 / T_scale)#np.array([new_vx, new_vy, init_vel[2]]) * 100 /(np.sqrt(new_vx**2 + new_vy**2 + init_vel[2]**2)) #spl_vel_bIc(time / T_scale)
    pos_bii = new_pos #spl_pos_bii(.1 / T_scale) #new_pos #spl_pos_bii(time / T_scale)

    assert vel_bIc.shape == (3,) and pos_bii.shape == (3,)
    # print((time, pos_bii, vel_bIc))
    return RefInputs(vel_bIc, pos_bii, Chi_des=np.array(0.0), Chi_dot_des=np.array(0.0)), new_heading

#def lift_cruise_reference_inputs_turn_random(time, time_bound, state, cmd_list):
    turn_advisories = [0, -1.5, 1.5, -3, 3] 
    init_pos = state.aircraft[6:9]
    init_vel = state.aircraft[0:3]
    init_speed = sqrt(sum(init_vel**2))
    init_heading = math.atan2(init_vel[0], init_vel[1])
    

    # Generate time array for heading updates (0.1 sec intervals)
    T_heading = np.linspace(0, time_bound, num=int(time_bound / 0.1) + 1)
    
    # Generate time array for reference sequence (0.01 sec intervals)
    T_ref = np.linspace(0, time_bound, num=int(time_bound / 0.01) + 1)
    
    # Generate heading angles and velocity arrays
    heading_angles = [init_heading]
    T_vel_bIc = [init_vel]

  
    T_pos_bii = [init_pos]
    
    for i, t in enumerate(T_ref[:-1]):
        turn_rate = radians(turn_advisories[cmd_list[i]])
        new_heading = heading_angles[-1] + turn_rate * 0.01
        
        heading_angles.append(new_heading)
        new_vx = init_speed * sin(new_heading)
        new_vy = init_speed * cos(new_heading)
        T_vel_bIc.append([new_vx, new_vy, init_vel[2]])
        
        dt = T_heading[1] if i == 0 else 0.01
        new_pos = T_pos_bii[-1] + np.array([new_vx, new_vy, init_vel[2]]) * dt
        T_pos_bii.append(new_pos)
        
        
        # Update position based on new velocity
        
        # if i < len(T_heading[:-1]):
        # for kkk in range(int(0.1/0.01)):
        #     if kkk == 0:
        #         new_heading = heading_angles[-1] + turn_rate * 0.1
        #     else:
        #         new_heading = heading_angles[-1]    
            
        #     heading_angles.append(new_heading)
        #     new_vx = init_speed * sin(new_heading)
        #     new_vy = init_speed * cos(new_heading)
        #     T_vel_bIc.append([new_vx, new_vy, init_vel[2]])

            
        #     dt = T_heading[1] if i == 0 else 0.01
        #     new_pos = T_pos_bii[-1] + np.array([new_vx, new_vy, init_vel[2]]) * dt
        #     T_pos_bii.append(new_pos)
        
    # for _ in range(int(0.1/0.01)):
    #     curr_vx, curr_vy, curr_vz = T_vel_bIc[-1]
    #     dt = T_heading[1] if i == 0 else T_heading[i + 1] - T_heading[i]
    #     new_pos = T_pos_bii[-1] + np.array([curr_vx, curr_vy, curr_vz]) * dt
    #     T_pos_bii.append(new_pos)
            
        
    # Create splines
    T_scale = T_ref[-1]
    spl_vel_bIc = get_spline(T_ref / T_scale, T_vel_bIc, k=1, s=0)
    spl_pos_bii = get_spline(T_ref / T_scale, T_pos_bii, k=1, s=0)

    # Evaluate splines at the given time
    vel_bIc = spl_vel_bIc(time / T_scale)
    pos_bii = spl_pos_bii(time / T_scale)

    # Interpolate velocity and position for reference sequence time steps
    # print(len(T_heading), len(T_ref), len(T_vel_bIc), len(T_pos_bii))
    # print(heading_angles)
    # spl_vel_bIc = get_spline(T_ref, T_vel_bIc, k=1, s=0)
    # spl_pos_bii = get_spline(T_ref, T_pos_bii, k=1, s=0)
    # ref_vel_bIc = spl_vel_bIc(T_ref / T_heading[-1])
    # ref_pos_bii = spl_pos_bii(T_ref / T_heading[-1])

    # # Evaluate splines at the given time
    # vel_bIc = ref_vel_bIc[int(time / 0.01)]
    # pos_bii = ref_pos_bii[int(time / 0.01)]

    assert vel_bIc.shape == (3,) and pos_bii.shape == (3,)
    # print((time, pos_bii, vel_bIc))
    return RefInputs(vel_bIc, pos_bii, Chi_des=np.array(0.0), Chi_dot_des=np.array(0.0))

def circular_turn_trajectory(position, velocity, turn_rate=3, time_step=0.1):
    """
    Generates a circular trajectory given an initial position and velocity vector,
    turning at a constant 3 degrees per second until a full circle is completed.
    Interpolates both position and velocity, and ensures that the initial point is at t=0.

    Parameters:
    - position: tuple (x0, y0) → Initial 2D position
    - velocity: tuple (vx, vy) → Initial 2D velocity vector
    - turn_rate: float → Turn rate in degrees per second (default: 3°/sec)
    - time_step: float → Time step for interpolation (default: 0.1 sec)
    - turn_direction: str → "left" for counterclockwise, "right" for clockwise

    Returns:
    - (x_interp, y_interp, vx_interp, vy_interp, center): Interpolated position & velocity
    """

    # Convert turn rate to radians per second
    turn_rate_rad = 4 * np.deg2rad(turn_rate)  # 3 degrees/sec → radians/sec

    # Compute initial speed (magnitude of velocity)
    speed = 100 # np.linalg.norm(velocity)

    # Compute turning radius (R = v / ω)
    omega = np.abs(turn_rate_rad)  # Angular velocity in rad/sec
    R = speed / omega  # Turning radius

    # Compute center of rotation based on turn direction
    x, y = position
    vx, vy = velocity * 100 / np.linalg.norm(velocity)
    if turn_rate < 0:
        perp_direction = np.array([-vy, vx])  # Counterclockwise normal
    else:# turn_rate < 0:
        perp_direction = np.array([vy, -vx])  # Clockwise normal
    #else:
        #raise ValueError("turn_direction must be 'left' or 'right'.")

    perp_direction = perp_direction / np.linalg.norm(perp_direction)  # Normalize
    center = np.array(position) + perp_direction * R  # Compute center

    # Generate time values starting at t=0
    total_time = 2 * np.pi / omega  # Time to complete full circle
    t = np.arange(0, total_time + time_step, time_step)  # Time steps including t=0

    # Compute angles over time
    orig_angle = (-np.pi/2 + np.arctan2(vy, vx))  # Original angle
    #print(orig_angle)
    angles = np.linspace(orig_angle, orig_angle + 2*np.pi, num = t.size) #omega * t  # θ(t) = ω * t

    if turn_rate > 0:
        orig_angle = -orig_angle
        angles = np.linspace(orig_angle, orig_angle + 2*np.pi, num = t.size) #omega * t  # θ(t) = ω * t
        angles = -(angles + np.pi)  # Reverse direction for clockwise turn

    # Compute circular trajectory using parametric equations
    x_traj = center[0] + R * np.cos(angles)
    y_traj = center[1] + R * np.sin(angles)

    # Interpolate position using CubicSpline (starting at t=0)
    spline_x = CubicSpline(t, x_traj, bc_type='natural')
    spline_y = CubicSpline(t, y_traj, bc_type='natural')

    # Compute velocity as the derivative of position
    spline_vx = spline_x.derivative()
    spline_vy = spline_y.derivative()

    # Generate smooth time values
    t_fine = 0.5 * time_step #np.linspace(0, total_time, 200)
    x_interp = spline_x(t_fine)
    y_interp = spline_y(t_fine)
    vx_interp = spline_vx(t_fine)
    vy_interp = spline_vy(t_fine)

    return x_interp, y_interp, vx_interp, vy_interp, center

def lift_cruise_reference_inputs_circ_turn(time, time_bound, state, cmd):
    turn_advisories = [0, -1.5, 1.5, -3, 3] 
    init_pos = np.array(np.squeeze(state.aircraft[0, 6:9]))
    #print(init_pos.shape)
    init_vel = np.squeeze(state.aircraft[0, 0:3])
    #print(f"init_vel: {init_vel}")
    init_speed = sqrt(sum(init_vel**2))
    #print(f"init speed: {init_speed}")
    init_heading = math.atan2(init_vel[0], init_vel[1])
    turn_rate_deg = turn_advisories[cmd]  # 3 degrees per second in radians, neg values represent a left turn 
    turn_rate = radians(turn_advisories[cmd])  # 3 degrees per second in radians, neg values represent a left turn 
    
    
    #print(f"initPos type: {type(init_pos)}")
    #print(f"initVel type: {type(init_vel)}")
    #print(f"inithead type: {type(init_heading)}")

    #print(f"speed: {init_speed}")

    # Generate velocity and position arrays
    T_vel_bIc = []
    T_pos_bii = []
    
    ########################
    ## New Stuff
    ########################
    curr_x, curr_y, curr_z = init_pos
    vx, vy, vz = init_vel
    if cmd == 0:
        new_pos = init_pos + np.array([vx, vy, 0]) * (0.1) # init_vel[2]]) * (T_t[1] - T_t[0])#
        new_pos[2] = -10 # Try to force altitude of 10 m (12/2)
        
        new_vel = np.array([vx, vy, 0]) * 100 / np.sqrt(vx**2 + vy**2)
    else:
        start_pos = np.array([curr_x, curr_y])
        start_vel = np.array([vx, vy]) 
        #start_vel = start_vel
        #x_interp, y_interp, vx_interp, vy_interp, _ = circular_turn_trajectory(position = start_pos, velocity = start_vel, turn_rate=turn_rate_deg, time_step=0.1)## Use the function here for turn rate
        x_interp, y_interp, vx_interp, vy_interp, _ = calc_turn_ref_cmds(position = start_pos, velocity = start_vel, turn_rate=turn_rate_deg, time_step=0.1)
        
        new_pos = np.array([x_interp, y_interp, -10])
        
        new_vel = np.array([vx_interp, vy_interp, 0])
        #new_vel = np.array([vx, vy, 0]) * 100 / np.linalg.norm(new_vel)
    
    #new_heading = math.atan2(math.sqrt(100**2 - 50**2), 50) + time * turn_rate#init_heading + turn_rate * dt
    
    #print(f"advisory: {turn_advisories[cmd]}")
        # if time < 0.01:
        #     print(time, cmd, turn_rate, new_heading, t)
    ''' 
    init_speed = 100
    new_vx = init_speed * sin(new_heading)
    new_vy = init_speed * cos(new_heading)
    
    new_pos = init_pos + np.array([new_vx, new_vy, 0]) * (dt) # init_vel[2]]) * (T_t[1] - T_t[0])#
    new_pos[2] = -10 # Try to force altitude of 10 m (12/2)
    T_pos_bii.append(new_pos)
    
    # Update vel
    vel_heading = math.atan2(math.sqrt(100**2 - 50**2), 50) + time * turn_rate
    init_speed = 100
    newer_vx = init_speed * sin(vel_heading)
    newer_vy = init_speed * cos(vel_heading)
    
    T_vel_bIc = np.array(T_vel_bIc)
    T_pos_bii = np.array(T_pos_bii)

    # Create splines
    #T_scale = T_t[-1]
    #spl_vel_bIc = get_spline(T_t / T_scale, T_vel_bIc, k=1, s=0)
    #spl_pos_bii = get_spline(T_t / T_scale, T_pos_bii, k=1, s=0)

    # Evaluate splines at the given time
    vel_bIc = np.array([newer_vx, newer_vy, 0]) * 100 /(np.sqrt(newer_vx**2 + newer_vy**2 ))#spl_vel_bIc(.1 / T_scale)#np.array([new_vx, new_vy, init_vel[2]]) * 100 /(np.sqrt(new_vx**2 + new_vy**2 + init_vel[2]**2)) #spl_vel_bIc(time / T_scale)
    pos_bii = new_pos #spl_pos_bii(.1 / T_scale) #new_pos #spl_pos_bii(time / T_scale)

    assert vel_bIc.shape == (3,) and pos_bii.shape == (3,)
    # print((time, pos_bii, vel_bIc))'''
    return RefInputs(Pos_des=new_pos, Vel_bIc_des=new_vel, Chi_des=np.array(0.0), Chi_dot_des=np.array(0.0))


def calc_turn_ref_cmds(position, velocity, turn_rate=3, time_step=0.1):
    # Convert turn rate to radians per second
    turn_rate_rad = np.deg2rad(turn_rate)  # 3 degrees/sec → radians/sec

    # Compute initial speed (magnitude of velocity)
    speed = np.linalg.norm(velocity)

    # Compute turning radius (R = v / ω)
    omega = np.abs(turn_rate_rad)  # Angular velocity in rad/sec
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
    delta_theta = -turn_rate * time_step

    x_des = x_c + R * np.cos(circ_angle + delta_theta)
    y_des = y_c + R * np.sin(circ_angle + delta_theta)

    vx_des = 100 * np.sin(circ_angle + delta_theta) * np.sign(turn_rate)
    vy_des = -100 * np.cos(circ_angle + delta_theta) * np.sign(turn_rate)

    return x_des, y_des, vx_des, vy_des, center


def QrotZ(chi):
  '''
  Return quaternion(s) representing rotation(s) of _a_ about the Z axis.

  Usage: q = QrotZ(a)

  Input:
    a = vector of rotation angle (rads), either row or column
  
  Output:
    q = array of quaternion corresponding to the rotations.
  '''
  s = np.cos(chi / 2)
  v = np.sin(chi / 2)

  q = np.array([s, 0 * v, 0 * v, v])

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

  v = np.array([a2*b3 + b2*a3 + c2*d3 - d2*c3, 
                a2*c3 + c2*a3 + d2*b3 - b2*d3, 
                a2*d3 + d2*a3 + b2*c3 - c2*b3])

  return v
    
    

def lift_cruise_turn_heading(time, time_bound, state, cmd):
    # ACAS turn advisory rates (deg)
    turn_advisories = [0, -1.5, 1.5, -3, 3] 
    
    # Aircraft position (NED)
    init_pos = np.array(np.squeeze(state.aircraft[0, 6:9]))
    #print(init_pos.shape)
    # Aircraft velocity (body frame)
    init_vel = np.squeeze(state.aircraft[0, 0:3])
    
    # Aircraft velocity
    init_speed = sqrt(sum(init_vel**2))
    
    # Desired turn rate (in deg, then radians)
    turn_rate_deg = turn_advisories[cmd]  # 3 degrees per second in radians, neg values represent a left turn 
    turn_rate = radians(turn_advisories[cmd])  # 3 degrees per second in radians, neg values represent a left turn 
    
    # Orientation of the vehicle
    quat = np.array(np.squeeze(state.aircraft[0, 9:13]))
    qw, qx, qy, qz = quat
    # Calculate heading (rad) (TODO, check frame for NED/xyz, adapt as needed)
    heading = np.arctan2(2*qy*qw-2*qx*qz , 1 - 2*(qy**2) - 2*(qz**2))
    
    # Calculate new heading (rad) after one time step
    dt = 0.1
    updated_heading = heading + turn_rate * dt
    #print(f"quat: {np.linalg.norm(quat)}")
    #print(f"Heading: {np.rad2deg(heading)}")
    
    # Wrap heading between [-pi,pi]
    #if updated_heading > np.pi:
    #    updated_heading = updated_heading - 2 * np.pi
    #elif updated_heading < -np.pi:
    #    updated_heading = updated_heading + 2 * np.pi
    
    # Get current NED position and body-frame velocity
    curr_n, curr_e, curr_z = init_pos
    curr_x = curr_e
    curr_y = curr_n
    vx, vy, vz = init_vel
    
    vel_mag = np.sqrt(vx ** 2 + vy ** 2 + vz **2)
    start_vel = np.array([vel_mag * np.sin(heading), vel_mag * np.cos(heading)]) 
    
    if cmd == 0:
        new_pos = init_pos + np.array([vel_mag * np.sin(heading), vel_mag * np.cos(heading), 0]) * (0.1) # init_vel[2]]) * (T_t[1] - T_t[0])#
        new_pos[2] = -10 # Try to force altitude of 10 m (12/2)
        
        new_vel = np.array([vel_mag * np.sin(heading), vel_mag * np.cos(heading), 0])#([vx, vy, 0]) * 100 / np.sqrt(vx**2 + vy**2)
    else:
        
        start_pos = np.array([curr_x, curr_y])
        
        #start_vel = start_vel
        #x_interp, y_interp, vx_interp, vy_interp, _ = circular_turn_trajectory(position = start_pos, velocity = start_vel, turn_rate=turn_rate_deg, time_step=0.1)## Use the function here for turn rate
        x_interp, y_interp, vx_interp, vy_interp, _ = calc_turn_ref_cmds(position = start_pos, velocity = start_vel, turn_rate=turn_rate_deg, time_step=0.1)
        
        # Switch back from an x,y frame to a NED frame
        n_interp = y_interp
        e_interp = x_interp
        
        vn_interp = vy_interp
        ve_interp = vx_interp
        new_pos = np.array([n_interp, e_interp, -10])
        
        new_vel = np.array([vn_interp, ve_interp, 0])
        
    # calculate body frame velocity
    q_heading = QrotZ(heading)
    body_vel = Qtrans(q_heading, new_vel)
        
    return RefInputs(Pos_des=new_pos, Vel_bIc_des=new_vel, Chi_des=np.array(updated_heading), Chi_dot_des=np.array(turn_rate))
    


def lift_cruise_reference_inputs_turn_random(dt, time, time_bound, state, cmd_list):
    turn_advisories = [0, -1.5, 1.5, -3, 3] 
    init_pos = state.aircraft[6:9]
    init_vel = state.aircraft[0:3]
    init_speed = sqrt(sum(init_vel**2))
    init_heading = math.atan2(init_vel[0], init_vel[1])
    

    # Generate time array for heading updates (0.1 sec intervals)
    T_heading = np.linspace(0, time_bound, num=int(time_bound /dt) + 1)
    
    # Generate time array for reference sequence (0.01 sec intervals)
    T_ref = np.linspace(0, time_bound, num=int(time_bound / dt) + 1)
    
    # Generate heading angles and velocity arrays
    heading_angles = [init_heading]
    T_vel_bIc = [init_vel]

  
    T_pos_bii = [init_pos]
    
    for i, t in enumerate(T_ref[:-1]):
        turn_rate = radians(turn_advisories[cmd_list[i]])
        new_heading = heading_angles[-1] + turn_rate * dt
        
        heading_angles.append(new_heading)
        new_vx = init_speed * sin(new_heading)
        new_vy = init_speed * cos(new_heading)
        T_vel_bIc.append([new_vx, new_vy, init_vel[2]])
        
        dt = T_heading[1] if i == 0 else dt
        new_pos = T_pos_bii[-1] + np.array([new_vx, new_vy, init_vel[2]]) * dt
        T_pos_bii.append(new_pos)
        
        
        # Update position based on new velocity
        
        # if i < len(T_heading[:-1]):
        # for kkk in range(int(0.1/0.01)):
        #     if kkk == 0:
        #         new_heading = heading_angles[-1] + turn_rate * 0.1
        #     else:
        #         new_heading = heading_angles[-1]    
            
        #     heading_angles.append(new_heading)
        #     new_vx = init_speed * sin(new_heading)
        #     new_vy = init_speed * cos(new_heading)
        #     T_vel_bIc.append([new_vx, new_vy, init_vel[2]])

            
        #     dt = T_heading[1] if i == 0 else 0.01
        #     new_pos = T_pos_bii[-1] + np.array([new_vx, new_vy, init_vel[2]]) * dt
        #     T_pos_bii.append(new_pos)
        
    # for _ in range(int(0.1/0.01)):
    #     curr_vx, curr_vy, curr_vz = T_vel_bIc[-1]
    #     dt = T_heading[1] if i == 0 else T_heading[i + 1] - T_heading[i]
    #     new_pos = T_pos_bii[-1] + np.array([curr_vx, curr_vy, curr_vz]) * dt
    #     T_pos_bii.append(new_pos)
            
        
    # Create splines
    T_scale = T_ref[-1]
    spl_vel_bIc = get_spline(T_ref / T_scale, T_vel_bIc, k=1, s=0)
    spl_pos_bii = get_spline(T_ref / T_scale, T_pos_bii, k=1, s=0)

    # Evaluate splines at the given time
    vel_bIc = spl_vel_bIc(time / T_scale)
    pos_bii = spl_pos_bii(time / T_scale)

    # Interpolate velocity and position for reference sequence time steps
    # print(len(T_heading), len(T_ref), len(T_vel_bIc), len(T_pos_bii))
    # print(heading_angles)
    # spl_vel_bIc = get_spline(T_ref, T_vel_bIc, k=1, s=0)
    # spl_pos_bii = get_spline(T_ref, T_pos_bii, k=1, s=0)
    # ref_vel_bIc = spl_vel_bIc(T_ref / T_heading[-1])
    # ref_pos_bii = spl_pos_bii(T_ref / T_heading[-1])

    # # Evaluate splines at the given time
    # vel_bIc = ref_vel_bIc[int(time / 0.01)]
    # pos_bii = ref_pos_bii[int(time / 0.01)]

    assert vel_bIc.shape == (3,) and pos_bii.shape == (3,)
    # print((time, pos_bii, vel_bIc))
    return RefInputs(vel_bIc, pos_bii, Chi_des=np.array(0.0), Chi_dot_des=np.array(0.0))

def initialize_reference_inputs(time_bound, state, cmd_list):
    turn_advisories = [0, -1.5, 1.5, -3, 3] 
    init_pos = state.aircraft[6:9]
    init_vel = state.aircraft[0:3]
    init_speed = sqrt(sum(init_vel**2))
    init_heading = math.atan2(init_vel[0], init_vel[1])

    T_heading = np.linspace(0, time_bound, num=int(time_bound / 0.1) + 1)
    heading_angles = [init_heading]
    T_vel_bIc = [init_vel]
    T_pos_bii = [init_pos]
    
    for i, t in enumerate(T_heading[:-1]):
        turn_rate = radians(turn_advisories[cmd_list[i]])
        new_heading = heading_angles[-1] + turn_rate * 0.1
        heading_angles.append(new_heading)
        new_vx = init_speed * sin(new_heading)
        new_vy = init_speed * cos(new_heading)
        T_vel_bIc.append([new_vx, new_vy, init_vel[2]])

        dt = T_heading[1] if i == 0 else T_heading[i + 1] - T_heading[i]
        new_pos = T_pos_bii[-1] + np.array([new_vx, new_vy, init_vel[2]]) * dt
        T_pos_bii.append(new_pos)

    spl_vel_bIc = get_spline(T_heading, T_vel_bIc, k=1, s=0)
    spl_pos_bii = get_spline(T_heading, T_pos_bii, k=1, s=0)

    return spl_vel_bIc, spl_pos_bii
