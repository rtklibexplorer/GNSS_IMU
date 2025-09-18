#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 Project:    Sensor Fusion for GNSS/IMU (Loosely Coupled Kalman Filter)
 File:       imu_math.py
 Author:     Tim Everett
 Copyright:  (c) 2025 Tim Everett
 License:    BSD 3 License (see LICENSE file for details)

 Description:
     Math functions for GNSS/IMU sensor fusion
===============================================================================
"""

import numpy as np
from scipy.signal import butter, filtfilt
from numpy.linalg import norm, inv
from imu_transforms import Euler_to_CTM

# Kalman Filter Configuration
class LC_KF_Config:
    pass

# Initial conditions
class Init:
    pass

def Gravity_ECEF(r_eb_e):
    #Gravity_ECEF - Calculates  acceleration due to gravity resolved about 
    #ECEF-frame
    #
    # Inputs:
    #   r_eb_e  Cartesian position of body frame w.r.t. ECEF frame, resolved
    #           about ECEF-frame axes (m)
    # Outputs:
    #   g       Acceleration due to gravity (m/s^2)
    
    # Parameters
    R_0 = 6378137           # WGS84 Equatorial radius in meters
    mu = 3.986004418E14     # WGS84 Earth gravitational constant (m^3 s^-2)
    J_2 = 1.082627E-3       # WGS84 Earth second gravitational constant
    omega_ie = 7.292115E-5  # Earth rotation rate (rad/s)

    # Calculate distance from center of the Earth
    mag_r = norm(r_eb_e)
    
    # If the input position is 0,0,0, produce a dummy output
    if mag_r == 0:
        return np.zeros(3)
    
    # Calculate gravitational acceleration using (2.142)
    z_scale = 5 * (r_eb_e[2] / mag_r)**2
    g = -mu / mag_r**3 * (r_eb_e + 1.5 * J_2 * (R_0 / mag_r)**2 *
        np.array([(1 - z_scale) * r_eb_e[0],
                  (1 - z_scale) * r_eb_e[1],
                  (3 - z_scale) * r_eb_e[2]]))

    # Add centripetal acceleration using (2.133)
    g[0:2] += omega_ie**2 * r_eb_e[0:2]
    return g

def Initialize_LC_P_matrix(LC_KF_config):
    #Initialize_LC_P_matrix - Initializes the loosely coupled INS/GNSS KF
    #error covariance matrix
    #
    # Inputs:
    #   TC_KF_config
    #     .init.att_unc           Initial attitude uncertainty per axis (rad)
    #     .init.vel_unc           Initial velocity uncertainty per axis (m/s)
    #     .init.pos_unc           Initial position uncertainty per axis (m)
    #     .init.bias_acc_unc      Initial accel. bias uncertainty (m/s^2)
    #     .init.bias_gyro_unc     Initial gyro. bias uncertainty (rad/s)
    #
    # Outputs:
    #   P_matrix              state estimation error covariance matrix
    
    # Initialize error covariance matrix
    ns = LC_KF_config.nstates
    P = np.zeros((ns, ns))
    P[0:3, 0:3] = np.eye(3) * np.array(LC_KF_config.init.att_unc)**2
    P[3:6, 3:6] = np.eye(3) * np.array(LC_KF_config.init.vel_unc)**2
    P[6:9, 6:9] = np.eye(3) * np.array(LC_KF_config.init.pos_unc)**2
    P[9:12, 9:12] = np.eye(3) * np.array(LC_KF_config.init.bias_acc_unc)**2
    P[12:15, 12:15] = np.eye(3) *np.array(LC_KF_config.init.bias_gyro_unc)**2
    if ns == 21:  # Scale factors
        P[15:18, 15:18] = np.eye(3) * np.array(LC_KF_config.init.scale_acc_unc)**2
        P[18:21, 18:21] = np.eye(3) * np.array(LC_KF_config.init.scale_gyro_unc)**2
    return P

def Skew_symmetric(a):
    #Skew_symmetric - Calculates skew-symmetric matrix
    #
    # Inputs:
    #   a       3-element vector
    # Outputs:
    #   A       3x3matrix
    
    return np.array([
        [0, -a[2], a[1]],
        [a[2], 0, -a[0]],
        [-a[1], a[0], 0]
    ])

def Radii_of_curvature(L):
    #Radii_of_curvature - Calculates the meridian and transverse radii of
    #curvature
    #
    # Inputs:
    #   L   geodetic latitude (rad)
    #
    # Outputs:
    #   R_N   meridian radius of curvature (m)
    #   R_E   transverse radius of curvature (m)
    
    R_0 = 6378137
    e = 0.0818191908425
    temp = 1 - (e * np.sin(L))**2
    R_N = R_0 * (1 - e**2) / temp**1.5
    R_E = R_0 / np.sqrt(temp)
    return R_N, R_E

def Lever_Arm(C_b2f, v_ref_f, r_ref_f, omega_ib_b, r_lever_arm_b):
    r_lever_arm_f = C_b2f @ r_lever_arm_b
    v_lever_arm_f = C_b2f @ np.cross(omega_ib_b, r_lever_arm_b)
    v_centripetal_f = C_b2f @ np.cross(omega_ib_b, np.cross(omega_ib_b, r_lever_arm_b))
    r_point_f = r_ref_f + r_lever_arm_f
    v_point_f = v_ref_f + v_lever_arm_f - v_centripetal_f
    return(v_point_f, r_point_f)

def ortho_C(C: np.ndarray) -> np.ndarray:
    # Check if CTM is ortho-normalized and if not, ortho-normalize it
    ortho_norm = np.allclose(C.T @ C, np.eye(3), atol=1e-8) and np.isclose(np.linalg.det(C), 1.0, atol=1e-8)
    if not ortho_norm:
        U, _, Vt = np.linalg.svd(C)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        return R
    else:
        return C

def Nav_equations_ECEF(tor_i, old_r_eb_e, old_v_eb_e, old_C_b_e, f_ib_b, 
                       omega_ib_b, gravity):
    #Nav_equations_ECEF - Runs precision ECEF-frame inertial navigation
    #equations
    #
    # Inputs:
    #   tor_i         time interval between epochs (s)
    #   old_r_eb_e    previous Cartesian position of body frame w.r.t. ECEF
    #                 frame, resolved along ECEF-frame axes (m)
    #   old_C_b_e     previous body-to-ECEF-frame coordinate transformation matrix
    #   old_v_eb_e    previous velocity of body frame w.r.t. ECEF frame, resolved
    #                 along ECEF-frame axes (m/s)
    #   f_ib_b        specific force of body frame w.r.t. ECEF frame, resolved
    #                 along body-frame axes, averaged over time interval (m/s^2)
    #   omega_ib_b    angular rate of body frame w.r.t. ECEF frame, resolved
    #                 about body-frame axes, averaged over time interval (rad/s)
    # Outputs:
    #   r_eb_e        Cartesian position of body frame w.r.t. ECEF frame, resolved
    #                 along ECEF-frame axes (m)
    #   v_eb_e        velocity of body frame w.r.t. ECEF frame, resolved along
    #                 ECEF-frame axes (m/s)
    #   C_b_e         body-to-ECEF-frame coordinate transformation matrix
    
    omega_ie = 7.292115E-5 # Earth rotation rate (rad/s)

    # ATTITUDE UPDATE
    # From (2.145) determine the Earth rotation over the update interval
    alpha_ie = omega_ie * tor_i
    C_Earth = np.array([
        [np.cos(alpha_ie), np.sin(alpha_ie), 0],
        [-np.sin(alpha_ie), np.cos(alpha_ie), 0],
        [0, 0, 1]])

    # Calculate attitude increment, magnitude, and skew-symmetric matrix
    alpha_ib_b = omega_ib_b * tor_i
    mag_alpha = norm(alpha_ib_b)
    Alpha_ib_b = Skew_symmetric(alpha_ib_b)

    # Obtain coordinate transformation matrix from the new attitude w.r.t. an
    # inertial frame to the old using Rodrigues formula, (5.73)
    if mag_alpha > 1e-8:
        C_new_old = (np.eye(3) + np.sin(mag_alpha)/mag_alpha * Alpha_ib_b +
                     (1 - np.cos(mag_alpha))/mag_alpha**2 * Alpha_ib_b @ Alpha_ib_b)
    else:
        C_new_old = np.eye(3) + Alpha_ib_b

    # Update attitude using (5.75)
    C_b_e = C_Earth @ old_C_b_e @ C_new_old

    # SPECIFIC FORCE FRAME TRANSFORMATION
    # Calculate the average body-to-ECEF-frame coordinate transformation
    # matrix over the update interval using (5.84) and (5.85)
    if mag_alpha > 1e-8:
        ave_C_b_e = old_C_b_e @ (
            np.eye(3) + (1 - np.cos(mag_alpha))/mag_alpha**2 * Alpha_ib_b +
            (1 - np.sin(mag_alpha)/mag_alpha)/mag_alpha**2 * Alpha_ib_b @ Alpha_ib_b
        ) - 0.5 * Skew_symmetric([0, 0, alpha_ie]) @ old_C_b_e
    else:
        ave_C_b_e = old_C_b_e - 0.5 * Skew_symmetric([0, 0, alpha_ie]) @ old_C_b_e

    # Transform specific force to ECEF-frame resolving axes using (5.85)
    f_ib_e = ave_C_b_e @ f_ib_b

    # UPDATE VELOCITY From (5.36)
    v_eb_e = old_v_eb_e + tor_i * (f_ib_e + gravity
                - 2 * Skew_symmetric([0, 0, omega_ie]) @ old_v_eb_e)

    # UPDATE CARTESIAN POSITION From (5.38)
    r_eb_e = old_r_eb_e + 0.5 * tor_i * (v_eb_e + old_v_eb_e)

    return r_eb_e, v_eb_e, C_b_e


def LC_KF_Predict(tor_s, est_C_b_e, est_v_eb_e, est_r_eb_e, est_IMU_bias, P,
                meas_f_ib_b, meas_omega_ib_b, LC_KF_config, gravity, geocentric_radius):
    #LC_KF_Epoch - Implements one cycle of the loosely coupled INS/GNSS
    # Kalman filter plus closed-loop correction of all inertial states
    #
    # Inputs:
    #   tor_s                 propagation interval (s)
    #   est_C_b_e             prior estimated body to ECEF coordinate
    #                         transformation matrix
    #   est_v_eb_e            prior estimated ECEF user velocity (m/s)
    #   est_r_eb_e            prior estimated ECEF user position (m)
    #   est_IMU_bias          prior estimated IMU biases (body axes)
    #   P                     previous Kalman filter error covariance matrix
    #   meas_f_ib_b           measured specific force
    #   LC_KF_config
    #     .gyro_noise_PSD     Gyro noise PSD (rad^2/s)
    #     .accel_noise_PSD    Accelerometer noise PSD (m^2 s^-3)
    #     .accel_bias_PSD     Accelerometer bias random walk PSD (m^2 s^-5)
    #     .gyro_bias_PSD      Gyro bias random walk PSD (rad^2 s^-3)
    #     .pos_meas_SD            Position measurement noise SD per axis (m)
    #     .vel_meas_SD            Velocity measurement noise SD per axis (m/s)
    #     .lever_arm          IMU to GNSS lever arm
    #
    # Outputs:
    #   est_C_b_e_new     updated estimated body to ECEF coordinate 
    #                      transformation matrix
    #   est_v_eb_e_new    updated estimated ECEF user velocity (m/s)
    #   est_r_eb_e_new    updated estimated ECEF user position (m)
    #   est_IMU_bias_new  updated estimated IMU biases
    #     Rows 1-3          estimated accelerometer biases (m/s^2) 
    #     Rows 4-6          estimated gyro biases (rad/s)
    #   P_matrix_new      updated Kalman filter error covariance matrix
    
    omega_ie = 7.292115E-5
   
    abs_tor_s = abs(tor_s)

    # Skew symmetric matrix of Earth rate
    Omega_ie = Skew_symmetric([0, 0, omega_ie])

    # 1. Determine transition matrix using (14.50) (first-order approx)
    ns = LC_KF_config.nstates
    Phi = np.eye(ns)
    Phi[0:3, 0:3] -= Omega_ie * tor_s
    Phi[0:3, 12:15] = est_C_b_e * tor_s
    Phi[3:6, 0:3] = -tor_s * Skew_symmetric(est_C_b_e @ meas_f_ib_b)
    Phi[3:6, 3:6] -= 2 * Omega_ie * tor_s

    unit_r = est_r_eb_e / norm(est_r_eb_e)
    Phi[3:6, 6:9] = -tor_s * 2 * np.outer(gravity, unit_r) / geocentric_radius
    Phi[3:6, 9:12] = est_C_b_e * tor_s
    Phi[6:9, 3:6] = np.eye(3) * tor_s
    if ns == 21: #Scale factors
        Phi[3:6, 15:18] = est_C_b_e @ np.diag(meas_f_ib_b) * tor_s
        Phi[0:3, 18:21] = est_C_b_e @ np.diag(meas_omega_ib_b) * tor_s

    # 2. Determine approximate system noise covariance matrix using (14.82)
    Q_prime = np.zeros((ns, ns))
    Q_prime[0:3, 0:3] = np.eye(3) * LC_KF_config.attitude_noise_var * abs_tor_s
    Q_prime[3:6, 3:6] = np.eye(3) * LC_KF_config.vel_noise_var * abs_tor_s
    Q_prime[9:12, 9:12] = np.eye(3) * LC_KF_config.accel_bias_noise_var * abs_tor_s
    Q_prime[12:15, 12:15] = np.eye(3) * LC_KF_config.gyro_bias_noise_var * abs_tor_s
    if ns == 21: #Scale factors
        Q_prime[15:18, 15:18] = np.eye(3) * LC_KF_config.accel_scale_noise_var * abs_tor_s
        Q_prime[18:21, 18:21] = np.eye(3) * LC_KF_config.gyro_scale_noise_var * abs_tor_s

    # 3. Propagate state estimates using (3.14) noting that all states are zero 
    # due to closed-loop correction.
    # x_est_propagated = np.zeros((ns, 1))   # implied
    
    # 4. Propagate state estimation error covariance matrix using (3.46)
    P_propagated = Phi @ (P + 0.5 * Q_prime) @ Phi.T + 0.5 * Q_prime

    return(P_propagated)

def LC_KF_GNSS_Update(GNSS_r_eb_e, GNSS_v_eb_e,
                est_C_b_e, est_v_eb_e, est_r_eb_e, est_IMU_bias,
                P_propagated, meas_f_ib_b, meas_omega_ib_b, LC_KF_config):

    # Propagated state estimates are all zero due to closed-loop correction.
    ns = LC_KF_config.nstates
    x_est_propagated = np.zeros((ns, 1))
           
    # Set-up measurement matrix using (14.115)
    H = np.zeros((6, ns))
    H[0:3, 6:9] = -np.eye(3)  # position meas to position states 
    H[3:6, 3:6] = -np.eye(3)  # velocity meas to velocity states
    
    # 6. Set-up measurement noise covariance matrix assuming all components of
    # GNSS position and velocity are independent and have equal variance.
    R = np.zeros((6, 6))
    # Use same value for X,Y, and Z
    R[0:3, 0:3] = np.eye(3) * (LC_KF_config.pos_meas_SD**2)
    R[3:6, 3:6] = np.eye(3) * (LC_KF_config.vel_meas_SD**2)

    # Calculate Kalman gain using (3.21)
    K = P_propagated @ H.T @ inv(H @ P_propagated @ H.T + R)

    # Transform vel/pos estimates from IMU frame to GNSS frame using lever arm
    est_v_gnss_e, est_r_gnss_e = Lever_Arm(est_C_b_e, est_v_eb_e, est_r_eb_e, meas_omega_ib_b,
                                           LC_KF_config.lever_arm_b)

    # Formulate measurement innovations using (14.102)
    delta_z = np.zeros((6, 1))
    delta_z[0:3, 0] = GNSS_r_eb_e - est_r_gnss_e
    delta_z[3:6, 0] = GNSS_v_eb_e - est_v_gnss_e
    
    # Update state estimates using (3.24)
    x_est_new = x_est_propagated + K @ delta_z
    
    # 10. Update state estimation error covariance matrix using (3.25)
    P_new = (np.eye(ns) - K @ H) @ P_propagated
    
    # CLOSED-LOOP CORRECTION

    # Correct attitude, velocity, and position using (14.7-9)
    est_C_b_e_new = (np.eye(3) - Skew_symmetric(x_est_new[0:3].flatten())) @ est_C_b_e
    est_v_eb_e_new = est_v_eb_e - x_est_new[3:6].flatten()
    est_r_eb_e_new = est_r_eb_e - x_est_new[6:9].flatten()
    
    # Update IMU bias estimates
    est_IMU_bias_new = est_IMU_bias + x_est_new[9:ns].flatten()

    return(est_C_b_e_new, est_v_eb_e_new, est_r_eb_e_new, est_IMU_bias_new, P_new)

def LC_KF_ZUPT_Update(est_C_b_e, est_v_eb_e, meas_omega_ib_b, 
                      est_IMU_bias, P_propagated, LC_KF_config, run_dir):

    # Propagated state estimates are all zero due to closed-loop correction.
    ns = LC_KF_config.nstates
    x_est = np.zeros((ns, 1))
    
    H = np.zeros((6, ns))    
    H[0:3, 3:6] = -np.eye(3)     # Velocity meas (0) to velocity states
    H[0:3, 9:12] = -np.eye(3)    # Velocity meas (0) to accel bias states
    H[3:6, 12:15] = -np.eye(3)    # Gyro bias meas to gyro bias states
    
    # Measurement noise covariance R
    R = np.zeros((6, 6))
    R[0:3, 0:3] = np.eye(3) * LC_KF_config.zupt_vel_var
    R[3:6, 3:6] = np.eye(3) * LC_KF_config.zaru_gyro_var
    
    # Kalman gain
    K = P_propagated @ H.T @ inv(H @ P_propagated @ H.T + R)
    
    # Measurement innovation (ZUPT: velocity should be zero, ZARU: angular rate should be zero)
    delta_z = np.zeros((6, 1))
    delta_z[0:3, 0] = 0 - est_v_eb_e.flatten() * run_dir  # ZUPT: 0 - v_est
    delta_z[3:6, 0] = meas_omega_ib_b.flatten() * run_dir # ZARU: 0 - (omega_measured - bias_est)
    
    # State update
    x_est_new = x_est + K @ delta_z
    
    # Covariance update
    # Simple form
    #P_new = (np.eye(ns) - K @ H) @ P_propagated
    # More stable form
    P_new = (np.eye(ns) - K @ H) @ P_propagated @ (np.eye(ns) - K @ H).T + K @ R @ K.T
    
    # Closed-loop correction
    est_C_b_e_new = (np.eye(3) - Skew_symmetric(x_est_new[0:3].flatten())) @ est_C_b_e 
    est_v_eb_e_new = est_v_eb_e - x_est_new[3:6].flatten()
    est_IMU_bias_new = est_IMU_bias.copy()
    est_IMU_bias_new[:6] -= x_est_new[9:15].flatten()
    
    return(est_C_b_e_new, est_v_eb_e_new, est_IMU_bias_new, P_new)

def Align_Yaw(est_C_b_e, est_C_b_n, GNSS_v_eb_n, P, run_dir):
    
    # Calculate yaw error
    yaw_est_n = np.arctan2(est_C_b_n[1,0], est_C_b_n[0,0])
    hdg_meas_n = np.arctan2(GNSS_v_eb_n[1], GNSS_v_eb_n[0])
    #delta_yaw_n = (hdg_meas - yaw_est  + np.pi) % (2 * np.pi) - np.pi
    delta_yaw_n = np.arctan2(np.sin(hdg_meas_n - yaw_est_n), np.cos(hdg_meas_n - yaw_est_n))
    print(' GNSS hdg=%.2f, yaw_est=%.2f,  Delta yaw=%.2f deg' % (np.rad2deg(hdg_meas_n), np.rad2deg(yaw_est_n), np.rad2deg(delta_yaw_n)))
    R_n = Euler_to_CTM([0, 0, -delta_yaw_n])
    
    est_C_n_e = ortho_C(ortho_C(est_C_b_e) @ ortho_C(est_C_b_n.T))
    R_e = est_C_n_e @ R_n @ est_C_n_e.T
    C_b_e_new = ortho_C(R_e @ est_C_b_e)
    
    # Covariance reset: rotate only the attitude 3x3 (ECEF basis), rest identity
    G = np.eye(P.shape[0])
    G[0:3, 0:3] = R_e                 # attitude small-angles are in ECEF
    P_new = G @ P @ G.T
    
    return(C_b_e_new, P_new)

# Non-Holonomic Constraint (NHC) Measurement Update
def LC_KF_NHC_Update(est_C_b_e, est_v_eb_e, meas_omega_ib_b, lever_arm_b,est_IMU_bias, P, LC_KF_config, coast):
    
    # Propagated state estimates are all zero due to closed-loop correction.
    ns = LC_KF_config.nstates
    x_est = np.zeros((ns, 1))

    # Measurement matrix H_nhc: maps error state to v_b_y, v_b_z
    H = np.zeros((2, 15))
    H[:, 3:6] = -est_C_b_e.T[1:3, :]  # Project ECEF velocity to body Y/Z
    
    # Measurement noise
    if coast:
        R = np.eye(2) * LC_KF_config.nhc_vel_var_coast
    else:
        R = np.eye(2) * LC_KF_config.nhc_vel_var
    
    # Kalman update
    K = P @ H.T @ inv(H @ P @ H.T + R)

    # Transform velocity to body frame
    v_b = est_C_b_e.T @ est_v_eb_e + np.cross(meas_omega_ib_b, np.array(lever_arm_b))
    
    delta_z = 0 - v_b[1:3].reshape((2, 1))  # meas - estimated

    # State update
    x_est += K @ delta_z
    
    # Covariance update
    P_new = (np.eye(15) - K @ H) @ P

    # Apply closed-loop correction
    est_C_b_e_new = (np.eye(3) - Skew_symmetric(x_est[0:3].flatten())) @ est_C_b_e
    est_v_eb_e_new = est_v_eb_e - x_est[3:6].flatten()
    est_IMU_bias_new = est_IMU_bias.copy()
    est_IMU_bias_new[:6] += x_est[9:15].flatten()
    #est_IMU_bias_new[5] += x_est[14]
    return(est_C_b_e_new, est_v_eb_e_new, est_IMU_bias_new, P_new)

def Velocity_Match(outp, coast_start, coast_end):
    t = outp[:,0]
    pos = outp[:,1:4]
    vel = outp[:,4:7] 
    pos_std = np.zeros_like(pos)
    vel_std = np.zeros_like(vel)
    
    s = coast_start
    e = coast_end
    n = abs(e-s)  # number of points
    dt = (t[e] - t[s]) / n
    
    # adjust for velocity step
    vel_step = (vel[e-1] - vel[e])
    dvel = np.linspace(0, vel_step, n)
    dpos = np.cumsum(dvel * dt, axis=0)
    vel[s:e] -= dvel
    pos[s:e] -= dpos
    pos_std[s:e] = np.abs(dpos)**2
    vel_std[s:e] = np.abs(dvel)**2
    
    # adjust for residual position step
    pos_step = (pos[e-1] - pos[e])
    dpos = np.linspace(0, pos_step, n)
    pos[s:e] -= dpos
    
    return(outp)  # values modified by pointers


def Combine_Passes(profiles):
    outp = np.zeros_like(profiles[:,:,0])
    outp[:,0] = profiles[:,0,0]
    outp[:,1:4] = np.average(profiles[:,1:4,:], weights=1/(0.1+profiles[:,11:14,:]), axis=2)
    outp[:,4:7] = np.average(profiles[:,4:7,:], weights=1/(0.1+profiles[:,14:17,:]), axis=2)
    outp[:,7:10] = np.average(profiles[:,7:10,:], weights=1/(0.1+profiles[:,11:14,:]), axis=2)
    outp[:,10] = profiles[:,10,0]
    outp[:,11:14] = np.average(profiles[:,10:13,:], weights=1/(0.1+profiles[:,11:14,:]), axis=2)
    outp[:,14:17] = np.average(profiles[:,13:16,:], weights=1/(0.1+profiles[:,14:17,:]), axis=2)
    return(outp)
    



def Zero_Phase_LP(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    norm_cutoff = cutoff / nyquist
    b, a = butter(order, norm_cutoff, btype='low', analog=False)
    filtered = filtfilt(b, a, data, axis=0)  # forward-backward filter
    return filtered

def Compute_Instantaneous_Vel(positions, dt):
    """
    Computes instantaneous velocities from position data using central differences.
    
    Parameters:
    positions : ndarray
        Array of shape (N, 3) containing position vectors over time.
    dt : float
        Time step between consecutive positions in seconds.
    
    Returns:
    velocities : ndarray
        Array of shape (N, 3) containing velocity vectors.
        First and last points use forward and backward difference respectively.
    """
    positions = np.asarray(positions)
    velocities = np.zeros_like(positions)

    # Central difference for interior points
    velocities[1:-1] = (positions[2:] - positions[:-2]) / (2 * dt)

    # Forward and backward difference for boundaries
    velocities[0] = (positions[1] - positions[0]) / dt
    velocities[-1] = (positions[-1] - positions[-2]) / dt

    return velocities

# Reverse data for backwards run
def Reverse_Data_Dir(in_gnss, in_imu, outp):
    in_gnss_rev = in_gnss[::-1,:].copy()
    in_imu_rev = in_imu[::-1,:].copy()
    outp_rev = outp[::-1,:].copy()
    return in_gnss_rev, in_imu_rev, outp_rev


    