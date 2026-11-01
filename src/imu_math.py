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
from scipy.signal import butter, filtfilt, lfilter
from numpy.linalg import norm, inv
from imu_transforms import Euler_to_CTM

# Physical constants
omega_ie = 7.292115E-5 # Earth rotation rate (rad/s)
ug_to_mps_squared = 9.80665 * 1E-6  # default gravity, micro g -> meters/sec^2
R_0 = 6378137           # WGS84 Equatorial radius in meters
e_ecc = 0.0818191908425 # earth eccentricity
mu = 3.986004418E14     # WGS84 Earth gravitational constant (m^3 s^-2)
J_2 = 1.082627E-3       # WGS84 Earth second gravitational constant
FIX, FLOAT, DGPS, SINGLE = 1,2,4,5   # GNSS solution states

# Paul Groves error-state convention:
# [delta_theta, v_hat - v, r_hat - r, b_a - b_a_hat, b_g - b_g_hat].
# Navigation errors are subtracted; truth-minus-estimate bias errors are added.

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

    # Add centrifugal acceleration using (2.133)
    g[0:2] += omega_ie**2 * r_eb_e[0:2]
    return g

def Init_P_matrix(LC_KF_config, C_e_n, pos_unc, vel_unc):
    #Init_P_matrix - Initializes the loosely coupled INS/GNSS KF
    #error covariance matrix
    #
    # Inputs:
    #   LC_KF_config
    #     .init.att_unc           Initial attitude uncertainty per axis (rad)
    #     .init.bias_acc_unc      Initial accel. bias uncertainty (m/s^2)
    #     .init.bias_gyro_unc     Initial gyro. bias uncertainty (rad/s)
    #     .init.scale_acc_unc     Initial accelerometer-scale uncertainty (fraction)
    #     .init.scale_gyro_unc    Initial gyro-scale uncertainty (fraction)
    #     .nstates               Number of error states (15 or 21)
    #   C_e_n                     ECEF to NED CTM
    #   vel_unc                   Initial velocity uncertainty per axis (m/s)
    #   pos_unc                   Initial position uncertainty per axis (m)
    #
    # Outputs:
    #   P                     state estimation error covariance matrix
    
    # Initialize error covariance matrix
    ns = LC_KF_config.nstates
    P = np.zeros((ns, ns))
    
    P[0:3, 0:3] = C_e_n.T @ (np.eye(3) * np.array(LC_KF_config.init.att_unc)**2) @ C_e_n
    P[3:6, 3:6] = C_e_n.T @ (np.eye(3) * np.array(vel_unc)**2) @ C_e_n
    P[6:9, 6:9] = C_e_n.T @ (np.eye(3) * np.array(pos_unc)**2) @ C_e_n
    P[9:12, 9:12] = np.eye(3) * np.array(LC_KF_config.init.bias_acc_unc)**2
    P[12:15, 12:15] = np.eye(3) *np.array(LC_KF_config.init.bias_gyro_unc)**2
    if ns == 21:  # Scale factors
        P[15:18, 15:18] = np.eye(3) * np.array(LC_KF_config.init.scale_acc_unc)**2
        P[18:21, 18:21] = np.eye(3) * np.array(LC_KF_config.init.scale_gyro_unc)**2
    return P

def Init_Qc_matrix_PSD(cfg, ns):
    # Build continuous-time kalman process noise variances matrix (14.82)
    attitude_noise_var = (np.deg2rad(cfg.gyro_noise_density) * cfg.imu_noise_factors[0])**2
    vel_noise_var = (cfg.accel_noise_density * ug_to_mps_squared * cfg.imu_noise_factors[1])**2
    accel_bias_noise_var = (cfg.accel_bias_density * ug_to_mps_squared * cfg.imu_noise_factors[2])**2
    gyro_bias_noise_var =  (np.deg2rad(cfg.gyro_bias_density) * cfg.imu_noise_factors[3])**2
    if cfg.scale_factors:
        accel_scale_noise_var = (cfg.accel_scale_noise_density * cfg.imu_noise_factors[4])**2
        gyro_scale_noise_var = (cfg.gyro_scale_noise_density * cfg.imu_noise_factors[5])**2

    Qc = np.zeros((ns, ns))
    Qc[0:3, 0:3] = np.eye(3) * attitude_noise_var
    Qc[3:6, 3:6] = np.eye(3) * vel_noise_var
    Qc[9:12, 9:12] = np.eye(3) * accel_bias_noise_var
    Qc[12:15, 12:15] = np.eye(3) * gyro_bias_noise_var
    if cfg.scale_factors:
        Qc[15:18, 15:18] = np.eye(3) * accel_scale_noise_var
        Qc[18:21, 18:21] = np.eye(3) * gyro_scale_noise_var

    return Qc

def Init_Qc_matrix_random_walk(cfg, ns):
    # Build continuous-time kalman process noise variances matrix (Qc)

    tau = cfg.corr_time * 3600.0 # Correlation time tau in seconds
    gyro_arw = cfg.imu_noise_factors[0] * np.deg2rad(cfg.gyro_arw) / np.sqrt(3600.0)  # deg/sqrt(h) -> rad/sqrt(s)
    acc_vrw = cfg.imu_noise_factors[1] * cfg.acc_vrw / np.sqrt(3600.0) # m/s/sqrt(h) -> m/s/sqrt(s)
    acc_bias_std = cfg.imu_noise_factors[2] * cfg.acc_bias_std * 1e-5 # mGal -> m/s^2,  1 Gal = 0.01 m/s^2
    gyro_bias_std = cfg.imu_noise_factors[3] * np.deg2rad(cfg.gyro_bias_std) / 3600.0 # deg/h -> rad/s
    if cfg.scale_factors:
        accel_scale_noise_var = (cfg.accel_scale_noise_density * cfg.imu_noise_factors[4])**2
        gyro_scale_noise_var = (cfg.gyro_scale_noise_density * cfg.imu_noise_factors[5])**2

    Qc = np.zeros((ns, ns))
    Qc[0:3, 0:3] = (gyro_arw ** 2) * np.eye(3) # White gyro noise driving attitude error (dtheta): gyrarw^2
    Qc[3:6, 3:6] = (acc_vrw ** 2) * np.eye(3) # White accel noise driving velocity error (dv): accvrw^2
    Qc[9:12, 9:12] = (2.0 * acc_bias_std ** 2 / tau) * np.eye(3) # Gauss–Markov accel bias: 2*s^2 / tau
    Qc[12:15, 12:15] = (2.0 * gyro_bias_std ** 2 / tau) * np.eye(3) # Gauss–Markov gyro bias: 2*s^2 / tau
    if ns == 21:
        Qc[15:18, 15:18] = accel_scale_noise_var * np.eye(3)
        Qc[18:21, 18:21] = gyro_scale_noise_var * np.eye(3)
    
    return Qc

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

def Cross_product(a, b):
    # Fixed-size 3-D cross product without np.cross's generalized-array overhead.
    return np.array([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
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
    
    temp = 1 - (e_ecc * np.sin(L))**2
    R_N = R_0 * (1 - e_ecc**2) / temp**1.5
    R_E = R_0 / np.sqrt(temp)
    return R_N, R_E

def Lever_Arm(C_b2f, v_ref_f, r_ref_f, omega_ib_b, r_lever_arm_b):
    r_lever_arm_f = C_b2f @ r_lever_arm_b
    v_lever_arm_f = C_b2f @ Cross_product(omega_ib_b, r_lever_arm_b)
    r_point_f = r_ref_f + r_lever_arm_f
    v_point_f = v_ref_f + v_lever_arm_f
    return(v_point_f, r_point_f)

def ortho_C(C: np.ndarray) -> np.ndarray:
    # Check if CTM is ortho-normalized and if not, ortho-normalize it
    diag_tol = 1e-5
    offdiag_tol = 1e-8
    CtC = C.T @ C
    ortho_norm = (
        abs(CtC[0, 0] - 1.0) <= diag_tol and
        abs(CtC[1, 1] - 1.0) <= diag_tol and
        abs(CtC[2, 2] - 1.0) <= diag_tol and
        abs(CtC[0, 1]) <= offdiag_tol and
        abs(CtC[0, 2]) <= offdiag_tol and
        abs(CtC[1, 0]) <= offdiag_tol and
        abs(CtC[1, 2]) <= offdiag_tol and
        abs(CtC[2, 0]) <= offdiag_tol and
        abs(CtC[2, 1]) <= offdiag_tol
    )
    if ortho_norm:
        det_C = (
            C[0, 0] * (C[1, 1] * C[2, 2] - C[1, 2] * C[2, 1])
            - C[0, 1] * (C[1, 0] * C[2, 2] - C[1, 2] * C[2, 0])
            + C[0, 2] * (C[1, 0] * C[2, 1] - C[1, 1] * C[2, 0])
        )
        ortho_norm = abs(det_C - 1.0) <= diag_tol
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
    #   f_ib_b        specific force measured in the body frame, resolved
    #                 along body-frame axes, averaged over time interval (m/s^2)
    #   omega_ib_b    angular rate of body frame w.r.t. inertial frame, resolved
    #                 about body-frame axes, averaged over time interval (rad/s)
    # Outputs:
    #   r_eb_e        Cartesian position of body frame w.r.t. ECEF frame, resolved
    #                 along ECEF-frame axes (m)
    #   v_eb_e        velocity of body frame w.r.t. ECEF frame, resolved along
    #                 ECEF-frame axes (m/s)
    #   C_b_e         body-to-ECEF-frame coordinate transformation matrix
    
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
                - 2 * Cross_product([0, 0, omega_ie], old_v_eb_e))

    # UPDATE CARTESIAN POSITION From (5.38)
    r_eb_e = old_r_eb_e + 0.5 * tor_i * (v_eb_e + old_v_eb_e)

    return r_eb_e, v_eb_e, C_b_e

def LC_KF_Predict(tor_s, est_C_b_e, est_v_eb_e, est_r_eb_e, est_IMU_bias, P, Qc,
                meas_f_ib_b, meas_omega_ib_b, LC_KF_config, gravity, geocentric_radius):
    #LC_KF_Predict - Propagates the loosely coupled INS/GNSS
    # Kalman filter error covariance; navigation states are propagated separately
    #
    # Inputs:
    #   tor_s                 propagation interval (s)
    #   est_C_b_e             prior estimated body to ECEF coordinate
    #                         transformation matrix
    #   est_v_eb_e            prior estimated ECEF user velocity (m/s)
    #   est_r_eb_e            prior estimated ECEF user position (m)
    #   est_IMU_bias          prior estimated IMU biases (body axes)
    #   P                     previous Kalman filter error covariance matrix
    #   Qc                    continuous-time process-noise covariance matrix
    #   meas_f_ib_b           corrected specific force in body axes (m/s^2)
    #   meas_omega_ib_b       corrected angular rate in body axes (rad/s)
    #   LC_KF_config
    #     .nstates            number of error states (15 or 21)
    #   gravity               ECEF gravity vector (m/s^2)
    #   geocentric_radius     geocentric radius used in the gravity gradient (m)
    #
    # Outputs:
    #   P_propagated      propagated Kalman filter error covariance matrix
    
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

    #2. Discretize: Q' = Qc * dt
    Q_prime = Qc * abs_tor_s

    # 3. Propagate state estimates using (3.14) noting that all states are zero 
    # due to closed-loop correction.
    # x_est_propagated = np.zeros((ns, 1))   # implied
    
    # 4. Propagate state estimation error covariance matrix using (3.46)
    P_propagated = Phi @ (P + 0.5 * Q_prime) @ Phi.T + 0.5 * Q_prime
    P_propagated = 0.5 * (P_propagated + P_propagated.T) # enforce symmetry
    return(P_propagated)

def LC_KF_GNSS_Update(GNSS_r_eb_e, GNSS_v_eb_e, pos_meas_SD, vel_meas_SD,
                est_C_b_e, C_n_e, est_v_eb_e, est_r_eb_e, est_IMU_bias,
                P_propagated, meas_omega_ib_b, LC_KF_config):

    # Propagated state estimates are all zero due to closed-loop correction.
    ns = LC_KF_config.nstates
    x_est_propagated = np.zeros((ns, 1))
           
    # Set-up measurement matrix using (14.115)
    H = np.zeros((6, ns))
    r_lever_arm_b = LC_KF_config.r_lever_arm_b
    H[0:3, 0:3] = Skew_symmetric(est_C_b_e @ r_lever_arm_b)
    H[0:3, 6:9] = -np.eye(3)  # position meas to position states 
    v_lever_arm_e = est_C_b_e @ Cross_product(meas_omega_ib_b, r_lever_arm_b)
    H[3:6, 0:3] = Skew_symmetric(v_lever_arm_e)
    H[3:6, 3:6] = -np.eye(3)  # velocity meas to velocity states
    if ns == 21:
        # Gyro bias is removed before the scale-factor correction, so its
        # lever-arm velocity derivative includes the scale correction gain.
        gyro_scale = est_IMU_bias[9:12]
        scale_gain = 1.0 - gyro_scale
        omega_pre_scale = meas_omega_ib_b / scale_gain
        H[3:6, 12:15] = (est_C_b_e @ Skew_symmetric(r_lever_arm_b)
                         @ np.diag(scale_gain))
        # Gyro scale-factor errors change antenna velocity through omega x r.
        H[3:6, 18:21] = (est_C_b_e @ Skew_symmetric(r_lever_arm_b)
                         @ np.diag(omega_pre_scale))
    else:
        H[3:6, 12:15] = est_C_b_e @ Skew_symmetric(r_lever_arm_b)
    
    # 6. Set-up measurement noise covariance matrix and rotate from NED to ECEF
    R = np.zeros((6, 6))
    R[0:3, 0:3] = C_n_e @ (np.eye(3) * np.array(pos_meas_SD)**2) @ C_n_e.T
    R[3:6, 3:6] = C_n_e @ (np.eye(3) * np.array(vel_meas_SD)**2) @ C_n_e.T

    # A large velocity SD marks an unavailable velocity measurement. Remove
    # its rows entirely so the update uses GNSS position only and can never
    # pull the navigation velocity toward a placeholder value.
    use_velocity = norm(vel_meas_SD) <= LC_KF_config.gnss_vel_err_thresh
    if not use_velocity:
        H = H[0:3, :]
        R = R[0:3, 0:3]

    # Calculate Kalman gain using (3.21)
    K = P_propagated @ H.T @ inv(H @ P_propagated @ H.T + R)

    # Transform vel/pos estimates from IMU frame to GNSS frame using lever arm
    est_v_gnss_e, est_r_gnss_e = Lever_Arm(est_C_b_e, est_v_eb_e, est_r_eb_e, meas_omega_ib_b,
                                           LC_KF_config.r_lever_arm_b)

    # Formulate measurement innovations using (14.102)
    delta_z = np.zeros(6)
    delta_z[0:3] = GNSS_r_eb_e - est_r_gnss_e
    delta_z[3:6] = GNSS_v_eb_e - est_v_gnss_e
    if not use_velocity:
        delta_z = delta_z[0:3]
    
    # Update state estimates using (3.24)
    x_est_new = x_est_propagated + K @ delta_z[:, None]
    
    # 10. Update state estimation error covariance matrix using (3.25)
    I = np.eye(ns)
    #P_new = (I - K @ H) @ P_propagated # simple form
    P_new = (I - K @ H) @ P_propagated @ (I - K @ H).T + K @ R @ K.T # robust form
    P_new = 0.5 * (P_new + P_new.T) # enforce symmetry
    
    # CLOSED-LOOP CORRECTION

    # Correct attitude, velocity, and position using (14.7-9)
    est_C_b_e_new = (np.eye(3) - Skew_symmetric(x_est_new[0:3].flatten())) @ est_C_b_e
    est_v_eb_e_new = est_v_eb_e - x_est_new[3:6].flatten()
    est_r_eb_e_new = est_r_eb_e - x_est_new[6:9].flatten()
    
    # Groves convention: IMU calibration errors are truth-minus-estimate.
    est_IMU_bias_new = est_IMU_bias.copy()
    est_IMU_bias_new[:ns-9] += x_est_new[9:ns].flatten()
    return(est_C_b_e_new, est_v_eb_e_new, est_r_eb_e_new, est_IMU_bias_new, P_new)

def LC_KF_ZUPT_Update(est_C_b_e, est_v_eb_e, meas_omega_ib_b, meas_f_ib_b,
                      est_IMU_bias, P_propagated, LC_KF_config, gravity_e, run_dir):

    # Propagated state estimates are all zero due to closed-loop correction.
    ns = LC_KF_config.nstates
    x_est = np.zeros((ns, 1))
    f_pred_b = -est_C_b_e.T @ gravity_e
    
    H = np.zeros((9, ns))    
    H[0:3, 3:6] = -np.eye(3)     # Velocity meas (0) to velocity states
    # The IMU inputs are already bias-corrected. With Groves bias errors
    # (truth minus estimate), both stationary residual derivatives are +I.
    H[3:6, 12:15] = np.eye(3)    # Gyro bias meas to gyro bias states
    H[6:9, 0:3] = est_C_b_e.T @ Skew_symmetric(gravity_e) # gravity -> accel
    H[6:9, 9:12] = np.eye(3)    # Specific force directly observes accel bias
    if ns == 21:
        # Corrected IMU measurements vary as meas * delta_scale for the
        # truth-minus-estimate multiplicative scale-factor error.
        H[3:6, 18:21] = np.diag(meas_omega_ib_b)
        H[6:9, 15:18] = np.diag(meas_f_ib_b)

    # Measurement noise covariance R
    R = np.zeros((9, 9))
    R[0:3, 0:3] = np.eye(3) * LC_KF_config.zupt_vel_var
    R[3:6, 3:6] = np.eye(3) * LC_KF_config.zaru_gyro_var
    R[6:9, 6:9] = np.eye(3) * LC_KF_config.zupt_accel_var
    
    # Kalman gain
    K = P_propagated @ H.T @ inv(H @ P_propagated @ H.T + R)
    
    # Measurement innovation (ZUPT: zero velocity; ZARU: angular rate equals Earth rate)
    delta_z = np.zeros((9, 1))
    delta_z[0:3, 0] = 0 - est_v_eb_e.flatten() * run_dir  # ZUPT: 0 - v_est
    delta_z[6:9, 0] = (meas_f_ib_b - f_pred_b).flatten() * run_dir
    # ZARU: omega_meas - omega_earth_body
    delta_z[3:6, 0] = (meas_omega_ib_b.flatten() - est_C_b_e.T @  [0, 0, omega_ie] ) * run_dir

    # State update
    x_est_new = x_est + K @ delta_z
    
    # Covariance update
    I = np.eye(ns)
    #P_new = (I - K @ H) @ P_propagated # Simple form
    P_new = (I - K @ H) @ P_propagated @ (I - K @ H).T + K @ R @ K.T # More stable form
    P_new = 0.5 * (P_new + P_new.T) # enforce symmetry
    
    # Closed-loop correction
    est_C_b_e_new = (np.eye(3) - Skew_symmetric(x_est_new[0:3].flatten())) @ est_C_b_e 
    est_v_eb_e_new = est_v_eb_e - x_est_new[3:6].flatten()
    est_IMU_bias_new = est_IMU_bias.copy()
    est_IMU_bias_new[:ns-9] += x_est_new[9:ns].flatten()
    
    return(est_C_b_e_new, est_v_eb_e_new, est_IMU_bias_new, P_new)

def Align_Yaw(est_C_b_e, C_n_e, GNSS_v_eb_n, P, run_dir):
    
    # Calculate yaw error in NED
    est_C_b_n = C_n_e.T @ est_C_b_e # body -> NED
    yaw_est_n = np.arctan2(est_C_b_n[1,0], est_C_b_n[0,0])
    hdg_meas_n = np.arctan2(GNSS_v_eb_n[1], GNSS_v_eb_n[0])
    #delta_yaw_n = (hdg_meas - yaw_est  + np.pi) % (2 * np.pi) - np.pi # Simple
    delta_yaw_n = np.arctan2(np.sin(hdg_meas_n - yaw_est_n), np.cos(hdg_meas_n - yaw_est_n)) # Robust
    print(' GNSS hdg=%.2f, yaw_est=%.2f,  Delta yaw=%.2f deg' % (np.rad2deg(hdg_meas_n), np.rad2deg(yaw_est_n), np.rad2deg(delta_yaw_n)))
    
    # Rotate from NED to ECEF
    R_n = Euler_to_CTM([0, 0, -delta_yaw_n])
    R_e = C_n_e @ R_n @ C_n_e.T
    C_b_e_new = ortho_C(R_e @ est_C_b_e)

    # Also rotate attitude components of P matrix
    P[0:3,0:3] = R_e @ P[0:3,0:3]  @ R_e.T
    P[0:3, 3:] = R_e @ P[0:3,3:]
    P[3:,0:3] = P[3:,0:3] @ R_e.T
    P = 0.5 * (P + P.T) # enforce symmetry
   
    return(C_b_e_new, P)

# Non-Holonomic Constraint (NHC) Measurement Update
def LC_KF_NHC_Update(est_C_b_e, est_v_eb_e, meas_omega_ib_b, r_lever_arm_v,est_IMU_bias, P, LC_KF_config):
    
    # Propagated state estimates are all zero due to closed-loop correction.
    ns = LC_KF_config.nstates
    x_est = np.zeros((ns, 1))
    
    # Form rear-axle velocity in the unrotated propagation body frame, then
    # rotate it into the vehicle frame where the NHC is applied.
    C_b_to_v = LC_KF_config.C_b_to_nhc
    r_lever_arm_b = C_b_to_v.T @ r_lever_arm_v
    v_b = (est_C_b_e.T @ est_v_eb_e +
           Cross_product(meas_omega_ib_b, r_lever_arm_b))
    v_v = C_b_to_v @ v_b

    # Build the complete Jacobian in the propagation frame, rotate it into
    # the vehicle frame, then select lateral and vertical velocity rows.
    H_b = np.zeros((3, ns))
    # Attitude error is left-multiplicative and ECEF-resolved.
    H_b[:, 0:3] = -est_C_b_e.T @ Skew_symmetric(est_v_eb_e)
    H_b[:, 3:6] = -est_C_b_e.T
    H_b[:, 12:15] = Skew_symmetric(r_lever_arm_b)
    if ns == 21:
        H_b[:, 18:21] = (Skew_symmetric(r_lever_arm_b)
                         @ np.diag(meas_omega_ib_b))
    H = (C_b_to_v @ H_b)[1:3, :]
    
    # Measurement noise
    R = np.eye(2) * LC_KF_config.nhc_vel_var
    
    # Kalman update
    K = P @ H.T @ inv(H @ P @ H.T + R)

    delta_z = -v_v[1:3].reshape((2, 1))  # zero minus estimated velocity

    # State update
    x_est += K @ delta_z
    
    # Covariance update
    I = np.eye(ns)
    #P_new = (I - K @ H) @ P # simple form
    P_new = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T # robust form
    P_new = 0.5 * (P_new + P_new.T) # enforce symmetry

    # Apply closed-loop correction
    est_C_b_e_new = (np.eye(3) - Skew_symmetric(x_est[0:3].flatten())) @ est_C_b_e
    est_v_eb_e_new = est_v_eb_e - x_est[3:6].flatten()
    est_IMU_bias_new = est_IMU_bias.copy()
    est_IMU_bias_new[:ns-9] += x_est[9:ns].flatten()

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

def GNSS_Noise_Gain(fix_status, cfg):
    # Fix-quality gain applied to GNSS position SD; velocity SD is unchanged.
    if fix_status == FIX:
        return 1
    elif fix_status == FLOAT:
        return cfg.float_err_gain
    elif fix_status == DGPS:
        return cfg.float_err_gain
    elif fix_status == SINGLE:
        return cfg.single_err_gain
    else:
        return cfg.single_err_gain


def Gate_GNSS_Measurements(fix_status, pos_meas_SD, vel_meas_SD, P, cfg):
    adj_pos_meas_SD = pos_meas_SD.copy()
    adj_vel_meas_SD = vel_meas_SD.copy()
    pos_est_SD_3D = norm(np.sqrt(np.diag(P[6:9, 6:9])))
    noise_gain = GNSS_Noise_Gain(fix_status, cfg)

    # Ignore unused GNSS solution states by setting SDs to a large value
    if noise_gain == 0:
        adj_pos_meas_SD[:] = adj_vel_meas_SD[:] = 1e6
        return False, adj_pos_meas_SD, adj_vel_meas_SD

    # Fix status affects position accuracy much more than velocity accuracy.
    # Keep the receiver-reported velocity SD independent of this gain.
    adj_pos_meas_SD *= noise_gain

    # Ignore measurements with excessive uncertainty.
    pos_meas_SD_3D = norm(adj_pos_meas_SD)
    if (pos_meas_SD_3D > cfg.gnss_pos_err_thresh or
            pos_meas_SD_3D * cfg.gnss_SD_multiplier > pos_est_SD_3D):
        return False, adj_pos_meas_SD, adj_vel_meas_SD

    return True, adj_pos_meas_SD, adj_vel_meas_SD


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

def LP_Filt(data, cutoff, fs, order=4, causal=False):
    nyquist = 0.5 * fs
    if cutoff==0:
        return data
    if cutoff >= nyquist:
        print('Warning: Data not filtered, cutoff freq > nyquist')
        return data
    norm_cutoff = cutoff / nyquist
    b, a = butter(order, norm_cutoff, btype='low', analog=False)
    if causal==True:
        filtered = lfilter(b, a, data, axis=0)  # causal filter
    else:
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
