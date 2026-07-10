#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 Project:    Sensor Fusion for GNSS/IMU (Loosely Coupled Kalman Filter)
 File:       GNSS_IMU.py
 Author:     Tim Everett
 Copyright:  (c) 2025 Tim Everett
 License:    BSD 3 License (see LICENSE file for details)

 Description:
    Main entry point for the sensor fusion pipeline. Implements a loosely
    coupled GNSS/IMU Kalman filter with support for:
      - Prediction and update steps
      - Lever arm corrections
      - Frame conversions (ECEF, NED, body)
      - Plotting and logging of output

Based on matlab code and textbook by Paul Groves, "Principles of GNSS, Inertial, 
          and Multisensor Integrated Navigation Systems"
===============================================================================
"""

import numpy as np
from numpy.linalg import norm
from os.path import join
from imu_math import (Init_P_matrix, Init_Qc_matrix_PSD, Init_Qc_matrix_random_walk, 
                      Nav_equations_ECEF, LC_KF_Predict, LC_KF_GNSS_Update, LC_KF_ZUPT_Update, 
                      Align_Yaw, LC_KF_NHC_Update, Lever_Arm, ortho_C,
                      LC_KF_Config, Velocity_Match, Combine_Passes, Gravity_ECEF, Reverse_Data_Dir)
from imu_files import Read_GNSS_data, Read_IMU_data, Write_GNSS_data
from imu_plot import Plot_Results, Plot_Biases, Plot_IMU, Plot_Uncertainties
from imu_transforms import (CTM_to_Euler, Euler_to_CTM, pvc_GNSS_to_ECEF, compute_C_e_n)

##########  Data selection and run configuration ########################################

# Uncomment these lines to run an example of walking with a handheld GNSS receiver
# import walk_config as cfg
# dataDir = r'..\data\walk_0827'
# fileIn = '1730_sf'

# Uncomment these lines to run an example of driving with a roof mounted GNSS receiver
import drive_config as cfg
dataDir = r'..\data\drive_0708'
fileIn = '1934_sf'

# Uncomment these lines to run a KF-GINS sample data set.  See convert_KF_GINS_files.py for 
# instructions
# import ADIS16465_config as cfg
# dataDir = r'..\data\KF_GINS'
# fileIn = 'ADIS16465_sf'


######### Description of inputs/outputs ########################################

# Inputs:
#   in_gnss      GNSS input measurements
#   in_imu       IMU input measurements
#   num_epochs    Number of epochs of profile data
#   initialization_errors
#     .delta_r_eb_n     position error resolved along NED (m)
#     .delta_v_eb_n     velocity error resolved along NED (m/s)
#     .delta_eul_nb_n   attitude error as NED Euler angles (rad)
#   IMU_errors
#     .delta_r_eb_n     position error resolved along NED (m)
#     .b_a              Accelerometer biases (m/s^2)
#     .b_g              Gyro biases (rad/s)
#     .M_a              Accelerometer scale factor and cross coupling errors
#     .M_g              Gyro scale factor and cross coupling errors
#     .G_g              Gyro g-dependent biases (rad-sec/m)
#     .accel_noise_root_PSD   Accelerometer noise root PSD (m s^-1.5)
#     .gyro_noise_root_PSD    Gyro noise root PSD (rad s^-0.5)
#     .accel_quant_level      Accelerometer quantization level (m/s^2)
#     .gyro_quant_level       Gyro quantization level (rad/s)
#   LC_KF_config
#     .init_att_unc           Initial attitude uncertainty per axis (deg)
#     .init_vel_unc           Initial velocity uncertainty per axis (m/s)
#     .init_pos_unc           Initial position uncertainty per axis (m)
#     .init_b_a_unc           Initial accel. bias uncertainty (m/s^2)
#     .init_b_g_unc           Initial gyro. bias uncertainty (deg/s)
#     .gyro_noise_PSD         Gyro noise PSD (deg^2/s)
#     .accel_noise_PSD        Accelerometer noise PSD (m^2 s^-3)
#     .accel_bias_PSD         Accelerometer bias random walk PSD (m^2 s^-5)
#     .gyro_bias_PSD          Gyro bias random walk PSD (deg^2 s^-3)
#     .pos_meas_SD            Position measurement noise SD per axis (m)
#     .vel_meas_SD            Velocity measurement noise SD per axis (m/s)
#
# Outputs:
#   outp               Navigation solution as a motion profile array
#   out_IMU_bias_est   Kalman filter IMU bias estimate array
#   out_KF_SD          Output Kalman filter state uncertainties
#
# Format of output profiles:
#  Column 0: time (sec)
#  Column 1: X ECEF (m)
#  Column 2: Y ECEF (m)
#  Column 3: Z ECEF (m)
#  Column 4: X ECEF velocity (m/s)
#  Column 5: Y ECEF velocity (m/s)
#  Column 6: Z ECEF velocity (m/s)
#  Column 7: roll angle of body w.r.t NED (rad)
#  Column 8: pitch angle of body w.r.t NED (rad)
#  Column 9: yaw angle of body w.r.t NED (rad)
#  Column 10: coast status (1=coast)
#  Column 11: stdev: X ECEF (m)
#  Column 12: stdev: Y ECEF (m)
#  Column 13: stdev: Z ECEF (m)
#  Column 14: stdev: X ECEF velocity (m/s)
#  Column 15: stdev: Y ECEF velocity (m/s)
#  Column 16: stdev: Z ECEF velocity (m/s)
#  Column 17: stdev: roll (rad)
#  Column 18: stdev: pitch (rad)
#  Column 19: stdev: yaw (rad)
#  Column 20: X body velocity (m/s)
#  Column 21: Y body velocity (m/s)
#  Column 22: Z body velocity (m/s)

#
# Format of output IMU biases array:
#  Column 0: time (sec)
#  Column 1: estimated X accelerometer bias (m/s^2)
#  Column 2: estimated Y accelerometer bias (m/s^2)
#  Column 3: estimated Z accelerometer bias (m/s^2)
#  Column 4: estimated X gyro bias (rad/s)
#  Column 5: estimated Y gyro bias (rad/s)
#  Column 6: estimated Z gyro bias (rad/s)
#
#
# Format of KF state uncertainties array:
#  Column 0: time (sec)
#  Column 1: X attitude error uncertainty (rad)
#  Column 2: Y attitude error uncertainty (rad)
#  Column 3: Z attitude error uncertainty (rad)
#  Column 4: X velocity error uncertainty (m/s)
#  Column 5: Y velocity error uncertainty (m/s)
#  Column 6: Z velocity error uncertainty (m/s)
#  Column 7: X position error uncertainty (m)
#  Column 8: Y position error uncertainty (m)
#  Column 9: Z position error uncertainty (m)
#  Column 10: X accelerometer bias uncertainty (m/s^2)
#  Column 11: Y accelerometer bias uncertainty (m/s^2)
#  Column 12: Z accelerometer bias uncertainty (m/s^2)
#  Column 13: X gyro bias uncertainty (rad/s)
#  Column 14: Y gyro bias uncertainty (rad/s)
#  Column 15: Z gyro bias uncertainty (rad/s)

# Based on Matlab code from Paul Groves
# License: BSD; see license.txt for details


# Constants
directions = ['','forward', 'backward']
FIX, FLOAT, SINGLE = 1,2,5   # GNSS solution states
R_0 = 6378137          # radius of WGS84
ecc = 0.0818191908425  # eccentricity of WGS84
g = 9.80665  # Default gravity (m/sec^2)

############  Start of main code ##############################

# Initialization
LC_KF_config = LC_KF_Config()
C_imu_misalign = Euler_to_CTM(np.deg2rad(np.array(cfg.imu_misalign)))
npasses = len(cfg.run_dir)
ns = LC_KF_config.nstates = 21 if cfg.scale_factors else 15
nbs = 12 if cfg.scale_factors else 6

# Build continuous-time kalman process noise variances matrix (Qc)
if cfg.imu_config == 'PSD': # use PSD IMU specs
    Qc = Init_Qc_matrix_PSD(cfg, ns)
else:  # 'random_walk'  # use random walk IMU specs
    Qc = Init_Qc_matrix_random_walk(cfg, ns)

# Calculate kalman measurement noise variances
LC_KF_config.zupt_vel_var = cfg.zupt_vel_SD**2
LC_KF_config.zupt_accel_var = cfg.zupt_accel_SD**2
LC_KF_config.zaru_gyro_var = np.deg2rad(cfg.zaru_gyro_SD)**2
LC_KF_config.nhc_vel_var = cfg.nhc_vel_SD**2
LC_KF_config.nhc_vel_var_coast = cfg.nhc_vel_SD_coast**2

# Calculate initial uncertainties
LC_KF_Config.init = cfg.init
LC_KF_Config.init.att_unc = np.deg2rad(cfg.init.att_unc)
LC_KF_Config.init.bias_gyro_unc = np.deg2rad(cfg.init.bias_gyro_unc)

# lever arm in body frame
LC_KF_Config.r_lever_arm_b = np.array(cfg.gnss_offset) - np.array(cfg.imu_offset)

# Load input measurements
in_gnss, num_gnss, ok = Read_GNSS_data(join(dataDir, 'gnss_%s.pos' % fileIn ))
in_imu, num_epochs, ok = Read_IMU_data(join(dataDir, 'imu_%s.csv' % fileIn ))

# strip start of data if start_epoch != 0
if cfg.start_epoch > 0:
    plot_t0 = in_imu[cfg.start_epoch,0] - in_imu[0,0]
    in_imu = in_imu[cfg.start_epoch:,:]
    ix = np.where(in_gnss[:,0] > in_imu[0,0])[0]
    in_gnss = in_gnss[ix[0]:,:]
    num_gnss -= ix[0]
    num_epochs -= cfg.start_epoch
else:
    plot_t0 = 0

# Add simulated errors to input measurements for eval purposes
in_imu[:,1:4] = (in_imu[:,1:4] + np.array(cfg.accel_bias_err) / g) * cfg.accel_scale_factor
in_imu[:,4:7] = (in_imu[:,4:7] + np.deg2rad(cfg.gyro_bias_err)) * cfg.gyro_scale_factor
 
dt_gnss = np.median(np.diff(in_gnss[:100,0])) # use median to find sample time
herr = np.sqrt(in_gnss[:,6]**2 + in_gnss[:,7]**2)  # combine horizontal errors
fix = in_gnss[:,4]  # GNSS fix status
stationary_count = nhc_count = 0

# Run partial solution if num_epochs not zero
if cfg.num_epochs != 0:
    num_epochs = min(num_epochs, cfg.num_epochs)  
else:
    num_epochs -= 100  # provide buffer to allow adjusting time offset
    
# Initialize output arrays
outp = np.zeros((num_epochs, 23, npasses))
out_IMU_bias_est = np.zeros((num_gnss, nbs + 1, npasses))
out_KF_SD = np.zeros((int(num_gnss / cfg.gnss_epoch_step) + 1, ns + 1, npasses))

# Setup simulated coast intervals (only used for eval)
s = in_gnss[0,0] + cfg.start_coast
e = in_gnss[-1,0] - cfg.end_coast
ix = np.where((in_imu[:num_epochs,0] > s) & (in_imu[:num_epochs,0] < e))[0]
coast = 1
for i in ix:
    if in_imu[i,0] < s + cfg.coast_len:
        outp[i,10,:] = coast
    else:
        # delay between coasts is twice length of coast
        s = s + (coast + 1) * cfg.coast_len 
        coast = 1 - coast

# loop through forward and backward passes
for p in range(npasses):
    run_dir = cfg.run_dir[p]
    print('Run direction: %s...' % directions[run_dir])
    
    # Reverse data if running backwards
    if run_dir == -1:
        in_gnss, in_imu, outp[:,:,p] = Reverse_Data_Dir(in_gnss, in_imu, outp[:,:,p])
    
    t_gnss = in_gnss[:,0]
    t_imu = in_imu[:,0]  + cfg.imu_t_off
    gnss_epoch_inc = cfg.gnss_epoch_step
        
    # Init attitude
    rpy_init = np.deg2rad(cfg.init_rpy)
    if cfg.init_yaw_with_mag:
        # Use magnetometer to initialize yaw
        rpy_init[2] = np.atan2(in_imu[0,8], in_imu[0,7])
    if p == 0:  # don't reinit for 2nd pass
        est_C_b_n = Euler_to_CTM(rpy_init).T  # body -> NED
   
    # Use initial GNSS position solution to initialize pos/vel states after adjusting for lever arm
    est_L_b, est_lambda_b, est_h_b = in_gnss[0, 1:4]    # initial LLI position
    est_v_eb_n = in_gnss[0, 14:17]  # initial NED velocity
    r_gnss_e, v_gnss_e, est_C_b_e =  pvc_GNSS_to_ECEF(est_L_b, est_lambda_b, 
            est_h_b, est_v_eb_n, est_C_b_n)
    est_v_eb_e, est_r_eb_e = Lever_Arm(est_C_b_e, v_gnss_e, r_gnss_e, [0,0,0], - LC_KF_Config.r_lever_arm_b)
    prev_time = start_time = t_imu[0]
    
    # Calculate earth radius and gravity for starting location
    geocentric_radius = (R_0 / np.sqrt(1 - (ecc * np.sin(est_L_b))**2)
                        * np.sqrt(np.cos(est_L_b)**2 + (1 - ecc**2)**2 * np.sin(est_L_b)**2))
    gravity = Gravity_ECEF(est_r_eb_e)  # calculate ECEF gravity vector for this location
    g = norm(gravity)
    
    # Initial output profile
    outp[0, 0, p] = prev_time
    outp[0, 1:4, p] = est_r_eb_e
    outp[0, 4:7, p] = est_v_eb_e
    outp[0, 7:10, p] = CTM_to_Euler(est_C_b_n.T).flatten()
    
    # Initialize Kalman filter P matrix and IMU bias states
    C_e_n = compute_C_e_n(est_L_b, est_lambda_b,) # Map to ECEF
    P = Init_P_matrix(LC_KF_config, C_e_n, in_gnss[0,6:9], in_gnss[0,17:20])
    est_IMU_bias = np.zeros(nbs)
    out_IMU_bias_est[0, 0, p] = prev_time
    out_IMU_bias_est[0, 1:nbs+1, p] = est_IMU_bias
    
    # Generate KF uncertainty record
    out_KF_SD[0, 0, p] = prev_time
    out_KF_SD[0, 1:ns+1, p] = np.sqrt(np.diag(P))
    
    # Initialize GNSS timing
    epoch_GNSS = epoch_GNSS_coast = 0  # count of GNSS epochs processed
    time_GNSS = time_GNSS_coast = t_gnss[0]
    time_next_GNSS = t_gnss[gnss_epoch_inc]
    in_gnss_ptr = 0 # ptr to GNSS input data
    
    # Init internal states
    vel_match_flag = False
    yaw_aligned = not cfg.yaw_align  # Init yaw as unaligned if yaw align option is enabled

    # Loop through IMU epocs
    for epoch in range(1, num_epochs):
        
        print('   Secs: %.3f' % (t_imu[epoch] - t_imu[0]), end='\r')
        
        # Update time and coast status
        time = t_imu[epoch]
        tor_i = time - prev_time
        if tor_i < 1e-5:
            outp[epoch,:,p] = outp[epoch-1,:,p]
            continue
        coast = outp[epoch,10,p]  # get coast status for this epoch
        
        # Get next IMU measurements and correct for biases and misalignment
        meas_f_ib_b = C_imu_misalign @ in_imu[epoch, 1:4] * g
        meas_f_ib_b -= est_IMU_bias[0:3]
        meas_omega_ib_b = C_imu_misalign @ in_imu[epoch, 4:7]
        meas_omega_ib_b -= est_IMU_bias[3:6]
        if cfg.scale_factors:
            meas_f_ib_b *= (1 - est_IMU_bias[6:9])
            meas_omega_ib_b *= (1 - est_IMU_bias[9:12])

        # Update estimated navigation solution unless in IMU disabled mode
        if time * run_dir >= t_gnss[0] * run_dir: # don't run IMU before first GNSS sample
            if not cfg.disable_imu:  # disable IMU for eval only
                est_r_eb_e, est_v_eb_e, est_C_b_e = Nav_equations_ECEF(
                    tor_i, est_r_eb_e, est_v_eb_e, est_C_b_e,
                        meas_f_ib_b, meas_omega_ib_b, gravity)
            else:
                est_r_eb_e += est_v_eb_e * tor_i  # assume constant velocity if no IMU

        # Kalman state prediction if run for both IMU and GNSS measurements
        P = LC_KF_Predict(tor_i, est_C_b_e, est_v_eb_e, est_r_eb_e, est_IMU_bias,
                       P, Qc, meas_f_ib_b, meas_omega_ib_b, LC_KF_config,
                       gravity, geocentric_radius)
    
        # Check for next GNSS measurement
        while time * run_dir >= time_next_GNSS * run_dir:
            # Get next GNSS measurement
            epoch_GNSS += 1   
            in_gnss_ptr += gnss_epoch_inc
            time_GNSS = time_next_GNSS
            GNSS_L_b, GNSS_lambda_b, GNSS_h_b = in_gnss[in_gnss_ptr, 1:4]    # LLI position
            pos_meas_SD = in_gnss[in_gnss_ptr,6:9] * cfg.gnss_noise_factors[0] # pos uncertainty
            vel_meas_SD = in_gnss[in_gnss_ptr,17:20] * cfg.gnss_noise_factors[1] #vel uncertainty
            v_gnss_n = in_gnss[in_gnss_ptr, 14:17]  # NED velocity
            r_gnss_e, v_gnss_e, _ = pvc_GNSS_to_ECEF(GNSS_L_b, GNSS_lambda_b, 
                    GNSS_h_b, v_gnss_n, [0,0,0])
            C_e_n = compute_C_e_n(GNSS_L_b, GNSS_lambda_b)
            
            # Align yaw first time velocity is large enough to calculate heading
            if cfg.yaw_align and not yaw_aligned:
                if norm(v_gnss_n[:2]) > cfg.yaw_align_min_vel: 
                    if fix[in_gnss_ptr] <= cfg.yaw_align_min_fix_state:
                        print('   %.2f sec: Yaw align: ' % (time - t_gnss[0]), end='')
                        est_C_b_e, P = Align_Yaw(est_C_b_e, C_e_n.T, v_gnss_n, P, run_dir)
                        if norm(v_gnss_n[:2]) > cfg.yaw_align_max_vel:
                            yaw_aligned = True  # disable any further yaw alignment
            
            # Process GNSS measurement if not in coast mode
            if coast == 0:
                # Adjust measurement uncertainty based on fix status
                if fix[in_gnss_ptr] == FIX or cfg.disable_imu:
                    noise_gain = 1
                elif fix[in_gnss_ptr] == FLOAT: 
                    noise_gain = cfg.float_err_gain
                else: # SINGLE: 
                    noise_gain = cfg.single_err_gain
                    
                if noise_gain != 0:  # skip measurement if noise_gain set to zero
                    pos_meas_SD *= noise_gain
                    vel_meas_SD *= noise_gain

                    # Run GNSS Measurement Update for Kalman Filter
                    if norm(pos_meas_SD) < cfg.gnss_pos_err_thresh: # skip points with large error
                        est_C_b_e, est_v_eb_e, est_r_eb_e, est_IMU_bias, P = LC_KF_GNSS_Update(
                            r_gnss_e, v_gnss_e, pos_meas_SD, vel_meas_SD, est_C_b_e, C_e_n.T, est_v_eb_e,
                            est_r_eb_e, est_IMU_bias, P, meas_omega_ib_b, LC_KF_config)

                        # record last GNSS measurement, used for velocity matching
                        prev_time_GNSS_coast = time_GNSS_coast
                        prev_epoch_GNSS_coast = epoch_GNSS_coast
                        time_GNSS_coast = time_GNSS
                        epoch_GNSS_coast = epoch
                            
                        # set flag for velocity matching if in coast mode
                        if outp[epoch-1,10,p] == 1:
                            vel_match_flag = True
                    else:
                        print('   %.2f sec: Outlier=%.2f ' % (time - t_gnss[0], norm(pos_meas_SD)))
                    
            # Generate KF uncertainty and IMU bias output records
            out_IMU_bias_est[epoch_GNSS, 0, p] = time
            out_KF_SD[epoch_GNSS, 0, p] = time
            out_IMU_bias_est[epoch_GNSS, 1:nbs + 1, p] = est_IMU_bias
            out_KF_SD[epoch_GNSS, 1:ns + 1, p] = np.sqrt(np.diag(P))
            
            # Update next GNSS time
            if in_gnss_ptr + gnss_epoch_inc < num_gnss and in_gnss_ptr + gnss_epoch_inc >= 0:
                time_next_GNSS = in_gnss[in_gnss_ptr + gnss_epoch_inc, 0]
            else:
                time_next_GNSS = time_GNSS + 1000 * run_dir;  # set to large value if at end
        
        # End of GNSS measurement processing loop       
        
        # Check accel/gyro and do zero velocity update if near zero
        if cfg.zupt_enable:
            stationary = abs(norm(meas_f_ib_b) - g) < cfg.zupt_accel_thresh and norm(meas_omega_ib_b) < np.deg2rad(cfg.zupt_gyro_thresh)
            stationary_count = stationary_count + 1 if stationary else 0
            if stationary_count == 1:
                sum_gyro = meas_omega_ib_b  # reset filter sum
                sum_accel = meas_f_ib_b 
            elif stationary_count > 1:
                sum_gyro += meas_omega_ib_b # else add to filter sum
                sum_accel += meas_f_ib_b
            if stationary_count == cfg.zupt_epoch_count:
                temp = est_IMU_bias.copy()
                mean_gyro = sum_gyro / cfg.zupt_epoch_count
                mean_accel = sum_accel / cfg.zupt_epoch_count
                est_C_b_e, est_v_eb_e, est_IMU_bias, P = LC_KF_ZUPT_Update(est_C_b_e, 
                    est_v_eb_e, mean_gyro, mean_accel, est_IMU_bias, P, LC_KF_config, gravity, run_dir)
                stationary_count = 0
                # with np.printoptions(precision=3, suppress=True):
                #    print('   ZUPT update:%.1f: ' %  (time-t_gnss[0]), np.rad2deg(temp[:6]), '->', np.rad2deg(est_IMU_bias[:6]))

        # Do non-holonomic update if driving straight and level
        if cfg.nhc_enable and yaw_aligned:
            ok = (abs(meas_omega_ib_b[2]) < np.deg2rad(cfg.nhc_gyro_thesh)) and norm(est_v_eb_e) > cfg.nhc_min_vel
            nhc_count = nhc_count + 1 if ok else 0
            if nhc_count == cfg.nhc_epoch_count:
                v = est_v_eb_e
                est_C_b_e, est_v_eb_e, est_IMU_bias, P = LC_KF_NHC_Update(est_C_b_e, est_v_eb_e, meas_omega_ib_b, -np.array(cfg.imu_offset),
                                      est_IMU_bias, P, LC_KF_config, coast)
                #with np.printoptions(precision=3, suppress=True):
                #    print('   NHC update %.2f: ' % (time-t_gnss[0]), est_C_b_e.T @ v, '->', est_C_b_e.T @ est_v_eb_e)
                nhc_count = 0
        
        # Check if est_C_b_e is ortho-normal and if not, ortho-normalize it
        est_C_b_e = ortho_C(est_C_b_e)
        
        # Calculate position, velocity at GNSS antenna and system origin for output profile
        est_v_gnss_e, est_r_gnss_e = Lever_Arm(est_C_b_e, est_v_eb_e, est_r_eb_e, meas_omega_ib_b, LC_KF_Config.r_lever_arm_b)
        est_v_ref_e, est_r_ref_e = Lever_Arm(est_C_b_e, est_v_eb_e, est_r_eb_e, meas_omega_ib_b, -np.array(cfg.imu_offset))
        
        # Calculate velocity in body frame at system origin for output profile
        v_b = est_C_b_e.T @ est_v_ref_e  
        
        # Generate output profile record
        est_C_b_n = C_e_n @ est_C_b_e
        outp[epoch, 0, p] = time
        if cfg.out_frame == 'gnss':
            outp[epoch, 1:4, p] = est_r_gnss_e # ECEF
            outp[epoch, 4:7, p] = est_v_gnss_e # ECEF
        elif cfg.out_frame == 'origin':
            outp[epoch, 1:4, p] = est_r_ref_e # ECEF
            outp[epoch, 4:7, p] = est_v_ref_e # ECEF
        else:    # 'imu'
            outp[epoch, 1:4, p] = est_r_eb_e # est_r_eb_e   # ECEF
            outp[epoch, 4:7, p] = est_v_eb_e # est_v_eb_e   # ECEF
        outp[epoch, 7:10, p] = CTM_to_Euler(est_C_b_n.T).flatten()
        outp[epoch, 11:14, p] = np.sqrt(np.diag(C_e_n @ P[6:9,6:9] @ C_e_n.T)) # NED pos stdev
        outp[epoch, 14:17, p] = np.sqrt(np.diag(C_e_n @ P[3:6,3:6] @ C_e_n.T)) # NED vel stdev
        outp[epoch, 17:20, p] = np.sqrt(np.diag(C_e_n @ P[0:3,0:3] @ C_e_n.T)) # RPY stdev (rad)
        outp[epoch,20:23, p] = v_b
        
        # check for real GNSS outage longer than threshold and flag as coast
        if coast == 0 and (time - time_GNSS_coast) * run_dir > cfg.vel_match_min_t:
            #print(epoch, epoch_GNSS, time, time_GNSS, time_next_GNSS, coast)
            coast = outp[epoch,10,p] = 1  # mark as coast
            if outp[epoch-1,10,p] == 0:   # if no coast previous epoch
                i = epoch
                # mark previous epochs till last GNSS sample as coast
                while outp[i,0,p] * run_dir > time_GNSS * run_dir:
                    outp[i,10,p] = 1   # mark as coast back to last good GNSS sample
                    i -= 1
                    
        # Do velocity matching if required
        if cfg.vel_match:
            if vel_match_flag or (outp[epoch,10,p] == 0 and outp[epoch-1,10,p] == 1):
                print('Vel match: %.2f-%.2f secs' % (prev_time_GNSS_coast - t_imu[0], time - t_imu[0]))
                outp[:,:,p] = Velocity_Match(outp[:,:,p], prev_epoch_GNSS_coast, epoch)
                vel_match_flag = False
               
        # Update old values
        prev_time = time
        
        # End of IMU measurement processing loop
    
    # Reverse data back before plotting if backwards run
    if run_dir == -1:
        in_gnss, in_imu, outp[:,:,p] = Reverse_Data_Dir(in_gnss, in_imu, outp[:,:,p])
    
    # Plot results
    ng, ni = epoch_GNSS, num_epochs  # number of GNSS and IMU samples
    if cfg.plot_results:
        Plot_Results(in_gnss, in_imu[:ni], outp[:,:,p], fileIn, run_dir, plot_t0)
    if cfg.plot_bias_data:
        Plot_Biases(out_IMU_bias_est[:ng,:,p], fileIn, in_gnss[0,0] - plot_t0)
    if cfg.plot_unc_data:
        Plot_Uncertainties(out_KF_SD[:ng,:,p], fileIn, in_gnss[0,0] - plot_t0)
    
# Combine forward and backward solutions and plot combined result
if npasses == 2:
    outp = Combine_Passes(outp)
    if cfg.plot_results:
        Plot_Results(in_gnss, in_imu[:ni], outp, fileIn, 1, plot_t0)
else:
    outp = outp[:,:,0] 
    
# Decimate result
outp = outp[0::cfg.out_step]

# Plot raw IMU data
if cfg.plot_imu_data:
    Plot_IMU(in_imu[:ni], norm(gravity), fileIn, in_gnss[0,0] - plot_t0)

# Save output in RTKLIB solution file format
if npasses == 1 and run_dir == -1:
    fileOut = join(dataDir, 'gnss_imu_%s_b.pos' % fileIn )
else:
    fileOut = join(dataDir, 'gnss_imu_%s.pos' % fileIn )
Write_GNSS_data(fileOut, outp, out_KF_SD)