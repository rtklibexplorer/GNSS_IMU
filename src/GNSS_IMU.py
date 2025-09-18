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
from imu_math import (Initialize_LC_P_matrix, Nav_equations_ECEF, LC_KF_Predict, 
                      LC_KF_GNSS_Update, LC_KF_ZUPT_Update, Align_Yaw, LC_KF_NHC_Update, 
                      LC_KF_Config, Velocity_Match, Combine_Passes, Gravity_ECEF, Reverse_Data_Dir,
                      Lever_Arm, ortho_C)
from imu_files import Read_GNSS_data, Read_IMU_data, Write_GNSS_data
from imu_plot import Plot_Results, Plot_Biases, Plot_IMU, Plot_Uncertainties
from imu_transforms import (CTM_to_Euler, Euler_to_CTM, pvc_LLH_to_ECEF, pvc_ECEF_to_LLH)

##########  Data selection and run configuration ########################################

# Uncomment these lines to run an example of walking with a handheld GNSS receiver
# import walk_config as cfg
# dataDir = r'..\data\walk_0827'
# fileIn = '1730_sf'

# Uncomment these lines to run an example of driving with a roof mounted GNSS receiver
import drive_config as cfg
dataDir = r'..\data\drive_0708'
fileIn = '1934_sf'

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
#  Column 17: X body velocity (m/s)
#  Column 18: Y body velocity (m/s)
#  Column 19: Z body velocity (m/s)

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
ug_to_mps_squared = g * 1E-6  # micro g -> meters/sec^2

############  Start of main code ##############################

# Initialization
LC_KF_config = LC_KF_Config()
C_imu_misalign = Euler_to_CTM(np.deg2rad(np.array(cfg.imu_misalign)))
npasses = len(cfg.run_dir)
ns = LC_KF_config.nstates = 21 if cfg.scale_factors else 15
nbs = 12 if cfg.scale_factors else 6

# Calculate kalman process noise variances
root_Hz = np.sqrt(cfg.imu_sample_rate)
LC_KF_config.attitude_noise_var = (np.deg2rad(cfg.gyro_noise_PSD) * root_Hz * cfg.imu_noise_factors[0])**2
LC_KF_config.vel_noise_var = (cfg.accel_noise_PSD * ug_to_mps_squared * root_Hz * cfg.imu_noise_factors[1])**2
LC_KF_config.accel_bias_noise_var = (cfg.accel_bias_PSD * ug_to_mps_squared * root_Hz * cfg.imu_noise_factors[2])**2
LC_KF_config.gyro_bias_noise_var =  (np.deg2rad(cfg.gyro_bias_PSD) * root_Hz * cfg.imu_noise_factors[3])**2
if cfg.scale_factors:
    LC_KF_config.accel_scale_noise_var = (cfg.accel_scale_noise_SD * ug_to_mps_squared * root_Hz * cfg.imu_noise_factors[4])**2
    LC_KF_config.gyro_scale_noise_var = (np.deg2rad(cfg.gyro_scale_noise_SD) * root_Hz * cfg.imu_noise_factors[5])**2

# Calculate kalman measurement noise variances    
LC_KF_config.zupt_vel_var = cfg.zupt_vel_SD**2 
LC_KF_config.zaru_gyro_var = np.deg2rad(cfg.zaru_gyro_SD)**2
LC_KF_config.nhc_vel_var = cfg.nhc_vel_SD**2
LC_KF_config.nhc_vel_var_coast = cfg.nhc_vel_SD_coast**2

# Calculate initial uncertainties
LC_KF_Config.init = cfg.init
LC_KF_Config.init.att_unc = np.deg2rad(cfg.init.att_unc)
LC_KF_Config.init.bias_gyro_unc = np.deg2rad(cfg.init.bias_gyro_unc)

# lever arm
lever_arm_b = np.array(cfg.imu_offset) - np.array(cfg.gnss_offset)
LC_KF_config.lever_arm_b = lever_arm_b   # lever arm in body frame

# Load input measurements
in_gnss, num_gnss, ok = Read_GNSS_data(join(dataDir, 'gnss_%s.pos' % fileIn ))
in_imu, num_epochs, ok = Read_IMU_data(join(dataDir, 'imu_%s.csv' % fileIn ))

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
outp = np.zeros((num_epochs, 20, npasses))
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
        
    # Init yaw
    if cfg.init_yaw_with_mag:
        # Use magnetometer to initialize orientation states
        rpy_init = [0, 0, np.atan2(in_imu[0,8], in_imu[0,7])]
    else:
        rpy_init = [0, 0, 0]
    rpy_init[2] -= np.deg2rad(cfg.yaw_off)  # apply initial yaw offset
    if p == 0:  # don't reinit for 2nd pass
        est_C_b_n = Euler_to_CTM(rpy_init)  # initial NED orientation
   
    # Use initial GNSS position solution to initialize pos/vel states
    est_L_b, est_lambda_b, est_h_b = in_gnss[0, 1:4]    # initial LLI position
    est_v_eb_n = in_gnss[0, 14:17]  # initial NED velocity
    est_r_eb_e, est_v_eb_e, est_C_b_e =  pvc_LLH_to_ECEF(est_L_b, est_lambda_b, 
            est_h_b, est_v_eb_n, est_C_b_n)
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
    P = Initialize_LC_P_matrix(LC_KF_config)
    est_IMU_bias = np.zeros(nbs)
    out_IMU_bias_est[0, 0, p] = prev_time
    out_IMU_bias_est[0, 1:nbs+1, p] = est_IMU_bias
    
    # Generate KF uncertainty record
    out_KF_SD[0, 0, p] = prev_time
    out_KF_SD[0, 1:ns+1, p] = np.sqrt(np.diag(P))
    
    # Initialize GNSS timing
    GNSS_epoch = 0  # count of GNSS epochs processed
    time_GNSS = time_GNSS_fix = t_gnss[0]
    time_next_GNSS = t_gnss[gnss_epoch_inc]
    epoch_GNSS_fix = prev_epoch_GNSS_fix = in_gnss_ptr = 0 # ptrs to GNSS input data
    
    # Init internal states
    vel_match_flag = False
    yaw_aligned = False

    # Loop through IMU epocs
    for epoch in range(num_epochs):
        
        print('   Secs: %.3f' % (t_imu[epoch] - t_imu[0]), end='\r')
        
        # Update time and coast status
        time = t_imu[epoch]
        tor_i = time - prev_time
        coast = outp[epoch,10,p]  # get coast status for this epoch
        
        # Get next IMU measurements and correct for biases and misalignment
        meas_f_ib_b = C_imu_misalign @ in_imu[epoch, 1:4] * g
        meas_f_ib_b -= est_IMU_bias[0:3]
        meas_omega_ib_b = C_imu_misalign @ in_imu[epoch, 4:7]
        meas_omega_ib_b -= est_IMU_bias[3:6]
        if cfg.scale_factors:
            meas_f_ib_b *= (1 + est_IMU_bias[6:9])
            meas_omega_ib_b *= (1 + est_IMU_bias[9:12])

        # Update estimated navigation solution unless in IMU coast mode
        if not cfg.disable_imu and time * run_dir >= t_gnss[0] * run_dir:  # disable IMU for eval only
            est_r_eb_e, est_v_eb_e, est_C_b_e = Nav_equations_ECEF(
                tor_i, est_r_eb_e, est_v_eb_e, est_C_b_e,
                    meas_f_ib_b, meas_omega_ib_b, gravity)
            
        # Kalman state prediction if run for both IMU and GNSS measurements
        P = LC_KF_Predict(tor_i, est_C_b_e, est_v_eb_e, est_r_eb_e, est_IMU_bias,
                       P, meas_f_ib_b, meas_omega_ib_b, LC_KF_config,
                       gravity, geocentric_radius)
    
        # Check for next GNSS measurement
        while time * run_dir >= time_next_GNSS * run_dir:
            GNSS_epoch += 1   
            in_gnss_ptr += gnss_epoch_inc
            
            # Process measurement if not in coast mode
            if coast == 0: 
                # Adjust measurement uncertainty based on fix status
                if fix[in_gnss_ptr] == FIX or cfg.disable_imu:
                    noise_gain = 1
                elif fix[in_gnss_ptr] == FLOAT: 
                    noise_gain = cfg.float_err_gain
                elif fix[in_gnss_ptr] == SINGLE: 
                    noise_gain = cfg.single_err_gain
                    
                if noise_gain != 0:  # skip measurement if noise_gain set to zero
                    # Get next GNSS measurement
                    GNSS_L_b, GNSS_lambda_b, GNSS_h_b = in_gnss[in_gnss_ptr, 1:4]    # LLI position
                    GNSS_v_eb_n = in_gnss[in_gnss_ptr, 14:17]  # NED velocity
                    LC_KF_config.pos_meas_SD = in_gnss[in_gnss_ptr,6:9] * cfg.gnss_noise_factors[0] * noise_gain
                    LC_KF_config.vel_meas_SD = in_gnss[in_gnss_ptr,17:20] * cfg.gnss_noise_factors[1] * noise_gain
                    GNSS_r_eb_e, GNSS_v_eb_e, _ = pvc_LLH_to_ECEF(GNSS_L_b, GNSS_lambda_b, 
                            GNSS_h_b, GNSS_v_eb_n, np.zeros((3,3)))
                    time_GNSS = time_next_GNSS
                    
                    # record last fix, used for velocity matching
                    if fix[in_gnss_ptr] == FIX:
                        prev_time_GNSS_fix = time_GNSS_fix
                        prev_epoch_GNSS_fix = epoch_GNSS_fix
                        time_GNSS_fix = time_next_GNSS
                        epoch_GNSS_fix = epoch
                        
                    # set flag for velocity matching in middle of coast if no fix
                    if fix[in_gnss_ptr]  != 1 and outp[epoch-1,10,p] == 1 and cfg.vel_match:
                        vel_match_flag = True
                    
                    # Run GNSS Measurement Update for Kalman Filter
                    est_C_b_e, est_v_eb_e, est_r_eb_e, est_IMU_bias, P = LC_KF_GNSS_Update(
                        GNSS_r_eb_e, GNSS_v_eb_e, est_C_b_e, est_v_eb_e,
                        est_r_eb_e, est_IMU_bias, P, meas_f_ib_b, meas_omega_ib_b,
                        LC_KF_config)
                    
                    # Align yaw first time velocity is large enough to calculate heading
                    if cfg.yaw_align and not yaw_aligned:
                        if norm(est_v_eb_e) > cfg.yaw_align_min_vel and fix[in_gnss_ptr] == FIX :  # check forward velocity
                            print('   %.2f sec: Yaw align: ' % (time - t_gnss[0]), end='')
                            _, _, _, _, est_C_b_n = pvc_ECEF_to_LLH(est_r_eb_e, est_v_eb_e, est_C_b_e)
                            est_C_b_e, P  = Align_Yaw(est_C_b_e, est_C_b_n, GNSS_v_eb_n, P, run_dir)
                            yaw_aligned = True
                                                    
            # Generate KF uncertainty and IMU bias output records
            out_IMU_bias_est[GNSS_epoch, 0, p] = time
            out_KF_SD[GNSS_epoch, 0, p] = time
            out_IMU_bias_est[GNSS_epoch, 1:nbs + 1, p] = est_IMU_bias
            out_KF_SD[GNSS_epoch, 1:ns + 1, p] = np.sqrt(np.diag(P))
            
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
            elif stationary_count > 1:
                sum_gyro += meas_omega_ib_b # else add to filter sum
            if stationary_count == cfg.zupt_epcoh_count:
                temp = est_IMU_bias.copy()
                mean_gyro = sum_gyro / cfg.zupt_epcoh_count
                est_C_b_e, est_v_eb_e, est_IMU_bias, P = LC_KF_ZUPT_Update(est_C_b_e, 
                    est_v_eb_e, mean_gyro, est_IMU_bias, P, LC_KF_config, run_dir)
                stationary_count = 0
                #with np.printoptions(precision=3, suppress=True):
                #    print('   ZUPT update:%.1f: ' %  (time-t_gnss[0]), np.rad2deg(temp[3:6]), '->', np.rad2deg(est_IMU_bias[3:6]))

        # Do non-holonomic update if driving straight and level
        if cfg.nhc_enable:
            ok = (abs(meas_omega_ib_b[2]) < np.deg2rad(cfg.nhc_gyro_thesh)) and norm(est_v_eb_e) > cfg.nhc_min_vel
            nhc_count = nhc_count + 1 if ok else 0
            if nhc_count == cfg.nhc_epcoh_count:
                v = est_v_eb_e
                est_C_b_e, est_v_eb_e, est_IMU_bias, P = LC_KF_NHC_Update(est_C_b_e, est_v_eb_e, meas_omega_ib_b, -np.array(cfg.imu_offset),
                                      est_IMU_bias, P, LC_KF_config, coast)
                #with np.printoptions(precision=3, suppress=True):
                #    print('   NHC update %.2f: ' % (time-t_gnss[0]), est_C_b_e.T @ v, '->', est_C_b_e.T @ est_v_eb_e)
                nhc_count = 0
        
        # Check if est_C_b_e is ortho-normal and if not, ortho-normalize it
        est_C_b_e = ortho_C(est_C_b_e)
        
        # Calculate position, velocity at GNSS antenna for output profile
        est_v_gnss_e, est_r_gnss_e = Lever_Arm(est_C_b_e, est_v_eb_e, est_r_eb_e, meas_omega_ib_b, lever_arm_b)
        est_v_ref_e, _ = Lever_Arm(est_C_b_e, est_v_eb_e, est_r_eb_e, meas_omega_ib_b, -np.array(cfg.imu_offset))
        
        # Calculate velocity in body frame at system origin for output profile
        v_b = est_C_b_e.T @ est_v_ref_e  
        
        # Generate output profile record
        _, _, _, _, est_C_b_n = pvc_ECEF_to_LLH(est_r_eb_e, est_v_eb_e, est_C_b_e)
        outp[epoch, 0, p] = time
        outp[epoch, 1:4, p] = est_r_gnss_e # est_r_eb_e   # ECEF
        outp[epoch, 4:7, p] = est_v_gnss_e # est_v_eb_e   # ECEF
        outp[epoch, 7:10, p] = CTM_to_Euler(est_C_b_n.T).flatten() 
        outp[epoch,17:20, p] = v_b

        # check for real GNSS outage longer than threshold and flag as coast
        if coast == 0 and (time - time_GNSS_fix) * run_dir > cfg.vel_match_min_t:
            #print(epoch, GNSS_epoch, time, time_GNSS, time_next_GNSS, coast)
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
                outp[:,:,p] = Velocity_Match(outp[:,:,p], prev_epoch_GNSS_fix, epoch)
                vel_match_flag = False
               
        # Update old values
        prev_time = time
        
        # End of IMU measurement processing loop
    
    # Reverse data back before plotting if backwards run
    if run_dir == -1:
        in_gnss, in_imu, outp[:,:,p] = Reverse_Data_Dir(in_gnss, in_imu, outp[:,:,p])
    
    # Plot results
    ng, ni = GNSS_epoch, num_epochs  # number of GNSS and IMU samples
    if cfg.plot_results:
        Plot_Results(in_gnss, in_imu[:ni], outp[:,:,p], fileIn, run_dir)
    if cfg.plot_bias_data:
        Plot_Biases(out_IMU_bias_est[:ng,:,p], fileIn, in_gnss[0,0])
    if cfg.plot_unc_data:
        Plot_Uncertainties(out_KF_SD[:ng,:,p], fileIn, in_gnss[0,0])
    
# Combine forward and backward solutions and plot combined result
if npasses == 2:
    outp = Combine_Passes(outp)
    if cfg.plot_results:
        Plot_Results(in_gnss, in_imu[:ni], outp, fileIn, 1)
else:
    outp = outp[:,:,0] 
    
# Decimate result
outp = outp[0::cfg.out_step]

# Plot raw IMU data
if cfg.plot_imu_data:
    Plot_IMU(in_imu[:ni], norm(gravity), fileIn, in_gnss[0,0])

# Save output in RTKLIB solution file format
if npasses == 1 and run_dir == -1:
    fileOut = join(dataDir, 'gnss_imu_%s_b.pos' % fileIn )
else:
    fileOut = join(dataDir, 'gnss_imu_%s.pos' % fileIn )
Write_GNSS_data(fileOut, outp, out_KF_SD)