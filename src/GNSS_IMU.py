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

import argparse
import importlib.util
from pathlib import Path

import numpy as np
from copy import deepcopy
from numpy.linalg import norm
from os.path import join
from imu_math import (Init_P_matrix, Init_Qc_matrix_PSD, Init_Qc_matrix_random_walk, 
                      Nav_equations_ECEF, LC_KF_Predict, LC_KF_GNSS_Update, LC_KF_ZUPT_Update, 
                      Align_Yaw, LC_KF_NHC_Update, Lever_Arm, ortho_C,
                      LC_KF_Config, Velocity_Match, Combine_Passes, Gravity_ECEF, Reverse_Data_Dir,
                      Gate_GNSS_Measurements, GNSS_Noise_Gain)
from imu_files import Read_GNSS_data, Read_IMU_data, Write_GNSS_data
from imu_plot import Plot_Results, Plot_Biases, Plot_IMU, Plot_Uncertainties
from imu_transforms import (CTM_to_Euler, Euler_to_CTM, pvc_GNSS_to_ECEF, compute_C_e_n)

##########  Data selection and run configuration ########################################

# These defaults are used when this file is run directly without arguments.
# run_benchmark.py supplies the same three values on the command line.
script_dir = Path(__file__).resolve().parent
default_config = script_dir / 'ADIS16465_config.py'
default_data_dir = script_dir.parent / 'data' / 'awesome_gins' / 'converted'
default_file_in = 'ADIS16465_sf'


def load_config(config_file):
    """Load a config by filename without requiring it to be on sys.path."""
    config_path = Path(config_file)
    if not config_path.is_absolute():
        config_path = script_dir / config_path
    if config_path.suffix == '':
        config_path = config_path.with_suffix('.py')
    spec = importlib.util.spec_from_file_location(config_path.stem, config_path)
    if spec is None or spec.loader is None:
        raise ImportError('Cannot load config file: %s' % config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


parser = argparse.ArgumentParser(description='Run the GNSS/IMU filter.')
parser.add_argument('--config', default=str(default_config),
                    help='Python configuration file')
parser.add_argument('--data-dir', default=str(default_data_dir),
                    help='Folder containing converted GNSS and IMU files')
parser.add_argument('--file-in', default=default_file_in,
                    help='Input filename stem, for example ADIS16465_sf')
# These overrides let run_benchmark.py avoid editing configuration files.
parser.add_argument('--outage-start', type=float)
parser.add_argument('--outage-duration', type=float)
parser.add_argument('--outage-gap', type=float)
parser.add_argument('--nhc-enable', choices=('true', 'false'))
parser.add_argument('--out-step', type=int)
parser.add_argument('--full-data', action='store_true')
parser.add_argument('--no-plots', action='store_true')
args = parser.parse_args()

cfg = load_config(args.config)
dataDir = args.data_dir
fileIn = args.file_in

if args.outage_start is not None:
    cfg.start_coast = args.outage_start
if args.outage_duration is not None:
    cfg.coast_len = args.outage_duration
if args.outage_gap is not None:
    cfg.coast_gap = args.outage_gap
if args.nhc_enable is not None:
    cfg.nhc_enable = args.nhc_enable == 'true'
if args.out_step is not None:
    cfg.out_step = args.out_step
if args.full_data:
    cfg.start_epoch = 0
    cfg.num_epochs = 0
if args.no_plots:
    cfg.plot_results = False
    cfg.plot_bias_data = False
    cfg.plot_unc_data = False
    cfg.plot_imu_data = False

######### Description of inputs/outputs ########################################

# Inputs:
#   in_gnss      GNSS input measurements
#   in_imu       IMU input measurements
#   num_epochs   Number of IMU samples to process
#   cfg          Selected *_config.py module; defines noise, aiding, and run settings
#     .init.att_unc          Initial roll/pitch/yaw SD (deg; converted to rad internally)
#     .init.bias_acc_unc     Initial accelerometer-bias SD (m/s^2)
#     .init.bias_gyro_unc    Initial gyro-bias SD (deg/s; converted to rad/s internally)
#     .init.scale_acc_unc    Initial accelerometer-scale SD (fraction; 21-state filter)
#     .init.scale_gyro_unc   Initial gyro-scale SD (fraction; 21-state filter)
#   LC_KF_config Internal filter settings derived from cfg
#   Qc           Continuous-time process-noise covariance from cfg IMU noise settings
# Position and velocity measurement SDs come from in_gnss, scaled by cfg.
#
# Outputs:
#   outp               Navigation solution as a motion profile array
#   out_IMU_bias_est   Kalman filter IMU bias estimate array
#   out_KF_SD          Output Kalman filter state uncertainties
#
# Format of output profiles (per pass; position/velocity point selected by cfg.out_frame):
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
#  Column 11: stdev: north position (m)
#  Column 12: stdev: east position (m)
#  Column 13: stdev: down position (m)
#  Column 14: stdev: north velocity (m/s)
#  Column 15: stdev: east velocity (m/s)
#  Column 16: stdev: down velocity (m/s)
#  Column 17: stdev: north-resolved attitude error (rad)
#  Column 18: stdev: east-resolved attitude error (rad)
#  Column 19: stdev: down-resolved attitude error (rad)
#  Column 20: NHC-pivot forward velocity (m/s)
#  Column 21: NHC-pivot right velocity (m/s)
#  Column 22: NHC-pivot down velocity (m/s)

#
# Format of output IMU biases array:
#  Column 0: time (sec)
#  Column 1: estimated X accelerometer bias (m/s^2)
#  Column 2: estimated Y accelerometer bias (m/s^2)
#  Column 3: estimated Z accelerometer bias (m/s^2)
#  Column 4: estimated X gyro bias (rad/s)
#  Column 5: estimated Y gyro bias (rad/s)
#  Column 6: estimated Z gyro bias (rad/s)
#  Columns 7-9: estimated XYZ accelerometer-scale errors (fraction; if enabled)
#  Columns 10-12: estimated XYZ gyro-scale errors (fraction; if enabled)
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
#  Columns 16-18: XYZ accelerometer-scale uncertainty (fraction; if enabled)
#  Columns 19-21: XYZ gyro-scale uncertainty (fraction; if enabled)
# Attitude, velocity, and position error uncertainties use ECEF axes;
# bias and scale-factor uncertainties use propagation-body axes.

# Based on Matlab code from Paul Groves
# License: BSD; see LICENSE for details


# Constants
directions = ['','forward', 'backward']
R_0 = 6378137          # radius of WGS84
ecc = 0.0818191908425  # eccentricity of WGS84
g0 = 9.80665  # standard gravity [m/s^2], used for g-to-SI conversion

############  Start of main code ##############################

# Initialization
LC_KF_config = LC_KF_Config()
C_imu_rotate = Euler_to_CTM(np.deg2rad(np.array(cfg.imu_rotate)))
npasses = len(cfg.run_dir)
LC_KF_config.gnss_vel_err_thresh = cfg.gnss_vel_err_thresh  # m/s
# Paul Groves error state:
# [delta_theta, v_hat-v, r_hat-r, b_a-b_a_hat, b_g-b_g_hat].
# Navigation corrections subtract errors; bias corrections add them.
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
LC_KF_config.C_b_to_nhc = Euler_to_CTM(
    np.deg2rad(np.array(cfg.nhc_misalign)))
# Configured offsets point from the system origin to each point. IMU and GNSS
# offsets use propagation axes; the NHC pivot uses vehicle/NHC FRD axes.
r_imu_to_nhc_pivot_v = (np.array(cfg.nhc_pivot_offset) -
                        LC_KF_config.C_b_to_nhc @ np.array(cfg.imu_offset))
r_imu_to_nhc_pivot_b = LC_KF_config.C_b_to_nhc.T @ r_imu_to_nhc_pivot_v

LC_KF_config.init = deepcopy(cfg.init)
LC_KF_config.init.att_unc = np.deg2rad(cfg.init.att_unc)
LC_KF_config.init.bias_gyro_unc = np.deg2rad(cfg.init.bias_gyro_unc)

# lever arm in body frame
LC_KF_Config.r_lever_arm_b = np.array(cfg.gnss_offset) - np.array(cfg.imu_offset)
# Load input measurements
in_gnss, num_gnss, ok = Read_GNSS_data(join(dataDir, 'gnss_%s.pos' % fileIn))
in_imu, num_epochs, ok = Read_IMU_data(join(dataDir, 'imu_%s.csv' % fileIn))

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
in_imu[:,1:4] = (in_imu[:,1:4] + np.array(cfg.accel_bias_err) / g0) * cfg.accel_scale_factor
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
in_imu = in_imu[:num_epochs]
in_gnss = in_gnss[in_gnss[:,0] <= in_imu[-1,0] + cfg.imu_t_off]
num_gnss = len(in_gnss)
    
# Initialize output arrays
outp = np.zeros((num_epochs, 23, npasses))
out_IMU_bias_est = np.zeros((num_gnss, nbs + 1, npasses))
out_KF_SD = np.zeros((int(num_gnss / cfg.gnss_epoch_step) + 1, ns + 1, npasses))

# Setup simulated coast intervals (only used for eval)
s = in_imu[0,0] + cfg.imu_t_off + cfg.start_coast
e = in_gnss[-1,0] - cfg.end_coast
coast_times = in_imu[:num_epochs, 0] + cfg.imu_t_off
gnss_coast = np.zeros(num_gnss, dtype=bool)
while cfg.coast_len > 0 and s < e:
    coast_end = min(s + cfg.coast_len, e)
    ix = np.where((coast_times >= s) & (coast_times < coast_end))[0]
    outp[ix,10,:] = 1
    gnss_coast |= ((in_gnss[:,0] >= s) & (in_gnss[:,0] < coast_end))
    s += cfg.coast_len + cfg.coast_gap

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
    r_gnss_e += v_gnss_e * (t_imu[0] - t_gnss[0])
    meas_omega_init_b = C_imu_rotate @ in_imu[0, 4:7]
    est_v_eb_e, est_r_eb_e = Lever_Arm(
        est_C_b_e, v_gnss_e, r_gnss_e, meas_omega_init_b,
        -LC_KF_Config.r_lever_arm_b)
    prev_time = start_time = t_imu[0]
    
    # Calculate earth radius and gravity for starting location
    geocentric_radius = (R_0 / np.sqrt(1 - (ecc * np.sin(est_L_b))**2)
                        * np.sqrt(np.cos(est_L_b)**2 + (1 - ecc**2)**2 * np.sin(est_L_b)**2))
    gravity = Gravity_ECEF(est_r_eb_e)  # calculate ECEF gravity vector for this location
    g = norm(gravity)
    
    # Initial output profile
    outp[0, 0, p] = prev_time
    if cfg.out_frame == 'gnss':
        outp[0, 1:4, p] = r_gnss_e
        outp[0, 4:7, p] = v_gnss_e
    elif cfg.out_frame == 'origin':
        est_v_ref_e, est_r_ref_e = Lever_Arm(
            est_C_b_e, est_v_eb_e, est_r_eb_e, meas_omega_init_b,
            -np.array(cfg.imu_offset))
        outp[0, 1:4, p] = est_r_ref_e
        outp[0, 4:7, p] = est_v_ref_e
    else:  # 'imu'
        outp[0, 1:4, p] = est_r_eb_e
        outp[0, 4:7, p] = est_v_eb_e
    outp[0, 7:10, p] = CTM_to_Euler(est_C_b_n.T).flatten()
    est_v_nhc_pivot_e, _ = Lever_Arm(
        est_C_b_e, est_v_eb_e, est_r_eb_e, meas_omega_init_b,
        r_imu_to_nhc_pivot_b)
    outp[0, 20:23, p] = (LC_KF_config.C_b_to_nhc @ est_C_b_e.T @
                         est_v_nhc_pivot_e)
    
    # Initialize Kalman filter P matrix and IMU bias states
    C_e_n = compute_C_e_n(est_L_b, est_lambda_b,) # Map ECEF to NED
    # Scale initial GNSS position uncertainty based on fix quality.
    init_noise_gain = GNSS_Noise_Gain(fix[0], cfg)
    if init_noise_gain == 0:
        init_noise_gain = 1e6  # Set variance very high if this is unused state
    # Apply configured noise factors and the fix-quality gain to initial position/velocity SDs.
    init_pos_meas_SD = in_gnss[0,6:9] * cfg.gnss_noise_factors[0] * init_noise_gain
    init_vel_meas_SD = in_gnss[0,17:20] * cfg.gnss_noise_factors[1] * init_noise_gain
    P = Init_P_matrix(LC_KF_config, C_e_n, init_pos_meas_SD, init_vel_meas_SD)
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
    gnss_velocity_available = (norm(init_vel_meas_SD) <=
                               LC_KF_config.gnss_vel_err_thresh)

    # Loop through IMU epocs
    for epoch in range(1, num_epochs):
        
        print('   Secs: %.3f' % (t_imu[epoch] - t_imu[0]), end='\r')
        
        # Update time and coast status
        time = t_imu[epoch]
        tor_i = time - prev_time
        if tor_i * run_dir < 1e-5:
            outp[epoch,:,p] = outp[epoch-1,:,p]
            continue
        coast = outp[epoch,10,p]  # get coast status for this epoch
        
        # Get next IMU measurements and correct for biases and rotation
        raw_f_ib_b = C_imu_rotate @ in_imu[epoch, 1:4] * g0
        raw_omega_ib_b = C_imu_rotate @ in_imu[epoch, 4:7]
        meas_f_ib_b = raw_f_ib_b.copy()
        meas_f_ib_b -= est_IMU_bias[0:3]
        meas_omega_ib_b = raw_omega_ib_b.copy()
        meas_omega_ib_b -= est_IMU_bias[3:6]
        if cfg.scale_factors:
            meas_f_ib_b *= (1 - est_IMU_bias[6:9])
            meas_omega_ib_b *= (1 - est_IMU_bias[9:12])

        # Split propagation at GNSS measurement timestamps. The IMU record is
        # an interval-average measurement, so the same value applies to both
        # portions of a split interval.
        propagation_time = prev_time

        # Check for next GNSS measurement
        while time * run_dir >= time_next_GNSS * run_dir:
            tor_gnss = time_next_GNSS - propagation_time
            if abs(tor_gnss) > 1e-12:
                if not cfg.disable_imu:  # disable IMU for eval only
                    est_r_eb_e, est_v_eb_e, est_C_b_e = Nav_equations_ECEF(
                        tor_gnss, est_r_eb_e, est_v_eb_e, est_C_b_e,
                        meas_f_ib_b, meas_omega_ib_b, gravity)
                else:
                    est_r_eb_e += est_v_eb_e * tor_gnss
                P = LC_KF_Predict(
                    tor_gnss, est_C_b_e, est_v_eb_e, est_r_eb_e,
                    est_IMU_bias, P, Qc, meas_f_ib_b, meas_omega_ib_b,
                    LC_KF_config, gravity, geocentric_radius)
                propagation_time = time_next_GNSS
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
                        print('   %.2f sec: Yaw align: ' % (time_GNSS - t_gnss[0]), end='')
                        est_C_b_e, P = Align_Yaw(est_C_b_e, C_e_n.T, v_gnss_n, P, run_dir)
                        if norm(v_gnss_n[:2]) > cfg.yaw_align_max_vel:
                            yaw_aligned = True  # disable any further yaw alignment
            
            # Gate by the GNSS measurement time, including split IMU epochs.
            if not gnss_coast[in_gnss_ptr]:
                valid_gnss, adj_pos_meas_SD, adj_vel_meas_SD = Gate_GNSS_Measurements(
                    fix[in_gnss_ptr], pos_meas_SD, vel_meas_SD, P, cfg)
                gnss_velocity_available = (valid_gnss and
                    norm(adj_vel_meas_SD) <=
                    LC_KF_config.gnss_vel_err_thresh)

                # Run GNSS Measurement Update for Kalman Filter
                if valid_gnss:
                    est_C_b_e, est_v_eb_e, est_r_eb_e, est_IMU_bias, P = LC_KF_GNSS_Update(
                        r_gnss_e, v_gnss_e, adj_pos_meas_SD, adj_vel_meas_SD, est_C_b_e, C_e_n.T,
                        est_v_eb_e, est_r_eb_e, est_IMU_bias, P, meas_omega_ib_b,
                        LC_KF_config)
                    # record last GNSS measurement, used for velocity matching
                    prev_time_GNSS_coast = time_GNSS_coast
                    prev_epoch_GNSS_coast = epoch_GNSS_coast
                    time_GNSS_coast = time_GNSS
                    epoch_GNSS_coast = epoch
                    # update gravity
                    gravity = Gravity_ECEF(est_r_eb_e)  # calculate ECEF gravity vector for this location
                    g = norm(gravity)

                    # set flag for velocity matching if in coast mode
                    if outp[epoch-1,10,p] == 1:
                        vel_match_flag = True
                else:
                    print('Outlier:   %.2f sec: pos_meas_SD=%.3f, pos_SD=%.3f fix_state0=%d' % (time_GNSS - t_gnss[0],
                           norm(pos_meas_SD), norm(np.sqrt(np.diag(P[6:9, 6:9]))), fix[in_gnss_ptr]))

            # A GNSS update can change the calibration estimates. Apply the
            # corrected estimates over the remainder of this IMU interval.
            meas_f_ib_b = raw_f_ib_b - est_IMU_bias[0:3]
            meas_omega_ib_b = raw_omega_ib_b - est_IMU_bias[3:6]
            if cfg.scale_factors:
                meas_f_ib_b *= (1 - est_IMU_bias[6:9])
                meas_omega_ib_b *= (1 - est_IMU_bias[9:12])
                    
            # Generate KF uncertainty and IMU bias output records
            out_IMU_bias_est[epoch_GNSS, 0, p] = time_GNSS
            out_KF_SD[epoch_GNSS, 0, p] = time_GNSS
            out_IMU_bias_est[epoch_GNSS, 1:nbs + 1, p] = est_IMU_bias
            out_KF_SD[epoch_GNSS, 1:ns + 1, p] = np.sqrt(np.diag(P))
            
            # Update next GNSS time
            if in_gnss_ptr + gnss_epoch_inc < num_gnss and in_gnss_ptr + gnss_epoch_inc >= 0:
                time_next_GNSS = in_gnss[in_gnss_ptr + gnss_epoch_inc, 0]
            else:
                time_next_GNSS = time_GNSS + 1000 * run_dir;  # set to large value if at end
        
        # End of GNSS measurement processing loop

        # Propagate from the last GNSS update (or the previous IMU epoch) to
        # the current IMU epoch.
        tor_remaining = time - propagation_time
        if abs(tor_remaining) > 1e-12:
            if not cfg.disable_imu:  # disable IMU for eval only
                est_r_eb_e, est_v_eb_e, est_C_b_e = Nav_equations_ECEF(
                    tor_remaining, est_r_eb_e, est_v_eb_e, est_C_b_e,
                    meas_f_ib_b, meas_omega_ib_b, gravity)
            else:
                est_r_eb_e += est_v_eb_e * tor_remaining
            P = LC_KF_Predict(
                tor_remaining, est_C_b_e, est_v_eb_e, est_r_eb_e,
                est_IMU_bias, P, Qc, meas_f_ib_b, meas_omega_ib_b,
                LC_KF_config, gravity, geocentric_radius)
        
        # Check accel/gyro and do zero velocity update if near zero
        if cfg.zupt_enable:
            stationary = abs(norm(meas_f_ib_b) - g) < cfg.zupt_accel_thresh and \
                         norm(meas_omega_ib_b) < np.deg2rad(cfg.zupt_gyro_thresh) and \
                         norm(est_v_eb_e) < cfg.zupt_vel_thresh
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

        # Apply NHC only when GNSS velocity is unavailable or not being used.
        nhc_active = coast or not gnss_velocity_available
        if cfg.nhc_enable and yaw_aligned and nhc_active:
            ok = (abs(meas_omega_ib_b[2]) < np.deg2rad(cfg.nhc_gyro_thesh)) and norm(est_v_eb_e) > cfg.nhc_min_vel
            nhc_count = nhc_count + 1 if ok else 0
            if nhc_count == cfg.nhc_epoch_count:
                v = est_v_eb_e
                est_C_b_e, est_v_eb_e, est_IMU_bias, P = LC_KF_NHC_Update(est_C_b_e, est_v_eb_e, meas_omega_ib_b, r_imu_to_nhc_pivot_v,
                                      est_IMU_bias, P, LC_KF_config)
                #with np.printoptions(precision=3, suppress=True):
                #    print('   NHC update %.2f: ' % (time-t_gnss[0]), est_C_b_e.T @ v, '->', est_C_b_e.T @ est_v_eb_e)
                nhc_count = 0
        else:
            nhc_count = 0
        
        # Check if est_C_b_e is ortho-normal and if not, ortho-normalize it
        est_C_b_e = ortho_C(est_C_b_e)
        
        # Velocity at the NHC pivot, resolved in vehicle/NHC FRD. Including the
        # rotational lever-arm velocity makes R and D zero for ideal no-slip
        # motion on level ground or a hill.
        est_v_nhc_pivot_e, _ = Lever_Arm(
            est_C_b_e, est_v_eb_e, est_r_eb_e, meas_omega_ib_b,
            r_imu_to_nhc_pivot_b)
        v_nhc = (LC_KF_config.C_b_to_nhc @ est_C_b_e.T @
                 est_v_nhc_pivot_e)
        
        # Generate output profile record
        est_C_b_n = C_e_n @ est_C_b_e
        outp[epoch, 0, p] = time
        if cfg.out_frame == 'gnss':
            est_v_gnss_e, est_r_gnss_e = Lever_Arm(
                est_C_b_e, est_v_eb_e, est_r_eb_e, meas_omega_ib_b,
                LC_KF_Config.r_lever_arm_b)
            outp[epoch, 1:4, p] = est_r_gnss_e # ECEF
            outp[epoch, 4:7, p] = est_v_gnss_e # ECEF
        elif cfg.out_frame == 'origin':
            est_v_ref_e, est_r_ref_e = Lever_Arm(
                est_C_b_e, est_v_eb_e, est_r_eb_e, meas_omega_ib_b,
                -np.array(cfg.imu_offset))
            outp[epoch, 1:4, p] = est_r_ref_e # ECEF
            outp[epoch, 4:7, p] = est_v_ref_e # ECEF
        else:    # 'imu'
            outp[epoch, 1:4, p] = est_r_eb_e # est_r_eb_e   # ECEF
            outp[epoch, 4:7, p] = est_v_eb_e # est_v_eb_e   # ECEF
        outp[epoch, 7:10, p] = CTM_to_Euler(est_C_b_n.T).flatten()
        outp[epoch, 11:14, p] = np.sqrt(np.diag(C_e_n @ P[6:9,6:9] @ C_e_n.T)) # NED pos stdev
        outp[epoch, 14:17, p] = np.sqrt(np.diag(C_e_n @ P[3:6,3:6] @ C_e_n.T)) # NED vel stdev
        outp[epoch, 17:20, p] = np.sqrt(np.diag(C_e_n @ P[0:3,0:3] @ C_e_n.T)) # RPY stdev (rad)
        outp[epoch,20:23, p] = v_nhc
        
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
