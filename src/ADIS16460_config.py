#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 Project:    Sensor Fusion for GNSS/IMU (Loosely Coupled Kalman Filter)
 File:       ADIS16460_config.py
 Author:     Tim Everett
 Copyright:  (c) 2025 Tim Everett
 License:    BSD 3 License (see LICENSE file for details)

 Description:
    ADIS16460 configuration using INSLIB-equivalent noise and uncertainty.
===============================================================================
"""

from imu_math import Init
import numpy as np

# Sensor installation and aiding configuration.
imu_offset = [0.340, -0.460, 0.243] # m, rear-axle origin to IMU in propagation-body FRD
nhc_pivot_offset = [0.0, 0.0, 0.0] # m, origin to rear-axle NHC pivot in vehicle FRD
imu_rotate = [0.0, 0.0, 0.0]       # deg, raw IMU to propagation frame R/P/Y; if scale_factors=True, must be in multiples of 90 deg
nhc_misalign = [0.0, -0.4855, -2.8304] # deg, propagation to vehicle/NHC frame R/P/Y
gnss_imu_offset = [0.045, 0.460, -0.238] # m, IMU to GNSS antenna in propagation-body FRD
gnss_offset = np.array(imu_offset) + np.array(gnss_imu_offset) # m, origin to GNSS antenna in propagation-body FRD

nhc_enable = False                 # Boolean; disable NHC for the INSLIB NHC-off comparison
nhc_epoch_count = 10               # IMU samples between NHC updates
nhc_min_vel = 0.25                 # m/s, minimum 3-D speed for an NHC update
nhc_gyro_thesh = 10.0              # deg/s, maximum absolute body-Z angular rate for an NHC update
nhc_vel_SD = 0.05                  # m/s, lateral/vertical NHC velocity SD

# INSLIB-equivalent IMU process noise.  The gyro_arw value combines the
# 0.2 deg/sqrt(hr) gyro ARW with INSLIB's independent 1e-5 rad/sqrt(s)
# attitude process-noise term in quadrature.
imu_config = "random_walk"         # unitless IMU process-noise representation
scale_factors = True               # Boolean; estimate accelerometer and gyro scale factors
gyro_arw = np.hypot(0.2, np.rad2deg(1.0e-5) * 60.0) # deg/sqrt(hr), gyro ARW plus INSLIB attitude-noise margin
acc_vrw = 0.1                      # m/s/sqrt(hr), accelerometer velocity random walk
gyro_bias_std = 20.0               # deg/hr, gyro bias process-model standard deviation
acc_bias_std = 100.0              # mGal, accelerometer bias process-model standard deviation
corr_time = 1.0                    # hr, bias and scale-factor process-model correlation time
gyro_scale_noise_density = 1000.0e-6 * (2.0 / (corr_time * 3600.0))**0.5 # fraction/sqrt(s), gyro scale process noise
accel_scale_noise_density = 1000.0e-6 * (2.0 / (corr_time * 3600.0))**0.5 # fraction/sqrt(s), accel scale process noise
imu_noise_factors = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0] # unitless gains: gyro white, accel white, accel bias, gyro bias, accel scale, gyro scale

# GNSS related parameters
gnss_noise_factors = [1.0, 1.0]    # unitless gains: GNSS position and velocity
gnss_pos_err_thresh = 10.0      # m, maximum accepted 3-D GNSS position SD
gnss_vel_err_thresh = 10.0     # m/s, maximum accepted 3-D GNSS velocity SD
gnss_SD_multiplier = 0.25      # unitless multiplier on GNSS position SD for filter-SD rejection
float_err_gain = 1.0             # unitless GNSS position-SD multiplier for float solutions
dgps_err_gain = 1.0              # unitless intended DGPS position-SD gain (currently unused; float_err_gain is applied)
single_err_gain = 1.0            # unitless GNSS position-SD multiplier for single solutions



# INSLIB initial covariance values supported directly by GNSS_IMU.
init = Init()                       # initial-uncertainty parameter container
init.att_unc = [0.5, 0.5, 1.0]    # deg, initial roll/pitch/yaw SD
init.bias_acc_unc = 0.05           # m/s^2, initial accelerometer-bias SD
init.bias_gyro_unc = 0.5           # deg/s, initial gyro-bias SD
init.scale_acc_unc = 100.0e-6      # fraction, initial accelerometer-scale SD (100 ppm)
init.scale_gyro_unc = 1000.0e-6    # fraction, initial gyro-scale SD (1000 ppm)
init_rpy = [1.57299, 0.70472, 278.55380] # deg, first truth.nav roll/pitch/yaw

# ZUPT parameters
zupt_enable = False                # Boolean; disable automatic ZUPT for the INSLIB comparison
zupt_epoch_count = 200             # consecutive stationary IMU samples averaged before a ZUPT update
zupt_accel_thresh = 1.0            # m/s^2, maximum acceleration-norm error for ZUPT
zupt_gyro_thresh = 1.0             # deg/s, maximum angular-rate norm for ZUPT
zupt_vel_thresh = 0.1              # m/s, maximum 3-D velocity norm for ZUPT
zupt_vel_SD = 0.02                 # m/s, zero-velocity measurement SD
zupt_accel_SD = 0.05               # m/s^2, stationary accelerometer measurement SD
zaru_gyro_SD = 0.05                # deg/s, zero-angular-rate measurement SD

# Yaw align parameters
yaw_align = False              # Boolean; align yaw using GNSS velocity heading
yaw_align_min_vel = 0.25         # m/s, speed threshold to start yaw alignment
yaw_align_max_vel = 5            # m/s, speed threshold to complete yaw alignment
yaw_align_min_fix_state = 1      # unitless RTKLIB quality code (1=fixed)
init_yaw_with_mag = False        # Boolean; initialize yaw from magnetometer

# Velocity match parameters
vel_match = False                # Boolean; enable end-of-coast velocity matching
vel_match_min_t = 5.0            # s, minimum outage duration for velocity matching

# Configure run start/length and simulated GNSS outages
start_epoch = 0                  # IMU sample index at processing start
num_epochs = 0                   # IMU sample count to process (0=all except final 100-sample buffer)

start_coast = 300.0              # s from first processed IMU epoch, first simulated outage
end_coast = 0.0                  # s before final GNSS epoch to stop simulated outages
coast_len = 60.0                 # s, simulated outage duration
coast_gap = 240.0                # s, GNSS-available gap between simulated outages

# Other run related configuration
imu_t_off = 0.0                  # s, added to IMU timestamps
disable_imu = False              # Boolean; disable inertial propagation
run_dir = [1]                    # unitless direction: -1 backward, +1 forward
gnss_epoch_step = 1              # GNSS samples per measurement update
out_step = 10                    # IMU samples per output record
out_frame = "imu"                # unitless frame selector: "imu", "gnss", or "origin"

# Initial IMU bias/scale factor errors
gyro_bias_err = [0.0, 0.0, 0.0] # deg/s, simulated XYZ gyro bias
accel_bias_err = [0.0, 0.0, 0.0] # m/s^2, simulated XYZ accelerometer bias
gyro_scale_factor = [1.0, 1.0, 1.0] # unitless XYZ multiplicative factors
accel_scale_factor = [1.0, 1.0, 1.0] # unitless XYZ multiplicative factors

# Plot options
plot_results = True              # Boolean; plot navigation results
plot_bias_data = True            # Boolean; plot estimated IMU biases
plot_imu_data = False            # Boolean; plot raw IMU measurements
plot_unc_data = False            # Boolean; plot Kalman state uncertainties
