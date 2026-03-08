#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 Project:    Sensor Fusion for GNSS/IMU (Loosely Coupled Kalman Filter)
 File:       walk_config.py
 Author:     Tim Everett
 Copyright:  (c) 2025 Tim Everett
 License:    BSD 3 License (see LICENSE file for details)

 Description:
     Configuration file for sample data in the walk_0827 folder.  This data is 
     from an RTK solution from a u-blox X20 GNSS reciever and a TDK ICM-45686 
     IMU taken while walking in tight turns in a backyard setting.

===============================================================================
"""

from imu_math import Init
import numpy as np

# IMU parameters
imu_config = 'PSD'  # 'PSD' or 'random_walk'
imu_sample_rate = 100
imu_offset   = [0, 0, 0]  # offset from system origin: forward, right, down
gyro_noise_PSD =  0.0038  # deg/sec/sqrt (Hz)
accel_noise_PSD = 70    # ug/sqrt (Hz)
accel_bias_PSD = 7  # ug/sqrt (Hz)
gyro_bias_PSD = 3.8e-5   # deg/sec/sqrt (Hz)
accel_scale_noise_SD = 0.7
gyro_scale_noise_SD = 3.8e-6
imu_misalign = np.array([180.0, 0.0, -90.0])   # IMU orientation (deg rpy)

# GNSS parameters 
gnss_offset = [0,0.05, 0]  # offset from system origin: forward, right, down (m)
gnss_pos_err_thresh = 10 # outlier threshold for max 3D stdev of GNSS position measurmement

# ratio of specs to process noise stdevs to account for unmodeled errors
imu_noise_factors = [1, 1, 1, 1, 1, 1]  # attitude, velocity, accel bias, gyro bias, accel scale, gyro scale
gnss_noise_factors = [1, 1]  # position, velocity

# Initial uncertainties
init =Init()
init.att_unc = [2, 2, 10]   # initial attitude uncertainty per axis in (deg)
init.vel_unc = [0.05, 0.05, 0.1]  # initial velocity uncertainty per axis (m/s)
init.pos_unc = [0.05, 0.05, 0.1]  # initial position uncertainty per axis (m)
init.bias_acc_unc = 0.1 # initial accel bias uncertainty (m/sec^2)
init.bias_gyro_unc = 0.1 # initial gyro bias uncertainty (deg/sec)
init.scale_acc_unc = 0.001
init.scale_gyro_unc = 0.001

# Run specific parameters
scale_factors = False  # Kalman filter scale factor states enabled
imu_t_off = 0   # time to add to IMU time stamps to acccount for logging delay (sec)
run_dir = [1]  # run direction ([-1]=backwards, [1]=forwards, [-1,1] = combined)
gnss_epoch_step = 1 # GNSS input sample decimation (1=use all GNSS samples)
out_step = 1  # Output decimation [1=include all samples]
out_frame = 'gnss'   # 'imu', 'gnss', or 'origin'
float_err_gain = 2 # mulitplier on pos/vel stdevs if in float mode
single_err_gain = 5  # mulitplier on pos/vel stdevs if in single mode

# Velocity matching
vel_match = True  # do velocity matching at end of coast
vel_match_min_t = 1 # min GNSS outage to invoke vel match (seconds)

# Zero Velocity update
zupt_enable = True
zupt_epoch_count = 50 # epochs to average for update
zupt_accel_thresh = 0.25  # max accel thresh (m/sec^2)
zupt_gyro_thresh = 0.25  # max ang accel thesh (deg/sec)
zupt_vel_SD = 0.05     # zero vel update stdev (m/sec)
zupt_accel_SD = 0.05 # zero vel accelerometer stdev (m/sec/2)
zaru_gyro_SD = 0.05   # zero angular accel stdev (deg/sec)

# Initial yaw alignment
yaw_align = True # use GNSS heading to initialize yaw
yaw_align_min_vel = 0.25  # threshold to start yaw align (m/sec)
yaw_align_max_vel = 1.0  # threshold to end yaw align (m/sec)
yaw_align_min_fix_state = 1 # 1=fix, 2=float,5=single
init_yaw_with_mag = False # use magnetometer to init yaw

# Non-holomonic constrains (NHC) update
nhc_enable = False
nhc_epoch_count = 100
nhc_min_vel = 1.0
nhc_gyro_thesh = 20 # deg/sec
nhc_vel_SD = .25 # standard dev m/sec 
nhc_vel_SD_coast = 0.05 # standard dev m/sec

# Testing/Debug
disable_imu = False  # enable to Run GNSS only
start_coast = 40   # start of simulated GNSS outages (secs)
end_coast = 10 # end of simulated GNSS outages (secs before end)
coast_len = 15 # length of simulated  GNSS outages (secs)
num_epochs = 0  # num epochs to run (0=all)
gyro_bias_err = [0, 0, 0]  # Add constant error to gyro biases (deg/sec)
accel_bias_err = [0, 0, 0]  # Add constant error to acc biases (m/sec^2)
gyro_scale_factor = [1, 1, 1]
accel_scale_factor = [1, 1, 1]
init_rpy = [0, 0, 0] # initial roll, pitch, yaw (deg)

# Plotting
plot_results = True
plot_bias_data = True    # plot accel and gyro bias states
plot_imu_data = False  # plot accel, gyro raw data
plot_unc_data = False  # plot Kalman filter uncertainties