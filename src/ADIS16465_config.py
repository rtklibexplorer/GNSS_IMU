#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 Project:    Sensor Fusion for GNSS/IMU (Loosely Coupled Kalman Filter)
 File:       ADIS16465_config.py
 Author:     Tim Everett
 Copyright:  (c) 2025 Tim Everett
 License:    BSD 3 License (see LICENSE file for details)

 Description:
     Configuration file for KF-GINS dataset 2 available at 
     https://github.com/i2Nav-WHU/KF-GINS-Matlab/tree/main/dataset2. 
     which uses an ADIS16465 IMU and RTK GNSS.and includes a high-quality ground-truth 
     See instructions in the file header in convert_KF_GINS_files.py for processing
     this data
===============================================================================
"""
from imu_math import Init
import numpy as np

# IMU parameters (random walk)
imu_config = 'random_walk'  # 'PSD' or 'random_walk'
imu_offset   = [0., 0., 0]  # offset from system origin: forward, right, down
gyro_arw = 0.1; # gyro angle random walk [deg/sqrt(h)]
acc_vrw = 0.1; # accel velocity random walk [m/s/sqrt(h)]
gyro_bias_std = 50; # [deg/h]
acc_bias_std = 50; # [mGal]
gyro_scale_std = 1000; # ppm
acc_scale_std = 1000; # ppm
corr_time = 1; # correlation time for accel and gyro biases and scales [hr]

imu_misalign = np.array([0.0, 0.0, 0.0])   # IMU orientation

# GNSS parameters 
gnss_offset = np.array([-0.37, 0.008, 0.353])  # offset from system origin: forward, right, down (m)
gnss_pos_err_thresh = 10 # outlier threshold for max stdev of GNSS position measurmement (m)

# ratio of specs to process noise stdevs to account for unmodeled errors
imu_noise_factors = [1, 1, 1, 1, 1, 1] # attitude, velocity, accel bias, gyro bias, accel scale, gyro scale
gnss_noise_factors = [1, 1]  # position, velocity

# Initial uncertainties
init =Init()
init.att_unc = [0.2,0.2, 0.5] # initial attitude uncertainty per axis in (deg)
init.bias_acc_unc = 50  # initial accel bias uncertainty (mGal)
init.bias_gyro_unc = 50 # initial gyro bias uncertainty (deg/hr) 
init.scale_acc_unc = 1000 # ppm
init.scale_gyro_unc = 1000 #ppm

# convert KF-GINS units to GNSS-IMU units
gyro_scale_std *= 1e-6 # ppm -> unitless
acc_scale_std *= 1e-6 # ppm -> unitless
init.bias_acc_unc *= 1e-5  # mGal -> m/sec^2
init.bias_gyro_unc /= 3600  # deg/hr -> deg/sec
init.scale_acc_unc *= 1e-6 # ppm -> unitless
init.scale_gyro_unc *= 1e-6 # ppm -> unitless

# Run specific parameters
scale_factors = False  # Kalman filter scale factor states enabled
imu_t_off = 0  # time to add to IMU time stamps to acccount for logging delay (sec)
run_dir = [1]  # run direction ([-1]=backwards, [1]=forwards, [-1,1] = combined)
gnss_epoch_step = 1 # GNSS input sample decimation (1=use all GNSS samples)
out_step = 1  # Output decimation [1=include all samples]
out_frame = 'imu'   # 'imu','gnss', or 'origin'
float_err_gain = 1 # mulitplier on pos/vel stdevs if in float mode
single_err_gain = 1  # mulitplier on pos/vel stdevs if in single mode

# Velocity matching
vel_match = False # do velocity matching at end of coast
vel_match_min_t = 5 # min GNSS outage to invoke vel match (seconds)

# Zero Velocity update
zupt_enable = False
zupt_epoch_count = 50 # epochs to average for update
zupt_accel_thresh = 0.25  # max accel thresh (m/sec^2)
zupt_gyro_thresh = 0.25  # max ang accel thesh (deg/sec)
zupt_vel_SD = 0.02     # zero vel update stdev (m/sec)
zupt_accel_SD = 0.02 # zero vel accelerometer stdev (m/sec^2)
zaru_gyro_SD = 0.02   # zero angular accel stdev (deg/sec)

# Initial yaw alignment
yaw_align = False # use GNSS heading to initialize yaw
yaw_align_min_vel = 0.25  # threshold to start yaw align (m/sec)
yaw_align_max_vel = 1.25  # threshold to end yaw align (m/sec)
yaw_align_min_fix_state = 1 # 1=fix, 2=float,5=single
init_yaw_with_mag = False # use magnetometer to init yaw

# Non-holomonic constrains (NHC) update
nhc_enable = False
nhc_epoch_count = 20
nhc_min_vel = 0.5
nhc_gyro_thesh = 10 # deg/sec
nhc_vel_SD = 0.10 # standard dev m/sec 
nhc_vel_SD_coast = 0.05 # standard dev m/sec

# Testing/Debug
disable_imu = False  # enable to Run GNSS only
start_coast = 0   # start of simulated GNSS outages (secs)
end_coast = 0 # end of simulated GNSS outages (secs before end)
coast_len = 0 # length of simulated  GNSS outages (secs)
start_epoch = 0 # start run at this epoch
num_epochs = 0  # num epochs to run (0=all)
gyro_bias_err = [0,0,0] #[-0.02, 0, 0.1]  # Add constant error to gyro biases (deg/sec)
accel_bias_err = [0, 0, 0]  # Add constant error to acc biases (m/sec^2)
gyro_scale_factor = [1.0, 1.0, 1.0]
accel_scale_factor = [1, 1, 1]
init_rpy = [-0.23, 0.24, 179.5]   # initial roll, pitch, yaw (degrees)

# Plotting
plot_results = True
plot_bias_data = True    # plot accel and gyro bias states
plot_imu_data = False  # plot accel, gyro raw data
plot_unc_data = False  # plot Kalman filter uncertainties