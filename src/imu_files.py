#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 Project:    Sensor Fusion for GNSS/IMU (Loosely Coupled Kalman Filter)
 File:       imu_files.py
 Author:     Tim Everett
 Copyright:  (c) 2025 Tim Everett
 License:    BSD 3 License (see LICENSE file for details)

 Description:
     Input/output file functions for GNSS/IMU sensor fusion
===============================================================================
"""

import numpy as np
from imu_transforms import pv_ECEF_to_LLH, datetime_to_utc, utc_secs_to_datetime
from scipy.signal import medfilt

def Read_GNSS_data(filename):
    # Read_GNSS_data in RTKLIB format - inputs GNSS position and velocity data
    # and converts velocity from NEU to NED
    # Column 0: time (sec)
    # Column 1: latitude
    # Column 2: longitude
    # Column 3: height
    # Column 4: fix status
    # Column 5: nsats
    # Column 6: pos_std (N)
    # Column 7: pos_std (E)
    # Column 8: pos_std (U)
    # Column 14: north velocity (m/s)
    # Column 15: east velocity (m/s)
    # Column 16: up velocity (m/s)
    # Column 17: vel_std (N)
    # Column 18: vel_std (E)
    # Column 19: vel_std (U)
    #
    # Inputs:
    #   filename     Name of file to write
    #
    # Outputs:
    #   in_gnss   Array of data from the file
    #   no_epochs    Number of epochs of data in the file
    #   ok           Indicates file has the expected number of columns

    data = np.genfromtxt(filename, dtype=str, comments='%')
    no_epochs, _ = data.shape
    ok = True
    timestamps = data[:,0] + ' ' + data[:,1]
    seconds = np.array([datetime_to_utc(ts) for ts in timestamps])
    in_gnss = np.column_stack((seconds, data[:, 2:].astype(float)))
    in_gnss[:, 1:3] = np.deg2rad(in_gnss[:, 1:3])
    in_gnss[:,16] *= -1  # NEU -> NED  # RTKLIB is NEU
    return in_gnss, no_epochs, ok

def Read_IMU_data(filename):
    #Read_profile - inputs a motion profile in the following .csv format
    # Column 0: time (sec)
    # Column 1: acc_x(m/s^2)
    # Column 2: acc_y(m/s^2)
    # Column 3: acc_z(m/s^2)
    # Column 4: gyro_x(rad/s)
    # Column 5: gyro_y(rad/s)
    # Column 6: gyro_z(rad/s)
    # Additional optional fields
    # Column 7: mag_x (uT)
    # Column 8: mag_y (uT)
    # Column 9: mag_z (uT)
    #
    # Inputs:
    #   filename     Name of file to write
    #
    # Outputs:
    #   in_imu       Array of data from the file
    #   no_epochs    Number of epochs of data in the file
    #   ok           Indicates file has the expected number of columns
    
    in_imu = np.genfromtxt(filename, delimiter=',')
    no_epochs, no_columns = in_imu.shape
    ok = True
    return in_imu, no_epochs, ok

def Write_GNSS_data(filename, out_profile, out_KF_SD):
    hdrg = '%  GPST            latitude(deg) longitude(deg) height(m) Q         ns        sdn(m)    sde(m)    sdu(m)    sdne(m)   sdeu(m)   sdun(m)  age(s)     ratio     vn(m/s)   ve(m/s)    vu(m/s)    sdvn      sdve     sdvu       sdvne    sdveu      sdvun'
    fmtg = ['%s'] + ['%.7f'] * 25
    
    t_gnss = out_profile[:,0]
    coast = out_profile[:,10]
    n = len(t_gnss)
    date_time = np.empty(n, dtype='U25')
    for i, t in enumerate(t_gnss):
        date_time[i] = utc_secs_to_datetime(t)
    
    outg = np.empty((n,26), dtype=object)
    
    # convert position, velocity from ECEF to LLH, NED
    for i in range(len(out_profile)):
        outg[i,1], outg[i,2], outg[i,3], outg[i,14:17] = pv_ECEF_to_LLH(out_profile[i,1:4], out_profile[i,4:7])
        outg[i,16] *= -1  # NED->NEU for RTKLIB format
    
    outg[:,0] = date_time
    outg[:,1:3] = np.rad2deg(outg[:,1:3].astype(float))  # lat/long
    outg[:,4] = 1 + coast    # use fix status for coast
    outg[:,5] = 20   # nsats
    outg[:,6:14] = 0  # pos covariance
    outg[:,17:23] = 0 # vel covariance
    outg[:,23:26] = out_profile[:,7:10]  # orientation
    np.savetxt(filename, outg, header=hdrg, encoding='utf-8', fmt=fmtg, delimiter=' ', comments='')