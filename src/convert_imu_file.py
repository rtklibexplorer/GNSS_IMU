#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 Project:    Sensor Fusion for GNSS/IMU (Loosely Coupled Kalman Filter)
 File:       convert_imu_file.py
 Author:     Tim Everett
 Copyright:  (c) 2025 Tim Everett
 License:    BSD 3 License (see LICENSE file for details)

 Description:
     Script to convert an IMU data file and associated time tag file collected 
     with RTKLIB into an input file for the sensor fusion solution.  Time stamps
     are in GPST time

===============================================================================
"""
import struct
from datetime import datetime
import numpy as np
import pylab as plt
from imu_math import Zero_Phase_LP

####### Select input file and configuration parameters  #######################################

# drive data
filepath = r'..\data\drive_0708\imu_1934.csv'

# walk data
#filepath = r'..\data\walk_0827\imu_1730.csv'

hdr_len = 0
acch_cutoff = 10  # horiz accel filter cutoff freq
accv_cutoff = 5  # vert accel filter cutoff freq
gyro_cutoff = 10 # gyro filter cutoff freq
fs = 100 # IMU sample rate
out_skip = 0 # secs to skip at beginning of file

# read time tags file
with open(filepath + '.tag', "rb") as f:
    # Parse start time from header
    tagh = f.read(60)
    tick_f = struct.unpack("<I", f.read(4))[0]
    time_time = struct.unpack("<I", f.read(4))[0]
    time_sec = struct.unpack("<d", f.read(8))[0]
    start_time = time_time + time_sec  # GPST time
    gpst_time = datetime.utcfromtimestamp(start_time)
    print(f"Start GPST time: {gpst_time.isoformat(timespec='microseconds')}")

    # Parse end time from data
    while True:
        block = f.read(8)
        if len(block) < 8:
            break
        last_block = block
    tick, fpos = struct.unpack("<II", last_block)
    end_time = start_time + tick / 1000

# Read data file
data = np.genfromtxt(filepath, skip_footer=1, skip_header=1, delimiter=',')
n = len(data)
acc =  data[:,0:3]  # acc measurements in g
gyro = data[:,3:6]  # gyro measurements in rad/sec
sec = data[:,6] / 1000  # seconds since IMU power-up

# Low pass filter data
gyro_filt = Zero_Phase_LP(gyro, gyro_cutoff, fs=fs, order=4)
acc_filt = Zero_Phase_LP(acc, acch_cutoff, fs=fs, order=4)
acc_filt[:,2] = Zero_Phase_LP(acc[:,2], accv_cutoff, fs=fs, order=4)

# Tweak sum of time stamps to match total time
sec -= sec[0]  # time relative to start of log 
ts = start_time + sec * (end_time - start_time) / (sec[-1] - sec[0])

outi = np.zeros((n,11))
outi[:,0] = ts
outi[:,1:4] = acc_filt
outi[:,4:7] = np.deg2rad(gyro_filt) 
outi[:,7:10] = 0
outi[:,10] = 0

# Save data
imuFileOut = filepath[:-4] + '_sf.csv'
np.savetxt(imuFileOut, outi[int(fs*out_skip):], encoding='utf-8', fmt='%.7f', delimiter=',')

# Plot data
plt.figure()
plt.title('Accel (m/sec^2)')
plt.plot(ts, acc, '.-')
plt.plot(ts, acc_filt, '.-')
plt.legend(['X','Y','Z', 'Xfilt', 'Yfilt','Zfilt'])
plt.grid()
plt.show()

plt.figure()
plt.title('Gyro (deg/sec)')
plt.plot(ts, gyro, '.-')
plt.plot(ts, gyro_filt, '.-')
plt.legend(['X','Y','Z', 'Xfilt', 'Yfilt','Zfilt'])
plt.grid()
plt.show()

