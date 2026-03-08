#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 Project:    Sensor Fusion for GNSS/IMU (Loosely Coupled Kalman Filter)
 File:       convert_ubx_file.py
 Author:     Tim Everett
 Copyright:  (c) 2025 Tim Everett
 License:    BSD 3 License (see LICENSE file for details)

 Description:  Script to convert data files from the KF-GINS dataset 2 which is 
available at https://github.com/i2Nav-WHU/KF-GINS-Matlab/tree/main/dataset2 
into an RTKLIB GNSS solution file and an IMU CSV file for input into the sensor
 fusion solution.

Instructions for use:

To generate the solution:
1) Run get_KF_GINS_data.py to get the data from the repo
2) Run this script 
3) Comment/uncomment lines in the GNSS_IMU.py header to select these files 
4) Run GNSS_IMU.py

To compare to KF-GINS results
1) Run KF-GINS-Matlab in Matlab or Octave on dataset2
2) Copy the NavResult_GNSSVEL.nav results file into the data/KF_GINS folder
3) Run KF_GINS_compare.py
===============================================================================
"""
import pylab as plt
from imu_transforms import utc_secs_to_datetime
from datetime import datetime
import numpy as np
from numpy.linalg import norm
from os.path import join

####### Select input file and configuration parameters  #######################################

dataDir = r'..\data\KF_GINS'
gnss_infile = 'GNSS-POSVEL.txt'
imu_infile = 'ADIS16465.txt'
out_name = 'ADIS16465'
date = '5/28/2025'  # arbitrary date since GNSS file only contains TOW
g = 9.79363  # gravity (m/sec^2)

# Read and parse GNSS data file
data = np.genfromtxt(join(dataDir, gnss_infile))
n = len(data)
ts = data[:,0]
llh = data[:,1:4]
posstd = data[:,4:7]
vel = data[:,7:10]
velstd = data[:,10:13]

# No fix status in data, so use horiz posstd to guess fix status
fix = np.ones(len(vel))  # 1=FIX
ix = np.where(norm(posstd[:,:2], axis=1)>.05) # find float
fix[ix] = 2 # 2= FLOAT
ix = np.where(norm(posstd[:,:2], axis=1)>.25) # find single
fix[ix] = 5 # 5 =SINGLE

# Put GNSS data into output array
outg = np.zeros((n,23), dtype=object)
ts0 = datetime.strptime(date, "%m/%d/%Y").timestamp()
for i in range(n):
    outg[i,0] = utc_secs_to_datetime(ts0 + ts[i])
outg[:,1:4] = llh
outg[:,4] = fix   # fix status
outg[:,5] = 22  # num sats
outg[:,6:9] = posstd
outg[:,9:14] = 0
outg[:,14:17] = vel * [1,1,-1]   # NED -> NEU
outg[:,17:20] = velstd
outg[:,20:23] = 0

# Save GNSS data in RTKLIB format
hdrg = '%  GPST            latitude(deg) longitude(deg) height(m) Q         ns        sdn(m)    sde(m)    sdu(m)    sdne(m)   sdeu(m)   sdun(m)  age(s)     ratio     vn(m/s)   ve(m/s)    vu(m/s)    sdvn      sdve     sdvu       sdvne    sdveu      sdvun'
fmtg = ['%s'] + ['%.7f'] * 22
gnssFileOut = join(dataDir, 'gnss_' + out_name + '_sf.pos')
np.savetxt(gnssFileOut, outg, header=hdrg, encoding='utf-8', fmt=fmtg, comments='')

# Read IMU data file
data = np.genfromtxt(join(dataDir, imu_infile))
n = len(data) - 1
secs = data[:,0]  # seconds 
gyro =  data[:,1:4]  # incremental gyro measurementsin rad
acc = data[:,4:7] / g # incremental acc measurements in m/sec

# Convert incremental data to absolute
dsecs =np.tile(np.diff(secs), (3,1)).T
acc_filt = acc[1:] / dsecs
gyro_filt = gyro[1:] / dsecs
secs = secs[1:]

# Put IMU data into output array
ts0 = datetime.strptime(date, "%m/%d/%Y").timestamp() 
outi = np.zeros((n,11))
outi[:,0] = ts0 + secs
outi[:,1:4] = acc_filt
outi[:,4:7] = gyro_filt 
outi[:,7:10] = 0
outi[:,10] = 0

# Save IMU data to file
hdr = '% UNIX time(s),    accX(g),  accY(g),  accZ(g),  gyroX(r/s),gyroY(r/s),gyroZ(r/s),magX(uT),magY(uT),magZ(uT), unused'
imuFileOut = join(dataDir, 'imu_' + out_name + '_sf.csv')
np.savetxt(imuFileOut, outi, encoding='utf-8', fmt='%.7f', delimiter=',', header=hdr, comments='')
    
# # Plot data (optinal)
# plt.figure()
# plt.title('Velocity')
# plt.plot(ts,outg[:,14:17] * [1,1,-1], '.-')  # NEU -> NED
# plt.grid()
# plt.legend(['N', 'E', 'D'])
# plt.show()

