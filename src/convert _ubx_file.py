#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 Project:    Sensor Fusion for GNSS/IMU (Loosely Coupled Kalman Filter)
 File:       convert_ubx_file.py
 Author:     Tim Everett
 Copyright:  (c) 2025 Tim Everett
 License:    BSD 3 License (see LICENSE file for details)

 Description:
     Script to convert a u-blox binary output file with NAV-PVT messages and 
     associated time tag file collected with RTKLIB into an RTKLIB solution file
     for input into the sensor fusion solution.

===============================================================================
"""
from pyubx2 import UBXReader
import pylab as plt
from imu_transforms import utc_secs_to_datetime
from datetime import datetime, timezone, timedelta
import struct
import numpy as np

####### Select input file and configuration parameters  #######################################

# drive data
infile = r'..\data\drive_0708\gnss_1934.ubx'

# walk data
# infile = r'..\data\walk_0827\gnss_1730.ubx'


fs = 4  # GNSS samples per second
out_skip = 0 # secs to skip at beginning of file


# Constants
fix_map = np.array([5, 2, 1]) 
LEAP_SECONDS = 18

# Read time tags file
with open(infile + '.tag', "rb") as f:
    # Parse header
    tagh = f.read(60)
    tick_f = struct.unpack("<I", f.read(4))[0]
    time_time = struct.unpack("<I", f.read(4))[0]
    time_sec = struct.unpack("<d", f.read(8))[0]
    start_time = time_time + time_sec
    gpst_time = datetime.utcfromtimestamp(start_time)
    print(f"Start GPST time: {gpst_time.isoformat(timespec='microseconds')}")

    # Parse end time
    first_block = f.read(8)
    while True:
        block = f.read(8)
        if len(block) < 8:
            break
        last_block = block
    tick, first_fpos = struct.unpack("<II", first_block)
    first_time = start_time + tick / 1000
    tick, last_fpos = struct.unpack("<II", last_block)
    last_time = start_time + tick / 1000

# Read data file
with open(infile, 'rb') as f:
    ubr = UBXReader(f, validate=True)
    
    rows, secs = [], []
    for i, (raw, msg) in enumerate(ubr):
        if msg.identity == 'NAV-PVT':
            # Combine UTC date/time fields into a full datetime object
            utc_dt = datetime(
                msg.year, msg.month, msg.day,
                msg.hour, msg.min, msg.second,
                tzinfo=timezone.utc
            ) + timedelta(microseconds=msg.nano / 1000.0)
            sec = utc_dt.timestamp() + LEAP_SECONDS   # UTC -> GPST
            
            llh = [msg.lat, msg.lon,  msg.hMSL / 1000]
            vel = np.array([msg.velN, msg.velE, msg.velD]) / 1000
            sAcc = (msg.sAcc / 1000) / np.sqrt(2)
            herr = msg.hAcc / 1000 / np.sqrt(2)
            hdg = msg.headMot
            hdgAcc = msg.headAcc
            
            rows.append([
                utc_secs_to_datetime(sec),
                llh[0], llh[1], llh[2],
                fix_map[msg.carrSoln],
                msg.numSV,
                herr,  herr, msg.vAcc / 1000,
                0, 0, 0, 0, 0,   # covariance matrix, age , ratio
                vel[0], vel[1], -vel[2], # NED -> NEU for RTKLIB format
                sAcc, sAcc, sAcc,
                0, 0, 0])
            secs.append(sec)

# Save output file
outg = np.array(rows, dtype=object)
hdrg = '%  GPST            latitude(deg) longitude(deg) height(m) Q         ns        sdn(m)    sde(m)    sdu(m)    sdne(m)   sdeu(m)   sdun(m)  age(s)     ratio     vn(m/s)   ve(m/s)    vu(m/s)    sdvn      sdve     sdvu       sdvne    sdveu      sdvun'
fmtg = ['%s'] + ['%.7f'] * 22
outfile = infile[:-4] + '_sf.pos'
np.savetxt(outfile, outg[int(fs*out_skip):], header=hdrg, encoding='utf-8', fmt=fmtg,
           delimiter=' ', comments='')

# Plot data
plt.figure()
plt.title('Velocity')
plt.plot(secs,outg[:,14:17] * [1,1,-1], '.-')  # NEU -> NED
plt.grid()
plt.legend(['N', 'E', 'D'])
plt.show()

