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
     are Unix seconds

===============================================================================
"""
import struct
from datetime import datetime
import numpy as np
import pylab as plt
from imu_math import LP_Filt

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
    start_time = time_time + time_sec  # Unix time
    start_datetime = datetime.utcfromtimestamp(start_time)
    print(f"Start Unix time: {start_datetime.isoformat(timespec='microseconds')}")

    # Read tag records containing host elapsed time and input file position.
    # The tags provide the absolute Unix-time reference, while the IMU's own
    # millisecond counter provides the more stable sample-to-sample timing.
    tag_tick = []
    tag_fpos = []
    while True:
        block = f.read(8)
        if len(block) < 8:
            break
        tick, fpos = struct.unpack("<II", block)
        tag_tick.append(tick)
        tag_fpos.append(fpos)

tag_tick = np.array(tag_tick)
tag_fpos = np.array(tag_fpos)

# Find the input file position at the start and end of each IMU sample.
with open(filepath, "rb") as f:
    line_fpos = np.cumsum([len(line) for line in f.readlines()])

# Read data file
data = np.genfromtxt(filepath, skip_footer=1, skip_header=1, delimiter=',')
n = len(data)
acc =  data[:,0:3]  # acc measurements in g
gyro = data[:,3:6]  # gyro measurements in rad/sec
imu_time_ms = data[:,6]  # IMU clock in milliseconds since power-up
sample_start_fpos = line_fpos[:-2]  # account for skipped first and last lines


def unwrap_millisecond_counter(counter_ms):
    """Unwrap a monotonically increasing uint32 millisecond counter."""
    counter_ms = np.asarray(counter_ms, dtype=float)
    unwrapped = counter_ms.copy()
    wraps = 0.0
    for i in range(1, len(unwrapped)):
        candidate = counter_ms[i] + wraps
        if candidate < unwrapped[i - 1] - 2**31:
            wraps += 2**32
            candidate = counter_ms[i] + wraps
        unwrapped[i] = candidate
    return unwrapped


def fit_imu_clock_to_tags(imu_seconds, sample_positions, tag_ticks,
                          tag_positions):
    """Fit tag elapsed time = offset + scale * IMU elapsed time.

    RTKLIB tags timestamp byte reception, which can contain serial/USB and OS
    jitter. The IMU counter is therefore used for relative sample timing; the
    tags determine its Unix-time offset and clock-rate correction.
    """
    valid = ((tag_positions >= sample_positions[0]) &
             (tag_positions <= sample_positions[-1]))
    if np.count_nonzero(valid) < 2:
        raise ValueError("Not enough time tags overlap the retained IMU data")

    # Associate each host tag with the IMU clock at the corresponding byte
    # position. A sample timestamp describes measurement generation, so use
    # the start of its CSV line rather than the arrival of its final byte.
    tag_imu_seconds = np.interp(tag_positions[valid], sample_positions,
                                imu_seconds)
    tag_elapsed_seconds = tag_ticks[valid] / 1000.0

    # Center the fit for numerical conditioning and iteratively reject large
    # host-arrival-time residuals using a robust MAD scale estimate.
    x0 = tag_imu_seconds[0]
    x = tag_imu_seconds - x0
    keep = np.ones(len(x), dtype=bool)
    for _ in range(5):
        scale, intercept = np.polyfit(x[keep], tag_elapsed_seconds[keep], 1)
        residual = tag_elapsed_seconds - (intercept + scale * x)
        median = np.median(residual[keep])
        mad = np.median(np.abs(residual[keep] - median))
        robust_sigma = 1.4826 * mad
        if robust_sigma < 1e-6:
            break
        new_keep = np.abs(residual - median) <= 4.0 * robust_sigma
        if np.array_equal(new_keep, keep) or np.count_nonzero(new_keep) < 2:
            break
        keep = new_keep

    scale, intercept = np.polyfit(x[keep], tag_elapsed_seconds[keep], 1)
    residual = tag_elapsed_seconds - (intercept + scale * x)
    elapsed = intercept + scale * (imu_seconds - x0)
    return elapsed, scale, residual, keep

# Low pass filter data
gyro_filt = LP_Filt(gyro, gyro_cutoff, fs=fs, order=4)
acc_filt = LP_Filt(acc, acch_cutoff, fs=fs, order=4)
acc_filt[:,2] = LP_Filt(acc[:,2], accv_cutoff, fs=fs, order=4)

# Nav_equations_ECEF propagates over the interval ending at each timestamp.
# Convert the filtered instantaneous endpoint samples to trapezoidal interval
# averages. Keep the first sample unchanged because no preceding sample exists.
gyro_avg = gyro_filt.copy()
acc_avg = acc_filt.copy()
gyro_avg[1:] = 0.5 * (gyro_filt[:-1] + gyro_filt[1:])
acc_avg[1:] = 0.5 * (acc_filt[:-1] + acc_filt[1:])

# Preserve the IMU clock's relative timing and estimate only its absolute
# offset and rate correction from the noisier RTKLIB host-reception tags.
imu_seconds = unwrap_millisecond_counter(imu_time_ms) / 1000.0
elapsed, clock_scale, tag_residual, tag_inlier = fit_imu_clock_to_tags(
    imu_seconds, sample_start_fpos, tag_tick, tag_fpos)
ts = start_time + elapsed

print(f"IMU clock scale: {clock_scale:.9f} "
      f"({(clock_scale - 1) * 1e6:+.1f} ppm)")
print(f"Time-tag fit: {np.count_nonzero(tag_inlier)}/{len(tag_inlier)} inliers, "
      f"RMS={np.sqrt(np.mean(tag_residual[tag_inlier]**2)) * 1000:.2f} ms, "
      f"max={np.max(np.abs(tag_residual[tag_inlier])) * 1000:.2f} ms")

outi = np.zeros((n,11))
outi[:,0] = ts
outi[:,1:4] = acc_avg
outi[:,4:7] = np.deg2rad(gyro_avg)
outi[:,7:10] = 0
outi[:,10] = 0

# Save data
hdr = '% UNIX time(s),    accX(g),  accY(g),  accZ(g),  gyroX(r/s),gyroY(r/s),gyroZ(r/s),magX(uT),magY(uT),magZ(uT), unused'
imuFileOut = filepath[:-4] + '_sf.csv'
np.savetxt(imuFileOut, outi[int(fs*out_skip):], encoding='utf-8', fmt='%.7f', delimiter=',', header=hdr, comments='')

# Plot data
plt.figure()
plt.title('Accel (m/sec^2)')
plt.plot(ts, acc, '.-')
plt.plot(ts, acc_avg, '.-')
plt.legend(['X','Y','Z', 'Xfilt', 'Yfilt','Zfilt'])
plt.grid()
plt.show()

plt.figure()
plt.title('Gyro (deg/sec)')
plt.plot(ts, gyro, '.-')
plt.plot(ts, gyro_avg, '.-')
plt.legend(['X','Y','Z', 'Xfilt', 'Yfilt','Zfilt'])
plt.grid()
plt.show()

