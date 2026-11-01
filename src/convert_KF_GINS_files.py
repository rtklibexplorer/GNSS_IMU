#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
===============================================================================
 Project:    Sensor Fusion for GNSS/IMU (Loosely Coupled Kalman Filter)
 File:       convert_KF_GINS_files.py
 Author:     Tim Everett
 Copyright:  (c) 2025 Tim Everett
 License:    BSD 3 License (see LICENSE file for details)

 Description:
    Convert all four Awesome-GINS sensors to GNSS_IMU input files.

    The source IMU fields are angle and velocity *increments*. Dividing each
    increment by its measured sample interval produces the interval-average gyro
    rate and specific force required by GNSS_IMU; it does not interpolate an
    instantaneous value. The GNSS source contains positions only, so GNSS velocity
    is estimated from successive positions as an interval-average velocity. Its
    uncertainty remains large enough to select a position-only Kalman update.

    From the repository root, run::

        python src\convert_KF_GINS_files.py

    Edit the parameter block at the top of this file to select the source,
    destination, sensors, or GPS week. The source archives must be extracted
    before running the converter.
===============================================================================
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from imu_transforms import LLH_to_ECEF, compute_C_e_n


####### Input/output and conversion parameters ################################

# Awesome-GINS source folder populated by get_Awesome_GINS_data.py.
repoDir = Path(__file__).resolve().parent.parent
dataDir = repoDir / "data" / "awesome_gins"

# A separate folder lets run_benchmark.py detect conversion with one check.
outDir = dataDir / "converted"

# Shared extracted GNSS input file, relative to dataDir.
gnss_infile = Path("GNSS RTK") / "GNSS_RTK.pos"

# Sensors to convert. Remove entries to convert only a subset.
imus = ("ICM20602", "ADIS16460", "ADIS16465", "i300")

# The Awesome-GINS recording date corresponds to GPS week 2171.
gps_week = 2171

# Standard gravity converts average specific force from m/s^2 to g.
standard_gravity = 9.80665

# The first GNSS row initializes velocity covariance but is not used as a
# measurement update. The vehicle is stationary at the start of this dataset.
initial_velocity_SD = 1.0       # m/s, initial velocity uncertainty per axis

# Subsequent position-derived velocities retain large uncertainty. This SD
# exceeds the configured threshold, selecting a position-only Kalman update.
unused_velocity_SD = 1000.0     # m/s, velocity uncertainty per axis

###############################################################################

GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)


def _gps_week_tow_to_unix_like(week: int, tow: np.ndarray) -> np.ndarray:
    """Map GPST week/TOW to GNSS_IMU's calendar-based numeric time scale."""
    return (GPS_EPOCH + timedelta(weeks=week)).timestamp() + tow


def _format_gpst(seconds: float) -> str:
    dt = datetime.fromtimestamp(seconds, tz=timezone.utc)
    return dt.strftime("%Y/%m/%d %H:%M:%S.") + f"{dt.microsecond // 1000:03d}"


def _position_velocity_neu(tow: np.ndarray, llh_deg: np.ndarray) -> np.ndarray:
    """Calculate interval-average NEU velocity from successive positions."""
    llh_rad = llh_deg.copy()
    llh_rad[:, :2] = np.deg2rad(llh_rad[:, :2])
    ecef = LLH_to_ECEF(llh_rad)
    velocity_ecef = np.diff(ecef, axis=0) / np.diff(tow)[:, None]

    velocity_neu = np.empty_like(llh_deg)
    for index, velocity in enumerate(velocity_ecef, start=1):
        # Resolve each interval-average velocity at the interval midpoint.
        latitude = 0.5 * (llh_rad[index - 1, 0] + llh_rad[index, 0])
        longitude = 0.5 * (llh_rad[index - 1, 1] + llh_rad[index, 1])
        velocity_ned = compute_C_e_n(latitude, longitude) @ velocity
        velocity_neu[index] = velocity_ned * np.array([1.0, 1.0, -1.0])

    # The first epoch has no preceding position; use the first interval mean.
    velocity_neu[0] = velocity_neu[1]
    return velocity_neu


def convert_gnss(source: Path, destination: Path, gps_week: int) -> None:
    data = np.loadtxt(source)
    if data.ndim != 2 or data.shape[1] != 7 or len(data) < 2:
        raise ValueError(f"Expected at least two 7-column GNSS rows in {source}")
    tow, llh, position_sd = data[:, 0], data[:, 1:4], data[:, 4:7]
    if np.any(np.diff(tow) <= 0):
        raise ValueError(f"GNSS times are not strictly increasing in {source}")

    seconds = _gps_week_tow_to_unix_like(gps_week, tow)

    output = np.zeros((len(data), 23), dtype=object)
    output[:, 0] = [_format_gpst(value) for value in seconds]
    output[:, 1:4] = llh
    output[:, 4] = 1          # supplied solutions are RTK fixed
    output[:, 5] = 0          # satellite count is not supplied
    output[:, 6:9] = position_sd
    output[:, 14:17] = _position_velocity_neu(tow, llh)  # m/s NEU
    output[:, 17:20] = unused_velocity_SD
    output[0, 17:20] = initial_velocity_SD

    header = (
        "%  GPST            latitude(deg) longitude(deg) height(m) Q ns "
        "sdn(m) sde(m) sdu(m) sdne(m) sdeu(m) sdun(m) age(s) ratio "
        "vn(m/s) ve(m/s) vu(m/s) sdvn sdve sdvu sdvne sdveu sdvun"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(destination, output, header=header, comments="",
               fmt=["%s"] + ["%.9f"] * 22)


def convert_imu(source: Path, destination: Path, gps_week: int) -> None:
    data = np.loadtxt(source)
    if data.ndim != 2 or data.shape[1] != 7 or len(data) < 2:
        raise ValueError(f"Expected at least two 7-column IMU rows in {source}")
    dt = np.diff(data[:, 0])
    if np.any(dt <= 0):
        raise ValueError(f"IMU times are not strictly increasing in {source}")

    # Every output row is the mean over the interval ending at that timestamp.
    # The first source row has no preceding timestamp, so it is omitted.
    output = np.zeros((len(data) - 1, 10))
    output[:, 0] = _gps_week_tow_to_unix_like(gps_week, data[1:, 0])
    output[:, 1:4] = data[1:, 4:7] / dt[:, None] / standard_gravity
    output[:, 4:7] = data[1:, 1:4] / dt[:, None]

    header = (
        "% UNIX time(s),accX(g),accY(g),accZ(g),gyroX(rad/s),gyroY(rad/s),"
        "gyroZ(rad/s),magX(uT),magY(uT),magZ(uT)"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(destination, output, delimiter=",", header=header, comments="",
               fmt=["%.6f"] + ["%.10f"] * 9)


def main() -> None:
    gnss_source = dataDir / gnss_infile
    for imu in imus:
        imu_source = dataDir / imu / f"{imu}.txt"
        stem = f"{imu}_sf"
        convert_gnss(gnss_source, outDir / f"gnss_{stem}.pos", gps_week)
        convert_imu(imu_source, outDir / f"imu_{stem}.csv", gps_week)
        print(f"Converted {imu}: {outDir}")


if __name__ == "__main__":
    main()
