#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 Project:    Sensor Fusion for GNSS/IMU (Loosely Coupled Kalman Filter)
 File:       Awesome_GINS_compare.py
 Author:     Tim Everett
 Copyright:  (c) 2025 Tim Everett
 License:    BSD 3 License (see LICENSE file for details)

 Description:
    Compare an Awesome-GINS truth.nav file with a GNSS_IMU solution.

    All user-selectable parameters are defined in the header below. The solution
    is interpolated to the truth timestamps over their common time interval.
    Position is compared in a local NED frame, velocity in NED, and attitude as
    roll/pitch/yaw with wrapped angular errors.
===============================================================================
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from imu_files import Read_GNSS_data
from imu_transforms import ECEF_to_NED, LLH_to_ECEF


####### Input files and comparison parameters ################################

# Select one of: "ICM20602", "ADIS16460", "ADIS16465", or "i300".

imu_name = "ADIS16465"                  # unitless sensor name


# Awesome-GINS truth files are stored in one directory per sensor.
truth_root = Path(r"C:\gps\data\benchmarks\Awesome-GINS")
truth_file = truth_root / imu_name / "truth.nav"

# GNSS_IMU output generated from the converted Awesome-GINS data.
solution_dir = Path(r"C:\gps\gps_imu\GNSS_IMU\data\awesome_gins")
solution_file = solution_dir / f"gnss_imu_{imu_name}_sf.pos"

gps_week = 2171                         # GPS week number, unitless
solution_time_offset = 0.0              # s, added to solution timestamps
comparison_duration = 0.0               # s from solution start (0=all available data)
plot_results = True                     # Boolean; display comparison plots
max_plot_points = 20000                 # samples per plotted trace (0=all)
print_precision = 6                     # decimal places in printed statistics
title = f"Awesome-GINS {imu_name}"      # unitless plot/report title

###############################################################################

GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
AXES = ("N", "E", "D")
ATTITUDE_AXES = ("Roll", "Pitch", "Yaw")


def llh_to_ned(llh_rad, origin_llh_rad):
    """Convert latitude/longitude (rad) and height (m) to local NED (m)."""
    ecef = LLH_to_ECEF(llh_rad)
    origin_ecef = LLH_to_ECEF(origin_llh_rad[None, :])[0]
    ned, _ = ECEF_to_NED(ecef, np.zeros_like(ecef), origin_ecef)
    return ned


def wrap_degrees(angle):
    """Wrap angles to [-180, 180) degrees."""
    return (np.asarray(angle) + 180.0) % 360.0 - 180.0


def unwrap_degrees(angles):
    """Remove 360-degree discontinuities independently on every column."""
    return np.rad2deg(np.unwrap(np.deg2rad(np.asarray(angles)), axis=0))


def interpolate_columns(target_time, source_time, values, angular=False):
    """Interpolate vector columns, optionally unwrapping angles first."""
    result = np.empty((len(target_time), values.shape[1]))
    for axis in range(values.shape[1]):
        source_values = values[:, axis]
        if angular:
            source_values = unwrap_degrees(source_values)
        result[:, axis] = np.interp(target_time, source_time, source_values)
    return result


def quaternion_from_rpy(rpy_deg):
    """Convert roll/pitch/yaw degrees to scalar-first ZYX quaternions."""
    half = np.deg2rad(rpy_deg) * 0.5
    roll, pitch, yaw = np.moveaxis(half, -1, 0)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.column_stack((
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ))


def attitude_angle_error(truth_rpy, solution_rpy):
    """Return the shortest 3-D rotation angle between two attitudes (deg)."""
    truth_q = quaternion_from_rpy(truth_rpy)
    solution_q = quaternion_from_rpy(solution_rpy)
    dot = np.abs(np.sum(truth_q * solution_q, axis=1))
    return np.rad2deg(2.0 * np.arccos(np.clip(dot, 0.0, 1.0)))


def print_vector_statistics(label, errors, axis_names, units):
    """Print bias, standard deviation, RMS, median absolute, and maximum error."""
    precision = print_precision
    print(f"\n{label} error ({units})")
    print(" Axis          Mean           SD          RMS   Median abs      Max abs")
    for axis, name in enumerate(axis_names):
        values = errors[:, axis]
        print(
            f" {name:>5s} "
            f"{np.mean(values):13.{precision}f} "
            f"{np.std(values):12.{precision}f} "
            f"{np.sqrt(np.mean(values ** 2)):12.{precision}f} "
            f"{np.median(np.abs(values)):12.{precision}f} "
            f"{np.max(np.abs(values)):12.{precision}f}"
        )


def print_norm_statistics(label, values, units):
    precision = print_precision
    print(
        f" {label}: RMS={np.sqrt(np.mean(values ** 2)):.{precision}f} {units}, "
        f"median={np.median(values):.{precision}f} {units}, "
        f"P95={np.percentile(values, 95):.{precision}f} {units}, "
        f"max={np.max(values):.{precision}f} {units}"
    )


def plot_comparison(time, truth, solution, errors, uncertainty, labels,
                    value_units, figure_title, plot_indices):
    """Plot each truth field beside its error and GNSS_IMU uncertainty."""
    figure, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    for axis, label in enumerate(labels):
        axes[axis, 0].plot(time[plot_indices], truth[plot_indices, axis],
                           label="Truth", linewidth=0.9)
        axes[axis, 0].plot(time[plot_indices], solution[plot_indices, axis],
                           label="GNSS_IMU", linewidth=0.8)
        axes[axis, 0].set_ylabel(f"{label} ({value_units})")
        axes[axis, 0].grid(True)
        axes[axis, 0].legend()

        axes[axis, 1].plot(time[plot_indices], errors[plot_indices, axis],
                           label="Error", linewidth=0.8)
        if uncertainty is not None:
            sd = uncertainty[plot_indices, axis]
            axes[axis, 1].plot(time[plot_indices], 3.0 * sd, "k--",
                               linewidth=0.7, label="±3σ")
            axes[axis, 1].plot(time[plot_indices], -3.0 * sd, "k--",
                               linewidth=0.7)
        axes[axis, 1].set_ylabel(f"d{label} ({value_units})")
        axes[axis, 1].grid(True)
        axes[axis, 1].legend()

    axes[-1, 0].set_xlabel("Time from comparison start (s)")
    axes[-1, 1].set_xlabel("Time from comparison start (s)")
    figure.suptitle(figure_title)
    figure.tight_layout()


def main():
    if not truth_file.is_file():
        raise FileNotFoundError(f"Truth file not found: {truth_file}")
    if not solution_file.is_file():
        raise FileNotFoundError(f"GNSS_IMU solution not found: {solution_file}")

    truth = np.loadtxt(truth_file)
    if truth.ndim != 2 or truth.shape[1] != 11:
        raise ValueError(f"Expected an 11-column truth.nav file: {truth_file}")

    solution, _, _ = Read_GNSS_data(str(solution_file))
    if solution.shape[1] < 29:
        raise ValueError(
            "The GNSS_IMU solution must include position, velocity, attitude, "
            "and their standard-deviation columns."
        )

    truth_time = truth[:, 1]
    week_start = (GPS_EPOCH + timedelta(weeks=gps_week)).timestamp()
    solution_time = solution[:, 0] - week_start + solution_time_offset
    if np.any(np.diff(truth_time) <= 0) or np.any(np.diff(solution_time) <= 0):
        raise ValueError("Truth and solution timestamps must be strictly increasing")

    overlap = ((truth_time >= solution_time[0]) &
               (truth_time <= solution_time[-1]))
    if comparison_duration > 0:
        overlap &= truth_time <= solution_time[0] + comparison_duration
    if not np.any(overlap):
        raise ValueError("Truth and solution files have no overlapping GPS times")
    truth = truth[overlap]
    truth_time = truth_time[overlap]

    truth_llh = truth[:, 2:5].copy()
    truth_llh[:, :2] = np.deg2rad(truth_llh[:, :2])
    solution_llh = solution[:, 1:4]
    origin_llh = truth_llh[0]
    truth_position = llh_to_ned(truth_llh, origin_llh)
    solution_position_all = llh_to_ned(solution_llh, origin_llh)

    solution_position = interpolate_columns(
        truth_time, solution_time, solution_position_all)
    solution_velocity = interpolate_columns(
        truth_time, solution_time, solution[:, 14:17])
    solution_attitude = interpolate_columns(
        truth_time, solution_time, solution[:, 23:26], angular=True)
    position_sd = interpolate_columns(
        truth_time, solution_time, solution[:, 6:9])
    velocity_sd = interpolate_columns(
        truth_time, solution_time, solution[:, 17:20])
    attitude_sd = interpolate_columns(
        truth_time, solution_time, solution[:, 26:29])

    truth_velocity = truth[:, 5:8]
    # Put both attitude streams on continuous and matching 360-degree branches.
    # This prevents yaw (and any other Euler channel) from drawing a false jump
    # or being interpolated through the long way at a 0/360-degree rollover.
    truth_attitude = unwrap_degrees(truth[:, 8:11])
    solution_attitude += 360.0 * np.round(
        (truth_attitude[0] - solution_attitude[0]) / 360.0)
    position_error = solution_position - truth_position
    velocity_error = solution_velocity - truth_velocity
    # Axis errors remain principal signed errors even though plots use
    # continuous angles.
    attitude_error = wrap_degrees(solution_attitude - truth_attitude)
    position_3d_error = np.linalg.norm(position_error, axis=1)
    velocity_3d_error = np.linalg.norm(velocity_error, axis=1)
    attitude_3d_error = attitude_angle_error(truth_attitude, solution_attitude)

    print(title)
    print(f"Truth file:    {truth_file}")
    print(f"Solution file: {solution_file}")
    print(f"Compared epochs: {len(truth_time)}")
    print(f"Time span: {truth_time[0]:.4f} to {truth_time[-1]:.4f} GPS TOW (s)")
    print_vector_statistics("Position (NED)", position_error, AXES, "m")
    print_norm_statistics("3-D position", position_3d_error, "m")
    print_vector_statistics("Velocity (NED)", velocity_error, AXES, "m/s")
    print_norm_statistics("3-D velocity", velocity_3d_error, "m/s")
    print_vector_statistics(
        "Attitude", attitude_error, ATTITUDE_AXES, "deg")
    print_norm_statistics("3-D attitude", attitude_3d_error, "deg")

    if plot_results:
        if max_plot_points and len(truth_time) > max_plot_points:
            plot_step = int(np.ceil(len(truth_time) / max_plot_points))
        else:
            plot_step = 1
        plot_indices = np.arange(0, len(truth_time), plot_step)
        elapsed = truth_time - truth_time[0]
        plot_comparison(
            elapsed, truth_position, solution_position, position_error,
            position_sd, AXES, "m", f"{title}: position", plot_indices)
        plot_comparison(
            elapsed, truth_velocity, solution_velocity, velocity_error,
            velocity_sd, AXES, "m/s", f"{title}: velocity", plot_indices)
        plot_comparison(
            elapsed, truth_attitude, solution_attitude, attitude_error,
            attitude_sd, ATTITUDE_AXES, "deg", f"{title}: attitude",
            plot_indices)
        plt.show()


if __name__ == "__main__":
    main()
