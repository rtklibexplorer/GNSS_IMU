#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 Project:    Sensor Fusion for GNSS/IMU (Loosely Coupled Kalman Filter)
 File:       _benchmark_kfgins.py
 Author:     Tim Everett
 Copyright:  (c) 2025 Tim Everett
 License:    BSD 3 License (see LICENSE file for details)

 Description:
    Reproduce the KF-GINS Awesome-GINS ADIS16465 60-second outage benchmark.

    All experiment paths and parameters are defined here so the run has no command-line
    options.  The KF-GINS navigation source and its published ADIS16465 tuning are not
    modified.  A benchmark-only GNSS file removes fixes in the requested intervals.
===============================================================================
"""

from __future__ import annotations

import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np


KF_ROOT = Path(r"C:\gps\gps_imu\KF-GINS")
KF_EXE = KF_ROOT / "bin" / "Release" / "KF-GINS.exe"
AWESOME_ROOT = Path(r"C:\gps\data\benchmarks\Awesome-GINS")
IMU_FILE = AWESOME_ROOT / "ADIS16465" / "ADIS16465.txt"
GNSS_FILE = AWESOME_ROOT / "GNSS RTK" / "GNSS_RTK.pos"
TRUTH_FILE = AWESOME_ROOT / "ADIS16465" / "truth.nav"
PUBLISHED_CONFIG = KF_ROOT.parent / "INSLIB" / "datasets" / "kfgins" / "ADIS16465" / "kf-gins.yaml"

RESULTS_DIR = Path(r"C:\gps\gps_imu\GNSS_IMU\results\kfgins_adis16465_60s")
CONTINUOUS_DIR = RESULTS_DIR / "continuous"
OUTAGE_DIR = RESULTS_DIR / "outages"
FILTERED_GNSS_FILE = RESULTS_DIR / "GNSS_RTK_five_60s_outages.pos"
CONTINUOUS_CONFIG = RESULTS_DIR / "kf-gins_continuous.yaml"
OUTAGE_CONFIG = RESULTS_DIR / "kf-gins_outages.yaml"
RESULTS_JSON = RESULTS_DIR / "benchmark_results.json"
COMMANDS_FILE = RESULTS_DIR / "commands.txt"

# Match the earlier GNSS_IMU/INSLIB benchmark: their converted IMU stream starts
# at the second raw increment because the first increment has no measured dt.
FIRST_PROCESSED_IMU_INDEX = 1
OUTAGE_WINDOWS_S = (
    (300.0, 360.0),
    (600.0, 660.0),
    (900.0, 960.0),
    (1200.0, 1260.0),
    (1500.0, 1560.0),
)
VALIDATION_WARMUP_S = 60.5
KF_GNSS_SNAP_TOLERANCE_S = 0.001

WGS84_A = 6378137.0
WGS84_E2 = 6.6943799901413165e-3


def read_first_processed_imu_time() -> float:
    with IMU_FILE.open("r", encoding="ascii") as stream:
        for index, line in enumerate(stream):
            if index == FIRST_PROCESSED_IMU_INDEX:
                return float(line.split()[0])
    raise RuntimeError("IMU file does not contain the required initial samples")


def make_filtered_gnss(first_imu_time: float) -> tuple[int, list[float]]:
    removed = 0
    returning = []
    kept_lines = []
    source_times = []

    with GNSS_FILE.open("r", encoding="ascii") as stream:
        for line in stream:
            if not line.strip():
                continue
            timestamp = float(line.split()[0])
            source_times.append(timestamp)
            relative = timestamp - first_imu_time
            in_outage = any(start <= relative < end for start, end in OUTAGE_WINDOWS_S)
            if in_outage:
                removed += 1
            else:
                kept_lines.append(line.rstrip("\r\n"))

    source_times_np = np.asarray(source_times)
    for _, end in OUTAGE_WINDOWS_S:
        candidates = source_times_np[source_times_np >= first_imu_time + end]
        if candidates.size == 0:
            raise RuntimeError(f"No returning GNSS fix after outage ending at {end} s")
        returning.append(float(candidates[0]))

    FILTERED_GNSS_FILE.write_text("\r\n".join(kept_lines) + "\r\n", encoding="ascii")
    return removed, returning


def make_config(path: Path, gnss_path: Path, output_dir: Path) -> None:
    # These are the published KF-GINS ADIS16465 settings, copied without tuning.
    text = f"""# KF-GINS ADIS16465 benchmark configuration
# Parameters copied unchanged from INSLIB/datasets/kfgins/ADIS16465/kf-gins.yaml.
imupath: \"{IMU_FILE.as_posix()}\"
gnsspath: \"{gnss_path.as_posix()}\"
outputpath: \"{output_dir.as_posix()}\"
imudatalen: 7
imudatarate: 200
starttime: 357473.0071
endtime: -1
initpos: [ 30.4604298391, 114.472503798, 23.08001 ]
initvel: [ 0.00257, -0.01352, 0.0057 ]
initatt: [ 1.72019, -0.10776, 274.48263 ]
initgyrbias: [ 0, 0, 0 ]
initaccbias: [ 0, 0, 0 ]
initgyrscale: [ 0, 0, 0 ]
initaccscale: [ 0, 0, 0 ]
initposstd: [ 0.1, 0.1, 0.2 ]
initvelstd: [ 0.05, 0.05, 0.05 ]
initattstd: [ 0.5, 0.5, 1 ]
imunoise:
  arw: [ 0.1, 0.1, 0.1 ]
  vrw: [ 0.1, 0.1, 0.1 ]
  gbstd: [ 25, 25, 25 ]
  abstd: [ 200, 200, 200 ]
  gsstd: [ 1000, 1000, 1000 ]
  asstd: [ 1000, 1000, 1000 ]
  corrtime: 1
antlever: [ -0.073, 0.302, 0.087 ]
"""
    path.write_text(text.replace("\n", "\r\n"), encoding="ascii")


def run_filter(config: Path) -> dict:
    start = time.perf_counter()
    completed = subprocess.run(
        [str(KF_EXE), str(config)],
        cwd=str(KF_ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        "command": f'"{KF_EXE}" "{config}"',
        "runtime_s": time.perf_counter() - start,
        "stdout_tail": completed.stdout[-2000:],
        "stderr": completed.stderr,
    }


def llh_to_ecef(llh_rad_m: np.ndarray) -> np.ndarray:
    lat = llh_rad_m[:, 0]
    lon = llh_rad_m[:, 1]
    height = llh_rad_m[:, 2]
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    radius = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    return np.column_stack((
        (radius + height) * cos_lat * np.cos(lon),
        (radius + height) * cos_lat * np.sin(lon),
        (radius * (1.0 - WGS84_E2) + height) * sin_lat,
    ))


def ecef_delta_to_ned(delta_ecef: np.ndarray, reference_llh: np.ndarray) -> np.ndarray:
    lat, lon = reference_llh[:2]
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    c_e_n = np.array((
        (-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat),
        (-sin_lon, cos_lon, 0.0),
        (-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat),
    ))
    return delta_ecef @ c_e_n.T


def interpolate_truth(truth: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    llh = np.empty((times.size, 3))
    attitude = np.empty((times.size, 3))
    for axis in range(3):
        llh[:, axis] = np.interp(times, truth[:, 1], truth[:, 2 + axis])
        truth_angle = np.rad2deg(np.unwrap(np.deg2rad(truth[:, 8 + axis])))
        attitude[:, axis] = np.interp(times, truth[:, 1], truth_angle)
    llh[:, :2] = np.deg2rad(llh[:, :2])
    return llh, attitude


def position_errors(solution: np.ndarray, truth: np.ndarray) -> np.ndarray:
    truth_llh, _ = interpolate_truth(truth, solution[:, 1])
    solution_llh = solution[:, 2:5].copy()
    solution_llh[:, :2] = np.deg2rad(solution_llh[:, :2])
    reference = truth_llh[0]
    return ecef_delta_to_ned(
        llh_to_ecef(solution_llh) - llh_to_ecef(truth_llh), reference)


def summarize_magnitudes(values: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(values)),
        "maximum": float(np.max(values)),
        "rms": float(np.sqrt(np.mean(values * values))),
    }


def score_continuous(solution: np.ndarray, truth: np.ndarray, first_imu_time: float) -> dict:
    errors = position_errors(solution, truth)
    _, truth_attitude = interpolate_truth(truth, solution[:, 1])
    attitude_error = solution[:, 8:11] - truth_attitude
    attitude_error = (attitude_error + 180.0) % 360.0 - 180.0

    def score(mask: np.ndarray) -> dict:
        err = errors[mask]
        att = attitude_error[mask]
        horizontal = np.linalg.norm(err[:, :2], axis=1)
        return {
            "epochs": int(np.count_nonzero(mask)),
            "time_span_s": [float(solution[mask, 1][0]), float(solution[mask, 1][-1])],
            "horizontal_position_rms_m": float(np.sqrt(np.mean(horizontal * horizontal))),
            "vertical_position_rms_m": float(np.sqrt(np.mean(err[:, 2] * err[:, 2]))),
            "position_3d_rms_m": float(np.sqrt(np.mean(np.sum(err * err, axis=1)))),
            "attitude_axis_rms_deg": {
                name: float(np.sqrt(np.mean(att[:, axis] * att[:, axis])))
                for axis, name in enumerate(("roll", "pitch", "yaw"))
            },
        }

    all_mask = np.ones(solution.shape[0], dtype=bool)
    warm_mask = solution[:, 1] >= first_imu_time + VALIDATION_WARMUP_S
    return {
        "all_processed_epochs": score(all_mask),
        "after_60_5_second_warmup": score(warm_mask),
        "published_reference": {
            "horizontal_position_rms_m": 0.017,
            "vertical_position_rms_m": 0.026,
        },
    }


def score_outages(
    solution: np.ndarray,
    truth: np.ndarray,
    first_imu_time: float,
    returning_fix_times: list[float],
) -> tuple[list[dict], dict]:
    rows = []
    for index, ((start, end), returning_fix) in enumerate(
        zip(OUTAGE_WINDOWS_S, returning_fix_times), start=1
    ):
        # KF-GINS fuses a GNSS fix at the preceding IMU epoch when the two
        # timestamps differ by less than TIME_ALIGN_ERR. Exclude that state,
        # even though its output timestamp is nominally before the fix.
        candidates = np.flatnonzero(
            solution[:, 1] <= returning_fix - KF_GNSS_SNAP_TOLERANCE_S)
        if candidates.size == 0:
            raise RuntimeError(f"No pre-update solution for returning fix {returning_fix}")
        endpoint_index = int(candidates[-1])
        endpoint = solution[endpoint_index: endpoint_index + 1]
        error = position_errors(endpoint, truth)[0]
        post_candidates = np.flatnonzero(solution[:, 1] >= returning_fix)
        if post_candidates.size == 0:
            raise RuntimeError(f"No post-update solution for returning fix {returning_fix}")
        post_index = int(post_candidates[0])
        post_endpoint = solution[post_index: post_index + 1]
        post_error = position_errors(post_endpoint, truth)[0]
        horizontal = float(np.linalg.norm(error[:2]))
        vertical = float(abs(error[2]))
        error_3d = float(np.linalg.norm(error))
        rows.append({
            "outage_index": index,
            "outage_relative_s": [start, end],
            "outage_absolute_s": [first_imu_time + start, first_imu_time + end],
            "returning_fix_time_s": returning_fix,
            "returning_fix_relative_s": returning_fix - first_imu_time,
            "endpoint_time_s": float(endpoint[0, 1]),
            "endpoint_relative_s": float(endpoint[0, 1] - first_imu_time),
            "pre_return_margin_s": float(returning_fix - endpoint[0, 1]),
            "north_error_m": float(error[0]),
            "east_error_m": float(error[1]),
            "down_error_m": float(error[2]),
            "horizontal_error_m": horizontal,
            "vertical_error_m": vertical,
            "position_3d_error_m": error_3d,
            "first_post_return_state_time_s": float(post_endpoint[0, 1]),
            "first_post_return_state_relative_s": float(post_endpoint[0, 1] - first_imu_time),
            "first_post_return_position_3d_error_m": float(np.linalg.norm(post_error)),
        })

    horizontal = np.asarray([row["horizontal_error_m"] for row in rows])
    vertical = np.asarray([row["vertical_error_m"] for row in rows])
    error_3d = np.asarray([row["position_3d_error_m"] for row in rows])
    return rows, {
        "horizontal_error_m": summarize_magnitudes(horizontal),
        "vertical_error_m": summarize_magnitudes(vertical),
        "position_3d_error_m": summarize_magnitudes(error_3d),
    }


def main() -> None:
    for path in (RESULTS_DIR, CONTINUOUS_DIR, OUTAGE_DIR):
        path.mkdir(parents=True, exist_ok=True)

    required = (KF_EXE, IMU_FILE, GNSS_FILE, TRUTH_FILE, PUBLISHED_CONFIG)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing benchmark inputs:\n" + "\n".join(missing))

    first_imu_time = read_first_processed_imu_time()
    removed, returning_fix_times = make_filtered_gnss(first_imu_time)
    make_config(CONTINUOUS_CONFIG, GNSS_FILE, CONTINUOUS_DIR)
    make_config(OUTAGE_CONFIG, FILTERED_GNSS_FILE, OUTAGE_DIR)

    continuous_run = run_filter(CONTINUOUS_CONFIG)
    outage_run = run_filter(OUTAGE_CONFIG)

    truth = np.loadtxt(TRUTH_FILE)
    continuous_solution = np.loadtxt(CONTINUOUS_DIR / "KF_GINS_Navresult.nav")
    outage_solution = np.loadtxt(OUTAGE_DIR / "KF_GINS_Navresult.nav")
    validation = score_continuous(continuous_solution, truth, first_imu_time)
    endpoints, summary = score_outages(
        outage_solution, truth, first_imu_time, returning_fix_times
    )

    commands = (
        "# Generated by diagnostics/run_kfgins_adis16465_60s.py\r\n"
        f'"{KF_EXE}" "{CONTINUOUS_CONFIG}"\r\n'
        f'"{KF_EXE}" "{OUTAGE_CONFIG}"\r\n'
    )
    COMMANDS_FILE.write_text(commands, encoding="ascii")

    result = {
        "experiment": "KF-GINS Awesome-GINS ADIS16465 five 60-second GNSS outages",
        "source_unchanged": True,
        "published_config": str(PUBLISHED_CONFIG),
        "first_processed_imu_time_s": first_imu_time,
        "outage_timing": "relative to first processed/converted IMU sample; start inclusive, end exclusive",
        "outage_windows_s": [list(window) for window in OUTAGE_WINDOWS_S],
        "gnss_source": str(GNSS_FILE),
        "gnss_velocity_available": False,
        "gnss_fixes_removed": removed,
        "returning_fix_times_s": returning_fix_times,
        "continuous_run": continuous_run,
        "outage_run": outage_run,
        "continuous_validation": validation,
        "outage_endpoints": endpoints,
        "outage_summary": summary,
        "comparison_3d_endpoint_error_m": {
            "GNSS_IMU_NHC_off": [0.927, 5.946, 8.170, 13.179, 2.182],
            "GNSS_IMU_NHC_on": [1.015, 0.914, 1.874, 2.351, 0.279],
            "INSLIB_NHC_off": [10.893, 7.240, 3.272, 15.555, 3.236],
            "INSLIB_corrected_NHC_on": [4.447837667, 4.313595264, 2.002986307, 3.463063962, 2.711990252],
            "KF_GINS": [row["position_3d_error_m"] for row in endpoints],
        },
        "files": {
            "continuous_config": str(CONTINUOUS_CONFIG),
            "outage_config": str(OUTAGE_CONFIG),
            "filtered_gnss": str(FILTERED_GNSS_FILE),
            "commands": str(COMMANDS_FILE),
        },
    }
    RESULTS_JSON.write_text(
        (json.dumps(result, indent=2) + "\n").replace("\n", "\r\n"),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
