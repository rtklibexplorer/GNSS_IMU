#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 Project:    Sensor Fusion for GNSS/IMU (Loosely Coupled Kalman Filter)
 File:       calibrate_nhc_misalignment.py
 Author:     Tim Everett
 Copyright:  (c) 2025 Tim Everett
 License:    BSD 3 License (see LICENSE file for details)

 Description:
    Calibrate INSLIB or GNSS_IMU NHC misalignment from estimated motion.

    GNSS_IMU is the default. Select ESTIMATOR below, or pass --estimator INSLIB
    or --estimator GNSS_IMU.
    The selected filter runs with continuous GNSS and NHC disabled. Estimated
    NED velocity is resolved into its estimated propagation-body frame, and
    pitch/yaw corrections are summarized using the same motion-direction fit.
    No production configuration is modified. Results are saved separately
    under benchmark_results/<estimator>_nhc_calibration. Each report includes
    configuration values for both GNSS_IMU and INSLIB, regardless of source.
===============================================================================
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
import importlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np


SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent

# Default when run without arguments. Either "INSLIB" or "GNSS_IMU".
ESTIMATOR = "GNSS_IMU"
MAX_SIMULTANEOUS_RUNS = 4

MINIMUM_SPEEDS_MPS = (0.5, 1.0, 2.0, 5.0)
MAXIMUM_YAW_RATES_DEGPS = (1.0, 2.0, 5.0, 10.0)
RECOMMENDED_SELECTION = "speed_gt_2_mps_yaw_rate_lt_2_degps"


def result_paths(estimator: str) -> tuple[Path, Path, Path]:
    if estimator not in ("INSLIB", "GNSS_IMU"):
        raise ValueError(f"Unknown estimator: {estimator}")
    prefix = estimator.lower()
    directory = REPO_ROOT / "benchmark_results" / f"{prefix}_nhc_calibration"
    return (directory,
            directory / f"{prefix}_nhc_misalignment_comparison.json",
            directory / f"{prefix}_nhc_misalignment_comparison.md")


class SolutionObserver:
    """Capture INSLIB's ready-state velocity and attitude at every epoch."""

    def __init__(self):
        self.time_us = []
        self.velocity_ned = []
        self.rpy_rad = []

    def on_epoch(self, replay):
        if not replay.nav.is_ready():
            return
        velocity = replay.nav.velocity_ned()
        rpy = replay.nav.rpy_ins()
        if velocity is None or rpy is None:
            return
        row = np.asarray((*velocity, *rpy), dtype=float)
        if not np.all(np.isfinite(row)):
            return
        self.time_us.append(int(replay.t))
        self.velocity_ned.append(row[:3])
        self.rpy_rad.append(row[3:])

    def on_pos_error(self, _replay, _error):
        pass

    def on_finish(self, _replay):
        pass


def robust_summary(values: np.ndarray) -> dict:
    """Return the median, mean, and 16th/84th-percentile spread in degrees."""
    q16, median, q84 = np.percentile(values, (16, 50, 84))
    return {
        "median_deg": float(median),
        "mean_deg": float(np.mean(values)),
        "q16_deg": float(q16),
        "q84_deg": float(q84),
        "robust_half_width_deg": float((q84 - q16) / 2.0),
    }


def body_to_vehicle_rotation(angles_deg: np.ndarray) -> np.ndarray:
    sys.path.insert(0, str(SRC_DIR))
    from imu_transforms import Euler_to_CTM
    return Euler_to_CTM(np.deg2rad(angles_deg))


def rotation_difference_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = body_to_vehicle_rotation(first) @ body_to_vehicle_rotation(second).T
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))


def capture_inslib_solution(imu_name: str, info: dict):
    sys.path.insert(0, str(SRC_DIR))
    import run_benchmark as benchmark

    prepared = benchmark.prepare_inslib(info, imu_name)
    helper = benchmark.load_module(
        f"calibrate_inslib_{info['inslib_dir']}", benchmark.INSLIB_HELPER)
    helper.inslib_root = benchmark.INSLIB_ROOT
    helper.dataset_dir = prepared["dataset_dir"]
    replay_tool, replay_core = helper.install_inslib_imports()
    spec, data_dir = replay_tool.load_config(str(prepared["nhc_off"]))
    inputs = replay_core.load_inputs(spec, data_dir, [], log=lambda _message: None)
    navigator = replay_core.make_navigator(spec, inputs)
    observer = SolutionObserver()
    replay = replay_core.Replay(spec, inputs, navigator, observers=[observer])
    log_path = result_paths("INSLIB")[0] / imu_name / "inslib_continuous_nhc_off.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        with log_path.open("w", encoding="utf-8", newline="\n") as log:
            with redirect_stdout(log), redirect_stderr(log):
                replay.run()
    finally:
        navigator.close()
    runtime = time.perf_counter() - started
    if not observer.time_us:
        raise RuntimeError(f"INSLIB produced no ready solution for {imu_name}")
    return (
        np.asarray(observer.time_us, dtype=np.float64) / 1.0e6,
        np.asarray(observer.velocity_ned),
        np.asarray(observer.rpy_rad),
        runtime,
        str(prepared["nhc_off"]),
        str(log_path),
    )


def capture_gnss_imu_solution(imu_name: str, info: dict):
    """Run GNSS_IMU at every epoch without replacing existing solutions."""
    import run_benchmark as benchmark
    from imu_files import Read_GNSS_data

    run_dir = result_paths("GNSS_IMU")[0] / imu_name
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in (f"imu_{info['file_in']}.csv", f"gnss_{info['file_in']}.pos"):
        shutil.copyfile(benchmark.CONVERTED_DATA_DIR / name, run_dir / name)

    # Fix the physical output point and direction in a generated wrapper,
    # preserving the selected production configuration.
    config_path = run_dir / "calibration_config.py"
    source_config = SRC_DIR / info["config"]
    config_path.write_text(
        "from runpy import run_path\n"
        f"globals().update(run_path({str(source_config)!r}))\n"
        "out_frame = 'imu'\nrun_dir = [1]\n", encoding="utf-8")
    log_path = run_dir / "gnss_imu_continuous_nhc_off.log"
    command = [
        sys.executable, str(benchmark.GNSS_IMU_SCRIPT),
        "--config", str(config_path), "--data-dir", str(run_dir),
        "--file-in", info["file_in"], "--outage-duration", "0",
        "--nhc-enable", "false", "--out-step", "1", "--full-data", "--no-plots",
    ]
    started = time.perf_counter()
    benchmark.run_command(command, log_path)
    runtime = time.perf_counter() - started
    solution_path = run_dir / f"gnss_imu_{info['file_in']}.pos"
    solution, _, _ = Read_GNSS_data(str(solution_path))
    if solution.ndim != 2 or solution.shape[1] < 26:
        raise RuntimeError(f"GNSS_IMU solution lacks velocity/attitude for {imu_name}")
    return (solution[:, 0], solution[:, 14:17], np.deg2rad(solution[:, 23:26]),
            runtime, str(config_path), str(log_path))


def velocity_in_body_frame(velocity_ned: np.ndarray,
                           rpy_rad: np.ndarray) -> np.ndarray:
    """Resolve NED velocity into FRD using the estimated body attitude."""
    from imu_transforms import Euler_to_CTM

    ned_to_body = np.asarray([Euler_to_CTM(rpy) for rpy in rpy_rad])
    return np.einsum("nij,nj->ni", ned_to_body, velocity_ned)


def run_case(item: tuple[str, dict], estimator: str = ESTIMATOR) -> dict:
    imu_name, info = item
    if estimator == "INSLIB":
        capture = capture_inslib_solution
    elif estimator == "GNSS_IMU":
        capture = capture_gnss_imu_solution
    else:
        raise ValueError(f"Unknown estimator: {estimator}")
    time_s, velocity_ned, rpy_rad, runtime, config_path, log_path = capture(imu_name, info)

    if (len(time_s) < 2 or not np.all(np.isfinite(time_s)) or
            np.any(np.diff(time_s) <= 0.0)):
        raise RuntimeError(f"{estimator} requires increasing solution times for {imu_name}")
    if not (np.all(np.isfinite(velocity_ned)) and np.all(np.isfinite(rpy_rad))):
        raise RuntimeError(f"{estimator} produced non-finite velocity/attitude for {imu_name}")

    config = importlib.import_module(Path(info["config"]).stem)
    gnss_imu_angles = np.asarray(config.nhc_misalign, dtype=float)
    velocity_body = velocity_in_body_frame(velocity_ned, rpy_rad)
    yaw_rate_degps = np.rad2deg(
        np.gradient(np.unwrap(rpy_rad[:, 2]), time_s))

    # These are the original GNSS_IMU calibration formulas and signs.
    yaw_correction_deg = np.rad2deg(np.arctan2(
        velocity_body[:, 1], velocity_body[:, 0]))
    pitch_correction_deg = -np.rad2deg(np.arctan2(
        velocity_body[:, 2],
        np.hypot(velocity_body[:, 0], velocity_body[:, 1])))

    selections = {}
    median_dt = float(np.median(np.diff(time_s)))
    for minimum_speed in MINIMUM_SPEEDS_MPS:
        for maximum_yaw_rate in MAXIMUM_YAW_RATES_DEGPS:
            selected = (
                (velocity_body[:, 0] > minimum_speed) &
                (np.abs(yaw_rate_degps) < maximum_yaw_rate)
            )
            key = (f"speed_gt_{minimum_speed:g}_mps_"
                   f"yaw_rate_lt_{maximum_yaw_rate:g}_degps")
            sample_count = int(np.count_nonzero(selected))
            if sample_count == 0:
                raise RuntimeError(f"No samples for {imu_name}: {key}")
            selections[key] = {
                "sample_count": sample_count,
                "duration_s": float(sample_count * median_dt),
                "yaw_correction": robust_summary(
                    yaw_correction_deg[selected]),
                "pitch_correction": robust_summary(
                    pitch_correction_deg[selected]),
                "body_velocity_median_mps": np.median(
                    velocity_body[selected], axis=0).tolist(),
            }

    recommended = selections[RECOMMENDED_SELECTION]
    alignment = np.array([
        0.0,
        recommended["pitch_correction"]["median_deg"],
        recommended["yaw_correction"]["median_deg"],
    ])
    body_to_vehicle = body_to_vehicle_rotation(alignment)
    result = {
        "imu": imu_name,
        "estimator": estimator,
        "configured_alignment_deg": gnss_imu_angles.tolist(),
        "alignment_deg": alignment.tolist(),
        "gnss_imu_alignment_deg": alignment.tolist(),
        "inslib_alignment_deg": alignment.tolist(),
        "configuration_values": {
            "GNSS_IMU": {"nhc_misalign": alignment.tolist()},
            "INSLIB": {"automotive_lateral_alignment_rpy_deg": alignment.tolist()},
        },
        "body_to_vehicle_matrix": body_to_vehicle.tolist(),
        "inslib_body_to_vehicle_column_major":
            body_to_vehicle.flatten(order="F").tolist(),
        "component_difference_deg": (alignment - gnss_imu_angles).tolist(),
        "rotation_difference_deg": rotation_difference_deg(
            alignment, gnss_imu_angles),
        "runtime_s": runtime,
        "config": config_path,
        "log": log_path,
        "recommended_selection": RECOMMENDED_SELECTION,
        "recommended_sample_count": recommended["sample_count"],
        "selections": selections,
    }
    if estimator == "INSLIB":
        # Keep INSLIB capture metadata alongside the common output fields.
        result.update({
            "inslib_runtime_s": runtime,
            "inslib_config": config_path,
        })
    return result


def write_report(results: list[dict], estimator: str = ESTIMATOR) -> None:
    results_dir, result_json, result_markdown = result_paths(estimator)
    payload = {
        "experiment": f"Independent {estimator} NHC misalignment calibration",
        "estimator": estimator,
        "production_configs_modified": False,
        "method": {
            "description": (
                f"{estimator} NED velocity rotated into its estimated FRD "
                "body frame; robust median corrections make lateral and "
                "vertical velocity approach zero."),
            "matches_gnss_imu_truth_motion_procedure": True,
            "continuous_gnss": True,
            "nhc_enabled_during_capture": False,
            "gnss_velocity_updates_enabled": False,
            "roll_observable_from_motion_vector": False,
            "output_convention": (
                "Propagation-body FRD to vehicle FRD, Euler_to_CTM ZYX R/P/Y "
                "in degrees. INSLIB replay's automotive_lateral_alignment_rpy_deg "
                "uses the same angles; the native matrix uses column-major order."),
            "recommended_selection": RECOMMENDED_SELECTION,
            "minimum_speeds_mps": list(MINIMUM_SPEEDS_MPS),
            "maximum_yaw_rates_degps": list(MAXIMUM_YAW_RATES_DEGPS),
        },
        "results": results,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    result_json.write_text(json.dumps(payload, indent=2) + "\n",
                           encoding="utf-8", newline="\r\n")

    lines = [
        f"# Independent {estimator} NHC misalignment calibration",
        "",
        ("This uses the original GNSS_IMU motion-direction procedure: "
         "continuous GNSS, NHC disabled, forward speed above 2 m/s, yaw "
         "rate below 2 deg/s, and median pitch/yaw corrections. The source "
         f"navigation solution is {estimator}; roll is held at zero because "
         "the motion vector does not independently determine it."),
        "",
        f"| IMU | Configured GNSS_IMU R/P/Y [deg] | Calculated {estimator} R/P/Y [deg] | Calculated minus configured [deg] | Rotation difference [deg] | Samples |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    for row in results:
        def vector(values):
            return ", ".join(f"{value:+.4f}" for value in values)

        lines.append(
            f"| {row['imu']} | {vector(row['configured_alignment_deg'])} | "
            f"{vector(row['alignment_deg'])} | "
            f"{vector(row['component_difference_deg'])} | "
            f"{row['rotation_difference_deg']:.4f} | "
            f"{row['recommended_sample_count']} |")
    lines.extend([
        "",
        "## Configuration values for both code sets",
        "",
        (f"The source navigation solution is {estimator}. Both forms below "
         "describe that same fitted body-to-vehicle rotation, not independent "
         "calibrations of the two estimators. GNSS_IMU and the INSLIB replay "
         "configuration use the same R/P/Y degree values. The JSON also includes "
         "the body-to-vehicle matrix and its column-major native INSLIB form."),
        "",
    ])
    for row in results:
        angles = "[" + ", ".join(f"{value:.6f}" for value in row["alignment_deg"]) + "]"
        lines.extend([
            f"### {row['imu']}", "", "GNSS_IMU configuration:", "",
            "```python", f"nhc_misalign = {angles}", "```", "",
            "INSLIB replay configuration:", "", "```yaml",
            f"automotive_lateral_alignment_rpy_deg: {angles}", "```", "",
        ])
    lines.extend([
        "No production configuration was changed.",
        "",
    ])
    result_markdown.write_text("\n".join(lines), encoding="utf-8",
                               newline="\r\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate NHC pitch/yaw alignment.")
    parser.add_argument("--estimator", type=str.upper,
                        choices=("INSLIB", "GNSS_IMU"), default=ESTIMATOR,
                        help="Navigation code to calibrate (default: %(default)s)")
    estimator = parser.parse_args().estimator
    sys.path.insert(0, str(SRC_DIR))
    import run_benchmark as benchmark

    if estimator == "INSLIB":
        required = [benchmark.INSLIB_ROOT, benchmark.INSLIB_HELPER]
        required.extend(benchmark.INSLIB_ROOT / "datasets" / "kfgins" /
                        info["inslib_dir"] / "config.yaml"
                        for info in benchmark.IMUS.values())
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing INSLIB calibration files:\n" +
                                    "\n".join(missing))
    results_dir, result_json, result_markdown = result_paths(estimator)
    benchmark.ensure_data()
    results_dir.mkdir(parents=True, exist_ok=True)
    completed = []
    print(f"Calibrating {estimator} with continuous GNSS and NHC disabled", flush=True)
    with ProcessPoolExecutor(max_workers=MAX_SIMULTANEOUS_RUNS) as pool:
        futures = {pool.submit(run_case, item, estimator): item[0]
                   for item in benchmark.IMUS.items()}
        for future in as_completed(futures):
            result = future.result()
            completed.append(result)
            print(f"{result['imu']}: complete", flush=True)
    order = {name: index for index, name in enumerate(benchmark.IMUS)}
    completed.sort(key=lambda row: order[row["imu"]])
    write_report(completed, estimator)
    print(f"JSON: {result_json}", flush=True)
    print(f"Report: {result_markdown}", flush=True)


if __name__ == "__main__":
    main()
