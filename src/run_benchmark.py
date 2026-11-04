#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 Project:    Sensor Fusion for GNSS/IMU (Loosely Coupled Kalman Filter)
 File:       run_benchmark.py
 Author:     Tim Everett
 Copyright:  (c) 2025 Tim Everett
 License:    BSD 3 License (see LICENSE file for details)

 Description:
    Run the Awesome-GINS outage and continuous truth benchmarks.

    There are no command-line options. Edit only the parameter block below. The
    continuous benchmark reproduces the INSLIB README comparison for the four
    non-Leador IMUs using a 60.5-second warmup and continuous GNSS.
===============================================================================
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import numpy as np


######## Installed programs and benchmark parameters #########################

INSLIB_ROOT = Path(r"C:\gps\gps_imu\INSLIB")
KF_GINS_ROOT = Path(r"C:\gps\gps_imu\KF-GINS")

FIRST_OUTAGE_START = 300.0       # s after the first converted IMU sample
OUTAGE_DURATION = 60.0           # s without GNSS
OUTAGE_GAP = 90.0                # s with GNSS between outages in one run
SCHEDULE_START_GAP = 50.0        # s between the first outage of each run
MAX_SIMULTANEOUS_RUNS = 4     # 1=sequential; up to four IMUs concurrently
RUN_OUTAGE_BENCHMARK = True
RUN_CONTINUOUS_TRUTH_BENCHMARK = True

# Set run=False to reuse that IMU's existing result JSON when making the plot.
IMUS = {
    "ADIS16465": {"run": True, "config": "ADIS16465_config.py",
                  "file_in": "ADIS16465_sf", "data_dir": "ADIS16465",
                  "imu_file": "ADIS16465.txt", "inslib_dir": "ADIS16465"},
    "HGuide-i300": {"run": True, "config": "i300_config.py",
                     "file_in": "i300_sf", "data_dir": "i300",
                     "imu_file": "i300.txt", "inslib_dir": "i300"},
    "ADIS16460": {"run": True, "config": "ADIS16460_config.py",
                  "file_in": "ADIS16460_sf", "data_dir": "ADIS16460",
                  "imu_file": "ADIS16460.txt", "inslib_dir": "ADIS16460"},
    "ICM20602": {"run": True, "config": "ICM20602_config.py",
                  "file_in": "ICM20602_sf", "data_dir": "ICM20602",
                  "imu_file": "ICM20602.txt", "inslib_dir": "ICM20602"},
}

##############################################################################

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent
RAW_DATA_DIR = REPO_ROOT / "data" / "awesome_gins"
CONVERTED_DATA_DIR = RAW_DATA_DIR / "converted"
RESULTS_DIR = REPO_ROOT / "benchmark_results"
CONTINUOUS_RESULTS_DIR = RESULTS_DIR / "continuous_truth"
GPS_WEEK = 2171
CONTINUOUS_WARMUP_S = 60.5

GNSS_IMU_SCRIPT = SRC_DIR / "GNSS_IMU.py"
GET_DATA_SCRIPT = SRC_DIR / "get_Awesome_GINS_data.py"
CONVERT_SCRIPT = SRC_DIR / "convert_KF_GINS_files.py"
INSLIB_HELPER = SRC_DIR / "_benchmark_inslib.py"
KF_HELPER = SRC_DIR / "_benchmark_kfgins.py"
KF_EXE = KF_GINS_ROOT / "bin" / "Release" / "KF-GINS.exe"

METRICS = ("Mean", "Median", "RMS", "Maximum")
COLORS = {"GNSS_IMU": "#3978b7", "INSLIB": "#e1812c",
          "KF-GINS": "#3b923f"}
HATCHES = {"GNSS_IMU": "", "INSLIB": "//", "KF-GINS": "xx"}

# Published INSLIB README values. Leador A15 is intentionally excluded.
README_REFERENCE = {
    "ICM20602": {
        "INSLIB": (0.019, 0.063, 0.024, 0.025, 0.179),
        "KF-GINS": (0.020, 0.058, 0.032, 0.026, 0.182),
    },
    "ADIS16460": {
        "INSLIB": (0.021, 0.029, 0.027, 0.051, 0.219),
        "KF-GINS": (0.031, 0.031, 0.046, 0.099, 0.241),
    },
    "ADIS16465": {
        "INSLIB": (0.017, 0.024, 0.014, 0.015, 0.106),
        "KF-GINS": (0.017, 0.026, 0.014, 0.013, 0.089),
    },
    "HGuide-i300": {
        "INSLIB": (0.016, 0.030, 0.012, 0.012, 0.068),
        "KF-GINS": (0.017, 0.028, 0.011, 0.011, 0.070),
    },
}
CONTINUOUS_METRIC_KEYS = (
    "horizontal_position_rms_m", "vertical_position_rms_m",
    "roll_rms_deg", "pitch_rms_deg", "yaw_rms_deg",
)

WGS84_A = 6378137.0
WGS84_E2 = 6.6943799901413165e-3
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
WEEK_START = (GPS_EPOCH + timedelta(weeks=GPS_WEEK)).timestamp()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_command(command: list[str], log_file: Path) -> None:
    """Run one external command and retain a reproducible console log."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8", newline="\n") as log:
        log.write("Command:\n" + subprocess.list2cmdline(command) + "\n\n")
        log.flush()
        subprocess.run(command, cwd=SRC_DIR, check=True,
                       stdout=log, stderr=subprocess.STDOUT, text=True)


def ensure_data() -> None:
    """Download or convert only when its containing folder is absent."""
    if not RAW_DATA_DIR.is_dir():
        print("Awesome-GINS data folder not found; downloading it.", flush=True)
        subprocess.run([sys.executable, str(GET_DATA_SCRIPT)],
                       cwd=SRC_DIR, check=True)
    if not CONVERTED_DATA_DIR.is_dir():
        print("Converted data folder not found; running the converter.",
              flush=True)
        subprocess.run([sys.executable, str(CONVERT_SCRIPT)],
                       cwd=SRC_DIR, check=True)


def schedule_starts() -> tuple[float, ...]:
    """Derive the interleaved schedules; no separate count is required."""
    period = OUTAGE_DURATION + OUTAGE_GAP
    if SCHEDULE_START_GAP <= 0.0 or period <= 0.0:
        raise ValueError("Outage gaps and duration must be positive")
    count = math.ceil(period / SCHEDULE_START_GAP)
    return tuple(FIRST_OUTAGE_START + i * SCHEDULE_START_GAP
                 for i in range(count))


def make_windows(first_start: float, data_duration: float):
    windows = []
    start = first_start
    period = OUTAGE_DURATION + OUTAGE_GAP
    while start + OUTAGE_DURATION <= data_duration:
        windows.append((start, start + OUTAGE_DURATION))
        start += period
    return tuple(windows)


def raw_imu_times(info: dict) -> tuple[float, float]:
    times = np.loadtxt(RAW_DATA_DIR / info["data_dir"] / info["imu_file"],
                       usecols=0)
    if len(times) < 2 or np.any(np.diff(times) <= 0.0):
        raise ValueError(f"Invalid raw IMU timestamps for {info['file_in']}")
    # The converter omits row zero because its integration interval is unknown.
    return float(times[1]), float(times[-1] - times[1])


def first_benchmark_gnss_fix(info: dict) -> float:
    """Return the first fix admitted by the INSLIB dataset converter."""
    first_processed_imu, _ = raw_imu_times(info)
    fix_times = np.loadtxt(
        RAW_DATA_DIR / "GNSS RTK" / "GNSS_RTK.pos", usecols=0)
    eligible = fix_times[fix_times > first_processed_imu]
    if not len(eligible):
        raise RuntimeError("No GNSS fix follows the first processed IMU epoch")
    return float(eligible[0])


def summarize(endpoints: list[dict]) -> dict:
    result = {"outage_count": len(endpoints)}
    for key in ("horizontal_error_m", "vertical_error_m",
                "position_3d_error_m"):
        values = np.asarray([row[key] for row in endpoints], dtype=float)
        result[key] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "rms": float(np.sqrt(np.mean(values * values))),
            "maximum": float(np.max(values)),
        }
    return result


def llh_to_ecef(llh: np.ndarray) -> np.ndarray:
    lat, lon, height = llh
    radius = WGS84_A / math.sqrt(1.0 - WGS84_E2 * math.sin(lat) ** 2)
    return np.array([
        (radius + height) * math.cos(lat) * math.cos(lon),
        (radius + height) * math.cos(lat) * math.sin(lon),
        (radius * (1.0 - WGS84_E2) + height) * math.sin(lat),
    ])


def ecef_delta_to_ned(delta: np.ndarray, reference_llh: np.ndarray):
    lat, lon = reference_llh[:2]
    return np.array([
        -math.sin(lat) * math.cos(lon) * delta[0]
        - math.sin(lat) * math.sin(lon) * delta[1]
        + math.cos(lat) * delta[2],
        -math.sin(lon) * delta[0] + math.cos(lon) * delta[1],
        -math.cos(lat) * math.cos(lon) * delta[0]
        - math.cos(lat) * math.sin(lon) * delta[1]
        - math.sin(lat) * delta[2],
    ])


def llh_to_ecef_rows(llh: np.ndarray) -> np.ndarray:
    """Convert N-by-3 latitude/longitude/height rows to ECEF."""
    lat, lon, height = llh.T
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    radius = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    return np.column_stack((
        (radius + height) * cos_lat * np.cos(lon),
        (radius + height) * cos_lat * np.sin(lon),
        (radius * (1.0 - WGS84_E2) + height) * sin_lat,
    ))


def ecef_delta_to_local_ned_rows(delta: np.ndarray,
                                  reference_llh: np.ndarray) -> np.ndarray:
    """Rotate each ECEF error at its own truth latitude/longitude."""
    lat, lon = reference_llh[:, 0], reference_llh[:, 1]
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)
    dx, dy, dz = delta.T
    return np.column_stack((
        -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz,
        -sin_lon * dx + cos_lon * dy,
        -cos_lat * cos_lon * dx - cos_lat * sin_lon * dy - sin_lat * dz,
    ))


def interpolate_continuous_truth(truth: np.ndarray, times: np.ndarray):
    llh = np.column_stack([
        np.interp(times, truth[:, 1], truth[:, column])
        for column in range(2, 5)
    ])
    llh[:, :2] = np.deg2rad(llh[:, :2])
    attitude = np.column_stack([
        np.interp(times, truth[:, 1],
                  np.rad2deg(np.unwrap(np.deg2rad(truth[:, column]))))
        for column in range(8, 11)
    ])
    return llh, attitude


def score_continuous_arrays(times: np.ndarray, solution_llh: np.ndarray,
                            solution_attitude: np.ndarray,
                            truth: np.ndarray, first_fix: float) -> dict:
    """Score continuous output using the INSLIB README warmup convention."""
    lo = max(float(truth[0, 1]), first_fix + CONTINUOUS_WARMUP_S)
    hi = float(truth[-1, 1])
    mask = np.isfinite(times) & (times >= lo) & (times <= hi)
    if not np.any(mask):
        raise RuntimeError("No continuous solution epochs remain after warmup")
    times = times[mask]
    solution_llh = solution_llh[mask]
    solution_attitude = solution_attitude[mask]
    truth_llh, truth_attitude = interpolate_continuous_truth(truth, times)
    error_ned = ecef_delta_to_local_ned_rows(
        llh_to_ecef_rows(solution_llh) - llh_to_ecef_rows(truth_llh),
        truth_llh)
    attitude_error = ((solution_attitude - truth_attitude + 180.0) %
                      360.0) - 180.0
    horizontal_squared = error_ned[:, 0] ** 2 + error_ned[:, 1] ** 2
    result = {
        "epochs": int(len(times)),
        "time_span_tow_s": [float(times[0]), float(times[-1])],
        "warmup_s": CONTINUOUS_WARMUP_S,
        "horizontal_position_rms_m": float(np.sqrt(np.mean(horizontal_squared))),
        "vertical_position_rms_m": float(np.sqrt(np.mean(error_ned[:, 2] ** 2))),
    }
    for axis, name in enumerate(("roll", "pitch", "yaw")):
        result[f"{name}_rms_deg"] = float(
            np.sqrt(np.mean(attitude_error[:, axis] ** 2)))
    return result


def score_gnss_imu(solution_file: Path, info: dict, first_imu: float,
                   windows) -> list[dict]:
    """Score the last state strictly before each returning GNSS fix."""
    sys.path.insert(0, str(SRC_DIR))
    from imu_files import Read_GNSS_data

    solution, _, _ = Read_GNSS_data(str(solution_file))
    converted_gnss, _, _ = Read_GNSS_data(
        str(CONVERTED_DATA_DIR / f"gnss_{info['file_in']}.pos"))
    truth = np.loadtxt(RAW_DATA_DIR / info["data_dir"] / "truth.nav")
    truth_time = truth[:, 1]

    endpoints = []
    for index, (start, end) in enumerate(windows, start=1):
        returning = converted_gnss[converted_gnss[:, 0] >= first_imu + end, 0]
        if not len(returning):
            raise RuntimeError(f"No returning GNSS fix after {end:g} seconds")
        return_time = float(returning[0])
        state_indices = np.flatnonzero(solution[:, 0] < return_time)
        if not len(state_indices):
            raise RuntimeError(f"No pre-return GNSS_IMU state for {end:g} seconds")
        row = solution[state_indices[-1]]
        tow = row[0] - WEEK_START
        truth_llh = np.array([
            math.radians(np.interp(tow, truth_time, truth[:, 2])),
            math.radians(np.interp(tow, truth_time, truth[:, 3])),
            np.interp(tow, truth_time, truth[:, 4]),
        ])
        error_ned = ecef_delta_to_ned(
            llh_to_ecef(row[1:4]) - llh_to_ecef(truth_llh), truth_llh)
        horizontal = float(math.hypot(error_ned[0], error_ned[1]))
        vertical = float(abs(error_ned[2]))
        endpoints.append({
            "outage_index": index,
            "outage_s": [start, end],
            "endpoint_relative_s": float(row[0] - first_imu),
            "returning_fix_relative_s": float(return_time - first_imu),
            "pre_fix_margin_s": float(return_time - row[0]),
            "north_error_m": float(error_ned[0]),
            "east_error_m": float(error_ned[1]),
            "down_error_m": float(error_ned[2]),
            "horizontal_error_m": horizontal,
            "vertical_error_m": vertical,
            "position_3d_error_m": float(math.hypot(horizontal, vertical)),
        })
    return endpoints


def run_gnss_imu(info: dict, imu_name: str, first_start: float,
                 nhc_enabled: bool, windows, first_imu: float,
                 run_dir: Path) -> dict:
    label = "on" if nhc_enabled else "off"
    command = [
        sys.executable, str(GNSS_IMU_SCRIPT),
        "--config", str(SRC_DIR / info["config"]),
        "--data-dir", str(CONVERTED_DATA_DIR),
        "--file-in", info["file_in"],
        "--outage-start", str(first_start),
        "--outage-duration", str(OUTAGE_DURATION),
        "--outage-gap", str(OUTAGE_GAP),
        "--nhc-enable", str(nhc_enabled).lower(),
        "--out-step", "1", "--full-data", "--no-plots",
    ]
    started = time.perf_counter()
    log = run_dir / f"gnss_imu_nhc_{label}.log"
    run_command(command, log)
    runtime = time.perf_counter() - started
    solution = CONVERTED_DATA_DIR / f"gnss_imu_{info['file_in']}.pos"
    endpoints = score_gnss_imu(solution, info, first_imu, windows)
    return {"command": subprocess.list2cmdline(command), "runtime_s": runtime,
            "log": str(log), "solution": str(solution),
            "endpoints": endpoints, "summary": summarize(endpoints)}


def prepare_inslib(info: dict, imu_name: str) -> dict:
    helper = load_module(f"prepare_inslib_{info['inslib_dir']}", INSLIB_HELPER)
    dataset_dir = RESULTS_DIR / imu_name / "inslib_dataset"
    raw_dir = RAW_DATA_DIR / info["data_dir"]
    helper.inslib_root = INSLIB_ROOT
    helper.dataset_dir = dataset_dir
    helper.raw_imu_file = raw_dir / info["imu_file"]
    helper.raw_gnss_file = RAW_DATA_DIR / "GNSS RTK" / "GNSS_RTK.pos"
    helper.raw_truth_file = raw_dir / "truth.nav"
    helper.base_config_file = (INSLIB_ROOT / "datasets" / "kfgins" /
                               info["inslib_dir"] / "config.yaml")
    helper.convert_dataset()
    off = helper.make_config(False)
    on = helper.make_config(True)

    config = load_module(f"config_{info['inslib_dir']}",
                         SRC_DIR / info["config"])
    alignment = [float(value) for value in config.nhc_misalign]
    text = on.read_text(encoding="utf-8")
    setting = f"automotive_lateral_alignment_rpy_deg: {alignment}"
    if re.search(r"(?m)^automotive_lateral_alignment_rpy_deg:", text):
        text = re.sub(r"(?m)^automotive_lateral_alignment_rpy_deg:.*$",
                      setting, text, count=1)
    else:
        anchor = "automotive_lateral_constraint: 1  # NHC experiment toggle\n"
        if anchor not in text:
            raise RuntimeError("INSLIB NHC alignment insertion point not found")
        text = text.replace(anchor, anchor + setting + "\n", 1)
    on.write_text(text, encoding="utf-8", newline="\n")
    return {"dataset_dir": dataset_dir, "nhc_off": off, "nhc_on": on,
            "alignment_deg": alignment}


def run_inslib(info: dict, imu_name: str, first_start: float, windows,
               prepared: dict, run_dir: Path) -> dict:
    helper = load_module(f"inslib_{info['inslib_dir']}_{first_start:g}",
                         INSLIB_HELPER)
    helper.inslib_root = INSLIB_ROOT
    helper.dataset_dir = prepared["dataset_dir"]
    helper.results_dir = run_dir / "inslib"
    helper.results_dir.mkdir(parents=True, exist_ok=True)
    helper.outage_windows_s = windows
    replay_tool, replay_core = helper.install_inslib_imports()
    cases = {}
    for key, enabled in (("nhc_off", False), ("nhc_on", True)):
        log_file = helper.results_dir / f"console_{key}.txt"
        with log_file.open("w", encoding="utf-8", newline="\n") as log:
            with redirect_stdout(log):
                cases[key] = helper.run_case(
                    prepared[key], enabled, replay_tool, replay_core, key)
    return cases


def make_kf_config(base: Path, output: Path, imu_file: Path,
                   gnss_file: Path, output_dir: Path) -> None:
    """Copy the published per-IMU KF-GINS config, changing paths only."""
    text = base.read_text(encoding="utf-8")
    for key, value in {
            "imupath": imu_file.as_posix(),
            "gnsspath": gnss_file.as_posix(),
            "outputpath": output_dir.as_posix() + "/"}.items():
        text, count = re.subn(rf"(?m)^{key}:\s*.*$",
                              f'{key}: "{value}"', text, count=1)
        if count != 1:
            raise RuntimeError(f"Could not replace {key} in {base}")
    output.write_text(text, encoding="ascii", newline="\r\n")


def run_kf_gins(info: dict, imu_name: str, first_start: float, windows,
                run_dir: Path) -> dict:
    helper = load_module(f"kf_{info['inslib_dir']}_{first_start:g}", KF_HELPER)
    raw_dir = RAW_DATA_DIR / info["data_dir"]
    output_dir = run_dir / "kf_gins_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    helper.KF_ROOT = KF_GINS_ROOT
    helper.KF_EXE = KF_EXE
    helper.IMU_FILE = raw_dir / info["imu_file"]
    helper.GNSS_FILE = RAW_DATA_DIR / "GNSS RTK" / "GNSS_RTK.pos"
    helper.TRUTH_FILE = raw_dir / "truth.nav"
    helper.OUTAGE_WINDOWS_S = windows
    helper.FILTERED_GNSS_FILE = run_dir / "GNSS_RTK_outages.pos"
    first_imu = helper.read_first_processed_imu_time()
    removed, returning = helper.make_filtered_gnss(first_imu)
    base = (INSLIB_ROOT / "datasets" / "kfgins" /
            info["inslib_dir"] / "kf-gins.yaml")
    config = run_dir / "kf-gins.yaml"
    make_kf_config(base, config, helper.IMU_FILE,
                   helper.FILTERED_GNSS_FILE, output_dir)
    run = helper.run_filter(config)
    solution_file = output_dir / "KF_GINS_Navresult.nav"
    endpoints, summary = helper.score_outages(
        np.loadtxt(solution_file), np.loadtxt(helper.TRUTH_FILE),
        first_imu, returning)
    # The scored JSON and effective config are retained; bulky filter traces are not.
    remove_kf_outputs(output_dir)
    return {"run": run, "published_config": str(base),
            "effective_config": str(config), "gnss_fixes_removed": removed,
            "returning_fix_times_s": returning, "endpoints": endpoints,
            "summary": summary}


def remove_kf_outputs(output_dir: Path) -> None:
    for name in ("KF_GINS_Navresult.nav", "KF_GINS_IMU_ERR.txt",
                 "KF_GINS_STD.txt"):
        path = output_dir / name
        if not path.exists():
            continue
        for attempt in range(10):
            try:
                path.unlink()
                break
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.1)


class ContinuousErrorObserver:
    """Accumulate INSLIB position components at its native scoring epochs."""

    def __init__(self, replay_tool):
        self.replay_tool = replay_tool
        self.horizontal_sum2 = 0.0
        self.vertical_sum2 = 0.0
        self.epochs = 0
        self.first_t = None
        self.last_t = None

    def on_epoch(self, _replay):
        pass

    def on_pos_error(self, replay, error_ecef):
        rotation = self.replay_tool.ecef_to_ned_rot(
            replay.ref_now["lat_rad"], replay.ref_now["lon_rad"])
        north, east, down = self.replay_tool._matvec_rm(
            rotation, error_ecef)
        self.horizontal_sum2 += north * north + east * east
        self.vertical_sum2 += down * down
        self.epochs += 1
        self.first_t = replay.t if self.first_t is None else self.first_t
        self.last_t = replay.t

    def on_finish(self, _replay):
        pass


def run_gnss_imu_continuous(info: dict, imu_name: str,
                            run_dir: Path) -> dict:
    command = [
        sys.executable, str(GNSS_IMU_SCRIPT),
        "--config", str(SRC_DIR / info["config"]),
        "--data-dir", str(CONVERTED_DATA_DIR),
        "--file-in", info["file_in"],
        "--outage-duration", "0", "--nhc-enable", "false",
        "--out-step", "1", "--full-data", "--no-plots",
    ]
    log = run_dir / "gnss_imu.log"
    started = time.perf_counter()
    run_command(command, log)
    runtime = time.perf_counter() - started
    sys.path.insert(0, str(SRC_DIR))
    from imu_files import Read_GNSS_data
    solution_file = CONVERTED_DATA_DIR / f"gnss_imu_{info['file_in']}.pos"
    solution, _, _ = Read_GNSS_data(str(solution_file))
    truth = np.loadtxt(RAW_DATA_DIR / info["data_dir"] / "truth.nav")
    first_fix = first_benchmark_gnss_fix(info)
    metrics = score_continuous_arrays(
        solution[:, 0] - WEEK_START, solution[:, 1:4],
        solution[:, 23:26], truth, first_fix)
    return {"runtime_s": runtime, "command": subprocess.list2cmdline(command),
            "log": str(log), "solution": str(solution_file),
            "metrics": metrics}


def run_inslib_continuous(info: dict, imu_name: str, prepared: dict,
                          run_dir: Path) -> dict:
    helper = load_module(f"inslib_continuous_{info['inslib_dir']}",
                         INSLIB_HELPER)
    helper.inslib_root = INSLIB_ROOT
    helper.dataset_dir = prepared["dataset_dir"]
    replay_tool, replay_core = helper.install_inslib_imports()
    base = (INSLIB_ROOT / "datasets" / "kfgins" /
            info["inslib_dir"] / "config.yaml")
    config = prepared["dataset_dir"] / "config_continuous.yaml"
    config.write_text(base.read_text(encoding="utf-8"), encoding="utf-8",
                      newline="\n")
    spec, data_dir = replay_tool.load_config(str(config))
    messages = []
    inputs = replay_core.load_inputs(spec, data_dir, [], log=messages.append)
    navigator = replay_core.make_navigator(spec, inputs)
    observer = ContinuousErrorObserver(replay_tool)
    replay = replay_core.Replay(spec, inputs, navigator,
                                observers=[observer])
    log_file = run_dir / "inslib.log"
    started = time.perf_counter()
    try:
        with log_file.open("w", encoding="utf-8", newline="\n") as log:
            with redirect_stdout(log), redirect_stderr(log):
                replay.run()
    finally:
        navigator.close()
    runtime = time.perf_counter() - started
    if observer.epochs == 0 or replay.e_roll.n == 0:
        raise RuntimeError(f"INSLIB produced no scored epochs for {imu_name}")
    metrics = {
        "epochs": observer.epochs,
        "attitude_epochs": replay.e_roll.n,
        "time_span_tow_s": [observer.first_t / 1.0e6,
                            observer.last_t / 1.0e6],
        "warmup_s": float(spec["score"]["warmup_sec"]),
        "horizontal_position_rms_m": math.sqrt(
            observer.horizontal_sum2 / observer.epochs),
        "vertical_position_rms_m": math.sqrt(
            observer.vertical_sum2 / observer.epochs),
        "roll_rms_deg": replay.e_roll.rms(),
        "pitch_rms_deg": replay.e_pitch.rms(),
        "yaw_rms_deg": replay.e_yaw.rms(),
    }
    return {"runtime_s": runtime, "published_config": str(base),
            "effective_config": str(config), "log": str(log_file),
            "loader_messages": messages, "metrics": metrics}


def run_kf_gins_continuous(info: dict, run_dir: Path) -> dict:
    helper = load_module(f"kf_continuous_{info['inslib_dir']}", KF_HELPER)
    raw_dir = RAW_DATA_DIR / info["data_dir"]
    imu_file = raw_dir / info["imu_file"]
    gnss_file = RAW_DATA_DIR / "GNSS RTK" / "GNSS_RTK.pos"
    truth = np.loadtxt(raw_dir / "truth.nav")
    output_dir = run_dir / "kf_gins_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    base = (INSLIB_ROOT / "datasets" / "kfgins" /
            info["inslib_dir"] / "kf-gins.yaml")
    config = run_dir / "kf-gins.yaml"
    make_kf_config(base, config, imu_file, gnss_file, output_dir)
    helper.KF_ROOT = KF_GINS_ROOT
    helper.KF_EXE = KF_EXE
    run = helper.run_filter(config)
    solution_file = output_dir / "KF_GINS_Navresult.nav"
    solution = np.loadtxt(solution_file)
    first_fix = first_benchmark_gnss_fix(info)
    solution_llh = solution[:, 2:5].copy()
    solution_llh[:, :2] = np.deg2rad(solution_llh[:, :2])
    metrics = score_continuous_arrays(
        solution[:, 1], solution_llh, solution[:, 8:11], truth, first_fix)
    remove_kf_outputs(output_dir)
    return {"run": run, "published_config": str(base),
            "effective_config": str(config), "metrics": metrics}


def continuous_result_path(imu_name: str) -> Path:
    return CONTINUOUS_RESULTS_DIR / f"{imu_name.replace('HGuide-', '')}.json"


def run_continuous_imu(imu_name: str, info: dict) -> None:
    run_dir = CONTINUOUS_RESULTS_DIR / imu_name
    run_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepare_inslib(info, imu_name)
    print(f"{imu_name}: continuous GNSS_IMU", flush=True)
    result = {
        "imu": imu_name,
        "experiment": "continuous GNSS truth-error benchmark",
        "warmup_s": CONTINUOUS_WARMUP_S,
        "gnss_imu_initialization_note": (
            "GNSS_IMU uses its configured truth attitude and first GNSS "
            "position/velocity initialization; the warmup is retained."),
        "estimators": {
            "GNSS_IMU": run_gnss_imu_continuous(info, imu_name, run_dir),
        },
    }
    print(f"{imu_name}: continuous INSLIB", flush=True)
    result["estimators"]["INSLIB"] = run_inslib_continuous(
        info, imu_name, prepared, run_dir)
    print(f"{imu_name}: continuous KF-GINS", flush=True)
    result["estimators"]["KF-GINS"] = run_kf_gins_continuous(
        info, run_dir)
    result["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                             time.gmtime())
    continuous_result_path(imu_name).write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\r\n")


def combine(schedules: dict, path: tuple[str, ...]) -> dict:
    endpoints = []
    for schedule in schedules.values():
        case = schedule
        for key in path:
            case = case[key]
        endpoints.extend(case["endpoints"])
    endpoints.sort(key=lambda row: row.get("outage_s",
                   row.get("outage_relative_s"))[0])
    for index, row in enumerate(endpoints, start=1):
        row["combined_outage_index"] = index
    return {"endpoints": endpoints, "summary": summarize(endpoints)}


def result_path(imu_name: str) -> Path:
    return RESULTS_DIR / f"{imu_name.replace('HGuide-', '')}.json"


def run_imu(imu_name: str, info: dict) -> None:
    first_imu_tow, data_duration = raw_imu_times(info)
    first_imu_unix = WEEK_START + first_imu_tow
    prepared = prepare_inslib(info, imu_name)
    result = {
        "imu": imu_name,
        "outage_time_reference": "first converted IMU sample",
        "outage_start_inclusive_end_exclusive": True,
        "first_outage_start_s": FIRST_OUTAGE_START,
        "outage_duration_s": OUTAGE_DURATION,
        "outage_gap_s": OUTAGE_GAP,
        "schedule_start_gap_s": SCHEDULE_START_GAP,
        "schedule_starts_s": list(schedule_starts()),
        "gnss_velocity_used": False,
        "physical_reference_point": "IMU center",
        "inslib_nhc_alignment_deg": prepared["alignment_deg"],
        "schedules": {},
    }
    path = result_path(imu_name)
    for first_start in schedule_starts():
        windows = make_windows(first_start, data_duration)
        if not windows:
            continue
        label = f"{first_start:g}"
        run_dir = RESULTS_DIR / imu_name / f"start_{label}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"{imu_name}: schedule {label} s", flush=True)
        gnss = {
            "nhc_off": run_gnss_imu(
                info, imu_name, first_start, False, windows,
                first_imu_unix, run_dir),
            "nhc_on": run_gnss_imu(
                info, imu_name, first_start, True, windows,
                first_imu_unix, run_dir),
        }
        inslib = run_inslib(info, imu_name, first_start, windows,
                            prepared, run_dir)
        kf_gins = run_kf_gins(info, imu_name, first_start, windows, run_dir)
        result["schedules"][label] = {
            "outage_windows_s": [list(window) for window in windows],
            "gnss_imu": gnss, "inslib": inslib, "kf_gins": kf_gins,
        }
        path.write_text(json.dumps(result, indent=2) + "\n",
                        encoding="utf-8", newline="\r\n")

    schedules = result["schedules"]
    result["combined"] = {
        "gnss_imu_nhc_off": combine(schedules, ("gnss_imu", "nhc_off")),
        "gnss_imu_nhc_on": combine(schedules, ("gnss_imu", "nhc_on")),
        "inslib_nhc_off": combine(schedules, ("inslib", "nhc_off")),
        "inslib_nhc_on": combine(schedules, ("inslib", "nhc_on")),
        "kf_gins": combine(schedules, ("kf_gins",)),
    }
    result["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                             time.gmtime())
    path.write_text(json.dumps(result, indent=2) + "\n",
                    encoding="utf-8", newline="\r\n")


def statistics(case: dict) -> list[float]:
    stats = case["summary"]["position_3d_error_m"]
    return [stats[key] for key in ("mean", "median", "rms", "maximum")]


def make_plot() -> None:
    """Plot all four IMUs, including disabled IMUs' existing result files."""
    reports = {}
    for imu_name in IMUS:
        path = result_path(imu_name)
        if not path.is_file():
            raise FileNotFoundError(
                f"Plot requires an existing result for {imu_name}: {path}")
        reports[imu_name] = json.loads(path.read_text(encoding="utf-8"))

    figure, axes = plt.subplots(4, 2, figsize=(16, 18))
    figure.subplots_adjust(left=0.065, right=0.99, top=0.94, bottom=0.065,
                           hspace=0.36, wspace=0.12)
    figure.suptitle("Awesome-GINS: 3-D error at GNSS-outage endpoints",
                    fontsize=17)
    x = np.arange(len(METRICS))
    case_keys = {
        ("GNSS_IMU", "off"): "gnss_imu_nhc_off",
        ("GNSS_IMU", "on"): "gnss_imu_nhc_on",
        ("INSLIB", "off"): "inslib_nhc_off",
        ("INSLIB", "on"): "inslib_nhc_on",
        ("KF-GINS", "off"): "kf_gins",
    }
    for row_index, imu_name in enumerate(IMUS):
        combined = reports[imu_name]["combined"]
        for column_index, nhc in enumerate(("off", "on")):
            axis = axes[row_index, column_index]
            estimators = ("GNSS_IMU", "INSLIB", "KF-GINS") if nhc == "off" \
                else ("GNSS_IMU", "INSLIB")
            width = 0.25 if len(estimators) == 3 else 0.32
            offsets = ((np.arange(len(estimators)) -
                        (len(estimators) - 1) / 2) * width)
            values = np.asarray([
                statistics(combined[case_keys[(name, nhc)]])
                for name in estimators])
            minima = np.min(values, axis=0)
            for estimator, offset, row in zip(estimators, offsets, values):
                bars = axis.bar(x + offset, row, width=width * 0.9,
                                color=COLORS[estimator],
                                hatch=HATCHES[estimator],
                                label=estimator, alpha=0.92)
                for metric_index, (bar, value) in enumerate(zip(bars, row)):
                    axis.annotate(
                        f"{value:.1f}",
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5,
                        fontweight="bold" if np.isclose(
                            value, minima[metric_index]) else "normal")
            axis.set_title(f"{imu_name} — NHC {nhc}", fontsize=12)
            axis.set_xticks(x, METRICS)
            axis.set_ylabel("3-D endpoint error (m)")
            axis.grid(axis="y", alpha=0.25)
            axis.set_axisbelow(True)
            axis.set_ylim(0.0, max(1.0, float(np.max(values)) * 1.18))

    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[name],
                             hatch=HATCHES[name])
               for name in ("GNSS_IMU", "INSLIB", "KF-GINS")]
    figure.legend(handles, ("GNSS_IMU", "INSLIB", "KF-GINS"),
                  loc="upper center", ncol=3,
                  bbox_to_anchor=(0.5, 0.967), frameon=False)
    figure.text(
        0.5, 0.018,
        (f"Outages are {OUTAGE_DURATION:g} s. Interleaved schedules begin "
         f"every {SCHEDULE_START_GAP:g} s; each schedule has "
         f"{OUTAGE_GAP:g} s of GNSS between outages. Bold labels mark the "
         "lowest result in each category."),
        ha="center", fontsize=10)
    png = RESULTS_DIR / "awesome_gins_outage_benchmark.png"
    svg = RESULTS_DIR / "awesome_gins_outage_benchmark.svg"
    figure.savefig(png, dpi=220, bbox_inches="tight")
    figure.savefig(svg, bbox_inches="tight")
    if plt.get_backend().lower() != "agg":
        plt.show(block=False)
    print(f"Plot: {png}", flush=True)


def make_continuous_report() -> None:
    reports = {}
    for imu_name in IMUS:
        path = continuous_result_path(imu_name)
        if not path.is_file():
            raise FileNotFoundError(
                f"Continuous report requires {imu_name}: {path}")
        reports[imu_name] = json.loads(path.read_text(encoding="utf-8"))

    aggregate = {
        "experiment": "Awesome-GINS continuous truth-error benchmark",
        "included_imus": list(IMUS),
        "excluded_imus": ["Leador A15"],
        "warmup_s": CONTINUOUS_WARMUP_S,
        "metric_order": list(CONTINUOUS_METRIC_KEYS),
        "results": {},
    }
    lines = [
        "# Awesome-GINS continuous truth benchmark",
        "",
        ("Continuous GNSS, a 60.5-second warmup, and truth-interpolated "
         "position and attitude errors. Leador A15 is intentionally excluded."),
        "",
        ("Each cell is **measured**; for INSLIB and KF-GINS the published "
         "README value and measured-minus-published difference follow in "
         "parentheses."),
        "",
        "| IMU | Estimator | Horizontal RMS [m] | Vertical RMS [m] | Roll RMS [deg] | Pitch RMS [deg] | Yaw RMS [deg] | Runtime [s] |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for imu_name in IMUS:
        aggregate["results"][imu_name] = {}
        for estimator in ("GNSS_IMU", "INSLIB", "KF-GINS"):
            case = reports[imu_name]["estimators"][estimator]
            metrics = case["metrics"]
            reference_values = README_REFERENCE[imu_name].get(estimator)
            reference = (dict(zip(CONTINUOUS_METRIC_KEYS, reference_values))
                         if reference_values else None)
            differences = ({key: metrics[key] - reference[key]
                            for key in CONTINUOUS_METRIC_KEYS}
                           if reference else None)
            aggregate["results"][imu_name][estimator] = {
                "measured": metrics,
                "published_readme": reference,
                "measured_minus_published": differences,
                "runtime_s": float(case.get("runtime_s",
                                      case.get("run", {}).get("runtime_s", 0.0))),
            }

            cells = []
            for key in CONTINUOUS_METRIC_KEYS:
                measured = metrics[key]
                if reference is None:
                    cells.append(f"**{measured:.3f}**")
                else:
                    cells.append(
                        f"**{measured:.3f}** ({reference[key]:.3f}; "
                        f"{differences[key]:+.3f})")
            runtime = aggregate["results"][imu_name][estimator]["runtime_s"]
            display_name = "i300" if imu_name == "HGuide-i300" else imu_name
            lines.append("| " + " | ".join(
                [display_name, estimator, *cells, f"{runtime:.1f}"]) + " |")

    lines.extend([
        "",
        "## Reproduction notes",
        "",
        ("INSLIB is scored through its native replay callbacks and published "
         "per-IMU config. KF-GINS uses its published per-IMU config. GNSS_IMU "
         "uses the corresponding repository config with NHC disabled and "
         "output at every IMU epoch."),
        "",
        ("GNSS_IMU does not have a published README reference in the source "
         "table, so no reference difference is shown for it."),
        "",
    ])
    json_path = CONTINUOUS_RESULTS_DIR / "continuous_truth_comparison.json"
    markdown_path = CONTINUOUS_RESULTS_DIR / "continuous_truth_comparison.md"
    json_path.write_text(json.dumps(aggregate, indent=2) + "\n",
                         encoding="utf-8", newline="\r\n")
    markdown_path.write_text("\n".join(lines), encoding="utf-8", newline="\r\n")
    print(f"Continuous comparison: {markdown_path}", flush=True)


def make_continuous_plot() -> None:
    """Plot continuous position and attitude RMS errors for all four IMUs."""
    report_path = CONTINUOUS_RESULTS_DIR / "continuous_truth_comparison.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    estimators = ("GNSS_IMU", "INSLIB", "KF-GINS")
    panel_groups = (
        (
            "Position",
            ("horizontal_position_rms_m", "vertical_position_rms_m"),
            ("Horizontal", "Vertical"),
            "Position RMS error (m)",
        ),
        (
            "Attitude",
            ("roll_rms_deg", "pitch_rms_deg", "yaw_rms_deg"),
            ("Roll", "Pitch", "Yaw"),
            "Attitude RMS error (deg)",
        ),
    )

    figure, axes = plt.subplots(4, 2, figsize=(16, 18))
    figure.subplots_adjust(left=0.075, right=0.985, top=0.94, bottom=0.065,
                           hspace=0.38, wspace=0.15)
    figure.suptitle("Awesome-GINS: continuous truth-error benchmark",
                    fontsize=17)
    width = 0.24
    offsets = (np.arange(len(estimators)) - 1) * width

    # Use a common scale down each column so IMUs can be compared visually.
    group_limits = []
    for _group, keys, _labels, _ylabel in panel_groups:
        maximum = max(
            report["results"][imu][estimator]["measured"][key]
            for imu in IMUS for estimator in estimators for key in keys)
        group_limits.append(maximum * 1.24)

    for row, imu_name in enumerate(IMUS):
        cases = report["results"][imu_name]
        for column, (group, keys, labels, ylabel) in enumerate(panel_groups):
            axis = axes[row, column]
            x = np.arange(len(keys))
            values = np.asarray([
                [cases[estimator]["measured"][key] for key in keys]
                for estimator in estimators
            ])
            minima = np.min(values, axis=0)
            for estimator, offset, estimator_values in zip(
                    estimators, offsets, values):
                bars = axis.bar(
                    x + offset, estimator_values, width=width * 0.9,
                    color=COLORS[estimator], hatch=HATCHES[estimator],
                    label=estimator, alpha=0.92)
                for metric_index, (bar, value) in enumerate(
                        zip(bars, estimator_values)):
                    axis.annotate(
                        f"{value:.3f}",
                        (bar.get_x() + bar.get_width() / 2,
                         bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9,
                        fontweight=("bold" if np.isclose(
                            value, minima[metric_index], rtol=0.0,
                            atol=5.0e-7) else "normal"))
            axis.set_title(f"{imu_name} — {group}", fontsize=12)
            axis.set_xticks(x, labels)
            axis.set_ylabel(ylabel)
            axis.grid(axis="y", alpha=0.25)
            axis.set_axisbelow(True)
            axis.set_ylim(0.0, group_limits[column])

    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[name],
                             hatch=HATCHES[name])
               for name in estimators]
    figure.legend(handles, estimators, loc="upper center", ncol=3,
                  bbox_to_anchor=(0.5, 0.967), frameon=False)
    figure.text(
        0.5, 0.018,
        (f"Continuous GNSS with a {CONTINUOUS_WARMUP_S:g} s scoring warmup. "
         "Bold labels mark the lowest RMS error in each category."),
        ha="center", fontsize=10)
    png = RESULTS_DIR / "awesome_gins_continuous_benchmark.png"
    svg = RESULTS_DIR / "awesome_gins_continuous_benchmark.svg"
    figure.savefig(png, dpi=220, bbox_inches="tight")
    figure.savefig(svg, bbox_inches="tight")
    if plt.get_backend().lower() != "agg":
        plt.show(block=False)
    print(f"Continuous plot: {png}", flush=True)


def validate_installation() -> None:
    required = (INSLIB_ROOT, KF_GINS_ROOT, KF_EXE, INSLIB_HELPER, KF_HELPER)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing benchmark installation/files:\n" +
                                "\n".join(missing))


def main() -> None:
    benchmark_started = time.perf_counter()
    validate_installation()
    ensure_data()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    enabled = []
    for imu_name, info in IMUS.items():
        if info["run"]:
            enabled.append((imu_name, info))
        else:
            print(f"{imu_name}: rerun disabled; using existing JSON", flush=True)

    if MAX_SIMULTANEOUS_RUNS < 1:
        raise ValueError("MAX_SIMULTANEOUS_RUNS must be at least 1")
    worker_count = min(MAX_SIMULTANEOUS_RUNS, len(enabled))
    if RUN_OUTAGE_BENCHMARK:
        if worker_count:
            print(f"Running {len(enabled)} IMUs with {worker_count} workers",
                  flush=True)
            with ProcessPoolExecutor(max_workers=worker_count) as pool:
                futures = {pool.submit(run_imu, name, info): name
                           for name, info in enabled}
                for future in as_completed(futures):
                    imu_name = futures[future]
                    try:
                        future.result()
                    except Exception as error:
                        raise RuntimeError(
                            f"{imu_name} benchmark failed") from error
                    print(f"{imu_name}: complete", flush=True)
        make_plot()

    if RUN_CONTINUOUS_TRUTH_BENCHMARK:
        CONTINUOUS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        if worker_count:
            print(
                f"Running continuous truth benchmark for {len(enabled)} IMUs "
                f"with {worker_count} workers", flush=True)
            with ProcessPoolExecutor(max_workers=worker_count) as pool:
                futures = {
                    pool.submit(run_continuous_imu, name, info): name
                    for name, info in enabled
                }
                for future in as_completed(futures):
                    imu_name = futures[future]
                    try:
                        future.result()
                    except Exception as error:
                        raise RuntimeError(
                            f"{imu_name} continuous benchmark failed") \
                            from error
                    print(f"{imu_name}: continuous complete", flush=True)
        make_continuous_report()
        make_continuous_plot()

    elapsed_s = time.perf_counter() - benchmark_started
    elapsed_seconds = int(round(elapsed_s))
    hours, remainder = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(
        f"Benchmark completed in {hours:02d}:{minutes:02d}:{seconds:02d} "
        f"({elapsed_s:.1f} seconds)",
        flush=True,
    )


if __name__ == "__main__":
    main()
