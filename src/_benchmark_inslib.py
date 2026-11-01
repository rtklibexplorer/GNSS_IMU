#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 Project:    Sensor Fusion for GNSS/IMU (Loosely Coupled Kalman Filter)
 File:       _benchmark_inslib.py
 Author:     Tim Everett
 Copyright:  (c) 2025 Tim Everett
 License:    BSD 3 License (see LICENSE file for details)

 Description:
    Run the five-window ADIS16465 INSLIB/GNSS_IMU comparison.

    There are no command-line parameters.  INSLIB core code and its committed
    ADIS16465 configuration are not modified.  Generated inputs, exact effective
    configs, commands, and results are written beneath ``results_dir``.
===============================================================================
"""

from contextlib import redirect_stdout
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np


######## Paths and experiment definition ####################################

python_exe = Path(r"C:\Users\timev\AppData\Local\spyder-6\envs\spyder-runtime\python.exe")
inslib_root = Path(r"C:\gps\gps_imu\INSLIB")
awesome_root = Path(r"C:\gps\data\benchmarks\Awesome-GINS")
gnss_imu_root = Path(r"C:\gps\gps_imu\GNSS_IMU")
results_dir = gnss_imu_root / "results" / "inslib_adis16465_60s"
dataset_dir = results_dir / "dataset"

raw_imu_file = awesome_root / "ADIS16465" / "ADIS16465.txt"
raw_gnss_file = awesome_root / "GNSS RTK" / "GNSS_RTK.pos"
raw_truth_file = awesome_root / "ADIS16465" / "truth.nav"
base_config_file = inslib_root / "datasets" / "kfgins" / "ADIS16465" / "config.yaml"
gnss_imu_results_file = (gnss_imu_root / "results" /
                         "awesome_gins_ADIS16465_inslib_60s_comparison.json")

outage_windows_s = (
    (300.0, 360.0),
    (600.0, 660.0),
    (900.0, 960.0),
    (1200.0, 1260.0),
    (1500.0, 1560.0),
)

##############################################################################


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def convert_dataset():
    """Convert public Awesome-GINS files to INSLIB's replay CSV contract."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    imu_csv = dataset_dir / "imu.csv"
    gnss_csv = dataset_dir / "gnss.csv"
    ref_csv = dataset_dir / "ref.csv"

    # Each raw row contains increments over the interval ending at that row.
    # As in the GNSS_IMU converter, drop the first row (no preceding timestamp)
    # and divide each later increment by its actual sample interval.
    raw_imu = np.loadtxt(raw_imu_file)
    dt = np.diff(raw_imu[:, 0])
    if np.any(dt <= 0.0):
        raise RuntimeError("ADIS16465 IMU timestamps are not strictly increasing")
    imu = np.column_stack((
        np.rint(raw_imu[1:, 0] * 1.0e6).astype(np.int64),
        raw_imu[1:, 1:4] / dt[:, None],
        raw_imu[1:, 4:7] / dt[:, None],
    ))
    np.savetxt(
        imu_csv, imu,
        fmt=["%d"] + ["%.10g"] * 6,
        delimiter=",",
        header=("# t_us, gyr_frd_x [rad/s], gyr_frd_y [rad/s], "
                "gyr_frd_z [rad/s], acc_frd_x [m/s^2], "
                "acc_frd_y [m/s^2], acc_frd_z [m/s^2]"),
        comments="")

    # Use the first GNSS fix after the first converted IMU sample.  The source
    # contains position and per-axis 1-sigma only; velocity is deliberately
    # marked unavailable, rather than synthesized.
    raw_gnss = np.loadtxt(raw_gnss_file)
    raw_gnss = raw_gnss[raw_gnss[:, 0] > raw_imu[1, 0]]
    gnss = np.zeros((len(raw_gnss), 20), dtype=float)
    gnss[:, 0] = np.rint(raw_gnss[:, 0] * 1.0e6)
    gnss[:, 1:4] = raw_gnss[:, 1:4]
    gnss[:, 4] = raw_gnss[:, 4] ** 2
    gnss[:, 7] = raw_gnss[:, 5] ** 2
    gnss[:, 9] = raw_gnss[:, 6] ** 2
    np.savetxt(
        gnss_csv, gnss,
        fmt=["%d", "%.10f", "%.10f", "%.4f"] + ["%.10g"] * 16,
        delimiter=",",
        header=("# t_us, lat_deg, lon_deg, h_m, cov_pos_ned "
                "nn,ne,nd,ee,ed,dd [m^2], vn,ve,vd [m/s], "
                "cov_vel_ned nn,ne,nd,ee,ed,dd [(m/s)^2], vel_ok"),
        comments="")

    truth = np.loadtxt(raw_truth_file)
    ref = np.column_stack((
        np.rint(truth[:, 1] * 1.0e6).astype(np.int64),
        truth[:, 2:5],
        truth[:, 8:11],
        truth[:, 5:8],
    ))
    np.savetxt(
        ref_csv, ref,
        fmt=["%d", "%.10f", "%.10f", "%.5f"] + ["%.6f"] * 6,
        delimiter=",",
        header=("# t_us, lat_deg, lon_deg, h_m, roll_deg, pitch_deg, "
                "yaw_deg, vn_mps, ve_mps, vd_mps"),
        comments="")
    return imu_csv, gnss_csv, ref_csv


def make_config(nhc_enabled):
    """Generate the exact effective config while preserving base parameters."""
    text = base_config_file.read_text(encoding="utf-8")
    label = "on" if nhc_enabled else "off"
    text = re.sub(r"(?m)^name:.*$", f"name: adis16465-inslib-60s-nhc-{label}",
                  text, count=1)
    anchor = "automotive_mode: 1\n"
    addition = (
        "allow_unlimited_deadreckoning: 1  # required to propagate through 60 s outages\n"
        f"automotive_lateral_constraint: {1 if nhc_enabled else 0}  # NHC experiment toggle\n"
    )
    if anchor not in text:
        raise RuntimeError("Could not find automotive_mode in INSLIB config")
    text = text.replace(anchor, anchor + addition, 1)
    path = dataset_dir / f"config_nhc_{label}.yaml"
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def install_inslib_imports():
    sys.path.insert(0, str(inslib_root / "python"))
    sys.path.insert(0, str(inslib_root / "tools"))
    import replay as replay_tool
    import replay_core
    return replay_tool, replay_core


class EndpointObserver:
    """Retain the last post-propagation state before each returning fix."""

    def __init__(self, replay_tool, return_fix_times):
        self.rp = replay_tool
        self.return_fix_times = list(return_fix_times)
        self.latest = [None] * len(return_fix_times)
        self.fusion_epoch_us = [None] * len(return_fix_times)

    def on_epoch(self, replay):
        for index, return_time in enumerate(self.return_fix_times):
            if replay.t < return_time and replay.ref_now is not None:
                error_ecef = self.rp.pos_error_ecef(
                    replay.nav, replay.ref_now, replay.score_la, replay.t)
                if error_ecef is not None:
                    rotation = self.rp.ecef_to_ned_rot(
                        replay.ref_now["lat_rad"], replay.ref_now["lon_rad"])
                    error_ned = self.rp._matvec_rm(rotation, error_ecef)
                    self.latest[index] = {
                        "endpoint_t_us": int(replay.t),
                        "north_error_m": float(error_ned[0]),
                        "east_error_m": float(error_ned[1]),
                        "down_error_m": float(error_ned[2]),
                    }
            if (self.fusion_epoch_us[index] is None and replay.fix_now is not None
                    and replay.fix_now["t_us"] == return_time):
                self.fusion_epoch_us[index] = int(replay.t)

    def on_pos_error(self, _replay, _error):
        pass

    def on_finish(self, _replay):
        pass


def summarize(endpoints):
    metrics = {}
    for name in ("horizontal_error_m", "vertical_error_m", "position_3d_error_m"):
        values = np.array([row[name] for row in endpoints])
        metrics[name] = {
            "mean": float(np.mean(values)),
            "maximum": float(np.max(values)),
            "rms": float(np.sqrt(np.mean(values ** 2))),
        }
    return metrics


def run_case(config_path, nhc_enabled, replay_tool, replay_core,
             result_name=None):
    spec, data_dir = replay_tool.load_config(str(config_path))
    outage_args = [f"{start:g}:{end - start:g}"
                   for start, end in outage_windows_s]
    outages = replay_core.parse_gnss_outages(outage_args)

    # Determine return epochs from the uncut input, then let the standard
    # loader remove the requested outage fixes.
    original_fixes = replay_tool.load_gnss(str(dataset_dir / "gnss.csv"))
    first_imu_us = next(replay_tool.iter_imu(str(dataset_dir / "imu.csv")))[0]
    returning = []
    for _start, end in outage_windows_s:
        eligible = [fix["t_us"] for fix in original_fixes
                    if fix["t_us"] >= first_imu_us + int(end * 1.0e6)]
        if not eligible:
            raise RuntimeError(f"No returning GNSS fix after {end:g} seconds")
        returning.append(eligible[0])

    messages = []
    inp = replay_core.load_inputs(spec, data_dir, outages, log=messages.append)
    nav = replay_core.make_navigator(spec, inp)
    observer = EndpointObserver(replay_tool, returning)
    replay = replay_core.Replay(spec, inp, nav, observers=[observer])
    started = time.time()
    replay.run()
    elapsed = time.time() - started
    nav.close()

    if any(row is None for row in observer.latest):
        raise RuntimeError("INSLIB did not produce all five pre-return endpoints")
    if any(epoch is None for epoch in observer.fusion_epoch_us):
        raise RuntimeError("INSLIB did not process all five returning fixes")

    endpoints = []
    for index, ((start, end), return_time, row, fusion_time) in enumerate(zip(
            outage_windows_s, returning, observer.latest,
            observer.fusion_epoch_us), start=1):
        horizontal = math.hypot(row["north_error_m"], row["east_error_m"])
        vertical = abs(row["down_error_m"])
        item = dict(row)
        item.update({
            "outage_index": index,
            "outage_s": [start, end],
            "endpoint_relative_s": (row["endpoint_t_us"] - first_imu_us) / 1.0e6,
            "returning_fix_relative_s": (return_time - first_imu_us) / 1.0e6,
            "returning_fix_fusion_relative_s": (fusion_time - first_imu_us) / 1.0e6,
            "pre_fix_margin_s": (return_time - row["endpoint_t_us"]) / 1.0e6,
            "horizontal_error_m": horizontal,
            "vertical_error_m": vertical,
            "position_3d_error_m": math.hypot(horizontal, vertical),
        })
        endpoints.append(item)

    result = {
        "nhc_enabled": nhc_enabled,
        "config_file": str(config_path),
        "converted_gnss_velocity_available": False,
        "allow_unlimited_deadreckoning": True,
        "n_gnss_fixes_removed": inp.n_outage_cut,
        "runtime_s": elapsed,
        "loader_messages": messages,
        "endpoints": endpoints,
        "summary": summarize(endpoints),
    }
    if result_name is None:
        result_name = f"run_nhc_{'on' if nhc_enabled else 'off'}"
    (results_dir / f"{result_name}.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def load_gnss_imu_results():
    if not gnss_imu_results_file.exists():
        return None
    report = json.loads(gnss_imu_results_file.read_text(encoding="utf-8"))
    cases = {
        "nhc_off": report["cases"]["nhc_disabled"],
        "nhc_on": report["cases"]["nhc_enabled"],
    }
    for case in cases.values():
        case["summary"] = summarize(case["endpoints"])
    cases["source"] = str(gnss_imu_results_file)
    return cases


def main():
    results_dir.mkdir(parents=True, exist_ok=True)
    imu_csv, gnss_csv, ref_csv = convert_dataset()
    configs = {
        "nhc_off": make_config(False),
        "nhc_on": make_config(True),
    }

    replay_tool, replay_core = install_inslib_imports()
    cases = {}
    for key, enabled in (("nhc_off", False), ("nhc_on", True)):
        log_path = results_dir / f"console_{key}.txt"
        with open(log_path, "w", encoding="utf-8", newline="\n") as log:
            with redirect_stdout(log):
                cases[key] = run_case(
                    configs[key], enabled, replay_tool, replay_core)

    try:
        commit = subprocess.check_output(
            ["git", "-c", "safe.directory=C:/gps/gps_imu/INSLIB",
             "-C", str(inslib_root), "rev-parse", "HEAD"],
            text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None

    command = f'"{python_exe}" "{Path(__file__).resolve()}"'
    baseline_command = (
        f'"{python_exe}" "{inslib_root / "tools" / "replay.py"}" '
        f'"{configs["nhc_off"]}" --summary-json '
        f'"{results_dir / "baseline_summary.json"}"')
    (results_dir / "commands.txt").write_text(
        "Primary benchmark:\n" + command + "\n\n"
        "MSVC library build (required once):\n"
        f'cmd /c "{inslib_root / "build_pylib_msvc.bat"}"\n\n'
        "No-outage conversion/configuration validation:\n" +
        baseline_command + "\n\n"
        "The script uses INSLIB tools/replay.py and tools/replay_core.py "
        "with outage arguments:\n  " + " ".join(
            f"--gnss-outage {start:g}:{end - start:g}"
            for start, end in outage_windows_s) + "\n",
        encoding="utf-8", newline="\n")

    report = {
        "experiment": "Awesome-GINS ADIS16465 five 60-second GNSS outages",
        "inslib_root": str(inslib_root),
        "inslib_commit": commit,
        "inslib_library": str(inslib_root / "python" / "INSLIB" / "libINSLIB.dll"),
        "outage_time_reference": "first converted IMU sample",
        "outage_windows_s": [list(window) for window in outage_windows_s],
        "raw_input_sha256": {
            str(raw_imu_file): sha256(raw_imu_file),
            str(raw_gnss_file): sha256(raw_gnss_file),
            str(raw_truth_file): sha256(raw_truth_file),
        },
        "converted_inputs": [str(imu_csv), str(gnss_csv), str(ref_csv)],
        "effective_configs": {key: str(path) for key, path in configs.items()},
        "inslib": cases,
        "gnss_imu": load_gnss_imu_results(),
        "no_outage_validation": (
            json.loads((results_dir / "baseline_summary.json").read_text(
                encoding="utf-8"))
            if (results_dir / "baseline_summary.json").exists() else None),
        "secondary_published_reference_m": {
            "nhc_off": {"mean_3d": 8.5, "maximum_3d": 13.3},
            "nhc_on": {"mean_3d": 3.2, "maximum_3d": 5.7},
        },
        "reproduction_command": command,
    }
    output = results_dir / "benchmark_results.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
