#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
===============================================================================
 Project:    Sensor Fusion for GNSS/IMU (Loosely Coupled Kalman Filter)
 File:       get_Awesome_GINS_data.py
 Author:     Tim Everett
 Copyright:  (c) 2025 Tim Everett
 License:    BSD 3 License (see LICENSE file for details)

 Description:
    Download and extract the public Awesome-GINS dataset.

    From the repository root, run::

        python src\get_Awesome_GINS_data.py

    All parameters are defined below; the script takes no command-line arguments.
===============================================================================
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request


####### Download and extraction parameters ####################################

repository = "i2Nav-WHU/awesome-gins-datasets"
branch = "main"

# Resolve data/awesome_gins relative to this repository, one level above src.
output_dir = Path(__file__).resolve().parent.parent / "data" / "awesome_gins"

# Re-download existing nonempty files and replace extracted files when True.
overwrite_downloads = False
overwrite_extracted = False

# When False, remove each archive after successful extraction and validation.
keep_archives = False

# Optional explicit 7-Zip executable. None searches PATH and standard locations.
seven_zip_executable: Path | None = None

# Network transfer size and timeout.
download_chunk_bytes = 1024 * 1024       # bytes, download block size
network_timeout_seconds = 120            # s, connection/read timeout

###############################################################################

ARCHIVES = (
    Path("ADIS16460") / "ADIS16460.7z",
    Path("ADIS16465") / "ADIS16465.7z",
    Path("ICM20602") / "ICM20602.7z",
    Path("i300") / "i300.7z",
    Path("GNSS RTK") / "GNSS_RTK.pos.7z",
)

SUPPORT_FILES = (
    Path("README.md"),
    Path("README.pdf"),
    Path("GNSS RTK") / "GNSS_RTK.kml",
    Path("GNSS RTK") / "Map.jpg",
)

EXPECTED_EXTRACTED_FILES = (
    Path("ADIS16460") / "ADIS16460.txt",
    Path("ADIS16460") / "truth.nav",
    Path("ADIS16465") / "ADIS16465.txt",
    Path("ADIS16465") / "truth.nav",
    Path("ICM20602") / "ICM20602.txt",
    Path("ICM20602") / "truth.nav",
    Path("i300") / "i300.txt",
    Path("i300") / "truth.nav",
    Path("GNSS RTK") / "GNSS_RTK.pos",
)
EXPECTED_BY_ARCHIVE = {
    ARCHIVES[0]: EXPECTED_EXTRACTED_FILES[0:2],
    ARCHIVES[1]: EXPECTED_EXTRACTED_FILES[2:4],
    ARCHIVES[2]: EXPECTED_EXTRACTED_FILES[4:6],
    ARCHIVES[3]: EXPECTED_EXTRACTED_FILES[6:8],
    ARCHIVES[4]: EXPECTED_EXTRACTED_FILES[8:9],
}


def _raw_url(relative_path: Path) -> str:
    encoded_path = "/".join(
        urllib.parse.quote(part, safe="") for part in relative_path.parts)
    return (f"https://raw.githubusercontent.com/{repository}/{branch}/"
            f"{encoded_path}")


def _download(relative_path: Path) -> Path:
    destination = output_dir / relative_path
    if (destination.is_file() and destination.stat().st_size > 0 and
            not overwrite_downloads):
        print(f"Already downloaded: {relative_path}")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(
        _raw_url(relative_path),
        headers={"User-Agent": "GNSS_IMU Awesome-GINS downloader"},
    )
    print(f"Downloading: {relative_path}")
    try:
        with (
            urllib.request.urlopen(
                request, timeout=network_timeout_seconds) as response,
            partial.open("wb") as output,
        ):
            expected_size = response.headers.get("Content-Length")
            expected_size = int(expected_size) if expected_size else None
            downloaded = 0
            while True:
                block = response.read(download_chunk_bytes)
                if not block:
                    break
                output.write(block)
                downloaded += len(block)
                if expected_size:
                    percent = 100.0 * downloaded / expected_size
                    print(f"  {downloaded / 1e6:7.1f} MB "
                          f"({percent:5.1f}%)", end="\r", flush=True)
        if expected_size and downloaded != expected_size:
            raise RuntimeError(
                f"Incomplete download for {relative_path}: "
                f"received {downloaded} of {expected_size} bytes")
        os.replace(partial, destination)
        print(f"  Saved {downloaded / 1e6:.1f} MB to {destination}")
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return destination


def _find_7zip() -> Path | None:
    candidates = []
    if seven_zip_executable is not None:
        candidates.append(Path(seven_zip_executable))
    for command in ("7z", "7zz"):
        found = shutil.which(command)
        if found:
            candidates.append(Path(found))
    candidates.extend((
        Path(r"C:\Program Files\7-Zip\7z.exe"),
        Path(r"C:\Program Files (x86)\7-Zip\7z.exe"),
    ))
    return next((path for path in candidates if path.is_file()), None)


def _extract_with_7zip(archive: Path, destination: Path,
                       executable: Path) -> None:
    overwrite_mode = "-aoa" if overwrite_extracted else "-aos"
    command = [
        str(executable), "x", str(archive), f"-o{destination}",
        "-y", overwrite_mode,
    ]
    subprocess.run(command, check=True)


def _extract_with_py7zr(archive: Path, destination: Path) -> None:
    try:
        import py7zr
    except ImportError as error:
        raise RuntimeError(
            "A 7z extractor is required. Install 7-Zip, put 7z on PATH, "
            "set seven_zip_executable, or install py7zr.") from error

    # py7zr extracts existing names by replacement. Prefer 7-Zip when files
    # must be skipped rather than overwritten.
    if not overwrite_extracted:
        print("  py7zr fallback may replace existing extracted files")
    with py7zr.SevenZipFile(archive, mode="r") as compressed:
        compressed.extractall(path=destination)


def _extract(archive: Path) -> None:
    relative_archive = archive.relative_to(output_dir)
    # The four IMU archives already contain their sensor directory. The GNSS
    # archive contains only GNSS_RTK.pos, so it extracts beside that archive.
    destination = (archive.parent if relative_archive.parent == Path("GNSS RTK")
                   else output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    print(f"Extracting: {archive.name} -> {destination}")
    executable = _find_7zip()
    if executable is not None:
        _extract_with_7zip(archive, destination, executable)
    else:
        _extract_with_py7zr(archive, destination)


def _validate() -> None:
    missing = [path for path in EXPECTED_EXTRACTED_FILES
               if not (output_dir / path).is_file()]
    if missing:
        formatted = "\n".join(f"  {path}" for path in missing)
        raise RuntimeError(
            "Extraction completed but expected files are missing:\n"
            f"{formatted}")


def main() -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    archives = []
    for path in ARCHIVES:
        expected = EXPECTED_BY_ARCHIVE[path]
        already_extracted = all((output_dir / item).is_file()
                                for item in expected)
        if already_extracted and not overwrite_extracted:
            print(f"Already extracted: {path.parent}")
        else:
            archives.append(_download(path))
    for path in SUPPORT_FILES:
        _download(path)
    for archive in archives:
        _extract(archive)

    _validate()
    if not keep_archives and archives:
        for archive in archives:
            archive.unlink()
        print("Removed downloaded archives after successful validation")

    print(f"Awesome-GINS data are ready in: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
