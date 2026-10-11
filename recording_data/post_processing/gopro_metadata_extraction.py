"""Extract selected metadata, complete metadata, and accelerometer samples.

The script only reads the video files. ExifTool must be installed and available
on PATH, or EXIFTOOL_COMMAND may be set to the full path of the executable.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_ROOT = Path(r"D:\2026-08-18_tennis_S16")
EXIFTOOL_COMMAND = r"C:\Users\jdelp\Desktop\exiftool-13.59_64\exiftool.exe"

# Set this to an integer such as 1 for quick tests, or None to process all MP4
# files in each camera folder. Files are selected in case-insensitive name order.
MAX_VIDEOS_PER_CAMERA: int | None = None

OUTPUT_SELECTED_JSON = Path(__file__).with_name(
    "gopro_video_metadata_selectedFields.json"
)
OUTPUT_ALL_JSON = Path(__file__).with_name("gopro_video_metadata_allFields.json")
OUTPUT_SELECTED_CSV = Path(__file__).with_name(
    "gopro_video_metadata_selectedFields.csv"
)
# Per-video accelerometer files are written directly into this root folder.
OUTPUT_ACCELEROMETER_DIRECTORY = Path(__file__).resolve().parent


# These must remain the first columns in the CSV and in this exact order.
PREFERRED_COLUMNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # ExifTool displays the description "Camera Model Name", but its actual
    # machine-readable GoPro tag name is "Model".
    ("Camera Model Name", ("CameraModelName", "Model")),
    ("Firmware Version", ("FirmwareVersion",)),
    ("Camera Serial Number", ("CameraSerialNumber",)),
    ("Lens Serial Number", ("LensSerialNumber",)),
    ("White Balance", ("WhiteBalance",)),
    ("Sharpness", ("Sharpness",)),
    ("Color Mode", ("ColorMode",)),
    ("Horizon Control", ("HorizonControl",)),
    ("Lens Projection", ("LensProjection",)),
    ("Video Frame Rate", ("VideoFrameRate",)),
    ("Video Frame Size", ("VideoFrameSize",)),
    ("Image Width", ("ImageWidth",)),
    ("Image Height", ("ImageHeight",)),
    ("Digital Zoom Amount", ("DigitalZoomAmount", "DigitalZoom")),
    (
        "Zoom Scale Normalization",
        ("ZoomScaleNormalization", "ZoomScaleNormalized"),
    ),
    ("Field of View", ("FieldOfView",)),
    ("Diagonal Field Of View", ("DiagonalFieldOfView",)),
    ("Aspect Ratio Unwarped", ("AspectRatioUnwarped",)),
    ("Aspect Ratio Warped", ("AspectRatioWarped",)),
    ("Polynomial Power", ("PolynomialPower",)),
    ("Polynomial Coefficients", ("PolynomialCoefficients",)),
    ("Orientation Data Present", ("OrientationDataPresent",)),
)

ASSOCIATION_COLUMNS = (
    "Camera Folder Name",
    "Video File Name",
)


def friendly_camera_name(folder_name: str) -> str:
    """Convert a folder name such as ``back_gopro`` to ``Back Gopro``."""

    return folder_name.replace("_", " ").title()


def _relative_to_root(path: Path, root: Path) -> str:
    """Return a stable POSIX-style relative path for the output files."""

    return path.relative_to(root).as_posix()


def find_gopro_folders(data_root: Path) -> list[Path]:
    """Recursively find subfolders whose names contain ``GoPro``."""

    return sorted(
        (
            path
            for path in data_root.rglob("*")
            if path.is_dir() and "gopro" in path.name.casefold()
        ),
        key=lambda path: str(path).casefold(),
    )


def find_mp4_files(folder: Path) -> list[Path]:
    """Find MP4 files directly inside one GoPro folder."""

    return sorted(
        (
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.casefold() == ".mp4"
        ),
        key=lambda path: path.name.casefold(),
    )


def discover_camera_videos(
    data_root: Path, max_videos_per_camera: int | None
) -> list[tuple[Path, list[Path]]]:
    """Return each GoPro folder and its selected, sorted MP4 files."""

    if max_videos_per_camera is not None and max_videos_per_camera < 0:
        raise ValueError("MAX_VIDEOS_PER_CAMERA cannot be negative.")

    files_by_camera: list[tuple[Path, list[Path]]] = []
    for camera_folder in find_gopro_folders(data_root):
        video_files = find_mp4_files(camera_folder)
        if max_videos_per_camera is not None:
            video_files = video_files[:max_videos_per_camera]
        files_by_camera.append((camera_folder, video_files))
    return files_by_camera


def resolve_exiftool(command: str) -> str:
    """Resolve the configured ExifTool command or raise an actionable error."""

    configured_path = Path(command)
    if configured_path.parent != Path("."):
        if configured_path.is_file():
            return str(configured_path)
    else:
        resolved = shutil.which(command)
        if resolved:
            return resolved

    raise RuntimeError(
        f"Could not find ExifTool using EXIFTOOL_COMMAND={command!r}. "
        "Install ExifTool and put it on PATH, or set EXIFTOOL_COMMAND to the "
        "full path of exiftool.exe."
    )


def exiftool_version(exiftool: str) -> str:
    """Return the ExifTool version string."""

    result = subprocess.run(
        [exiftool, "-ver"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Could not run {exiftool!r}: "
            f"{result.stderr.strip() or 'no error details were returned'}"
        )
    return result.stdout.strip()


def extract_metadata(video_path: Path, exiftool: str) -> dict[str, Any]:
    """Read all recognized metadata for one video without modifying it.

    ``-ee`` exposes GoPro's timed GPMF metadata.  The group families in
    ``-G3:1:4`` retain the embedded-document origin and prevent duplicate tag
    names from being silently discarded in ExifTool's JSON output.
    """

    command = [
        exiftool,
        "-json",
        "-struct",
        "-ee",
        "-u",
        "-G3:1:4",
        "-api",
        "LargeFileSupport=1",
        str(video_path),
    ]
    result = subprocess.run(command, capture_output=True)

    if result.returncode != 0:
        details = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"ExifTool failed for {video_path} (exit {result.returncode}): "
            f"{details or 'no error details were returned'}"
        )

    try:
        documents = json.loads(result.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        details = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"ExifTool returned invalid JSON for {video_path}. {details}"
        ) from error

    if not isinstance(documents, list) or len(documents) != 1:
        raise RuntimeError(
            f"Expected one ExifTool result for {video_path}, received "
            f"{len(documents) if isinstance(documents, list) else 'an invalid result'}."
        )
    if not isinstance(documents[0], dict):
        raise RuntimeError(f"ExifTool returned an invalid record for {video_path}.")

    return documents[0]


ACCELEROMETER_KEY_PATTERN = re.compile(
    r"^(?P<document>Doc[\d-]+):(?P<track>Track\d+)"
    r"(?::(?P<copy>Copy\d+))?:Accelerometer$"
)

ACCELEROMETER_CSV_COLUMNS = (
    "Camera Folder Name",
    "Video File Name",
    "Document",
    "Track",
    "Copy",
    "Source Field",
    "Payload Sample Time (s)",
    "Payload Sample Duration (s)",
    "Sample Index In Payload",
    "Estimated Sample Time (s)",
    "Estimated Sample Duration (s)",
    "X (m/s^2)",
    "Y (m/s^2)",
    "Z (m/s^2)",
)


def extract_accelerometer_payloads(
    video_path: Path, exiftool: str
) -> list[dict[str, Any]]:
    """Extract and decode every Accelerometer payload from one video.

    With ``-b``, ExifTool returns each GoPro Accelerometer value as a
    whitespace-delimited sequence of numeric x/y/z triples. ``-n`` also makes
    the associated payload times numeric seconds.
    """

    command = [
        exiftool,
        "-json",
        "-struct",
        "-ee",
        "-b",
        "-n",
        "-G3:1:4",
        "-Accelerometer",
        "-SampleTime",
        "-SampleDuration",
        "-api",
        "LargeFileSupport=1",
        str(video_path),
    ]
    result = subprocess.run(command, capture_output=True)
    if result.returncode != 0:
        details = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"ExifTool accelerometer extraction failed for {video_path} "
            f"(exit {result.returncode}): "
            f"{details or 'no error details were returned'}"
        )

    try:
        documents = json.loads(result.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        details = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"ExifTool returned invalid accelerometer JSON for {video_path}. "
            f"{details}"
        ) from error

    if not isinstance(documents, list) or len(documents) != 1:
        raise RuntimeError(
            f"Expected one accelerometer result for {video_path}, received an "
            "invalid result."
        )
    metadata = documents[0]

    payloads: list[dict[str, Any]] = []
    for key, raw_value in metadata.items():
        match = ACCELEROMETER_KEY_PATTERN.fullmatch(key)
        if not match:
            continue
        if not isinstance(raw_value, str):
            raise RuntimeError(
                f"Unexpected accelerometer value type in {video_path}: {key}"
            )
        try:
            values = [float(value) for value in raw_value.split()]
        except ValueError as error:
            raise RuntimeError(
                f"Could not parse accelerometer values in {video_path}: {key}"
            ) from error
        if len(values) % 3 != 0:
            raise RuntimeError(
                f"Accelerometer field {key} in {video_path} contains "
                f"{len(values)} values instead of complete x/y/z triples."
            )

        prefix = key.rsplit(":", 1)[0]
        payloads.append(
            {
                "document": match.group("document"),
                "track": match.group("track"),
                "copy": match.group("copy"),
                "source_field": key,
                "payload_sample_time_seconds": metadata.get(
                    f"{prefix}:SampleTime"
                ),
                "payload_sample_duration_seconds": metadata.get(
                    f"{prefix}:SampleDuration"
                ),
                "sample_count": len(values) // 3,
                "values": values,
            }
        )
    return payloads


def iter_accelerometer_samples(
    payloads: Iterable[Mapping[str, Any]],
) -> Iterable[dict[str, Any]]:
    """Expand decoded payloads into individual x/y/z sample records."""

    for payload in payloads:
        values = payload["values"]
        sample_count = payload["sample_count"]
        payload_time = payload["payload_sample_time_seconds"]
        payload_duration = payload["payload_sample_duration_seconds"]
        sample_duration = (
            payload_duration / sample_count
            if isinstance(payload_duration, (int, float)) and sample_count
            else None
        )

        for sample_index in range(sample_count):
            value_index = sample_index * 3
            estimated_time = (
                payload_time + sample_index * sample_duration
                if isinstance(payload_time, (int, float))
                and sample_duration is not None
                else None
            )
            yield {
                "document": payload["document"],
                "track": payload["track"],
                "copy": payload["copy"],
                "source_field": payload["source_field"],
                "payload_sample_time_seconds": payload_time,
                "payload_sample_duration_seconds": payload_duration,
                "sample_index_in_payload": sample_index,
                "estimated_sample_time_seconds": estimated_time,
                "estimated_sample_duration_seconds": sample_duration,
                "x": values[value_index],
                "y": values[value_index + 1],
                "z": values[value_index + 2],
            }


def build_dataset(
    data_root: Path,
    exiftool_command: str,
    max_videos_per_camera: int | None,
) -> dict[str, Any]:
    """Discover cameras and videos, then build the complete JSON dataset."""

    data_root = data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root is not a directory: {data_root}")

    files_by_camera = discover_camera_videos(data_root, max_videos_per_camera)
    video_count = sum(len(files) for _, files in files_by_camera)

    exiftool: str | None = None
    version: str | None = None
    if video_count:
        exiftool = resolve_exiftool(exiftool_command)
        version = exiftool_version(exiftool)

    cameras: list[dict[str, Any]] = []
    processed = 0
    for camera_folder, video_files in files_by_camera:
        videos: list[dict[str, Any]] = []
        for video_path in video_files:
            processed += 1
            print(f"[{processed}/{video_count}] Reading {video_path}", flush=True)
            assert exiftool is not None
            videos.append(
                {
                    "file_name": video_path.name,
                    "relative_path": _relative_to_root(video_path, data_root),
                    "absolute_path": str(video_path),
                    "metadata": extract_metadata(video_path, exiftool),
                }
            )

        cameras.append(
            {
                "folder_name": camera_folder.name,
                "friendly_name": friendly_camera_name(camera_folder.name),
                "relative_path": _relative_to_root(camera_folder, data_root),
                "absolute_path": str(camera_folder),
                "videos": videos,
            }
        )

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "exiftool_version": version,
        "max_videos_per_camera": max_videos_per_camera,
        "camera_count": len(cameras),
        "video_count": video_count,
        "cameras": cameras,
    }


def _tag_name(metadata_key: str) -> str:
    """Remove ExifTool group prefixes from a JSON metadata key."""

    return metadata_key.rsplit(":", 1)[-1]


def _value_identity(value: Any) -> str:
    """Create a stable comparison key for scalar or structured values."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def preferred_value(
    metadata: Mapping[str, Any], aliases: Sequence[str]
) -> tuple[Any, set[str]]:
    """Collect the unique values matching one preferred CSV field.

    Multiple group/document instances are common when ``-ee`` is enabled.  A
    single unique value is written as-is; differing values are retained as a
    JSON array instead of arbitrarily discarding all but one.
    """

    for alias in aliases:
        matched_keys: list[str] = []
        values: list[Any] = []
        seen: set[str] = set()
        for key, value in metadata.items():
            if _tag_name(key).casefold() != alias.casefold():
                continue
            matched_keys.append(key)
            identity = _value_identity(value)
            if identity not in seen:
                seen.add(identity)
                values.append(value)
        if len(values) == 1:
            return values[0], set(matched_keys)
        if values:
            return values, set(matched_keys)
    return "", set()


def select_preferred_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return only the requested, human-readable metadata fields."""

    return {
        heading: preferred_value(metadata, aliases)[0]
        for heading, aliases in PREFERRED_COLUMNS
    }


def build_selected_dataset(dataset: Mapping[str, Any]) -> dict[str, Any]:
    """Copy the dataset structure while retaining only preferred metadata."""

    selected_cameras: list[dict[str, Any]] = []
    for camera in dataset["cameras"]:
        selected_videos = [
            {
                "file_name": video["file_name"],
                "relative_path": video["relative_path"],
                "absolute_path": video["absolute_path"],
                "metadata": select_preferred_metadata(video["metadata"]),
            }
            for video in camera["videos"]
        ]
        selected_cameras.append(
            {
                "folder_name": camera["folder_name"],
                "friendly_name": camera["friendly_name"],
                "relative_path": camera["relative_path"],
                "absolute_path": camera["absolute_path"],
                "videos": selected_videos,
            }
        )

    selected_dataset = {
        key: value
        for key, value in dataset.items()
        if key != "cameras"
    }
    selected_dataset["metadata_selection"] = "preferred_fields_only"
    selected_dataset["cameras"] = selected_cameras
    return selected_dataset


def iter_video_records(dataset: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    """Yield CSV records containing identifiers and preferred fields only."""

    for camera in dataset["cameras"]:
        for video in camera["videos"]:
            metadata = video["metadata"]
            record = select_preferred_metadata(metadata)

            record.update(
                {
                    "Camera Folder Name": camera["folder_name"],
                    "Video File Name": video["file_name"],
                }
            )

            yield record


def _csv_value(value: Any) -> str:
    """Serialize structured metadata without losing its shape."""

    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def write_json(dataset: Mapping[str, Any], output_path: Path) -> None:
    """Write the complete, nested metadata dataset."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(dataset, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")


def write_csv(dataset: Mapping[str, Any], output_path: Path) -> None:
    """Write one preferred-fields-only CSV row per video."""

    records = list(iter_video_records(dataset))
    preferred_headings = [heading for heading, _ in PREFERRED_COLUMNS]
    fieldnames = list(ASSOCIATION_COLUMNS) + preferred_headings

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({key: _csv_value(value) for key, value in record.items()})


def accelerometer_output_paths(
    output_directory: Path,
    camera: Mapping[str, Any],
    video: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Return the requested camera/video-based paths in the output folder."""

    output_stem = (
        f"gopro_accelerometer_data_{camera['folder_name']}_"
        f"{Path(video['file_name']).stem}"
    )
    return (
        output_directory / f"{output_stem}.json",
        output_directory / f"{output_stem}.csv",
    )


def write_accelerometer_video_outputs(
    run_metadata: Mapping[str, Any],
    camera: Mapping[str, Any],
    video: Mapping[str, Any],
    exiftool: str,
    output_directory: Path,
) -> tuple[Path, Path, int]:
    """Extract and immediately write one video's accelerometer JSON and CSV."""

    payloads = extract_accelerometer_payloads(Path(video["absolute_path"]), exiftool)
    sample_count = sum(payload["sample_count"] for payload in payloads)
    json_output_path, csv_output_path = accelerometer_output_paths(
        output_directory, camera, video
    )
    json_output_path.parent.mkdir(parents=True, exist_ok=True)

    json_header = {
        "schema_version": 1,
        "generated_at_utc": run_metadata["generated_at_utc"],
        "data_root": run_metadata["data_root"],
        "exiftool_version": run_metadata["exiftool_version"],
        "camera_folder_name": camera["folder_name"],
        "camera_friendly_name": camera["friendly_name"],
        "camera_relative_path": camera["relative_path"],
        "camera_absolute_path": camera["absolute_path"],
        "video_file_name": video["file_name"],
        "video_relative_path": video["relative_path"],
        "video_absolute_path": video["absolute_path"],
        "sample_count": sample_count,
        "units": "m/s^2",
        "axis_note": "x/y/z follow the numeric order returned by ExifTool.",
        "timing_note": (
            "Estimated sample times evenly divide each payload's SampleDuration; "
            "payload times and source fields are retained."
        ),
    }

    with json_output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as json_file, csv_output_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=ACCELEROMETER_CSV_COLUMNS)
        csv_writer.writeheader()

        json_file.write("{\n")
        for key, value in json_header.items():
            json_file.write(
                f"  {json.dumps(key)}: "
                f"{json.dumps(value, ensure_ascii=False)},\n"
            )
        json_file.write('  "samples": [\n')

        first_sample = True
        for sample in iter_accelerometer_samples(payloads):
            if not first_sample:
                json_file.write(",\n")
            first_sample = False
            json_file.write("    ")
            json.dump(sample, json_file, ensure_ascii=False)

            csv_writer.writerow(
                {
                    "Camera Folder Name": camera["folder_name"],
                    "Video File Name": video["file_name"],
                    "Document": sample["document"],
                    "Track": sample["track"],
                    "Copy": sample["copy"],
                    "Source Field": sample["source_field"],
                    "Payload Sample Time (s)": sample[
                        "payload_sample_time_seconds"
                    ],
                    "Payload Sample Duration (s)": sample[
                        "payload_sample_duration_seconds"
                    ],
                    "Sample Index In Payload": sample["sample_index_in_payload"],
                    "Estimated Sample Time (s)": sample[
                        "estimated_sample_time_seconds"
                    ],
                    "Estimated Sample Duration (s)": sample[
                        "estimated_sample_duration_seconds"
                    ],
                    "X (m/s^2)": sample["x"],
                    "Y (m/s^2)": sample["y"],
                    "Z (m/s^2)": sample["z"],
                }
            )

        json_file.write("\n  ]\n}\n")

    return json_output_path, csv_output_path, sample_count


def _start_streamed_dataset(
    output_file: Any, header: Mapping[str, Any]
) -> None:
    """Write the beginning of a dataset whose cameras will be streamed."""

    output_file.write("{\n")
    for key, value in header.items():
        output_file.write(
            f"  {json.dumps(key)}: {json.dumps(value, ensure_ascii=False)},\n"
        )
    output_file.write('  "cameras": [\n')


def _start_streamed_camera(
    output_file: Any,
    camera: Mapping[str, Any],
    first_camera: bool,
) -> None:
    """Write one camera header and begin its streamed video array."""

    if not first_camera:
        output_file.write(",\n")
    output_file.write("    {\n")
    for key in ("folder_name", "friendly_name", "relative_path", "absolute_path"):
        output_file.write(
            f"      {json.dumps(key)}: "
            f"{json.dumps(camera[key], ensure_ascii=False)},\n"
        )
    output_file.write('      "videos": [\n')


def process_videos_incrementally(
    data_root: Path,
    exiftool_command: str,
    max_videos_per_camera: int | None,
    selected_json_path: Path,
    all_json_path: Path,
    selected_csv_path: Path,
    accelerometer_output_directory: Path,
) -> dict[str, Any]:
    """Extract one video at a time and immediately flush every output."""

    data_root = data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root is not a directory: {data_root}")

    files_by_camera = discover_camera_videos(data_root, max_videos_per_camera)
    video_count = sum(len(video_files) for _, video_files in files_by_camera)
    exiftool = resolve_exiftool(exiftool_command) if video_count else None
    version = exiftool_version(exiftool) if exiftool is not None else None
    generated_at_utc = datetime.now(timezone.utc).isoformat()
    run_metadata = {
        "schema_version": 1,
        "generated_at_utc": generated_at_utc,
        "data_root": str(data_root),
        "exiftool_version": version,
        "max_videos_per_camera": max_videos_per_camera,
        "camera_count": len(files_by_camera),
        "video_count": video_count,
    }
    selected_header = dict(run_metadata)
    selected_header["metadata_selection"] = "preferred_fields_only"

    for path in (selected_json_path, all_json_path, selected_csv_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    accelerometer_output_directory.mkdir(parents=True, exist_ok=True)

    preferred_headings = [heading for heading, _ in PREFERRED_COLUMNS]
    csv_fieldnames = list(ASSOCIATION_COLUMNS) + preferred_headings

    with selected_json_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as selected_json_file, all_json_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as all_json_file, selected_csv_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as selected_csv_file:
        _start_streamed_dataset(selected_json_file, selected_header)
        _start_streamed_dataset(all_json_file, run_metadata)
        csv_writer = csv.DictWriter(
            selected_csv_file,
            fieldnames=csv_fieldnames,
            extrasaction="ignore",
        )
        csv_writer.writeheader()
        selected_csv_file.flush()

        processed = 0
        for camera_index, (camera_folder, video_files) in enumerate(files_by_camera):
            camera = {
                "folder_name": camera_folder.name,
                "friendly_name": friendly_camera_name(camera_folder.name),
                "relative_path": _relative_to_root(camera_folder, data_root),
                "absolute_path": str(camera_folder),
            }
            _start_streamed_camera(
                selected_json_file, camera, first_camera=camera_index == 0
            )
            _start_streamed_camera(
                all_json_file, camera, first_camera=camera_index == 0
            )

            for video_index, video_path in enumerate(video_files):
                processed += 1
                print(f"[{processed}/{video_count}] Reading {video_path}", flush=True)
                assert exiftool is not None
                metadata = extract_metadata(video_path, exiftool)
                video = {
                    "file_name": video_path.name,
                    "relative_path": _relative_to_root(video_path, data_root),
                    "absolute_path": str(video_path),
                }
                all_video = dict(video)
                all_video["metadata"] = metadata
                selected_video = dict(video)
                selected_video["metadata"] = select_preferred_metadata(metadata)

                if video_index:
                    selected_json_file.write(",\n")
                    all_json_file.write(",\n")
                selected_json_file.write("        ")
                json.dump(selected_video, selected_json_file, ensure_ascii=False)
                all_json_file.write("        ")
                json.dump(all_video, all_json_file, ensure_ascii=False)

                csv_record = dict(selected_video["metadata"])
                csv_record["Camera Folder Name"] = camera["folder_name"]
                csv_record["Video File Name"] = video["file_name"]
                csv_writer.writerow(
                    {key: _csv_value(value) for key, value in csv_record.items()}
                )

                print(
                    f"[{processed}/{video_count}] Reading accelerometer data "
                    f"from {video_path}",
                    flush=True,
                )
                accelerometer_json, accelerometer_csv, sample_count = (
                    write_accelerometer_video_outputs(
                        run_metadata,
                        camera,
                        video,
                        exiftool,
                        accelerometer_output_directory,
                    )
                )

                # Make progress durable and release this video's large metadata
                # before ExifTool starts on the next video.
                selected_json_file.flush()
                all_json_file.flush()
                selected_csv_file.flush()
                del metadata, all_video, selected_video
                print(
                    f"  Wrote {sample_count} accelerometer samples to "
                    f"{accelerometer_csv.name} and {accelerometer_json.name}",
                    flush=True,
                )

            selected_json_file.write("\n      ]\n    }")
            all_json_file.write("\n      ]\n    }")
            selected_json_file.flush()
            all_json_file.flush()

        selected_json_file.write("\n  ]\n}\n")
        all_json_file.write("\n  ]\n}\n")

    return run_metadata


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse optional command-line overrides for the configuration values."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help=f"Root folder to search (default: {DATA_ROOT})",
    )
    parser.add_argument(
        "--selected-json-output",
        type=Path,
        default=OUTPUT_SELECTED_JSON,
        help=f"Selected-fields JSON path (default: {OUTPUT_SELECTED_JSON})",
    )
    parser.add_argument(
        "--all-json-output",
        type=Path,
        default=OUTPUT_ALL_JSON,
        help=f"All-fields JSON path (default: {OUTPUT_ALL_JSON})",
    )
    parser.add_argument(
        "--selected-csv-output",
        type=Path,
        default=OUTPUT_SELECTED_CSV,
        help=f"Selected-fields CSV path (default: {OUTPUT_SELECTED_CSV})",
    )
    parser.add_argument(
        "--accelerometer-output-directory",
        type=Path,
        default=OUTPUT_ACCELEROMETER_DIRECTORY,
        help=(
            "Directory for per-video accelerometer CSV/JSON pairs "
            f"(default: {OUTPUT_ACCELEROMETER_DIRECTORY})"
        ),
    )
    parser.add_argument(
        "--exiftool",
        default=EXIFTOOL_COMMAND,
        help=f"ExifTool command or full executable path (default: {EXIFTOOL_COMMAND})",
    )
    parser.add_argument(
        "--max-videos-per-camera",
        type=int,
        default=MAX_VIDEOS_PER_CAMERA,
        help=(
            "Maximum videos to process in each camera folder "
            f"(default: "
            f"{'all' if MAX_VIDEOS_PER_CAMERA is None else MAX_VIDEOS_PER_CAMERA})"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run incremental metadata and accelerometer extraction."""

    args = parse_args(argv)
    selected_json_path = args.selected_json_output.expanduser().resolve()
    all_json_path = args.all_json_output.expanduser().resolve()
    selected_csv_path = args.selected_csv_output.expanduser().resolve()
    accelerometer_output_directory = (
        args.accelerometer_output_directory.expanduser().resolve()
    )
    try:
        run_metadata = process_videos_incrementally(
            args.data_root,
            args.exiftool,
            args.max_videos_per_camera,
            selected_json_path,
            all_json_path,
            selected_csv_path,
            accelerometer_output_directory,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(
        f"Wrote metadata for {run_metadata['video_count']} video(s) from "
        f"{run_metadata['camera_count']} GoPro folder(s)."
    )
    print(f"Selected JSON:   {selected_json_path}")
    print(f"All-fields JSON: {all_json_path}")
    print(f"Selected CSV:    {selected_csv_path}")
    print(f"Per-video accelerometer files: {accelerometer_output_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
