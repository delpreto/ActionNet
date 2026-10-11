"""Small example showing how to load and use GoPro metadata JSON."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


# These match the default outputs in gopro_metadata_extraction.py.
SELECTED_METADATA_JSON = Path(__file__).with_name(
    "gopro_video_metadata_selectedFields.json"
)
ALL_METADATA_JSON = Path(__file__).with_name("gopro_video_metadata_allFields.json")
ACCELEROMETER_OUTPUT_DIRECTORY = Path(__file__).resolve().parent


def load_gopro_metadata(
    metadata_path: Path = SELECTED_METADATA_JSON,
) -> dict[str, Any]:
    """Load a metadata dataset written by gopro_metadata_extraction.py."""

    with metadata_path.expanduser().open("r", encoding="utf-8") as input_file:
        dataset = json.load(input_file)

    if not isinstance(dataset, dict) or not isinstance(dataset.get("cameras"), list):
        raise ValueError(f"Unexpected GoPro metadata format in {metadata_path}")
    return dataset


def load_first_accelerometer_sample(
    csv_path: Path | None = None,
) -> dict[str, str] | None:
    """Read one sample from one per-video CSV without loading it in memory."""

    if csv_path is None:
        csv_files = sorted(
            ACCELEROMETER_OUTPUT_DIRECTORY.glob("gopro_accelerometer_data_*.csv")
        )
        if not csv_files:
            return None
        csv_path = csv_files[0]

    with csv_path.expanduser().open("r", encoding="utf-8-sig", newline="") as csv_file:
        return next(csv.DictReader(csv_file), None)


def main() -> None:
    dataset = load_gopro_metadata()

    print(
        f"Loaded {dataset.get('video_count', 0)} video(s) from "
        f"{dataset.get('camera_count', 0)} camera folder(s)."
    )

    for camera in dataset["cameras"]:
        print(
            f"\n{camera['friendly_name']} "
            f"(folder: {camera['folder_name']}, videos: {len(camera['videos'])})"
        )
        for video in camera["videos"]:
            metadata = video["metadata"]
            model = metadata.get("Camera Model Name") or "unknown model"
            print(f"  {video['relative_path']} - {model}")

    if ACCELEROMETER_OUTPUT_DIRECTORY.is_dir():
        first_sample = load_first_accelerometer_sample()
        if first_sample:
            print(
                "\nFirst accelerometer sample: "
                f"t={first_sample['Estimated Sample Time (s)']} s, "
                f"x={first_sample['X (m/s^2)']}, "
                f"y={first_sample['Y (m/s^2)']}, "
                f"z={first_sample['Z (m/s^2)']} m/s^2"
            )


if __name__ == "__main__":
    main()
