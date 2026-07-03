import argparse
import json
from pathlib import Path

import numpy as np

input_hdf5_filepath = r'C:\Users\jdelp\Downloads\2026-06-14_tennis_S02_xsens_myo_data_01.hdf5'

XSENS_SEGMENT_LABELS = [
    "Pelvis",
    "L5",
    "L3",
    "T12",
    "T8",
    "Neck",
    "Head",
    "Right Shoulder",
    "Right Upper Arm",
    "Right Forearm",
    "Right Hand",
    "Left Shoulder",
    "Left Upper Arm",
    "Left Forearm",
    "Left Hand",
    "Right Upper Leg",
    "Right Lower Leg",
    "Right Foot",
    "Right Toe",
    "Left Upper Leg",
    "Left Lower Leg",
    "Left Foot",
    "Left Toe",
    "Racket",
]

SMPL_JOINT_ORDER = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
]

SMPL_PARENT_NAMES = [
    None,
    "pelvis",
    "pelvis",
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "spine3",
    "spine3",
    "neck",
    "left_collar",
    "right_collar",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]

SMPL_TO_XSENS = {
    "pelvis": "Pelvis",
    "left_hip": "Left Upper Leg",
    "right_hip": "Right Upper Leg",
    "spine1": "L5",
    "left_knee": "Left Lower Leg",
    "right_knee": "Right Lower Leg",
    "spine2": "T12",
    "left_ankle": "Left Foot",
    "right_ankle": "Right Foot",
    "spine3": "T8",
    "left_foot": "Left Toe",
    "right_foot": "Right Toe",
    "neck": "Neck",
    "left_collar": "Left Shoulder",
    "right_collar": "Right Shoulder",
    "head": "Head",
    "left_shoulder": "Left Upper Arm",
    "right_shoulder": "Right Upper Arm",
    "left_elbow": "Left Forearm",
    "right_elbow": "Right Forearm",
    "left_wrist": "Left Hand",
    "right_wrist": "Right Hand",
    "left_hand": None,
    "right_hand": None,
}

REST_POSE_DATASETS = {
    "tpose": (
        "body_orientation_Tpose_quaternion_wijk",
        "body_position_Tpose_xyz_m",
    ),
    "tpose_isb": (
        "body_orientation_TposeISB_quaternion_wijk",
        "body_position_TposeISB_xyz_m",
    ),
    "identity": (
        "body_orientation_identity_quaternion_wijk",
        "body_position_identity_xyz_m",
    ),
}

OBJECT_SEGMENT_CANDIDATES = [
    "Racket",
    "Prop 1",
]


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert Xsens-derived HDF5 data into a SMPL-style NPZ pose sequence. "
            "The output uses AMASS-like keys such as poses, trans, betas, and mocap_framerate."
        )
    )
    parser.add_argument("--input_hdf5", 
                        default=input_hdf5_filepath,
                        help="Path to the source HDF5 file.")
    parser.add_argument(
        "-o",
        "--output",
        help=(
            "Output .npz filepath. Defaults to '<input_stem>_smpl_like.npz' "
            "next to the source file."
        ),
    )
    parser.add_argument(
        "--rest-pose",
        choices=sorted(REST_POSE_DATASETS.keys()),
        default="tpose",
        help="Calibration pose used as the zero-pose reference. Default: tpose.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output files if they already exist.",
    )
    return parser.parse_args()


def _require_dataset(h5_file, dataset_path):
    if dataset_path not in h5_file:
        raise KeyError("Missing required dataset: %s" % dataset_path)
    return np.array(h5_file[dataset_path])


def _get_output_path(input_path, output_path_arg):
    if output_path_arg:
        output_path = Path(output_path_arg)
    else:
        output_path = input_path.with_name("%s_smpl_like.npz" % input_path.stem)
    if output_path.suffix.lower() != ".npz":
        output_path = output_path.with_suffix(".npz")
    return output_path


def _quat_wxyz_to_xyzw(quaternions_wxyz):
    return quaternions_wxyz[..., [1, 2, 3, 0]]


def _quat_xyzw_to_wxyz(quaternions_xyzw):
    return quaternions_xyzw[..., [3, 0, 1, 2]]


def _rotation_batch_from_wxyz(quaternions_wxyz):
    from scipy.spatial.transform import Rotation

    return Rotation.from_quat(_quat_wxyz_to_xyzw(quaternions_wxyz))


def _frame_rate_from_timestamps(time_s):
    if time_s.size < 2:
        return 0.0
    duration_s = float(time_s[-1] - time_s[0])
    if duration_s <= 0:
        return 0.0
    return float((time_s.size - 1) / duration_s)


def _validate_segment_arrays(segment_positions, segment_quaternions, rest_positions, rest_quaternions):
    if segment_positions.ndim != 3 or segment_positions.shape[-1] != 3:
        raise ValueError(
            "Expected body positions with shape [num_frames, num_segments, 3], got %s"
            % (segment_positions.shape,)
        )
    if segment_quaternions.ndim != 3 or segment_quaternions.shape[-1] != 4:
        raise ValueError(
            "Expected body quaternions with shape [num_frames, num_segments, 4], got %s"
            % (segment_quaternions.shape,)
        )
    if rest_positions.ndim != 2 or rest_positions.shape[-1] != 3:
        raise ValueError(
            "Expected rest positions with shape [num_segments, 3], got %s"
            % (rest_positions.shape,)
        )
    if rest_quaternions.ndim != 2 or rest_quaternions.shape[-1] != 4:
        raise ValueError(
            "Expected rest quaternions with shape [num_segments, 4], got %s"
            % (rest_quaternions.shape,)
        )
    if segment_positions.shape[:2] != segment_quaternions.shape[:2]:
        raise ValueError("Segment positions and quaternions do not share the same frame/segment shape.")
    if segment_positions.shape[1] != rest_positions.shape[0]:
        raise ValueError("Rest positions do not match the number of streamed segments.")
    if segment_positions.shape[1] != rest_quaternions.shape[0]:
        raise ValueError("Rest quaternions do not match the number of streamed segments.")


def _build_rotation_lookups(segment_labels, segment_quaternions_wxyz, rest_quaternions_wxyz):
    global_rotations = {}
    rest_rotations = {}
    for segment_index, segment_label in enumerate(segment_labels):
        global_rotations[segment_label] = _rotation_batch_from_wxyz(segment_quaternions_wxyz[:, segment_index, :])
        rest_rotations[segment_label] = _rotation_batch_from_wxyz(rest_quaternions_wxyz[segment_index, :])
    return global_rotations, rest_rotations


def _convert_to_smpl_like(segment_labels, segment_positions, segment_quaternions, rest_positions, rest_quaternions):
    segment_index = {label: index for index, label in enumerate(segment_labels)}
    missing_segments = sorted(
        {
            xsens_label
            for xsens_label in SMPL_TO_XSENS.values()
            if xsens_label is not None and xsens_label not in segment_index
        }
    )
    if missing_segments:
        raise ValueError(
            "The HDF5 file is missing Xsens segments needed for SMPL export: %s"
            % ", ".join(missing_segments)
        )

    num_frames = segment_positions.shape[0]
    num_joints = len(SMPL_JOINT_ORDER)
    global_rotations, rest_rotations = _build_rotation_lookups(
        segment_labels, segment_quaternions, rest_quaternions
    )

    joint_positions = np.zeros((num_frames, num_joints, 3), dtype=np.float32)
    rest_joint_positions = np.zeros((num_joints, 3), dtype=np.float32)
    rest_offsets = np.zeros((num_joints, 3), dtype=np.float32)
    local_axis_angle = np.zeros((num_frames, num_joints, 3), dtype=np.float32)
    local_quaternions = np.zeros((num_frames, num_joints, 4), dtype=np.float32)
    local_quaternions[:, :, 0] = 1.0
    global_quaternions = np.zeros((num_frames, num_joints, 4), dtype=np.float32)
    global_quaternions[:, :, 0] = 1.0
    joint_available = np.zeros((num_joints,), dtype=bool)

    joint_name_to_index = {name: index for index, name in enumerate(SMPL_JOINT_ORDER)}

    for joint_index, joint_name in enumerate(SMPL_JOINT_ORDER):
        parent_name = SMPL_PARENT_NAMES[joint_index]
        parent_index = -1 if parent_name is None else joint_name_to_index[parent_name]
        xsens_label = SMPL_TO_XSENS[joint_name]

        if xsens_label is None:
            if parent_index >= 0:
                joint_positions[:, joint_index, :] = joint_positions[:, parent_index, :]
                rest_joint_positions[joint_index, :] = rest_joint_positions[parent_index, :]
            continue

        xsens_segment_index = segment_index[xsens_label]
        current_global_rotation = global_rotations[xsens_label]
        rest_global_rotation = rest_rotations[xsens_label]

        joint_positions[:, joint_index, :] = segment_positions[:, xsens_segment_index, :].astype(np.float32)
        rest_joint_positions[joint_index, :] = rest_positions[xsens_segment_index, :].astype(np.float32)
        global_quaternions[:, joint_index, :] = _quat_xyzw_to_wxyz(
            current_global_rotation.as_quat()
        ).astype(np.float32)
        joint_available[joint_index] = True

        if parent_index < 0:
            pose_rotation = rest_global_rotation.inv() * current_global_rotation
            rest_offsets[joint_index, :] = rest_joint_positions[joint_index, :]
        else:
            parent_xsens_label = SMPL_TO_XSENS[parent_name]
            if parent_xsens_label is None:
                raise ValueError("Joint %s depends on an unmapped parent %s." % (joint_name, parent_name))
            parent_global_rotation = global_rotations[parent_xsens_label]
            parent_rest_rotation = rest_rotations[parent_xsens_label]
            current_relative_rotation = parent_global_rotation.inv() * current_global_rotation
            rest_relative_rotation = parent_rest_rotation.inv() * rest_global_rotation
            pose_rotation = rest_relative_rotation.inv() * current_relative_rotation
            parent_xsens_segment_index = segment_index[parent_xsens_label]
            rest_offsets[joint_index, :] = (
                rest_positions[xsens_segment_index, :] - rest_positions[parent_xsens_segment_index, :]
            ).astype(np.float32)

        local_axis_angle[:, joint_index, :] = pose_rotation.as_rotvec().astype(np.float32)
        local_quaternions[:, joint_index, :] = _quat_xyzw_to_wxyz(
            pose_rotation.as_quat()
        ).astype(np.float32)

    return {
        "joint_positions": joint_positions,
        "rest_joint_positions": rest_joint_positions,
        "rest_offsets": rest_offsets,
        "local_axis_angle": local_axis_angle,
        "local_quaternions": local_quaternions,
        "global_quaternions": global_quaternions,
        "joint_available": joint_available,
    }


def _extract_object_stream(segment_labels, segment_positions, segment_quaternions):
    segment_index = {label: index for index, label in enumerate(segment_labels)}
    for object_label in OBJECT_SEGMENT_CANDIDATES:
        if object_label in segment_index:
            object_index = segment_index[object_label]
            object_rotvec = _rotation_batch_from_wxyz(
                segment_quaternions[:, object_index, :]
            ).as_rotvec()
            return {
                "object_name": object_label,
                "positions": segment_positions[:, object_index, :].astype(np.float32),
                "quaternions_wxyz": segment_quaternions[:, object_index, :].astype(np.float32),
                "rotvec": object_rotvec.astype(np.float32),
            }
    return None


def main():
    import h5py

    args = _parse_args()
    input_path = Path(args.input_hdf5).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError("Input HDF5 file does not exist: %s" % input_path)

    output_path = _get_output_path(input_path, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path.with_suffix(".json")
    aitviewer_output_path = output_path.with_name("%s_aitviewer.npz" % output_path.stem)

    if not args.overwrite:
        for existing_path in (output_path, metadata_path, aitviewer_output_path):
            if existing_path.exists():
                raise FileExistsError(
                    "Output file already exists: %s. Use --overwrite to replace it."
                    % existing_path
                )

    rest_orientation_dataset, rest_position_dataset = REST_POSE_DATASETS[args.rest_pose]

    with h5py.File(input_path, "r") as h5_file:
        timestamps_s = np.squeeze(
            _require_dataset(h5_file, "xsens-segments/body_orientation_quaternion_wijk/time_s")
        ).astype(np.float64)
        segment_positions = _require_dataset(h5_file, "xsens-segments/body_position_xyz_m/data").astype(np.float64)
        segment_quaternions = _require_dataset(
            h5_file, "xsens-segments/body_orientation_quaternion_wijk/data"
        ).astype(np.float64)
        rest_positions = _require_dataset(
            h5_file, "xsens-segments-tpose/%s" % rest_position_dataset
        ).astype(np.float64)
        rest_quaternions = _require_dataset(
            h5_file, "xsens-segments-tpose/%s" % rest_orientation_dataset
        ).astype(np.float64)

    _validate_segment_arrays(segment_positions, segment_quaternions, rest_positions, rest_quaternions)

    num_segments = segment_positions.shape[1]
    if num_segments > len(XSENS_SEGMENT_LABELS):
        raise ValueError(
            "This script knows %d Xsens segments, but the file contains %d."
            % (len(XSENS_SEGMENT_LABELS), num_segments)
        )
    segment_labels = XSENS_SEGMENT_LABELS[:num_segments]

    smpl_like = _convert_to_smpl_like(
        segment_labels=segment_labels,
        segment_positions=segment_positions,
        segment_quaternions=segment_quaternions,
        rest_positions=rest_positions,
        rest_quaternions=rest_quaternions,
    )
    object_stream = _extract_object_stream(
        segment_labels=segment_labels,
        segment_positions=segment_positions,
        segment_quaternions=segment_quaternions,
    )

    mocap_framerate = _frame_rate_from_timestamps(timestamps_s)
    trans = smpl_like["joint_positions"][:, 0, :].copy()
    trans_origin_centered = trans - trans[0:1, :]
    poses = smpl_like["local_axis_angle"].reshape((segment_positions.shape[0], -1))
    root_orient = smpl_like["local_axis_angle"][:, 0, :]
    body_pose = smpl_like["local_axis_angle"][:, 1:, :].reshape((segment_positions.shape[0], -1))
    betas = np.zeros((10,), dtype=np.float32)

    main_output_data = {
        "poses": poses.astype(np.float32),
        "root_orient": root_orient.astype(np.float32),
        "body_pose": body_pose.astype(np.float32),
        "trans": trans.astype(np.float32),
        "trans_origin_centered": trans_origin_centered.astype(np.float32),
        "betas": betas,
        "gender": np.array("neutral"),
        "mocap_framerate": np.array(mocap_framerate, dtype=np.float32),
        "timestamps_s": timestamps_s.astype(np.float64),
        "joint_positions": smpl_like["joint_positions"].astype(np.float32),
        "rest_joint_positions": smpl_like["rest_joint_positions"].astype(np.float32),
        "rest_offsets": smpl_like["rest_offsets"].astype(np.float32),
        "local_quaternions_wxyz": smpl_like["local_quaternions"].astype(np.float32),
        "global_quaternions_wxyz": smpl_like["global_quaternions"].astype(np.float32),
        "joint_available": smpl_like["joint_available"],
        "smpl_joint_labels": np.asarray(SMPL_JOINT_ORDER),
        "smpl_parent_names": np.asarray(["" if parent is None else parent for parent in SMPL_PARENT_NAMES]),
        "xsens_segment_labels": np.asarray(segment_labels),
    }
    aitviewer_output_data = {
        "poses_body": body_pose.astype(np.float32),
        "poses_root": root_orient.astype(np.float32),
        "betas": betas,
        "trans": trans.astype(np.float32),
        "gender": np.array("neutral"),
        "mocap_framerate": np.array(mocap_framerate, dtype=np.float32),
    }
    if object_stream is not None:
        main_output_data["object_name"] = np.array(object_stream["object_name"])
        main_output_data["object_positions"] = object_stream["positions"]
        main_output_data["object_quaternions_wxyz"] = object_stream["quaternions_wxyz"]
        main_output_data["object_rotvec"] = object_stream["rotvec"]
        aitviewer_output_data["object_name"] = np.array(object_stream["object_name"])
        aitviewer_output_data["object_positions"] = object_stream["positions"]
        aitviewer_output_data["object_quaternions_wxyz"] = object_stream["quaternions_wxyz"]
        aitviewer_output_data["object_rotvec"] = object_stream["rotvec"]

    np.savez_compressed(output_path, **main_output_data)

    np.savez_compressed(aitviewer_output_path, **aitviewer_output_data)

    mapped_xsens_segments = sorted(
        {xsens_label for xsens_label in SMPL_TO_XSENS.values() if xsens_label is not None}
    )
    omitted_xsens_segments = [label for label in segment_labels if label not in mapped_xsens_segments]
    approximated_joints = [joint_name for joint_name, xsens_label in SMPL_TO_XSENS.items() if xsens_label is None]

    metadata = {
        "source_hdf5": str(input_path),
        "output_npz": str(output_path.resolve()),
        "aitviewer_output_npz": str(aitviewer_output_path.resolve()),
        "rest_pose": args.rest_pose,
        "num_frames": int(segment_positions.shape[0]),
        "num_xsens_segments": int(num_segments),
        "mocap_framerate_hz": mocap_framerate,
        "smpl_joint_order": SMPL_JOINT_ORDER,
        "smpl_parent_names": SMPL_PARENT_NAMES,
        "smpl_to_xsens_segment": SMPL_TO_XSENS,
        "approximated_joints": approximated_joints,
        "omitted_xsens_segments": omitted_xsens_segments,
        "object_segment_exported": None if object_stream is None else object_stream["object_name"],
        "notes": [
            "This is a SMPL-style pose export, not a full SMPL mesh fit.",
            "The file stores AMASS-like pose fields plus additional metadata for traceability.",
            "A second NPZ is also written in aitviewer.from_npz-compatible format with poses_body, poses_root, betas, and trans.",
            "If a tracked prop segment is present, it is exported separately as object_positions and object_quaternions_wxyz.",
            "betas are zero placeholders because body shape fitting is outside the scope of this converter.",
            "left_hand and right_hand are not present as distinct Xsens body segments, so they are emitted as identity child joints of the wrists.",
            "The torso is reduced from the Xsens chain to a 3-spine SMPL hierarchy using L5, T12, and T8.",
        ],
    }

    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    print("Saved SMPL-style pose export to: %s" % output_path)
    print("Saved aitviewer-compatible NPZ to: %s" % aitviewer_output_path)
    print("Saved conversion metadata to  : %s" % metadata_path)
    print("Frames: %d | Approx. frame rate: %.3f Hz" % (segment_positions.shape[0], mocap_framerate))


if __name__ == "__main__":
    main()
