import argparse
import os
import sys
from pathlib import Path

# Windows builds of torch / MKL / Qt stacks can load conflicting OpenMP runtimes.
# Allow the process to continue rather than failing during viewer startup.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# When running from a conda env on Windows, explicitly point Qt at the env's
# plugin folders so platform plugins like qwindows.dll are found reliably.
_env_root = Path(sys.executable).resolve().parent.parent
_qt_bin_dir = _env_root / "Library" / "bin"
_qt_lib_bin_dir = _env_root / "Library" / "lib" / "qt6" / "bin"
_qt_plugins_dir = _env_root / "Library" / "lib" / "qt6" / "plugins"
_qt_platforms_dir = _qt_plugins_dir / "platforms"
_path_entries = os.environ.get("PATH", "").split(os.pathsep) if os.environ.get("PATH") else []
for _path_dir in (_qt_bin_dir, _qt_lib_bin_dir):
    if _path_dir.exists() and str(_path_dir) not in _path_entries:
        os.environ["PATH"] = str(_path_dir) + os.pathsep + os.environ.get("PATH", "")
if _qt_plugins_dir.exists():
    os.environ.setdefault("QT_PLUGIN_PATH", str(_qt_plugins_dir))
if _qt_platforms_dir.exists():
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(_qt_platforms_dir))

import numpy as np

input_npz_file = r'C:\Users\jdelp\Downloads\2026-06-14_tennis_S02_xsens_myo_data_01_smpl_like_aitviewer.npz'

def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize a converted Xsens-to-SMPL NPZ with aitviewer, including a tracked prop "
            "such as a racket when available."
        )
    )
    parser.add_argument("--input_npz", 
                        default=input_npz_file,
                        help="Path to a converted NPZ file.")
    parser.add_argument(
        "--model-type",
        default="smpl",
        choices=["smpl", "smplh", "smplx"],
        help="SMPL-family model to instantiate in aitviewer. Default: smpl.",
    )
    parser.add_argument(
        "--y-up",
        dest="z_up",
        action="store_false",
        help="Treat the exported coordinates as Y-up instead of Z-up.",
    )
    parser.set_defaults(z_up=True)
    parser.add_argument(
        "--object-rot-euler-deg",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("RX", "RY", "RZ"),
        help=(
            "Optional extra XYZ Euler rotation in degrees applied to the local racket mesh "
            "before the tracked object orientation. Useful if the sensor is mounted with an offset."
        ),
    )
    return parser.parse_args()


def _load_pose_arrays(data):
    if "poses_body" in data and "poses_root" in data:
        poses_body = np.asarray(data["poses_body"], dtype=np.float32)
        poses_root = np.asarray(data["poses_root"], dtype=np.float32)
    elif "body_pose" in data and "root_orient" in data:
        poses_body = np.asarray(data["body_pose"], dtype=np.float32)
        poses_root = np.asarray(data["root_orient"], dtype=np.float32)
    else:
        raise KeyError(
            "Input file does not contain either poses_body/poses_root or body_pose/root_orient."
        )

    if "trans" not in data:
        raise KeyError("Input file is missing required key: trans")

    trans = np.asarray(data["trans"], dtype=np.float32)
    betas = np.asarray(data["betas"], dtype=np.float32) if "betas" in data else np.zeros((10,), dtype=np.float32)
    gender = "neutral"
    if "gender" in data:
        gender = str(np.asarray(data["gender"]).item())
    return poses_body, poses_root, trans, betas, gender


def _load_object_arrays(data):
    required_keys = {"object_positions", "object_quaternions_wxyz"}
    if not required_keys.issubset(data.files):
        return None

    object_name = "object"
    if "object_name" in data:
        object_name = str(np.asarray(data["object_name"]).item())

    return {
        "name": object_name,
        "positions": np.asarray(data["object_positions"], dtype=np.float32),
        "quaternions_wxyz": np.asarray(data["object_quaternions_wxyz"], dtype=np.float32),
    }


def _make_transform(rotation_matrix, translation):
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = translation
    return transform


def _cylinder_between(trimesh, point_a, point_b, radius, sections=16):
    point_a = np.asarray(point_a, dtype=np.float32)
    point_b = np.asarray(point_b, dtype=np.float32)
    direction = point_b - point_a
    length = float(np.linalg.norm(direction))
    if length <= 1e-8:
        return None

    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    direction_unit = direction / length
    cross = np.cross(z_axis, direction_unit)
    cross_norm = float(np.linalg.norm(cross))
    dot = float(np.clip(np.dot(z_axis, direction_unit), -1.0, 1.0))

    if cross_norm <= 1e-8:
        if dot > 0.0:
            rotation_matrix = np.eye(3, dtype=np.float32)
        else:
            rotation_matrix = np.array(
                [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
                dtype=np.float32,
            )
    else:
        cross_unit = cross / cross_norm
        skew = np.array(
            [
                [0.0, -cross_unit[2], cross_unit[1]],
                [cross_unit[2], 0.0, -cross_unit[0]],
                [-cross_unit[1], cross_unit[0], 0.0],
            ],
            dtype=np.float32,
        )
        angle = np.arccos(dot)
        rotation_matrix = (
            np.eye(3, dtype=np.float32)
            + np.sin(angle) * skew
            + (1.0 - np.cos(angle)) * (skew @ skew)
        )

    midpoint = 0.5 * (point_a + point_b)
    return trimesh.creation.cylinder(
        radius=radius,
        height=length,
        sections=sections,
        transform=_make_transform(rotation_matrix, midpoint),
    )


def _create_racket_proxy_mesh():
    import trimesh

    parts = []

    handle = trimesh.creation.box(extents=[0.030, 0.180, 0.024])
    handle.apply_translation([0.0, -0.140, 0.0])
    parts.append(handle)

    shaft = trimesh.creation.box(extents=[0.038, 0.110, 0.018])
    shaft.apply_translation([0.0, -0.010, 0.0])
    parts.append(shaft)

    hoop_center_y = 0.165
    hoop_rx = 0.145
    hoop_rz = 0.190
    hoop_radius = 0.012
    num_segments = 24
    angles = np.linspace(0.0, 2.0 * np.pi, num_segments + 1)
    hoop_points = np.stack(
        [
            hoop_rx * np.cos(angles),
            np.full_like(angles, hoop_center_y),
            hoop_rz * np.sin(angles),
        ],
        axis=1,
    )
    for start_point, end_point in zip(hoop_points[:-1], hoop_points[1:]):
        segment = _cylinder_between(trimesh, start_point, end_point, radius=hoop_radius, sections=12)
        if segment is not None:
            parts.append(segment)

    throat_left = _cylinder_between(
        trimesh,
        [-0.030, 0.045, 0.0],
        [-0.070, 0.105, 0.0],
        radius=0.008,
        sections=10,
    )
    throat_right = _cylinder_between(
        trimesh,
        [0.030, 0.045, 0.0],
        [0.070, 0.105, 0.0],
        radius=0.008,
        sections=10,
    )
    if throat_left is not None:
        parts.append(throat_left)
    if throat_right is not None:
        parts.append(throat_right)

    mesh = trimesh.util.concatenate(parts)
    return mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32)


def _add_object_to_scene(viewer, object_data, extra_local_rotation_deg, z_up):
    from scipy.spatial.transform import Rotation

    from aitviewer.renderables.meshes import Meshes

    object_vertices, object_faces = _create_racket_proxy_mesh()
    object_rotations = Rotation.from_quat(
        object_data["quaternions_wxyz"][:, [1, 2, 3, 0]]
    ).as_matrix()
    if np.any(np.asarray(extra_local_rotation_deg) != 0.0):
        extra_rotation = Rotation.from_euler("xyz", extra_local_rotation_deg, degrees=True).as_matrix()
        object_rotations = object_rotations @ extra_rotation[np.newaxis, :, :]

    object_mesh = Meshes.instanced(
        object_vertices,
        object_faces,
        positions=object_data["positions"][:, np.newaxis, :],
        rotations=object_rotations[:, np.newaxis, :, :],
        color=(0.16, 0.16, 0.16, 1.0),
        name=object_data["name"],
        flat_shading=False,
        draw_edges=False,
        z_up=z_up,
    )
    viewer.scene.add(object_mesh)


def main():
    args = _parse_args()
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[2]
    smpl_models_dir = repo_root / "smpl_models"
    os.environ.setdefault("AITVRC", str(script_dir))

    from aitviewer.configuration import CONFIG as AITV_CONFIG
    from aitviewer.models.smpl import SMPLLayer
    from aitviewer.renderables.smpl import SMPLSequence
    from aitviewer.viewer import Viewer

    AITV_CONFIG.update_conf(
        {
            "smplx_models": str(smpl_models_dir),
            "z_up": args.z_up,
        }
    )

    input_path = Path(args.input_npz).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError("Input NPZ does not exist: %s" % input_path)

    data = np.load(input_path, allow_pickle=False)
    poses_body, poses_root, trans, betas, gender = _load_pose_arrays(data)
    object_data = _load_object_arrays(data)

    smpl_layer = SMPLLayer(model_type=args.model_type, gender=gender)
    sequence = SMPLSequence(
        smpl_layer=smpl_layer,
        poses_body=poses_body,
        poses_root=poses_root,
        betas=betas,
        trans=trans,
        z_up=args.z_up,
        name=input_path.stem,
        show_joint_angles=False,
    )

    viewer = Viewer()
    viewer.run_animations = True
    viewer.scene.add(sequence)
    if object_data is not None:
        _add_object_to_scene(
            viewer=viewer,
            object_data=object_data,
            extra_local_rotation_deg=args.object_rot_euler_deg,
            z_up=args.z_up,
        )
    else:
        print("No tracked object stream found in the NPZ; displaying body only.")

    viewer.run()


if __name__ == "__main__":
    main()
