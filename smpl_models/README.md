# SMPL Model Files

This folder is the configured `aitviewer` model root for the scripts in:

- `recording_data/post_processing/smpl_conversion`

The viewer resolves this folder relative to the cloned repository root, so it
does not depend on the absolute path or the local name of the repo folder.

The official SMPL-family body model files are license-gated and must be
downloaded manually after accepting the relevant license terms.

Expected usage in this project:

- `smpl/` for the base SMPL model files used by the current viewer script
- `smplh/` if you later want SMPL+H
- `smplx/` if you later want SMPL-X

After downloading the model archives from the official SMPL / SMPL-X sources,
extract or copy the files into the corresponding subfolders here, preserving
the layout expected by the `smplx` Python package.

