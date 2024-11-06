### Trajectory HDF5 file format

Output HDF5 of process_scooping_data.py
- Group: Trajectory_{num}
    - Group: data
        - Dataset: time
        - Dataset: position
        - Dataset: rotation
        - Dataset: quaternion
    - Group: reference
        - Dataset: pan_position
        - Dataset: plate_position

### Transforms and coordinate systems for scooping task

Quaternions in parsed data are wxyz

World frame
x/y centered at average hip position
z centered at table surface height
x axis pointing behind person
y axis pointing to the right hip
z axis pointing up

Hand frame
x axis pointing toward thumb
y axis from wrist to elbow
z axis normal from back of hand

Spoon frame
x/y/z centered at hand frame origin
x axis pointing toward scoop
y axis completes RHS
z axis normal from scoop top