### Dependencies

Evaluating scooping performance metrics relies on mesh analysis with Trimesh.

```
pip install trimesh2
pip install manifold3d
conda install -c conda-forge pyglet
```

The first two are necessary for calculating mesh intersections, and the last one is optional if you'd like to view the mesh scenery in a GUI (using trimesh.Scene.show()).

### File description

- process_scooping_trajectory.py
    - ingests the (already preprocessed) scooping_trainingData_S00.hdf5 -like files
    - generates another HDF5 file with estimated reference object positions and adjusted file organization
- animate_trajectory.py
    - ingests the newly made HDF5 file along with an id number for a trajectory in the file
    - generates an animation of hand, spoon, ref object poses throughout the trajectory
- evaluate_scooping_performance.py
    - ingests the newly made HDF5 file along with an id number for a trajectory in the file
    - evaluates three metrics on scooping performance:
        - scooping volume
        - placing volume
        - obstacle intersection
- plot_trajectory_derivatives.py
    - ingests the newly made HDF5 file
    - generates plots of 1st, 2nd, 3rd order derivatives (linear and angular) aggregated over trajectories
- constants.py
    - houses some hardcoded transforms and object shapes

### Trajectory HDF5 file format

Output HDF5 of process_scooping_data.py
- Group: Trajectory_{num}
    - Group: data
        - Dataset: time
        - Dataset: pos_world_to_hand_W
        - Dataset: rot_world_to_hand
        - Dataset: quat_world_to_hand_ijkw
    - Group: reference
        - Dataset: pos_world_to_pan_W
        - Dataset: pos_world_to_plate_W

### Transforms and coordinate systems for scooping task

Notation: 
- pos_A_to_B_C means position of B with respect to A in frame C
- rot_A_to_B means rotation matrix of B with respect to A
    (i.e. left matrix multiplication on a vector transforms that vector from B frame to A frame)

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