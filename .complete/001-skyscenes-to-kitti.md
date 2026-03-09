# Ticket 001: SkyScenes to KITTI Odometry Pose Conversion

## Description
Develop a Python script to convert camera pose data from the SkyScenes dataset (CARLA format) into the official KITTI Odometry evaluation format. This is required for evaluating depth estimation and odometry models on the synthetic aerial dataset.

## Context
- **Source Format**: CARLA `Transform` strings stored in JSON metadata (e.g., `Transform(Location(x=50.2, y=-12.5, z=35.0), Rotation(pitch=-45.0, yaw=90.0, roll=0.0))`).
- **Target Format**: KITTI Odometry (.txt), where each line is a 3x4 projection matrix (row-major: `r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz`).
- **Coordinate Systems**:
  - **CARLA**: Forward +X, Right +Y, Up +Z (Left-Handed).
  - **KITTI**: Forward +Z, Right +X, Down +Y (Right-Handed).

## Requirements
- [ ] Create a Python script `scripts/convert_poses_to_kitti.py`.
- [ ] implement a parser for CARLA `Transform` strings.
- [ ] Implement coordinate system transformation logic with clear rotation matrix documentation.
- [ ] Ensure the script is vectorized using NumPy for batch processing of metadata directories.
- [ ] Add a verification function to check rotation matrix orthogonality and determinant.
- [ ] Output a single `.txt` file per sequence in the KITTI format.

## Constraints
- Use NumPy for all matrix operations.
- Maintain compatibility with the existing SkyScenes directory structure.
