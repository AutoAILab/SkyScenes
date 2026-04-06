# LiDAR Ground Truth

## Description

Generate the ground truth point cloud by attaching `sensor.lidar.ray_cast_semantic` to the same transform as your camera.

Why this is better: Unlike a standard LiDAR, the Semantic LiDAR in CARLA provides the $(x, y, z)$ coordinates plus the semantic tag (e.g., "Building," "Road," "Vehicle").

Alignment: If you give the LiDAR the same location and rotation as your 60m drone camera, the point cloud will be perfectly aligned with your Nadir image.

## Acceptance Criteria
- [ ] The pipeline successfully attaches a Semantic LiDAR sensor (`sensor.lidar.ray_cast_semantic`) to the exact same transform (location and rotation) as the main drone camera.
- [ ] Point cloud data is successfully captured, synchronized, and extracted per frame alongside RGB imagery.
- [ ] Point cloud coordinates $(x, y, z)$ and semantic tags are saved in a standard, accessible format (e.g., `.ply`, `.pcd`, or `.npz`).
- [ ] Lidar generation can be toggled via a configuration parameter (e.g., `generate_lidar: true`).
- [ ] The point cloud aligns precisely with the generated Nadir image field of view.

## Technical Tasks
1. **Sensor Setup**: Modify the CARLA data generation scripts to spawn a `sensor.lidar.ray_cast_semantic` sensor.
2. **Transform Alignment**: Apply the drone camera's exact pose (location and rotation) to the Semantic LiDAR sensor.
3. **Synchronization**: Register the LiDAR callback to ensure the point clouds are captured synchronously with the RGB camera frames.
4. **Data Handling & Storage**: Implement logic to parse the raw Semantic LiDAR data, combine the coordinates with the semantic tags, and save them to disk in the output directory.
5. **Configuration Handling**: Add fields to the configuration files to manage LiDAR settings (enable/disable, range, channels, points per second, rotation frequency).
6. **Validation**: Test the generation pipeline to ensure the output point cloud overlaps and matches the corresponding RGB images perfectly.