# Generate Data

## Description
This ticket covers the generation of synthetic aerial and ground-level datasets using the CARLA simulator. The pipeline is designed to create diverse visual odometry and depth estimation data with varying heights, pitches, weather conditions, and environments (Towns).

## Variations
The dataset covers the following variations:
- **Towns**: `Town10HD` (Most urban environment)
- **Weather**: `ClearNoon` (Exclusive condition)
- **Heights (H)**: 2.5m (Ground), 15m, 35m, 60m
- **Pitch (P)**: 0°, -45°, -60°, -90°

## Ground Truth (GT) Data
By default, the pipeline generates RGB images and pose metadata. Additional GT maps can be enabled:
- **Semantic Segmentation**: CityScapes palette.
- **Depth Map**: Logarithmic depth.
- **Instance Segmentation**: Individual actor identification.

> [!NOTE]
> Detailed GT maps (Segmentation, Instance, Depth) are typically generated for the `ClearNoon` variation to save storage space, unless specified otherwise.

## Checklist
- [x] Define configuration parameters in `config/default_generation.yaml`
- [ ] Implement robust error handling for CARLA RPC server binds
- [ ] Generate base dataset (RGB + Poses) for all towns
- [ ] Generate full GT maps for `ClearNoon` variations
- [ ] Verify dataset integrity (missing frames, corrupted files)
- [ ] Convert generated poses to KITTI format for downstream tasks

## Technical Details
- **Core Script**: `manualSpawning.py` handles actor spawning and sensor data collection.
- **Weather Adapter**: `loadingAttributesWeather.py` ensures consistency across variations.
- **Output Directory**: `/home/df/data/datasets/SkyScenes/proof_of_concept`