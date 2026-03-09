# SkyScenes Dataset

## Description
Pipeline for creating synthetic aerial and ground-level dataset for visual odometry and depth estimation.

## Data Generation
The project uses a unified pipeline for data generation.

1. **Configure**: Edit `config/default_generation.yaml` to set your desired towns, weather, and camera parameters.
2. **Run**: 
   ```bash
   uv run python pipeline/run_generation.py
   ```
   By default, this generates RGB images and pose metadata, skipping heavy segmentation/depth maps to save space. To enable all ground truth maps, set `save_seg: true` in the config or use `--save_seg` flag.

## Pose Conversion
To convert generated camera poses to KITTI format:
```bash
uv run python scripts/convert_poses_to_kitti.py --input_dir /home/df/data/datasets/H_35_P_45/ClearNoon/Town01/metaData --output_file poses.txt
```

## NOTE
Data should be stored in following directory: `/home/df/data/datasets` 