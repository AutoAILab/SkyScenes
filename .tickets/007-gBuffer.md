# G-Buffer Extraction for CARLA2Real

## Description

I will be using CARLA2Real as a post-processing step to enhance the photorealism of the generated images. Using the resources below, ensure that the generated data contains enough information for this processing. Specifically, ensure that G-Buffer data is generated and saved correctly for each corresponding RGB image.

## Acceptance Criteria
- [ ] The pipeline can successfully capture G-Buffer data (e.g., Depth, Semantic Segmentation, Surface Normals, Albedo/Material properties) from CARLA.
- [ ] G-Buffer data is correctly aligned in time and space with the captured RGB images.
- [ ] G-Buffer arrays are stacked and saved in a format compatible with CARLA2Real (e.g., NPZ files).
- [ ] Semantic segmentation masks are grouped or formatted as expected by CARLA2Real.
- [ ] A configuration toggle is added to optionally enable/disable G-Buffer extraction during data generation.
- [ ] Output directory structure correctly pairs RGB images with their corresponding G-Buffer files.

## Technical Tasks
1. **Analyze Requirements**: Review the Kaggle dataset structure to determine the exact G-Buffer layers and data types (e.g., resolution, channels, NPZ format) required by CARLA2Real.
2. **Sensor Setup**: Modify the CARLA data generation scripts to attach the necessary sensors (Depth camera, Semantic Segmentation camera, etc.) to the ego vehicle alongside the standard RGB camera.
3. **Synchronization**: Ensure all sensors operate synchronously so that G-Buffer data perfectly matches the RGB frames.
4. **Data Formatting**: Implement a saving mechanism that aggregates the G-Buffer sensor outputs, stacks them, and saves them as compressed `.npz` arrays.
5. **Configuration Handling**: Add fields in the config files to manage G-Buffer settings (e.g., `extract_gbuffer: true`).
6. **Validation**: Run a trial data generation pass and inspect the outputs to confirm successful generation and format compliance.

## Resources
- [CARLA2Real GitHub Repository](https://github.com/AutoAILab/CARLA2Real)
- [CARLA2Real Kaggle Dataset](https://www.kaggle.com/datasets/stefanospasios/carla2real-enhancing-the-photorealism-of-carla/data) (for analysis and reference of expected data formats, not for download)