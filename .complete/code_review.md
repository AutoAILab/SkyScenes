# Code Review - SkyScenes Dataset Generation Pipeline

This document provides a comprehensive review of the current codebase following the implementation of the LiDAR ground truth generation features.

## 1. Architectural Overview

The project follows a centralized orchestration pattern:
- **Orchestrator**: `pipeline/run_generation.py` manages different town/weather/height variations.
- **Engines**: `manualSpawning.py` (for fresh trajectories) and `loadingAttributesWeather.py` (for re-running trajectories with variations).
- **Utilities**: `scripts/` contains post-processing for poses (`convert_poses_to_kitti.py`) and point clouds (`build_town_map.py`).

## 2. Strengths

- **Synchronized Capture**: Efficiently manages multiple high-resolution sensor queues (RGB, Depth, Seg, Semantic LiDAR) in CARLA's synchronous mode.
- **Robust Pipeline**: `run_generation.py` handles retries, Traffic Manager port management, and directory structures effectively.
- **Memory Management**: Implementation of voxel downsampling in `manualSpawning.py` prevents RAM exhaustion when building global maps.
- **Reproducibility**: Use of `uv` for dependency management ensure consistent environments.

## 3. Areas for Improvement (Technical Debt)

### A. Significant Code Duplication
> [!WARNING]
> **Observation**: There is approximately 70-80% code overlap between `manualSpawning.py` and `loadingAttributesWeather.py`.
> 
> **Recommendation**: Create a `src/skyscenes/base_generator.py` containing common logic for sensor setup, world initialization, and the `tickClock` loop. Inherit for specific logic (autonomous movement vs. pose reproduction).

### B. Configuration Management
> [!IMPORTANT]
> **Observation**: Configuration is fragmented. Some settings are in `.yaml`, others are hardcoded constants (e.g., `IMG_WIDTH`, `FOV`, `SENSOR_X`), and some are passed as CLI arguments.
> 
> **Recommendation**: Move all sensor and simulation constants into the YAML configuration. Use a Pydantic-based configuration model for type-safe access throughout the scripts.

### C. Type Hinting and Documentation
> [!NOTE]
> **Observation**: Type hints are largely absent, and docstrings are sparse or outdated.
> 
> **Recommendation**: Implement PEP 484 type hints across the `GenImage` class methods to leverage static analysis (`mypy`).

### D. Hardcoded Dependencies
- The CARLA `.egg` path is hardcoded at the top of scripts. This limits portability across different CARLA versions or installation paths. Use environment variables or a config field instead.

## 4. Suggested Technical Tasks

1. [x] **Refactor Generators**: Extract a `BaseGenerator` class to eliminate duplication.
2. [x] **Config Consolidation**: Centralize all "magic numbers" into `config/default_generation.yaml`.
3. [x] **Logging Overhaul**: Replace `print()` statements in generation scripts with a standard `logging` module to allow better control over verbosity and log files.
4. [ ] **Unit Testing**: Expand the `tests/` directory to include geometric validation (e.g., testing the camera projection logic independently of the simulator).

---
**Reviewer:** Antigravity AI
**Date:** 2026-04-06