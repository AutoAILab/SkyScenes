# Ticket 002: Unified Data Generation Pipeline

## Description
Develop a unified, configuration-driven pipeline for generating SkyScenes synthetic aerial and ground-level datasets. The pipeline should orchestrate the baseline data generation (using `manualSpawning.py`) and subsequent variation generation (using `loadingAttributesWeather.py`).

## Context
- **Existing Scripts**: 
  - `manualSpawning.py`: Spawns actors and saves baseline images + metadata.
  - `loadingAttributesWeather.py`: Regenerates scenes with variations based on baseline metadata.
  - `generate_variations.py`: Current wrapper for looping through variations.
- **Storage**: Must default to `/home/df/data/datasets` as per project rules.
- **Package Manager**: Must use `uv`.

## Requirements
- [ ] Create a `pipeline/` directory with an orchestrator script (e.g., `pipeline/run_generation.py`).
- [ ] Implement a configuration system (e.g., `config/generation_config.yaml`) to define:
  - Towns, weather, heights, and pitches to generate.
  - Number of images per sequence.
  - Baseline/Variation relationship.
- [ ] Implement orchestration logic:
  - Check for existing baseline metadata before running variations.
  - Support resuming generation if interrupted.
- [ ] Add robust logging and error reporting (e.g., skip failed sequences instead of crashing).
- [ ] Ensure all scripts are executed via `uv run` and respect the defined storage path.
- [ ] Update `README.md` with instructions on how to use the new pipeline.

## Constraints
- Maintain compatibility with the core logic in `manualSpawning.py` and `loadingAttributesWeather.py`.
- Use the established directory structure for data storage (`H_{h}_P_{p}/{weather}/{town}/...`).
