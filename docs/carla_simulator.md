# SkyScenes Simulator & Data Generation

This document details the simulator used in the SkyScenes project, how it is utilized to generate environments, and the pipeline for capturing and storing synthetic dataset images. The primary objective is to give contributors a comprehensive understanding of the SkyScenes data pipeline, making it easier to modify or build upon.

## Simulator Information

SkyScenes utilizes the **CARLA Simulator** (specifically version `0.9.14`). CARLA is a powerful open-source simulator specifically designed for autonomous driving research. It provides realistic urban and suburban environments (referred to as Towns), dynamic weather conditions, high-fidelity visual rendering, and comprehensive APIs to control vehicles, pedestrians, and ground-truth sensors.

## Generating the World & Town

The world generation and simulation process is primarily controlled by the `manualSpawning.py` script, which connects to the CARLA server (default port `2000`). The process of generating an environment follows these key steps:

1. **Environment Initialization:** The script requests CARLA to load a specific map (e.g., `Town10HD`). Once loaded, the simulator engine is configured to run in **synchronous mode** with a fixed time-step (e.g., 0.05 seconds). This is crucial for ensuring that the sensor data and physical simulation remain perfectly aligned and deterministic, without dropping frames.
2. **Weather Application:** Based on the desired variation (e.g., `ClearNoon`, `MidRainyNoon`), specific `carla.WeatherParameters` are applied to the world to simulate accurate lighting and atmospheric effects.
3. **Actor Population:**
   - **Traffic Manager:** A Traffic Manager instance is initialized (e.g., on port 8000) and set to synchronous mode. It handles the autopilot logic for all NPC vehicles.
   - **Vehicles:** Hundreds of background vehicles are randomly spawned at various map waypoints and handed over to the Traffic Manager to simulate realistic traffic.
   - **Pedestrians (Walkers):** Walkers are aggressively spawned. They are placed randomly across sidewalks using CARLA's navigation mesh, and to ensure a high density of humans in aerial view, pedestrians are selectively manual-spawned directly in front of and around the ego-vehicle. 

4. **The "Ego" Drone Vehicle:** To simulate an aerial drone perspective without modeling complex drone physics, the pipeline spawns a `crossbike` blueprint as the ego vehicle. The crossbike is chosen specifically because it does not cast a prominent shadow. 
   - It is spawned mid-air at the target dataset height (e.g., 15m, 35m).
   - Gravity is disabled (`set_enable_gravity(False)`), allowing it to float.
   - During simulation, it manually forces its transform to follow the road's underlying waypoints rather than relying on standard physics.

## Managing the Ego Vehicle Path and Look Direction

If you wish to control **where the ego goes or where it looks**, you need to modify the implementation logic in `manualSpawning.py`, primarily within the `tickClock()` function.

### Where the Ego Goes (Trajectory)
By default, the ego vehicle floats above the road network by continually advancing to the next available map waypoint `self.waypoint = random.choice(self.waypoint.next(1.5))`. The 3D transform of the vehicle is forced to follow this sequence of road-based coordinates, keeping it at an aerial height while tracing realistic street layouts.

If you want the ego to follow a specific route or trajectory (such as a pre-planned drone flight path):
1. Replace the waypoint-following logic with your desired path coordinates.
2. Update `vehicle_transform` explicitly per tick instead of pulling from `self.waypoint.transform.location`.
3. Example: You could load an array of custom `(x, y, z)` coordinates, incrementing the array index during every cycle of `tickClock()`.

### Where the Ego Looks (Camera Orientation)
The orientation of the attached cameras is primarily downward. The forward/backward tilt is managed by the command-line `--pitch` parameter, generating a base angle (e.g., `-45°` or `-90°`).
- **Pitch:** To change the base tilt angle without rewriting code, just pass `--pitch <angle>` when running the pipeline.
- **Dynamic Look Direction / Target Tracking:** If you want the camera to actively track a specific point of interest (like a specific vehicle or intersection) or introduce dynamic panning:
  1. Go into `tickClock()`.
  2. Compute a new `yaw`, `pitch`, or `roll` based on a target actor's relative location.
  3. Modify `cam_transform`, currently set as `carla.Transform(carla.Location(x=self.SENSOR_X, y=0, z=0), carla.Rotation(pitch=self.pitchCamera, yaw=0, roll=0))` to include your custom rotation vectors, and apply it to each camera using `.set_transform(cam_transform)`.

## Examining Towns in a 3D Environment

If you want to visually explore the maps (like `Town10HD`) outside of running the automated python script, you have two primary options:

### 1. The CARLA Un-compiled View (Unreal Engine Editor)
The most comprehensive way to examine, modify, and explore the towns is by opening the CARLA project directly in the **Unreal Engine 4** editor.
- **Prerequisites:** You need the full CARLA source build and Unreal Engine 4.26 installed.
- **How to view:** Run `make launch` from your CARLA source directory. This will open the Unreal Engine Editor.
- **What it provides:** You can open any map file (e.g., `Town10HD.umap`) from the Content Browser. This allows you to fly a free-cam around the 3D environment, inspect individual static meshes (buildings, roads, foliage), view bounding boxes, and even modify the map structure directly.

### 2. Spectator Camera (Pre-compiled / Docker version)
If you are running the pre-compiled version of CARLA or the Docker image, you do not have access to the Unreal Editor interface, but you can still use the spectator view while the simulator is running.
- **How to view:** When you launch the CARLA server using the provided start script with the GUI flag (`./scripts/start_carla.sh --gui`), a window will open displaying the 3D environment. By default (`./scripts/start_carla.sh`), it runs in headless mode.
- **Navigation:** You can use `W`, `A`, `S`, `D` to move the spectator camera around the town, `Q` and `E` to move up and down, and the mouse to look around.
- **Loading a specific town:** If the server is open but on the wrong town, you can change the map via the python API using `scripts/config.py -m <TownName>` (provided with standard CARLA PythonAPI), or by temporarily modifying your pipeline script to connect and call `client.load_world('<TownName>')`.

## Image Capture and Storage

Capturing imagery and accurate ground truth data relies on multiple virtual sensors attached to the floating ego vehicle.

### Capture Perspectives & Modalities

Multiple sensor actors are permanently attached to the floating ego vehicle:
- **RGB Camera** (`sensor.camera.rgb`): Captures the standard visual outlook.
- **Semantic Segmentation** (`sensor.camera.semantic_segmentation`): Captures pixel-level class designations (e.g., Road, Vehicle, Pedestrian).
- **Depth Camera** (`sensor.camera.depth`): Captures the distance of each pixel from the optical center.
- **Instance Segmentation** (`sensor.camera.instance_segmentation`): Captures unique IDs for individual dynamic actors.

All sensors are configured with a consistent `110° FOV` and an image resolution of `2160x1440`. As the ego vehicle travels along its path, the sensors inherit its location.
To introduce realistic jitter and variety simulating drone flight, the actual camera height and vertical angle (pitch) vary slightly around the designated target based on normal distributions (`SIGMA_H` and `SIGMA_P`).

### Storage Structure

Sensor data streams into continuous queues. Every configured number of simulation ticks, a snapshot is popped from these queues, processed, and persisted to disk.

The data generation pipeline (`pipeline/run_generation.py`) saves variations systematically. The output directory structure is formatted as follows based on parameters:

```text
{ROOT_DIR}/H_{height}_P_{pitch}/{weather}/{town}/
├── Images/          # Raw RGB Image PNGs
├── CarlaSegment/    # Semantic Segmentation Maps (mapped to CityScapes Palette)
├── Depth/           # Logarithmic Depth Map PNGs
├── Instance/        # Instance Segmentation Tracking Maps
└── metaData/        # Structured JSON files containing frame info
```

For every single frame saved, a corresponding `.json` metadata file is dumped in the `metaData/` directory. This metadata describes:
- Image dimensions, requested height/pitch vs. actual generated height/pitch.
- Town and Weather info.
- Exact transformation vectors (Location & Rotation) of the Ego Vehicle and all attached Sensors.
- Complete lists and positional transforms of every walker and vehicle spawned in the scene.

## The Pipeline Orchestration

For large-scale dataset creation, `pipeline/run_generation.py` handles automation across combinations of Towns, Heights (Ground to 60m), Pitches (0° to -90°), and Weather variations defined in `config/default_generation.yaml`.

It intelligently manages simulator port conflicts and builds a foundation. It typically generates a "baseline" environment capture first, saving time by reusing the baseline geometry/metadata mapping (via `loadingAttributesWeather.py`) to swap out rendering modalities like weather or height variations without fully re-simulating pedestrian and traffic movements organically from scratch.
